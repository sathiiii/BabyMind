
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from multimodal.attention_maps import Hook
from multimodal.beam_search import beam_search
from multimodal.multimodal_data_module import EOS_TOKEN_ID, MAX_LEN_UTTERANCE, PAD_TOKEN_ID, SOS_TOKEN_ID
from multimodal.utils import apply_permutation, get_entropy, load_model, map_structure

# -----------------------------------------------------------------------------
# Defaults / constants
# -----------------------------------------------------------------------------

TEXT_ENCODER = "embedding"
ATTENTION_ACTIVATION = "relu"
EMBEDDING_TYPE = "flat"
EMBEDDING_DIM = 128
CRANGE = 1
DROPOUT_I = 0.0
DROPOUT_O = 0.0

PRETRAINED_CNN = True
FINETUNE_CNN = False

NORMALIZE_FEATURES = False
SIM = "max"
TEMPERATURE = 0.07
FIX_TEMPERATURE = False

# vision encoder arguments
CNN_MODEL = "models/TC-S-resnext.tar"  # link to TC resnext model
CNN_DINO = False  # boolean flag to use DINO resnext model
VIT_DINO = False  # boolean flag to use DINO vision transformer model

# text encoder arguments
POS_EMBED_TYPE = "no_pos_embed"


def set_parameter_requires_grad(model: nn.Module, feature_extracting: bool = True) -> None:
    """Freeze all parameters in a module when feature_extracting is True."""
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# -----------------------------------------------------------------------------
# DDP helpers for global contrastive learning (variable batch sizes supported)
# -----------------------------------------------------------------------------

def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _gather_batch_sizes(local_bs: int, device: torch.device) -> torch.Tensor:
    """Gather per-rank batch sizes into a (world_size,) int64 tensor on device."""
    if not _dist_is_initialized():
        return torch.tensor([local_bs], device=device, dtype=torch.long)

    world_size = dist.get_world_size()
    bs = torch.tensor([local_bs], device=device, dtype=torch.long)
    out = [torch.zeros_like(bs) for _ in range(world_size)]
    dist.all_gather(out, bs)
    return torch.cat(out, dim=0).view(-1)  # (world_size,)


class _AllGatherWithPad(torch.autograd.Function):
    """All-gather a tensor with variable first-dimension across ranks, with grad support.

    Forward:
        x: (N_local, ...) on each rank
        sizes: (world_size,) first-dim sizes for each rank

    Returns:
        x_all: (sum(sizes), ...) concatenated in rank order.

    Backward:
        all-reduce grad over x_all then slice out this rank's gradient.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if not _dist_is_initialized():
            ctx.world_size = 1
            ctx.rank = 0
            ctx.offset = 0
            ctx.local_n = int(x.size(0))
            return x

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        sizes_list = sizes.detach().cpu().tolist()
        max_n = int(max(sizes_list)) if sizes_list else int(x.size(0))

        pad_shape = (max_n,) + tuple(x.shape[1:])
        x_pad = x.new_zeros(pad_shape)
        if x.size(0) > 0:
            x_pad[: x.size(0)] = x

        gathered = [x.new_zeros(pad_shape) for _ in range(world_size)]
        dist.all_gather(gathered, x_pad)

        outs: List[torch.Tensor] = []
        for r, g in enumerate(gathered):
            n_r = int(sizes_list[r])
            if n_r > 0:
                outs.append(g[:n_r])

        x_all = torch.cat(outs, dim=0) if outs else x.new_zeros((0,) + tuple(x.shape[1:]))

        offset = int(sum(sizes_list[:rank])) if rank > 0 else 0
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.offset = offset
        ctx.local_n = int(x.size(0))
        return x_all

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        if getattr(ctx, "world_size", 1) == 1:
            return grad_output, None

        # Sum gradients across ranks so every rank receives the gradient
        # contributions from all other ranks' losses.
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)

        offset = int(ctx.offset)
        local_n = int(ctx.local_n)
        grad_input = grad_output[offset: offset + local_n].contiguous()
        return grad_input, None


class LockedDropout(nn.Module):
    """Locked dropout that applies the same dropout mask across the time dimension."""

    def forward(self, x: torch.Tensor, dropout: float, dim: int = 1) -> torch.Tensor:
        if not (self.training and dropout):
            return x
        mask_shape = x.shape[:dim] + (1,) + x.shape[dim + 1:]
        mask = x.new_empty(mask_shape).bernoulli_(1 - dropout) / (1 - dropout)
        return mask * x


class VisionEncoder(nn.Module):
    """Visual encoder with optional neuro-symbolic concept head."""

    def __init__(self, args: Any):
        super().__init__()
        self.args: Dict[str, Any] = vars(args) if args is not None else {}

        self.embedding_type: str = self.args.get("embedding_type", EMBEDDING_TYPE)
        self.embedding_dim: int = int(self.args.get("embedding_dim", EMBEDDING_DIM))
        self.pretrained_cnn: bool = bool(self.args.get("pretrained_cnn", PRETRAINED_CNN))

        self.cnn_model: str = str(self.args.get("cnn_model", CNN_MODEL))
        self.cnn_dino: bool = bool(self.args.get("cnn_dino", CNN_DINO))
        self.vit_dino: bool = bool(self.args.get("vit_dino", VIT_DINO))
        self.finetune_cnn: bool = bool(self.args.get("finetune_cnn", FINETUNE_CNN))

        self.model = self._load_pretrained_cnn()

        # Predicts C concept logits from a (B, E) pooled image feature.
        self.num_concepts: int = int(self.args.get("num_concepts", 22))

        # Isolate RNG so adding the optional head does not perturb other modules' init.
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        self.concept_head: Optional[nn.Linear] = None
        if bool(self.args.get("neurosym", False)):
            self.concept_head = nn.Linear(self.embedding_dim, self.num_concepts, bias=True)
            torch.nn.init.xavier_uniform_(self.concept_head.weight, gain=0.01)
            torch.nn.init.zeros_(self.concept_head.bias)

        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--pretrained_cnn", action="store_true", help="use pretrained CNN")
        parser.add_argument(
            "--cnn_model",
            type=str,
            default=CNN_MODEL,
            help="name in torchvision.models or the path to the CNN model checkpoint",
        )
        parser.add_argument("--cnn_dino", action="store_true", default=CNN_DINO, help="use DINO resnext model")
        parser.add_argument("--vit_dino", action="store_true", default=VIT_DINO, help="use DINO vision transformer model")
        parser.add_argument("--finetune_cnn", action="store_true", help="finetune CNN (frozen by default)")
        parser.add_argument("--num_concepts", type=int, default=22, help="number of concepts for neuro-symbolic head")

    def concept_logits(self, image_features: torch.Tensor, image_feature_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return concept logits from pooled image features."""
        if self.concept_head is None:
            raise RuntimeError("concept_head is not initialized (neurosym disabled).")
        pooled = image_features.mean(dim=(2, 3)) if image_features.dim() == 4 else image_features
        return self.concept_head(pooled)

    def forward_to_layer3(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,3,H,W)
        m = self.model
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)          # (N,1024,14,14) for 224 input
        return x

    def forward_from_layer3(self, f3: torch.Tensor) -> torch.Tensor:
        # f3: (N,1024,14,14)
        m = self.model
        x = m.layer4(f3)         # (N,2048,7,7)
        x = m.avgpool(x)         # (N,2048,1,1)
        x = torch.flatten(x, 1)  # (N,2048)
        x = m.fc(x)              # (N,embedding_dim)
        return x

    def forward(self, x: torch.Tensor, requires_grad_for_fmap: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the visual backbone.

        Args:
            x: Input images (B, 3, H, W).
            requires_grad_for_fmap:
                If True, marks the hooked conv feature map so that gradients are retained for
                GradCAM-style visualizations. This does not change whether CNN weights are trainable.
        """
        if requires_grad_for_fmap and torch.is_grad_enabled():
            x = x.requires_grad_()

        if self.vit_dino:
            x = self.model(x)
            features = self.model.head(x)  # type: ignore[attr-defined]
            feature_map = None
            return features, feature_map

        # CNN path
        if self.embedding_type == "spatial":
            # Sequential backbone for spatial embedding: penultimate is the last ResNeXt block.
            layer = self.model[-2]  # type: ignore[index], dim: 1024
        else:
            # layer = self.model.layer4  # type: ignore[attr-defined], dim: 20248
            layer = self.model.layer3

        with Hook(layer, requires_grad=requires_grad_for_fmap) as hook:
            features = self.model(x)
            feature_map = hook.activation

        return features, feature_map

    def _forward_unbatched(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i in x:
            i = i.unsqueeze(0)
            out = self.model(i).squeeze(0)
            outputs.append(out)
        return torch.stack(outputs, dim=0)

    @property
    def last_cnn_out_dim(self) -> int:
        """Dimension of the last CNN block output (before projection)."""
        return 768 if self.vit_dino else 2048

    def _load_pretrained_cnn(self) -> nn.Module:
        if self.cnn_dino:
            print("Loading DINO resnext model!")
            model_name = "dino_sfp_resnext50"
            model = load_model(model_name, self.pretrained_cnn)
        elif self.vit_dino:
            print("Loading DINO vision transformer model!")
            model_name = "dino_sfp_vitb14"
            model = load_model(model_name, self.pretrained_cnn)
        else:
            model_name = self.cnn_model
            checkpoint_path = None

            if not hasattr(torchvision.models, model_name):
                checkpoint_path = self.cnn_model
                name_to_model_name = {"resnext": "resnext50_32x4d"}
                for name, model_name_ in name_to_model_name.items():
                    if name in model_name:
                        model_name = model_name_
                        break
                else:
                    raise ValueError(f"Unable to recognize the model name of {model_name}")

            model = getattr(torchvision.models, model_name)(
                pretrained=self.pretrained_cnn and not checkpoint_path
            )
            model.fc = nn.Linear(in_features=self.last_cnn_out_dim, out_features=2765, bias=True)

            if self.pretrained_cnn and checkpoint_path:
                print("Loading pretrained CNN!")
                checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
                state_dict = {k.replace("module.", ""): v for k, v in checkpoint["teacher"].items()}
                state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
                state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=False)

        if not self.finetune_cnn:
            print("Freezing CNN layers!")
            set_parameter_requires_grad(model, feature_extracting=True)
        else:
            print("Fine-tuning CNN layers!")

        if self.embedding_type == "spatial":
            # Remove classifier head and add 1x1 conv to map to embedding_dim
            model = nn.Sequential(*list(model.children())[:-2], nn.Conv2d(self.last_cnn_out_dim, self.embedding_dim, 1))
        elif self.embedding_type == "flat":
            print("Adding linear layer to vision encoder!")
            if self.vit_dino:
                model.head = nn.Linear(self.last_cnn_out_dim, self.embedding_dim)  # type: ignore[attr-defined]
            else:
                model.fc = nn.Linear(self.last_cnn_out_dim, self.embedding_dim)

        return model


class Attention(nn.Module):
    """Soft attention over a spatial feature map."""

    def __init__(self, encoder_dim: int, decoder_dim: int, attn_dim: int, activation: str = ATTENTION_ACTIVATION):
        super().__init__()
        self.encoder_projection = nn.Linear(encoder_dim, attn_dim)
        self.decoder_projection = nn.Linear(decoder_dim, attn_dim)
        self.attn_layer = nn.Linear(attn_dim, 1)

        activation_mapping = {"relu": "ReLU", "tanh": "Tanh"}
        self.activation_fn = getattr(nn, activation_mapping[activation])()

    @staticmethod
    def permute(t: torch.Tensor) -> torch.Tensor:
        perm = tuple(range(t.dim()))
        perm = perm[:1] + perm[2:] + perm[1:2]
        return t.permute(*perm)

    @staticmethod
    def unpermute(t: torch.Tensor) -> torch.Tensor:
        perm = tuple(range(t.dim()))
        perm = perm[:1] + perm[-1:] + perm[1:-1]
        return t.permute(*perm)

    def project_encoder_features(self, encoder_features: torch.Tensor) -> torch.Tensor:
        projected = self.unpermute(self.encoder_projection(self.permute(encoder_features)))
        return projected

    def forward(
        self,
        encoder_features: torch.Tensor,
        projected_encoder_features: torch.Tensor,
        decoder_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [N, attn_dim]
        projected_decoder_features = self.decoder_projection(decoder_features)

        # [N, encoder_dim, -1]
        encoder_features_ = encoder_features.reshape(*(encoder_features.shape[:2] + (-1,)))
        # [N, attn_dim, -1]
        projected_encoder_features_ = projected_encoder_features.reshape(*(projected_encoder_features.shape[:2] + (-1,)))
        # [N, attn_dim, 1]
        projected_decoder_features_ = projected_decoder_features.unsqueeze(-1)

        # [N, -1]
        attn_logits_ = self.attn_layer(
            self.permute(self.activation_fn(projected_encoder_features_ + projected_decoder_features_))
        ).squeeze(-1)
        attns_ = F.softmax(attn_logits_, dim=-1)

        # [N, ...]
        attns = attns_.reshape(attns_.size(0), *encoder_features.shape[2:])
        # [N, encoder_dim]
        features = torch.matmul(encoder_features_, attns_.unsqueeze(-1)).squeeze(-1)
        return features, attns


class TextEncoder(nn.Module):
    """Text encoder supporting embedding/CBOW/LSTM/biLSTM/Transformer, with optional attention/captioning."""

    def __init__(self, vocab: Dict[str, int], image_feature_map_dim: int, args: Any):
        super().__init__()
        self.args: Dict[str, Any] = vars(args) if args is not None else {}

        self.text_encoder: str = self.args.get("text_encoder", TEXT_ENCODER)
        self._captioning: bool = bool(self.args.get("captioning", False))
        self._attention: bool = bool(self.args.get("attention", False))
        self._attention_gate: bool = bool(self.args.get("attention_gate", False))

        self.embedding_type: str = self.args.get("embedding_type", EMBEDDING_TYPE)
        self.embedding_dim: int = int(self.args.get("embedding_dim", EMBEDDING_DIM))
        self.hidden_dim: int = self.embedding_dim

        self.input_dim: int = self.embedding_dim + (image_feature_map_dim if self._attention else 0)

        self.crange: int = int(self.args.get("crange", CRANGE))
        self.dropout_i: float = float(self.args.get("dropout_i", DROPOUT_I))
        self.dropout_o: float = float(self.args.get("dropout_o", DROPOUT_O))
        self.pos_embed_type: str = str(self.args.get("pos_embed_type", POS_EMBED_TYPE))

        # vocab + mappings
        self.vocab: Dict[str, int] = vocab
        self.word2idx = self.vocab
        self.idx2word = {idx: word for word, idx in self.vocab.items()}

        # embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=PAD_TOKEN_ID)

        # core encoder
        if self.text_encoder in ["lstm", "bilstm"]:
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, bidirectional=(self.text_encoder == "bilstm"))
        elif self.text_encoder == "transformer":
            print("Building transformer text encoder!")
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=8)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

            if self.pos_embed_type == "sinusoidal":
                print("Initializing sinusoidal positional embeddings!")
                pos_embed = torch.zeros(MAX_LEN_UTTERANCE, self.embedding_dim)
                position = torch.arange(0, MAX_LEN_UTTERANCE).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, self.embedding_dim, 2) * -(math.log(10000.0) / self.embedding_dim))
                pos_embed[:, 0::2] = torch.sin(position * div_term)
                pos_embed[:, 1::2] = torch.cos(position * div_term)
                pos_embed = pos_embed.unsqueeze(0).permute(1, 0, 2)
                self.register_buffer("pos_embed", pos_embed)
            elif self.pos_embed_type == "learned":
                print("Initializing learned positional embeddings!")
                self.pos_embed = nn.Parameter(torch.zeros(MAX_LEN_UTTERANCE, 1, self.embedding_dim))
            else:
                print("Initializing no positional embeddings!")

        # captioning components (LSTM only)
        if self.captioning:
            assert self.regressional, "only regressional text encoder supports captioning"
            self.connector = nn.Linear(self.embedding_dim, 2 * self.lstm.num_layers * self.hidden_dim)  # type: ignore[attr-defined]

        # attention module (LSTM only)
        if self._attention:
            self.attention = Attention(image_feature_map_dim, self.hidden_dim, self.hidden_dim)
            if self.has_attention_gate:
                self.attention_gate_projection = nn.Linear(self.hidden_dim, image_feature_map_dim)

        self.lockdrop = LockedDropout()
        self.output_dropout = nn.Dropout(self.dropout_o)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--text_encoder",
            type=str,
            default=TEXT_ENCODER,
            choices=["embedding", "cbow", "lstm", "bilstm", "transformer"],
            help="text encoder architecture",
        )
        parser.add_argument("--captioning", action="store_true", help="initialize hidden states with image features")
        parser.add_argument("--attention", action="store_true", help="attend to the image feature map")
        parser.add_argument(
            "--attention_activation",
            type=str,
            default=ATTENTION_ACTIVATION,
            choices=["relu", "tanh"],
            help="activation in attention",
        )
        parser.add_argument("--attention_gate", action="store_true", help="use attention gate")
        parser.add_argument("--crange", type=int, default=CRANGE, help="context range for cbow")
        parser.add_argument("--dropout_i", type=float, default=DROPOUT_I, help="input dropout rate")
        parser.add_argument("--dropout_o", type=float, default=DROPOUT_O, help="output dropout rate")
        parser.add_argument(
            "--pos_embed_type",
            type=str,
            default=POS_EMBED_TYPE,
            choices=["no_pos_embed", "sinusoidal", "learned"],
            help="type of positional embedding to use",
        )

    # -----------------------------
    # LSTM stepping utilities
    # -----------------------------
    def inputs_to_outputs(
        self,
        inputs: torch.Tensor,
        states,
        image_feature_map: Optional[torch.Tensor] = None,
        projected_image_feature_map: Optional[torch.Tensor] = None,
    ):
        """Perform one LSTM step from embedded inputs to outputs and states."""
        if image_feature_map is not None:
            h = states[0][-1]
            attn_feature, attns = self.attention(image_feature_map, projected_image_feature_map, h)  # type: ignore[arg-type]
            if self.has_attention_gate:
                gate = torch.sigmoid(self.attention_gate_projection(h))
                attn_feature = gate * attn_feature
            inputs = torch.cat([inputs, attn_feature], dim=1)
        else:
            attns = None

        inputs = inputs.unsqueeze(0)
        outputs, states = self.lstm(inputs, states)  # type: ignore[attr-defined]
        outputs = outputs.squeeze(0)
        return outputs, states, attns

    def ids_to_outputs(
        self,
        ids: torch.Tensor,
        states,
        image_feature_map: Optional[torch.Tensor] = None,
        projected_image_feature_map: Optional[torch.Tensor] = None,
    ):
        """Perform one LSTM step from token ids to outputs and states."""
        inputs = self.embedding(ids)
        return self.inputs_to_outputs(
            inputs,
            states,
            image_feature_map=image_feature_map,
            projected_image_feature_map=projected_image_feature_map,
        )

    def train_greedy(self, inputs: PackedSequence, hidden, image_feature_map: Optional[torch.Tensor] = None):
        """Teacher-forcing training for attention models."""
        data, batch_sizes, sorted_indices, unsorted_indices = inputs
        hidden = self.lstm.permute_hidden(hidden, sorted_indices)  # type: ignore[attr-defined]

        if image_feature_map is not None:
            image_feature_map = apply_permutation(image_feature_map, sorted_indices, 0)
            projected_image_feature_map = self.attention.project_encoder_features(image_feature_map)
            attns_list: List[torch.Tensor] = []

        outputs_list: List[torch.Tensor] = []
        p = 0
        for batch_size in batch_sizes:
            p_ = p + batch_size
            input_batch = data[p:p_]
            hidden_batch = map_structure(lambda t: t[:, :batch_size], hidden)

            if image_feature_map is not None:
                image_feature_map_batch = image_feature_map[:batch_size]
                projected_image_feature_map_batch = projected_image_feature_map[:batch_size]
            else:
                image_feature_map_batch = None
                projected_image_feature_map_batch = None

            outputs_batch, hidden_batch, attn_batch = self.inputs_to_outputs(
                input_batch,
                hidden_batch,
                image_feature_map=image_feature_map_batch,
                projected_image_feature_map=projected_image_feature_map_batch,
            )

            hidden = map_structure(
                lambda h, h_batch: torch.cat((h_batch, h[:, batch_size:]), dim=1),
                hidden,
                hidden_batch,
            )
            outputs_list.append(outputs_batch)
            if image_feature_map is not None:
                attns_list.append(attn_batch)
            p = p_

        outputs = torch.cat(outputs_list, dim=0)
        outputs_packed = PackedSequence(outputs, batch_sizes, sorted_indices, unsorted_indices)

        if image_feature_map is not None:
            attns = torch.cat(attns_list, dim=0)
            attns_packed = PackedSequence(attns, batch_sizes, sorted_indices, unsorted_indices)
        else:
            attns_packed = None

        return outputs_packed, self.lstm.permute_hidden(hidden, unsorted_indices), attns_packed  # type: ignore[attr-defined]

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(
        self,
        x: torch.Tensor,
        x_len: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        image_feature_map: Optional[torch.Tensor] = None,
    ):
        attns = None
        embedding = self.embedding(x)  # (B, L, E)

        if self.text_encoder == "embedding":
            raw_output = embedding  # (B, L, E)
            if self.embedding_type == "flat":
                ret = torch.sum(raw_output, dim=1) / x_len.unsqueeze(1)

        elif self.text_encoder == "cbow":
            assert self.embedding_type != "flat", "cbow with flat embedding is not supported"
            presum = F.pad(embedding, (0, 0, self.crange + 1, self.crange)).cumsum(1)
            raw_output = (presum[:, 2 * self.crange + 1:] - presum[:, : -(2 * self.crange + 1)] - embedding) / (
                2 * self.crange
            )

        elif self.text_encoder in ["lstm", "bilstm"]:
            batch_size = int(x.size(0))
            hidden = self.init_hidden(batch_size, image_features=image_features)

            embedding = self.lockdrop(embedding, self.dropout_i)
            packed = pack_padded_sequence(embedding, x_len.cpu(), batch_first=True, enforce_sorted=False)

            if self.has_attention:
                raw_output, (hidden, _cell), attns_packed = self.train_greedy(packed, hidden, image_feature_map=image_feature_map)
                if attns_packed is not None:
                    attns, _ = pad_packed_sequence(attns_packed, batch_first=True)
            else:
                raw_output, (hidden, _cell) = self.lstm(packed, hidden)  # type: ignore[attr-defined]

            raw_output, _ = pad_packed_sequence(raw_output, batch_first=True)

            if self.text_encoder == "bilstm":
                raw_output_fwd = raw_output[:, :, : self.embedding_dim]
                raw_output_bwd = raw_output[:, :, self.embedding_dim :]
                raw_output = torch.mean(torch.stack([raw_output_fwd, raw_output_bwd], dim=0), dim=0)

            if self.embedding_type == "flat":
                ret = hidden.mean(dim=0)

        elif self.text_encoder == "transformer":
            src_key_padding_mask = (x == PAD_TOKEN_ID).bool()
            emb = embedding.permute(1, 0, 2)  # (L,B,E)

            if self.pos_embed_type in ["sinusoidal", "learned"]:
                pos_embed = self.pos_embed[: emb.size(0), :, :]  # type: ignore[attr-defined]
                emb = emb + pos_embed

            raw_output = self.transformer_encoder(emb, src_key_padding_mask=src_key_padding_mask)  # type: ignore[attr-defined]
            raw_output = raw_output.permute(1, 0, 2)  # (B,L,E)

            if self.embedding_type == "flat":
                ret = torch.sum(raw_output, dim=1) / x_len.unsqueeze(1)

        else:
            raise ValueError(f"Unknown text_encoder: {self.text_encoder}")

        output = self.lockdrop(raw_output, self.dropout_o)

        if self.embedding_type == "flat":
            ret = self.output_dropout(ret)
        elif self.embedding_type == "spatial":
            ret = output
        else:
            raise ValueError(f"Unknown embedding_type: {self.embedding_type}")

        return ret, output, attns

    def _forward_unbatched(self, x: torch.Tensor, x_len: torch.Tensor, image_features: Optional[torch.Tensor] = None):
        if self.text_encoder == "embedding":
            outputs = []
            for i, i_len in zip(x, x_len):
                if self.embedding_type == "flat":
                    out = self.embedding(i)
                    out = torch.sum(out, dim=0)
                    out = out / i_len
                else:
                    out = self.embedding(i)
                outputs.append(out)
            return torch.stack(outputs, dim=0)

        if self.text_encoder in ["lstm", "bilstm"]:
            outputs = []
            max_seq_len = int(torch.max(x_len).item())
            for i, i_len in zip(x, x_len):
                batch_size = 1
                i = i.unsqueeze(0)
                i_len_ = i_len.unsqueeze(0)
                hidden = self.init_hidden(batch_size, image_features=image_features)

                emb = self.embedding(i).transpose(0, 1)
                packed = pack_padded_sequence(emb, i_len_.cpu(), enforce_sorted=False)

                out_packed, (hidden, _cell) = self.lstm(packed, hidden)  # type: ignore[attr-defined]

                if self.embedding_type == "flat":
                    padded_out = hidden.mean(dim=0).squeeze(0)
                else:
                    out, _ = pad_packed_sequence(out_packed)
                    if self.text_encoder == "bilstm":
                        out_fwd = out[:, :, : self.embedding_dim]
                        out_bwd = out[:, :, self.embedding_dim :]
                        out = torch.mean(torch.stack([out_fwd, out_bwd], dim=0), dim=0)
                    out = out.transpose(0, 1).squeeze(0)

                    device = self.embedding.weight.device
                    padded_out = torch.zeros((max_seq_len, self.embedding_dim), device=device)
                    padded_out[: int(i_len_.item()), :] = out

                outputs.append(padded_out)

            return torch.stack(outputs, dim=0)

        raise NotImplementedError("Unbatched forward is only implemented for embedding and (bi)LSTM text encoders.")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def regressional(self) -> bool:
        return self.text_encoder == "lstm"

    # Backward-compat properties
    @property
    def captioning(self) -> bool:
        return getattr(self, "_captioning", False)

    @property
    def has_attention(self) -> bool:
        return getattr(self, "_attention", False)

    @property
    def has_attention_gate(self) -> bool:
        return getattr(self, "_attention_gate", False)

    def init_hidden(self, batch_size: int, image_features: Optional[torch.Tensor] = None):
        d = 2 if self.text_encoder == "bilstm" else 1

        device = image_features.device if image_features is not None else self.embedding.weight.device

        if image_features is not None:
            if image_features.dim() == 4:
                image_features = image_features.mean(dim=(2, 3))
            elif image_features.dim() != 2:
                raise ValueError(f"Unexpected image_features shape: {tuple(image_features.shape)}")

            return (
                self.connector(image_features)  # type: ignore[attr-defined]
                .reshape(image_features.size(0), 2, d * self.lstm.num_layers, self.hidden_dim)  # type: ignore[attr-defined]
                .permute(1, 2, 0, 3)
                .unbind()
            )

        return (
            torch.zeros(d * self.lstm.num_layers, batch_size, self.hidden_dim, device=device),  # type: ignore[attr-defined]
            torch.zeros(d * self.lstm.num_layers, batch_size, self.hidden_dim, device=device),  # type: ignore[attr-defined]
        )


class MultiModalModel(nn.Module):
    """Joint image-text model with DDP-safe contrastive loss."""

    def __init__(self, vision_encoder: nn.Module, text_encoder: nn.Module, args: Any):
        super().__init__()
        self.args: Dict[str, Any] = vars(args) if args is not None else {}

        self.sim: str = self.args.get("sim", SIM)
        self.embedding_type: str = self.args.get("embedding_type", EMBEDDING_TYPE)
        self.normalize_features: bool = bool(self.args.get("normalize_features", NORMALIZE_FEATURES))

        self.initial_temperature: float = float(self.args.get("temperature", TEMPERATURE))
        self.fix_temperature: bool = bool(self.args.get("fix_temperature", FIX_TEMPERATURE))

        self.image_embed = vision_encoder
        self.text_embed = text_encoder

        init = torch.tensor(-np.log(self.initial_temperature), dtype=torch.float32)
        if self.fix_temperature:
            self.register_buffer("logit_neg_log_temperature", init)
        else:
            self.logit_neg_log_temperature = nn.Parameter(init)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--embedding_type",
            type=str,
            default=EMBEDDING_TYPE,
            choices=["spatial", "flat"],
            help="type of embeddings to use (spatial or flat)",
        )
        parser.add_argument("--embedding_dim", type=int, default=EMBEDDING_DIM, help="embedding dimension")
        parser.add_argument("--normalize_features", action="store_true", help="L2-normalize embeddings after encoding")
        parser.add_argument("--sim", type=str, default=SIM, choices=["mean", "max"], help="matchmap similarity reduction")
        parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="initial temperature")
        parser.add_argument("--fix_temperature", action="store_true", help="fix temperature so it is not trained")

    def encode_image(self, image: torch.Tensor):
        image_features, image_feature_map = self.image_embed(image)
        if self.normalize_features:
            # normalize image features
            image_features = F.normalize(image_features, p=2, dim=1)
        return image_features, image_feature_map

    def encode_text(self, text: torch.Tensor, text_length: torch.Tensor):
        text_features, text_outputs, _attns = self.text_embed(text, text_length)
        if self.normalize_features:
            # normalize text features
            text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features, text_outputs

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        text_length: torch.Tensor,
        return_image_features: bool = False,
        return_text_outputs: bool = False,
    ):
        image_features, image_feature_map = self.encode_image(image)
        text_features, text_outputs = self.encode_text(text, text_length)

        if self.embedding_type == "flat":
            # image_features: (B, E), text_features: (B, E)

            # calculate match similarity
            match = image_features @ text_features.T

        elif self.embedding_type == "spatial":
            # image_features: (B, E, H, W), text_features: (B, L, E)

            # calculate batched similarity
            if self.sim == "mean":
                match_sum = torch.einsum("iehw,tle->it", [image_features, text_features])
                denom = image_features.size(-2) * image_features.size(-1) * text_length
                match = match_sum / denom.clamp(min=1.0)
            elif self.sim == "max":
                match_max = torch.einsum("iehw,tle->itlhw", [image_features, text_features])
                match_max = torch.amax(match_max, dim=(3, 4))
                match = torch.sum(match_max, dim=2) / text_length.clamp(min=1.0)
            else:
                raise ValueError(f"Unknown sim: {self.sim}")
        else:
            raise ValueError(f"Unknown embedding_type: {self.embedding_type}")

        logit_scale = self.logit_neg_log_temperature.exp()
        logits_per_image = match * logit_scale
        logits_per_text = match.t() * logit_scale

        ret = (logits_per_image, logits_per_text)
        if return_image_features:
            ret = ret + (image_features, image_feature_map)
        if return_text_outputs:
            ret = ret + (text_outputs,)
        return ret

    def calculate_contrastive_loss(self, x: torch.Tensor, y: torch.Tensor, y_len: torch.Tensor):
        """DDP-safe InfoNCE loss.

        Under DDP, we:
          * all-gather image/text features across ranks (with gradient support)
          * compute logits for local anchors against the global bank
          * use ground-truth indices that account for per-rank batch offsets
        """
        image_features, image_feature_map = self.encode_image(x)
        text_features, text_outputs = self.encode_text(y, y_len)

        device = image_features.device
        local_bs = int(x.size(0))

        sizes = _gather_batch_sizes(local_bs, device=device)
        if _dist_is_initialized():
            rank = dist.get_rank()
            sizes_list = sizes.detach().cpu().tolist()
            offset = int(sum(sizes_list[:rank])) if rank > 0 else 0
            global_bs = int(sum(sizes_list))
        else:
            offset = 0
            global_bs = local_bs

        all_image_features = _AllGatherWithPad.apply(image_features, sizes)
        all_text_features = _AllGatherWithPad.apply(text_features, sizes)

        all_y_len = None
        if self.embedding_type == "spatial":
            all_y_len = _AllGatherWithPad.apply(y_len.view(-1), sizes)

        # Similarity for local anchors against global bank
        if self.embedding_type == "flat":
            match_i2t = image_features @ all_text_features.T
            match_t2i = text_features @ all_image_features.T

        elif self.embedding_type == "spatial":
            if all_y_len is None:
                raise RuntimeError("all_y_len is required for spatial similarity.")

            if self.sim == "mean":
                match_sum_i2t = torch.einsum("iehw,tle->it", [image_features, all_text_features])
                denom_i2t = image_features.size(-2) * image_features.size(-1) * all_y_len.to(match_sum_i2t.dtype).view(1, -1)
                match_i2t = match_sum_i2t / denom_i2t.clamp(min=1.0)

                match_sum_g2l = torch.einsum("iehw,tle->it", [all_image_features, text_features])
                denom_g2l = all_image_features.size(-2) * all_image_features.size(-1) * y_len.to(match_sum_g2l.dtype).view(1, -1)
                match_t2i = (match_sum_g2l / denom_g2l.clamp(min=1.0)).T

            elif self.sim == "max":
                match_max_i2t = torch.einsum("iehw,tle->itlhw", [image_features, all_text_features])
                match_max_i2t = torch.amax(match_max_i2t, dim=(3, 4))
                match_i2t = torch.sum(match_max_i2t, dim=2) / all_y_len.to(match_max_i2t.dtype).view(1, -1).clamp(min=1.0)

                match_max_g2l = torch.einsum("iehw,tle->itlhw", [all_image_features, text_features])
                match_max_g2l = torch.amax(match_max_g2l, dim=(3, 4))
                match_t2i = (torch.sum(match_max_g2l, dim=2) / y_len.to(match_max_g2l.dtype).view(1, -1).clamp(min=1.0)).T

            else:
                raise ValueError(f"Unknown sim: {self.sim}")

        else:
            raise ValueError(f"Unknown embedding_type: {self.embedding_type}")

        logit_scale = self.logit_neg_log_temperature.exp()
        logits_per_image = match_i2t * logit_scale
        logits_per_text = match_t2i * logit_scale

        # Empty local batch safety
        if local_bs == 0:
            # Tie gradients to encoder outputs so DDP doesn't treat params as unused.
            infonce_loss = (image_features.sum() + text_features.sum() + logits_per_image.sum() + logits_per_text.sum()) * 0.0
            image_accuracy = torch.zeros((), device=device)
            text_accuracy = torch.zeros((), device=device)
            image_entropy = torch.zeros((), device=device)
            text_entropy = torch.zeros((), device=device)
            return (
                infonce_loss,
                image_accuracy,
                text_accuracy,
                image_entropy,
                text_entropy,
                logits_per_image,
                logits_per_text,
                image_features,
                image_feature_map,
                text_outputs,
            )

        ground_truth = torch.arange(local_bs, device=device, dtype=torch.long) + offset

        infonce_loss = (
            F.cross_entropy(logits_per_image, ground_truth) + F.cross_entropy(logits_per_text, ground_truth)
        ).div(2)

        # calculate accuracy (image and text separately)
        image_pred = torch.argmax(logits_per_image, dim=-1)
        text_pred = torch.argmax(logits_per_text, dim=-1)
        image_accuracy = (image_pred == ground_truth).float().mean()
        text_accuracy = (text_pred == ground_truth).float().mean()
        image_entropy = get_entropy(logits_per_image, dim=-1).mean()
        text_entropy = get_entropy(logits_per_text, dim=-1).mean()

        return (
            infonce_loss,
            image_accuracy,
            text_accuracy,
            image_entropy,
            text_entropy,
            logits_per_image,
            logits_per_text,
            image_features,
            image_feature_map,
            text_outputs,
        )


class LanguageModel(nn.Module):
    """Decoder head over text encoder outputs for next-token prediction / CE loss."""

    def __init__(self, text_encoder: TextEncoder, args: Any):
        super().__init__()
        self.args: Dict[str, Any] = vars(args) if args is not None else {}
        self.text_encoder = text_encoder

        self.output_layer = nn.Linear(self.text_encoder.hidden_dim, self.text_encoder.vocab_size, bias=bool(self.args.get("bias", True)))
        if bool(self.args.get("tie", True)):
            self.output_layer.weight = self.text_encoder.embedding.weight

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--tie", type=lambda s: bool(eval(s)), default=True, help="tie input embedding and output weights")
        parser.add_argument("--bias", type=lambda s: bool(eval(s)), default=True, help="use bias in output layer")

    def forward(
        self,
        y: torch.Tensor,
        y_len: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        image_feature_map: Optional[torch.Tensor] = None,
    ):
        if outputs is None:
            _text_features, outputs, attns = self.text_encoder(
                y,
                y_len,
                image_features=image_features,
                image_feature_map=image_feature_map,
            )
        else:
            # in this case the outputs is reused, so it mustn't be an attention
            # model.
            attns = None

        logits = self.output_layer(outputs)
        return outputs, logits, attns

    def calculate_ce_loss(
        self,
        y: torch.Tensor,
        y_len: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        image_feature_map: Optional[torch.Tensor] = None,
        tokenwise: bool = False,
        weight: Optional[torch.Tensor] = None,
    ):
        outputs, logits, attns = self(
            y,
            y_len,
            outputs=outputs,
            image_features=image_features,
            image_feature_map=image_feature_map,
        )

        if not self.text_encoder.regressional:
            labels = y
        else:
            logits = logits[:, :-1]
            labels = y[:, 1: 1 + logits.size(1)]

        loss = F.cross_entropy(
            logits.transpose(-2, -1),
            labels,
            weight=weight,
            ignore_index=PAD_TOKEN_ID,
            reduction="none" if tokenwise else "mean",
        )
        return loss, outputs, logits, attns, labels

    def beam_search_decode(
        self,
        batch_size: int,
        beam_width: int,
        decode_length: int,
        length_penalty_alpha: float,
        image_features: Optional[torch.Tensor] = None,
        image_feature_map: Optional[torch.Tensor] = None,
    ):
        """Beam search decoding (LSTM only)."""
        assert self.text_encoder.regressional, "only regressional text encoder supports beam search decoding"

        start_tokens = torch.full(
            (batch_size,),
            SOS_TOKEN_ID,
            dtype=torch.int,
            device=self.text_encoder.embedding.weight.device,
        )

        init_states = self.text_encoder.init_hidden(batch_size, image_features=image_features)
        init_states = map_structure(lambda t: t.transpose(0, 1), init_states)

        if self.text_encoder.has_attention:
            projected_image_feature_map = self.text_encoder.attention.project_encoder_features(image_feature_map)  # type: ignore[arg-type]
            init_states = (init_states, image_feature_map, projected_image_feature_map)

        def _symbols_to_logits_fn(ids: torch.Tensor, states):
            if self.text_encoder.has_attention:
                states, image_fmap, projected_fmap = states
            else:
                image_fmap, projected_fmap = None, None

            states = map_structure(lambda t: t.transpose(0, 1), states)
            outputs, states, _attns = self.text_encoder.ids_to_outputs(
                ids[:, -1],
                states,
                image_feature_map=image_fmap,
                projected_image_feature_map=projected_fmap,
            )
            states = map_structure(lambda t: t.transpose(0, 1), states)
            logits = self.output_layer(outputs)

            if self.text_encoder.has_attention:
                states = (states, image_fmap, projected_fmap)

            return logits, states

        return beam_search(
            _symbols_to_logits_fn,
            start_tokens,
            beam_width,
            decode_length,
            self.text_encoder.vocab_size,
            length_penalty_alpha,
            states=init_states,
            eos_id=EOS_TOKEN_ID,
        )


def calculate_attn_reg_loss(attns):
    return ((attns.sum(dim=1) - 1.) ** 2).mean()
