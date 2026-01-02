import argparse
import functools
import json
import numpy as np
import os
import torchvision
import spacy
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms
import pytorch_lightning as pl
from typing import List, Tuple
from huggingface_hub import hf_hub_download

from multimodal.multimodal import MultiModalModel, LanguageModel, calculate_attn_reg_loss
from multimodal.utils import get_entropy
from multimodal.textgen_eval import evaluate as textgen_eval
from multimodal.multimodal_data_module import (
    N_VAL_DATALOADERS_PER_SPLIT, MAX_LEN_UTTERANCE,
    PAD_TOKEN_ID, SOS_TOKEN_ID, EOS_TOKEN_ID
)
from multimodal.nesy_constraints import (
    build_targets, existential_soft_or_loss, implication_hinge_loss,
    build_default_rules, build_edge_weights
)
from multimodal.visual_memory import (
    VisualMemory,
    ObjectAppearanceEncoder,
    masked_spatial_pool,
    gaussian_blur2d,
    make_context_alpha,
    sample_masks_per_concept_for_viz,
    overlay_masks_on_images,
)

try:
    import wandb
except Exception:
    wandb = None


# --------------------------------------------------------------------------
# DDP helpers
# --------------------------------------------------------------------------

class _AllGatherWithGrad(torch.autograd.Function):
    """
    All-gather tensors across DDP processes with autograd support.

    This implementation supports variable batch sizes across ranks by padding
    to the max batch size, gathering, and then unpadding.

    Forward returns a concatenated tensor of shape (sum_i B_i, ...).

    Backward all-reduces the per-rank gradient tensor for every gathered chunk,
    ensuring that gradients for negatives propagate to the original rank.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            ctx.is_distributed = False
            return x

        ctx.is_distributed = True
        ctx.rank = dist.get_rank()
        ctx.world_size = dist.get_world_size()

        # gather batch sizes
        local_bs = torch.tensor([x.size(0)], device=x.device, dtype=torch.long)
        bs_list = [torch.zeros_like(local_bs) for _ in range(ctx.world_size)]
        dist.all_gather(bs_list, local_bs)
        sizes = [int(b.item()) for b in bs_list]

        ctx.sizes = sizes
        ctx.max_size = max(sizes) if sizes else x.size(0)

        # pad to max_size along dim 0
        if x.size(0) < ctx.max_size:
            pad_shape = (ctx.max_size - x.size(0),) + tuple(x.shape[1:])
            padding = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
            x_pad = torch.cat([x, padding], dim=0)
        else:
            x_pad = x

        # all_gather padded tensors
        gathered = [torch.zeros_like(x_pad) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, x_pad)

        # unpad and concatenate
        out = []
        for t, sz in zip(gathered, sizes):
            out.append(t[:sz])
        if len(out) == 0:
            return x
        return torch.cat(out, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not getattr(ctx, "is_distributed", False):
            return grad_output

        world_size = int(ctx.world_size)
        rank = int(ctx.rank)
        sizes = list(ctx.sizes)
        max_size = int(ctx.max_size)

        # split grad_output into per-rank chunks (unpadded)
        grads = []
        offset = 0
        for sz in sizes:
            grads.append(grad_output[offset: offset + sz])
            offset += sz

        # pad each chunk back to max_size
        padded = []
        for g, sz in zip(grads, sizes):
            if sz < max_size:
                pad_shape = (max_size - sz,) + tuple(g.shape[1:])
                padding = torch.zeros(pad_shape, device=g.device, dtype=g.dtype)
                g = torch.cat([g, padding], dim=0)
            padded.append(g)

        # (world, max_size, ...) tensor
        grad_stack = torch.stack(padded, dim=0)

        # Sum gradients for every gathered chunk across ranks.
        dist.all_reduce(grad_stack, op=dist.ReduceOp.SUM)

        # Return gradient for this rank's original (unpadded) input
        grad_input = grad_stack[rank][:sizes[rank]]
        return grad_input


def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _dist_rank() -> int:
    return dist.get_rank() if _dist_is_initialized() else 0


def _dist_world_size() -> int:
    return dist.get_world_size() if _dist_is_initialized() else 1


OPTIMIZER = torch.optim.AdamW
LR = 3e-4
FACTOR = 0.1
PATIENCE = 20
WEIGHT_DECAY = 0.01

# text generation evaluation arguments
BEAM_WIDTH = 3
DECODE_LENGTH = MAX_LEN_UTTERANCE
LENGTH_PENALTY_ALPHA = 0.0
# print arguments
PRINT_EVAL_TEXTGEN_EXAMPLE_IDS = range(10)


def mask_hypernyms_when_hyponyms_present(pos_mask: torch.Tensor,
                                         edges: List[Tuple[int, int]]) -> torch.Tensor:
    """
    Given pos_mask (B,C) and directed edges A->B, set mask[:,B]=0 for any
    example that mentions at least one of B's antecedents A.
    """
    if not edges:
        return pos_mask
    pm = pos_mask.clone()
    from collections import defaultdict
    ant = defaultdict(list)
    for a, b in edges:
        ant[b].append(a)
    for b, As in ant.items():
        if len(As) == 0:
            continue
        present = (pm[:, As].max(dim=1).values > 0.5)
        pm[present, b] = 0.0
    return pm


class MultiModalLitModel(pl.LightningModule):
    """
    PyTorch Lightning class for MultiModal SAYCam model.
    """

    def __init__(self, vision_encoder, text_encoder, args):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.optimizer_class = self.args.get("optimizer", OPTIMIZER)
        self.lr = self.args.get("lr", LR)
        self.lr_scheduler = self.args.get("lr_scheduler", False)
        self.factor = self.args.get("factor", FACTOR)
        self.patience = self.args.get("patience", PATIENCE)
        self.weight_decay = self.args.get("weight_decay", WEIGHT_DECAY)

        self.lambda_mm = self.args.get("lambda_mm", 1.0)
        self.lambda_lm = self.args.get("lambda_lm", 0.0)
        self.lambda_ar = self.args.get("lambda_ar", 0.0)
        self.optimize_unused = self.args.get("optimize_unused", False)
        self.eval_textgen = self.args.get("eval_textgen", False)
        self.beam_width = self.args.get("beam_width", BEAM_WIDTH)
        self.decode_length = self.args.get("decode_length", DECODE_LENGTH)
        self.length_penalty_alpha = self.args.get("length_penalty_alpha", LENGTH_PENALTY_ALPHA)

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.model = MultiModalModel(self.vision_encoder, self.text_encoder, args)
        self.language_model = LanguageModel(self.text_encoder, args)

        # vocab
        self.vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab.json")
        with open(self.vocab_path) as f:
            self.vocab = json.load(f)
        self.nlp = spacy.load("en_core_web_sm")

        # id -> token
        self.id2tok = None
        if isinstance(self.vocab, dict):
            self.id2tok = {int(i): tok for tok, i in self.vocab.items()}
        elif hasattr(self.vocab, "itos"):
            itos = list(self.vocab.itos)
            self.id2tok = {i: tok for i, tok in enumerate(itos)}

        # tokens to ignore
        ignore_tokens = ["<pad>", "<unk>", "<sos>", "<eos>", ".", ",", "?", "!", "...", "..", "...."]
        self.ignore_ids = {self.vocab[t] for t in ignore_tokens if t in self.vocab}

        # concepts for NeSy (optional)
        self.concepts = None
        self.concept2idx = None

        cpath = self.args.get("concept_list_file", None)
        if cpath and os.path.exists(cpath):
            with open(cpath) as f:
                raw = json.load(f)
            if isinstance(raw, list):
                self.concepts = [str(c) for c in raw]
            elif isinstance(raw, dict):
                # interpret keys as concept names to stay aligned with DataModule
                self.concepts = [str(k) for k in raw.keys()]
            else:
                raise ValueError("concept_list_file must contain a JSON list or dict")
            assert len(self.concepts) > 0, "Empty concept_list_file"
            self.concept2idx = {c.lower(): i for i, c in enumerate(self.concepts)}
            self.ns_edges = build_default_rules(self.concepts)
        else:
            self.ns_edges = []

        # NeSy class/edge weights (optional, from utterance mention counts)
        self.ns_class_weights = None
        self.ns_edge_weights = None
        counts_path = self.args.get("ns_class_count_file", None)
        if counts_path and os.path.exists(counts_path) and self.concepts is not None:
            import csv
            concept_names = [c.lower() for c in self.concepts]
            name2idx = {n: i for i, n in enumerate(concept_names)}
            counts = np.zeros(len(concept_names), dtype=np.float64)
            with open(counts_path, newline="") as f:
                reader = csv.DictReader(f)
                assert "concept" in reader.fieldnames and "count" in reader.fieldnames, \
                    "Counts CSV must have headers: concept,count"
                for row in reader:
                    n = row["concept"].strip().lower()
                    if n in name2idx:
                        try:
                            cval = float(row["count"])
                        except Exception:
                            cval = 0.0
                        counts[name2idx[n]] += cval
            counts = np.maximum(counts, 1.0)
            alpha = float(self.args.get("ns_weight_power", 1.0))
            w = (np.median(counts) / counts) ** alpha
            w = np.clip(
                w,
                self.args.get("ns_weight_min", 0.33),
                self.args.get("ns_weight_max", 3.0),
            )
            # normalize so average weight is 1
            w = w * (len(w) / w.sum())

            self.ns_class_weights = torch.tensor(w, dtype=torch.float32)
            if len(self.ns_edges):
                scheme = self.args.get("ns_edge_weight_scheme", "mean")
                self.ns_edge_weights = build_edge_weights(self.ns_class_weights, self.ns_edges, scheme=scheme)
            try:
                if self.global_rank == 0:
                    print("[NeSy] Loaded class weights from", counts_path)
                    for c, wt in zip(self.concepts, w):
                        print(f"  {c:>10s}: {wt:.3f}")
            except Exception:
                pass

        # -------- Visual Memory (object-centric, concept-conditioned) --------
        self.vm = None
        self.vm_lambda = float(self.args.get("vm_lambda", 0.2))
        self.vm_warmup_steps = int(self.args.get("vm_warmup_steps", 0))

        # embedding dimension of the VM space; object encoder projects into this
        vm_dim = self.args.get("vm_embedding_dim", None)
        if vm_dim is None:
            vm_dim = self.args.get("embedding_dim", 128)
        self.vm_obj_out_dim = int(vm_dim)
        # ObjectAppearanceEncoder is instantiated lazily on first batch when dims are known
        self.obj_encoder: ObjectAppearanceEncoder | None = None

        if self.args.get("vm_enable", False):
            if self.concepts is not None and len(self.concepts) > 0:
                vm_num_concepts = len(self.concepts)
            else:
                vm_num_concepts = int(self.args.get("vm_num_concepts", 0))

            vm_protos_per_concept = int(self.args.get("vm_protos_per_concept", 0))

            # VisualMemory does usage-based reweighting internally using usage_* hyperparameters
            self.vm = VisualMemory(
                embedding_dim=self.vm_obj_out_dim,
                num_concepts=vm_num_concepts,
                protos_per_concept=vm_protos_per_concept,
                bank_capacity=int(self.args.get("vm_bank_capacity", 512)),
                tau=float(self.args.get("vm_tau", 0.5)),
                beta=float(self.args.get("vm_beta", 0.25)),
                use_soft_assignment=not bool(self.args.get("vm_hard_assignment", False)),
                enable_temporal=bool(self.args.get("vm_temporal", True)),
                t_window=int(self.args.get("vm_t_window", 2)),
                temporal_weight=float(self.args.get("vm_t_weight", 1.0)),
                sep_margin=float(self.args.get("vm_sep_margin", 0.5)),
                sep_weight=float(self.args.get("vm_sep_weight", 0.1)),
                enable_usage_reweighting=self.args.get("vm_enable_usage_reweighting", False),
                usage_smoothing=float(self.args.get("vm_usage_smoothing", 1.0)),
                usage_power=float(self.args.get("vm_usage_power", 1.0)),
                usage_min=float(self.args.get("vm_usage_min", 0.33)),
                usage_max=float(self.args.get("vm_usage_max", 3.0)),
                use_sinkhorn=bool(self.args.get("vm_use_sinkhorn", True)),
                sinkhorn_iters=int(self.args.get("vm_sinkhorn_iters", 3)),
                sinkhorn_epsilon=float(self.args.get("vm_sinkhorn_epsilon", 0.05)),
                sinkhorn_min_samples=int(self.args.get("vm_sinkhorn_min_samples", 2)),
                temporal_same_concept_only=bool(self.args.get("vm_temp_same_concept_only", True)),
                temporal_sim_thresh=float(self.args.get("vm_temp_sim_thresh", 0.3)),
                temporal_detach_target=bool(self.args.get("vm_temp_detach_target", True)),
            )

            if self.concepts is not None:
                self.vm.set_concept_names(self.concepts)

            if self.obj_encoder is None:
                dim_global = int(self.args.get("embedding_dim", 128))
                dim_local = self.args.get("vm_fmap_dim", None)

                if dim_local is None:
                    try:
                        with torch.no_grad():
                            dummy_h = 224
                            dummy_w = 224
                            dummy = torch.zeros(1, 3, dummy_h, dummy_w)
                            dummy_global, dummy_fmap = self.model.encode_image(dummy)
                            dim_global = int(dummy_global.size(-1))
                            if dummy_fmap is not None:
                                dim_local = int(dummy_fmap.size(1))
                    except Exception:
                        dim_local = None

                if dim_local is None:
                    raise ValueError(
                        "vm_enable requires vm_fmap_dim (channel dimension of the conv feature map), "
                        "or a vision encoder whose encode_image returns a non-None feature map."
                    )

                self.obj_encoder = ObjectAppearanceEncoder(
                    dim_global=int(dim_global),
                    dim_local=int(dim_local),
                    dim_out=self.vm_obj_out_dim,
                    use_local=True,
                    local_weight=float(self.args.get("vm_local_weight", 1.0)),
                    learn_local_weight=bool(self.args.get("vm_learn_local_weight", False)),
                )

        self.vm_log_gradcam = bool(getattr(args, "vm_log_gradcam", False))
        # VM debug flags
        self.vm_debug_verbose = bool(self.args.get("vm_debug_verbose", False))
        self.vm_debug_log_images_every = int(self.args.get("vm_debug_log_images_every", 0))

        # save hyperparameters to logger
        self.save_hyperparameters()

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _unnorm_for_viz(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized images back to [0,1] range for visualization or edge maps.
        Assumes ImageNet-style normalization.
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 3, 1, 1)
        imgs = imgs.float()
        return (imgs * std + mean).clamp(0, 1)

    def _get_global_rank(self) -> int:
        """
        Try trainer.global_rank first, fall back to self.global_rank, default 0.
        """
        trainer = getattr(self, "trainer", None)
        if trainer is not None and hasattr(trainer, "global_rank"):
            try:
                return int(trainer.global_rank)
            except Exception:
                pass
        try:
            return int(getattr(self, "global_rank", 0))
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # DDP-safe contrastive loss
    # ------------------------------------------------------------------

    def _gather_batch_sizes(self, local_bs: int, device: torch.device) -> List[int]:
        """
        Gather per-rank batch sizes (B_i) and return as a Python list in rank order.
        """
        if not _dist_is_initialized():
            return [int(local_bs)]
        bs = torch.tensor([int(local_bs)], device=device, dtype=torch.long)
        bs_list = [torch.zeros_like(bs) for _ in range(_dist_world_size())]
        dist.all_gather(bs_list, bs)
        return [int(b.item()) for b in bs_list]

    def calculate_contrastive_loss_ddp(self, x, y, y_len):
        """
        Contrastive loss with global (multi-GPU) negatives.

        Returns:
            infonce_loss, image_accuracy, text_accuracy,
            image_entropy, text_entropy,
            logits_per_image, logits_per_text,
            image_features, image_feature_map, text_outputs
        """
        # encode
        image_features, image_feature_map = self.model.encode_image(x)
        text_features, text_outputs = self.model.encode_text(y, y_len)

        local_bs = int(image_features.size(0))
        device = image_features.device

        # gather features (with grad) for global negatives
        if _dist_is_initialized() and _dist_world_size() > 1:
            sizes = self._gather_batch_sizes(local_bs, device=device)
            rank = _dist_rank()
            world_size = _dist_world_size()
            global_bs = int(sum(sizes))
            offset = int(sum(sizes[:rank]))

            all_image_features = _AllGatherWithGrad.apply(image_features)
            all_text_features = _AllGatherWithGrad.apply(text_features)

            labels = torch.arange(local_bs, device=device, dtype=torch.long) + offset

            # Correct scaling under DDP gradient averaging (robust to variable local batch sizes):
            # Each rank computes a scaled local sum so that DDP's gradient averaging yields the global mean.
            scale = float(world_size) / float(max(global_bs, 1))
        else:
            all_image_features = image_features
            all_text_features = text_features
            labels = torch.arange(local_bs, device=device, dtype=torch.long)
            scale = 1.0

        # logits
        logit_scale = (-self.model.logit_neg_log_temperature).exp()
        logits_per_image = logit_scale * image_features @ all_text_features.t()
        logits_per_text = logit_scale * text_features @ all_image_features.t()

        # losses (sum then scale so DDP average gives global mean)
        loss_i = F.cross_entropy(logits_per_image, labels, reduction="sum") * scale
        loss_t = F.cross_entropy(logits_per_text, labels, reduction="sum") * scale
        infonce_loss = (loss_i + loss_t) / 2.0

        # metrics on local anchors vs global candidates
        with torch.no_grad():
            image_accuracy = (logits_per_image.argmax(dim=1) == labels).float().mean()
            text_accuracy = (logits_per_text.argmax(dim=1) == labels).float().mean()

            image_entropy = get_entropy(logits_per_image)
            if torch.is_tensor(image_entropy) and image_entropy.ndim > 0:
                image_entropy = image_entropy.mean()

            text_entropy = get_entropy(logits_per_text)
            if torch.is_tensor(text_entropy) and text_entropy.ndim > 0:
                text_entropy = text_entropy.mean()

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

    def _log_vm_histogram_to_wandb(self, stage: str) -> None:
        """
        Log histogram of running concept mask-instance counts to W&B using wandb.log.
        Safe under DDP (rank 0 only) and no-op if no active run.
        """
        if wandb is None:
            return
        run = getattr(wandb, "run", None)
        if run is None:
            # WandbLogger has not called wandb.init() yet
            return
        if self.vm is None or not hasattr(self.vm, "concept_mask_counts"):
            return
        if self.vm.concept_mask_counts is None or self.vm.concept_mask_counts.numel() == 0:
            return

        global_rank = self._get_global_rank()
        if global_rank != 0:
            return

        counts = (
            self.vm.concept_mask_counts
            .clone()
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        step_int = int(getattr(self, "global_step", 0))

        if self.vm_debug_verbose and step_int < 10:
            print(
                f"[VM-hist] logging histogram at step={step_int}, "
                f"num_bins={len(counts)}, sum={counts.sum()}"
            )

        try:
            wandb.log(
                {f"{stage}/vm_concept_mask_hist": wandb.Histogram(counts)},
                step=step_int,
            )
        except Exception as e:
            if self.vm_debug_verbose:
                print("[VM-hist] failed to log histogram:", repr(e))

    def _log_vm_prompt_panels_to_wandb(
        self,
        tag: str,
        imgs: torch.Tensor,        # (N,3,H,W) in [0,1]
        masks: torch.Tensor,       # (N,1,H,W) in {0,1}
        concept_ids: torch.Tensor, # (N,)
        max_images: int = 12,
        overlay_alpha: float = 0.4,
    ) -> None:
        if wandb is None:
            return
        logger = getattr(self, "logger", None)
        if logger is None:
            return
        experiment = getattr(logger, "experiment", None)
        if experiment is None:
            return

        if imgs.ndim != 4 or masks.ndim != 4 or imgs.size(0) == 0:
            return

        n = min(int(max_images), int(imgs.size(0)))
        imgs = imgs[:n].detach().cpu()
        masks = masks[:n].detach().cpu()
        concept_ids = concept_ids[:n].detach().cpu()

        # overlay on original
        overlays = overlay_masks_on_images(imgs, masks, alpha=float(overlay_alpha)).clamp(0.0, 1.0)

        # blur-filled prompt (with ring + feather)
        blur_ksize = int(self.args.get("vm_bg_blur_kernel", 23))
        blur_sigma = float(self.args.get("vm_bg_blur_sigma", 5.0))
        bg = gaussian_blur2d(imgs, kernel_size=blur_ksize, sigma=blur_sigma)  # reflect padding default

        ring_px = int(self.args.get("vm_ctx_ring_px", 0))
        ring_strength = float(self.args.get("vm_ctx_ring_strength", 1.0))
        feather_sigma = float(self.args.get("vm_mask_feather_sigma", 0.0))
        feather_kernel = int(self.args.get("vm_mask_feather_kernel", 0))
        feather_kernel = None if feather_kernel <= 0 else feather_kernel

        alpha_mask = make_context_alpha(
            masks,
            ring_px=ring_px,
            ring_strength=ring_strength,
            feather_sigma=feather_sigma,
            feather_kernel=feather_kernel,
        )
        prompts = (imgs * alpha_mask + bg * (1.0 - alpha_mask)).clamp(0.0, 1.0)

        # side-by-side panels: [orig | overlay | prompt]
        panels = torch.cat([imgs, overlays, prompts], dim=3)  # concat along width

        images = []
        for i in range(n):
            panel = panels[i].permute(1, 2, 0).numpy()
            cid = int(concept_ids[i].item())
            caption = (
                f"cid={cid} | orig|overlay|prompt "
                f"(ring_px={ring_px}, ring_str={ring_strength}, feather_sigma={feather_sigma})"
            )
            images.append(wandb.Image(panel, caption=caption))

        step_int = int(getattr(self, "global_step", 0))
        try:
            experiment.log({tag: images}, step=step_int)
        except TypeError:
            experiment.log({tag: images})

    # ------------------------------------------------------------------
    # GradCAM helpers
    # ------------------------------------------------------------------

    def _compute_gradcam_from_logits(
        self,
        logits_per_image: torch.Tensor,
        image_feature_map: torch.Tensor,
        max_examples: int = 4,
    ):
        """
        Compute GradCAM maps for the diagonal image–text pairs in the batch.

        Args:
            logits_per_image: (B, B) contrastive logits matrix.
            image_feature_map: (B, E, H, W) last conv feature map from VisionEncoder.
            max_examples: number of examples to visualize from the batch.

        Returns:
            cam: (K, 1, H, W) GradCAM heatmaps normalized to [0, 1], or None.
            idx: (K,) indices of the examples used, or None.
        """
        if image_feature_map is None:
            return None, None
        if not torch.is_grad_enabled():
            return None, None
        if not image_feature_map.requires_grad:
            return None, None

        B = logits_per_image.size(0)
        device = logits_per_image.device

        diag_scores = logits_per_image[
            torch.arange(B, device=device),
            torch.arange(B, device=device)
        ]  # (B,)

        K = min(B, max_examples)
        idx = torch.arange(K, device=device)

        scores_sel = diag_scores[idx]      # (K,)
        fmap_sel = image_feature_map[idx]  # (K, E, H, W)

        grads = torch.autograd.grad(
            outputs=scores_sel.sum(),
            inputs=fmap_sel,
            retain_graph=False,
            create_graph=False,
            only_inputs=True,
            allow_unused=True,
        )[0]
        if grads is None:
            return None, None

        weights = grads.mean(dim=(2, 3), keepdim=True)      # (K, E, 1, 1)
        cam = (weights * fmap_sel).sum(dim=1, keepdim=True)  # (K, 1, H, W)
        cam = F.relu(cam)

        K_, _, H, W = cam.shape
        cam_flat = cam.view(K_, -1)
        cam_flat = cam_flat - cam_flat.min(dim=1, keepdim=True)[0]
        cam_max = cam_flat.max(dim=1, keepdim=True)[0]
        cam_max[cam_max < 1e-8] = 1.0
        cam_flat = cam_flat / cam_max
        cam = cam_flat.view(K_, 1, H, W)

        return cam.detach(), idx

    def _log_gradcam_debug_images(
        self,
        images: torch.Tensor,   # (B, 3, H, W) normalized batch
        y: torch.Tensor,
        y_len: torch.Tensor | None,
        stage: str,
        batch_idx: int,
        max_examples: int = 4,
    ):
        """
        Run a small separate forward pass with gradients to get a GradCAM
        visualization of the contrastive logits. This does not affect the main
        training graph or loss.
        """
        if not self.vm_log_gradcam or self.vm_debug_log_images_every <= 0:
            return
        if stage != "train":
            return

        trainer = getattr(self, "trainer", None)
        if trainer is not None and getattr(trainer, "global_rank", 0) != 0:
            return
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return

        step = int(getattr(self, "global_step", 0))
        if step % self.vm_debug_log_images_every != 0:
            return

        B = images.size(0)
        if B == 0:
            return

        K = min(B, max_examples)

        with torch.enable_grad():
            x_cam = images[:K].detach().to(self.device)
            x_cam.requires_grad_(True)

            y_cam = y[:K].detach().to(self.device)
            y_len_cam = y_len[:K].detach().to(self.device) if y_len is not None else None

            # fresh forward through encoders (no detaching here)
            img_feats, fmap = self.model.encode_image(x_cam)
            if fmap is None or not fmap.requires_grad:
                return

            txt_feats, _ = self.model.encode_text(y_cam, y_len_cam)

            logit_scale = (-self.model.logit_neg_log_temperature).exp()
            logits_per_image = logit_scale * img_feats @ txt_feats.t()  # (K,K)

            cam, idx = self._compute_gradcam_from_logits(
                logits_per_image, fmap, max_examples=K
            )
            if cam is None or idx is None:
                return

            images_sel = x_cam[idx].detach().cpu()
            cam_sel = cam.detach().cpu()

        # de-normalize for visualization
        imgs_vis = self._unnorm_for_viz(images_sel)

        # upsample CAM to image size
        cam_up = F.interpolate(
            cam_sel,
            size=imgs_vis.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )  # (K,1,H,W)

        alpha = 0.5
        cam_rgb = cam_up.repeat(1, 3, 1, 1)          # (K,3,H,W)
        overlay = (1.0 - alpha) * imgs_vis + alpha * cam_rgb
        overlay = overlay.clamp(0.0, 1.0)

        grid = torchvision.utils.make_grid(
            torch.cat([imgs_vis, cam_rgb, overlay], dim=0),
            nrow=K,
        )

        exp = self.logger.experiment
        try:
            import wandb

            exp.log(
                {
                    f"{stage}/gradcam_debug": wandb.Image(
                        grid,
                        caption=f"{stage} step={step} batch={batch_idx}",
                    )
                },
                step=step,
            )
        except ImportError:
            if hasattr(exp, "add_image"):
                exp.add_image(
                    f"{stage}/gradcam_debug",
                    grid,
                    global_step=step,
                )

    def decode_ids_to_tokens(self, u):
        """
        Accepts nested tensors/lists of ids or raw strings.
        Returns a flat list[str] of lowercase tokens with specials/punct removed.
        """
        import re

        def _flatten(obj):
            if torch.is_tensor(obj):
                obj = obj.tolist()
            if isinstance(obj, (list, tuple)):
                for x in obj:
                    yield from _flatten(x)
            else:
                yield obj

        toks = []
        for item in _flatten(u):
            if isinstance(item, int):
                if self.id2tok is None:
                    continue
                if int(item) in self.ignore_ids:
                    continue
                toks.append(self.id2tok.get(int(item), "<unk>"))
            elif isinstance(item, str):
                toks.extend([w for w in re.split(r"[^a-zA-Z]+", item.lower()) if w])
        return toks

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=lambda o: getattr(torch.optim, o), default=OPTIMIZER)
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--lr_scheduler", action="store_true")
        parser.add_argument("--factor", type=float, default=FACTOR)
        parser.add_argument("--patience", type=int, default=PATIENCE)
        parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
        parser.add_argument("--lambda_mm", type=float, default=1.0)
        parser.add_argument("--lambda_lm", type=float, default=0.0)
        parser.add_argument("--lambda_ar", type=float, default=0.0)
        parser.add_argument("--optimize_unused", action="store_true")
        parser.add_argument("--eval_textgen", action="store_true")
        parser.add_argument("--beam_width", type=int, default=BEAM_WIDTH)
        parser.add_argument("--decode_length", type=int, default=DECODE_LENGTH)
        parser.add_argument("--length_penalty_alpha", type=float, default=LENGTH_PENALTY_ALPHA)

        # ---- For Neuro Symbolic Experiment (From our previous attempt) ----
        parser.add_argument("--neurosym", action="store_true")
        parser.add_argument("--ns_lambda_exist", type=float, default=0.5)
        parser.add_argument("--ns_lambda_hier", type=float, default=0.1)
        parser.add_argument("--concept_list_file", type=str, default=None)
        parser.add_argument("--ns_class_count_file", type=str, default=None)
        parser.add_argument("--ns_weight_power", type=float, default=1.0)
        parser.add_argument("--ns_weight_min", type=float, default=0.33)
        parser.add_argument("--ns_weight_max", type=float, default=3.0)
        parser.add_argument("--ns_edge_weight_scheme", type=str, default="mean",
                            choices=["a", "b", "mean", "geomean"])
        # ----

        # ---- Visual Memory (train-only, object-centric) ----
        parser.add_argument("--vm_enable", action="store_true",
                            help="enable visual memory (prototype bank)")
        parser.add_argument("--vm_lambda", type=float, default=0.2,
                            help="weight for visual memory loss")
        parser.add_argument("--vm_embedding_dim", type=int, default=None,
                            help="embedding dimension for visual memory; "
                                 "defaults to model embedding_dim")
        parser.add_argument("--vm_fmap_dim", type=int, default=None,
                            help="channel dimension of conv feature map; kept for compatibility")

        parser.add_argument("--vm_bank_capacity", type=int, default=512,
                            help="capacity of prototype bank (ignored if protos_per_concept > 0)")

        # concept-conditioned layout
        parser.add_argument("--vm_num_concepts", type=int, default=0,
                            help="number of concepts for VM; if 0, inferred from concept_list_file when present")
        parser.add_argument("--vm_protos_per_concept", type=int, default=0,
                            help="number of prototypes per concept; if 0, memory is global")

        # temporal consistency
        parser.add_argument("--vm_temporal", dest="vm_temporal", action="store_true",
                            help="enable temporal consistency using clip/frame metadata")
        parser.add_argument("--vm_no_temporal", dest="vm_temporal", action="store_false",
                            help=argparse.SUPPRESS)

        parser.add_argument("--vm_t_window", type=int, default=5,
                            help="max frame distance inside a clip to form temporal positives")
        parser.add_argument("--vm_t_weight", type=float, default=1.0,
                            help="relative weight of temporal positives")

        parser.add_argument("--vm_warmup_steps", type=int, default=0,
                            help="steps before applying the VM loss")

        # VQ / assignment and separation
        parser.add_argument("--vm_tau", type=float, default=0.5,
                            help="temperature for VM soft assignment")
        parser.add_argument("--vm_beta", type=float, default=0.25,
                            help="commitment weight for VQ-style VM loss")
        parser.add_argument("--vm_hard_assignment", action="store_true",
                            help="use hard nearest-prototype assignment instead of soft")
        parser.add_argument("--vm_sep_margin", type=float, default=0.5,
                            help="margin for prototype separation loss")
        parser.add_argument("--vm_sep_weight", type=float, default=0.1,
                            help="weight for prototype separation loss")
        parser.add_argument("--vm_bg_blur_kernel", type=int, default=23,
                            help="Gaussian blur kernel size (odd int) for blurred background fill.")
        parser.add_argument("--vm_bg_blur_sigma", type=float, default=5.0,
                            help="Gaussian blur sigma for blurred background fill.")

        # --- Sinkhorn balancing ---
        parser.add_argument("--vm_use_sinkhorn", dest="vm_use_sinkhorn", action="store_true",
                            help="Use Sinkhorn-Knopp balanced assignments for VM.")
        parser.add_argument("--vm_no_sinkhorn", dest="vm_use_sinkhorn", action="store_false",
                            help=argparse.SUPPRESS)
        parser.add_argument("--vm_sinkhorn_iters", type=int, default=3)
        parser.add_argument("--vm_sinkhorn_epsilon", type=float, default=0.05)
        parser.add_argument("--vm_sinkhorn_min_samples", type=int, default=2)

        # --- Temporal assignment distillation controls ---
        parser.add_argument("--vm_temp_same_concept_only", dest="vm_temp_same_concept_only", action="store_true",
                            help="Temporal positives only within same concept id.")
        parser.add_argument("--vm_temp_allow_cross_concept", dest="vm_temp_same_concept_only", action="store_false",
                            help=argparse.SUPPRESS)
        parser.add_argument("--vm_temp_sim_thresh", type=float, default=0.3,
                            help="Cosine similarity threshold to accept temporal positives.")
        parser.add_argument("--vm_temp_detach_target", dest="vm_temp_detach_target", action="store_true",
                            help="Detach target distribution in temporal loss.")
        parser.add_argument("--vm_temp_no_detach_target", dest="vm_temp_detach_target", action="store_false",
                            help=argparse.SUPPRESS)

        # context ring + feathered boundary for blur-fill prompting
        parser.add_argument(
            "--vm_ctx_ring_px",
            type=int,
            default=0,
            help="Context ring width in pixels (at 224x224). 0 disables.",
        )
        parser.add_argument(
            "--vm_ctx_ring_strength",
            type=float,
            default=1.0,
            help="Alpha for ring region (0..1). Only used if vm_ctx_ring_px > 0.",
        )
        parser.add_argument(
            "--vm_mask_feather_sigma",
            type=float,
            default=0.0,
            help="Sigma for feathering the (object+ring) alpha boundary. 0 disables.",
        )
        parser.add_argument(
            "--vm_mask_feather_kernel",
            type=int,
            default=0,
            help="Kernel size for feathering blur. 0 means auto from sigma.",
        )

        # regularizer on learnable local_weight to prevent collapsing toward 0 too fast
        parser.add_argument(
            "--vm_local_weight_reg_lambda",
            type=float,
            default=0.0,
            help="Strength of (local_weight - init)^2 regularizer (with optional decay).",
        )
        parser.add_argument(
            "--vm_local_weight_reg_decay_steps",
            type=int,
            default=0,
            help="If >0, linearly decay the reg weight to 0 over this many steps.",
        )

        # usage-based VM weights (derived from per-slot usage_counts in VisualMemory)
        parser.add_argument(
            "--vm_enable_usage_reweighting",
            dest="vm_enable_usage_reweighting",
            action="store_true",
            help="Enable inverse-frequency reweighting.",
        )
        parser.add_argument("--vm_usage_smoothing", type=float, default=1.0,
                            help="additive smoothing for VM usage-based prototype weights")
        parser.add_argument("--vm_usage_power", type=float, default=1.0,
                            help="exponent for VM inverse-frequency prototype weights")
        parser.add_argument("--vm_usage_min", type=float, default=0.33,
                            help="lower clamp for VM usage-based prototype weights")
        parser.add_argument("--vm_usage_max", type=float, default=3.0,
                            help="upper clamp for VM usage-based prototype weights")

        # object encoder fusion
        parser.add_argument("--vm_local_weight", type=float, default=1.0,
                            help="relative weight of local object feature in fusion")
        parser.add_argument("--vm_learn_local_weight", action="store_true",
                            help="learn the local/global mixing weight in object encoder")

        # GradCAM logging flag used by MultiModalModel.encode_image
        parser.add_argument(
            "--vm_log_gradcam",
            action="store_true",
            help="keep gradients on the conv feature map to enable GradCAM logging"
        )

        # VM debugging
        parser.add_argument(
            "--vm_debug_verbose",
            action="store_true",
            help="print verbose visual memory debug info for the first few steps",
        )
        parser.add_argument(
            "--vm_debug_log_images_every",
            type=int,
           default=0,
            help="if >0, log SAM mask overlays every N training steps",
        )

        # legacy VM flags kept for CLI compatibility (no effect in object-centric VM)
        parser.add_argument("--vm_top_p", type=float, default=0.20,
                            help=argparse.SUPPRESS)
        parser.add_argument("--vm_use_masked_for_loss", action="store_true",
                            help=argparse.SUPPRESS)
        parser.add_argument("--vm_blend_alpha", type=float, default=0.20,
                            help=argparse.SUPPRESS)
        parser.add_argument("--vm_ema_alpha", type=float, default=0.20,
                            help=argparse.SUPPRESS)
        parser.add_argument("--vm_merge_sim", type=float, default=0.70,
                            help=argparse.SUPPRESS)
        parser.add_argument("--vm_debiased", action="store_true",
                            help=argparse.SUPPRESS)
        parser.add_argument("--vm_edge_prior", action="store_true",
                            help=argparse.SUPPRESS)
        parser.add_argument("--vm_edge_blend", type=float, default=0.35,
                            help=argparse.SUPPRESS)
        parser.add_argument("--vm_loss_mode", type=str, default="proto_nce",
                            choices=["proto_nce", "supcon"],
                            help=argparse.SUPPRESS)
        parser.add_argument("--vm_use_edges", action="store_true",
                            help=argparse.SUPPRESS)

        # defaults for boolean flags that should be on by default
        parser.set_defaults(vm_temporal=True)
        parser.set_defaults(vm_use_sinkhorn=True)
        parser.set_defaults(vm_temp_same_concept_only=True)
        parser.set_defaults(vm_temp_detach_target=True)

        return parser

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if not self.lr_scheduler:
            return optimizer
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.factor,
            patience=self.patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val/loss",
            }
        }

    def forward(self, x, y, y_len):
        return self.model(x, y, y_len)

    @staticmethod
    def load_model(model_name="cvcl"):
        """Load pre-trained CVCL model from HuggingFace Hub"""
        if model_name == "cvcl":
            checkpoint_name = "cvcl_s_dino_resnext50_embedding"
            checkpoint = hf_hub_download(
                repo_id="wkvong/" + checkpoint_name,
                filename=checkpoint_name + ".ckpt"
            )
            model = MultiModalLitModel.load_from_checkpoint(checkpoint_path=checkpoint)
        else:
            raise ValueError("Model name not found.")

        preprocess = transforms.Compose([
            transforms.Resize((224, 224),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        return model, preprocess

    def encode_image(self, x):
        """Encode images to obtain image features"""
        image_features, _ = self.model.encode_image(x)
        return image_features

    def encode_text(self, y, y_len=None):
        """Encode text to obtain text features"""
        text_features, _ = self.model.encode_text(y, y_len)
        return text_features

    def tokenize(self, texts):
        """Tokenize texts to obtain tokens and token lengths"""
        max_seq_len = 25

        if isinstance(texts, str):
            texts = [texts]

        all_tokens = []
        token_lengths = []

        for text in texts:
            doc = self.nlp(text)
            word_tokens = [token.text for token in doc]

            if len(word_tokens) > max_seq_len - 2:
                word_tokens = word_tokens[:max_seq_len - 2]

            token_length = len(word_tokens) + 2  # +2 for <sos> and <eos>

            tokens = [self.vocab["<sos>"]] + [
                self.vocab.get(token, self.vocab["<unk>"]) for token in word_tokens
            ] + [self.vocab["<eos>"]] + [
                self.vocab["<pad>"]
            ] * (max_seq_len - len(word_tokens) - 2)

            all_tokens.append(tokens)
            token_lengths.append(token_length)

        tokens = torch.tensor(all_tokens, dtype=torch.long)
        token_lengths = torch.tensor(token_lengths, dtype=torch.long)
        return tokens, token_lengths

    def _split_batch(self, batch):
        """
        Accept batches in one of three shapes:
        1) (x, y, y_len, raw_y)
        2) (x, y, y_len, raw_y, meta)
        3) ((x, y, y_len, raw_y), meta)
        Returns (x, y, y_len, raw_y, meta_or_None).
        """
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2 and isinstance(batch[0], (list, tuple)) and len(batch[0]) == 4:
                (x, y, y_len, raw_y), meta = batch
                return x, y, y_len, raw_y, meta
            if len(batch) >= 5:
                x, y, y_len, raw_y, meta = batch[:5]
                return x, y, y_len, raw_y, meta
            if len(batch) == 4:
                x, y, y_len, raw_y = batch
                return x, y, y_len, raw_y, None
        raise ValueError(
            f"Unexpected batch structure: type={type(batch)} len={len(batch) if isinstance(batch, (list, tuple)) else 'n/a'}"
        )

    def calculate_ce_loss(
        self, y, y_len, x=None,
        outputs=None,
        image_features=None,
        image_feature_map=None,
        return_image_features=False,
        **kwargs
    ):
        """Wraps self.language_model.calculate_ce_loss."""
        if self.language_model.text_encoder.captioning or \
                self.language_model.text_encoder.has_attention:
            # get image_features and image_feature_map if needed
            if image_features is None:
                image_features, image_feature_map = self.model.encode_image(x)
            # text_outputs is not reusable since it is not obtained from
            # captioning in the contrastive module
            outputs = None
        else:
            image_features, image_feature_map = None, None

        # calculate language model ce loss
        ret = self.language_model.calculate_ce_loss(
            y, y_len,
            outputs=outputs,
            image_features=image_features
                if self.language_model.text_encoder.captioning else None,
            image_feature_map=image_feature_map
                if self.language_model.text_encoder.has_attention else None,
            **kwargs
        )
        if return_image_features:
            ret = ret + (image_features, image_feature_map)
        return ret

    def build_vm_inputs_from_batch(
        self,
        batch,
        device=None,
    ):
        """
        Rebuild object appearance embeddings and metadata from a dataloader batch.

        Returns None if there are no valid SAM masks in the batch.

        Output dict has:
            "embeds":        (N_obj, D_vm)
            "concept_ids":   (N_obj,)
            "clip_id":       (N_obj,)  long (hashed clip ids)
            "frame_idx":     (N_obj,)  long
            "frame_relpath": list[str] of length N_obj
        """
        if device is None:
            device = next(self.parameters()).device

        # Use the same batch splitter as in training
        images, utt_idxs, utt_lens, raw_y, meta = self._split_batch(batch)
        x = images.to(device)

        if not isinstance(meta, dict):
            return None
        if "sam_mask" not in meta or "sam_mask_concept_id" not in meta:
            return None

        sam_mask = meta["sam_mask"].to(device)           # (B, K, 1, H, W)
        sam_cid  = meta["sam_mask_concept_id"].to(device)  # (B, K)

        B, K, _, H, W = sam_mask.shape

        # ---- flatten masks and concepts ----
        sam_mask_flat = sam_mask.view(B * K, 1, H, W)    # (B*K, 1, H, W)
        cid_flat      = sam_cid.view(B * K)              # (B*K,)

        valid = cid_flat >= 0
        if valid.sum() == 0:
            return None

        # broadcast images to match masks
        x_exp  = x.unsqueeze(1).expand(-1, K, -1, -1, -1)    # (B, K, C, H, W)
        x_flat = x_exp.reshape(B * K, x.size(1), H, W)       # (B*K, C, H, W)

        x_valid     = x_flat[valid]                          # (N_obj, C, H, W)
        masks_valid = sam_mask_flat[valid]                   # (N_obj, 1, H, W)
        cids_valid  = cid_flat[valid]                        # (N_obj,)

        # ---- masked RGB with blurred background fill (match training) ----
        blur_ksize = int(self.args.get("vm_bg_blur_kernel", 23))
        blur_sigma = float(self.args.get("vm_bg_blur_sigma", 5.0))

        x_blur = gaussian_blur2d(x, kernel_size=blur_ksize, sigma=blur_sigma)  # (B,3,H,W)
        x_blur_exp = x_blur.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(B * K, x.size(1), H, W)
        x_bg_valid = x_blur_exp[valid]  # (N_obj,3,H,W)

        ring_px = int(self.args.get("vm_ctx_ring_px", 0))
        ring_strength = float(self.args.get("vm_ctx_ring_strength", 1.0))
        feather_sigma = float(self.args.get("vm_mask_feather_sigma", 0.0))
        feather_kernel = int(self.args.get("vm_mask_feather_kernel", 0))
        feather_kernel = None if feather_kernel <= 0 else feather_kernel

        alpha_mask = make_context_alpha(
            masks_valid,
            ring_px=ring_px,
            ring_strength=ring_strength,
            feather_sigma=feather_sigma,
            feather_kernel=feather_kernel,
        )

        x_masked = x_valid * alpha_mask + x_bg_valid * (1.0 - alpha_mask)

        # encode masked patches with the same path as training
        with torch.no_grad():
            obj_global_feat, obj_fmap = self.model.encode_image(x_masked)

        if obj_fmap is None:
            return None

        N_obj, Cf, Hf, Wf = obj_fmap.shape

        # downsample masks to feature map size
        masks_ds = torch.nn.functional.interpolate(
            masks_valid,
            size=(Hf, Wf),
            mode="nearest",
        )  # (N_obj, 1, Hf, Wf)

        # local feature by masked spatial pooling
        obj_local_feat = masked_spatial_pool(obj_fmap, masks_ds)  # (N_obj, Cf)

        # ---- visual memory appearance encoder (reuse training obj_encoder) ----
        if self.obj_encoder is None:
            self.obj_encoder = ObjectAppearanceEncoder(
                dim_global=obj_global_feat.size(-1),
                dim_local=obj_local_feat.size(-1),
                dim_out=self.vm_obj_out_dim,
                use_local=True,
                local_weight=float(self.args.get("vm_local_weight", 1.0)),
                learn_local_weight=bool(self.args.get("vm_learn_local_weight", False)),
            ).to(device)

        with torch.no_grad():
            embeds = self.obj_encoder(
                global_feat=obj_global_feat,
                local_feat=obj_local_feat,
            )  # (N_obj, D_vm)

        # ------------------------------------------------------------------
        # Temporal metadata: clip_id, frame_idx, frame_filename
        # ------------------------------------------------------------------
        clip_raw  = meta.get("clip_id", None)
        frame_raw = meta.get("frame_idx", None)

        # clip ids may be strings; follow the training code and hash to numeric ids
        if clip_raw is None:
            clip_ids_valid = torch.zeros_like(cids_valid, dtype=torch.long, device=device)
        else:
            if isinstance(clip_raw, torch.Tensor):
                clip_ids_list = clip_raw.tolist()
            else:
                clip_ids_list = list(clip_raw)

            clip_ids_numeric = [
                abs(hash(c)) % (1 << 31) if isinstance(c, str) else int(c)
                for c in clip_ids_list
            ]
            clip_tensor = torch.as_tensor(
                clip_ids_numeric, device=device, dtype=torch.long
            )  # (B,)
            clip_tensor = clip_tensor.view(B, 1).expand(-1, K).reshape(B * K)  # (B*K,)
            clip_ids_valid = clip_tensor[valid].to(torch.long)                 # (N_obj,)

        # frame indices should already be numeric, but be defensive
        if frame_raw is None:
            frame_idx_valid = torch.zeros_like(cids_valid, dtype=torch.long, device=device)
        else:
            if isinstance(frame_raw, torch.Tensor):
                frame_tensor = frame_raw.to(device)
            else:
                frame_tensor = torch.as_tensor(
                    [int(f) for f in frame_raw], device=device, dtype=torch.long
                )
            frame_tensor = frame_tensor.view(B, 1).expand(-1, K).reshape(B * K)
            frame_idx_valid = frame_tensor[valid].to(torch.long)

        # frame-level identifier, for later visualization
        if "frame_filename" in meta:
            relpaths = meta["frame_filename"]
        elif "img_relpath" in meta:
            relpaths = meta["img_relpath"]
        elif "image_relpath" in meta:
            relpaths = meta["image_relpath"]
        else:
            relpaths = [""] * B

        if isinstance(relpaths, torch.Tensor):
            relpaths = list(relpaths)

        relpaths_per_obj: list[str] = []
        valid_2d = valid.view(B, K)
        for b in range(B):
            rp_b = relpaths[b]
            for k in range(K):
                if valid_2d[b, k]:
                    relpaths_per_obj.append(rp_b)

        return {
            "embeds": embeds,                    # (N_obj, D_vm)
            "concept_ids": cids_valid,           # (N_obj,)
            "clip_id": clip_ids_valid,           # (N_obj,)
            "frame_idx": frame_idx_valid,        # (N_obj,)
            "frame_relpath": relpaths_per_obj,   # list[str]
        }

    def calculate_joint_loss(self, batch, stage, log, batch_idx, eval_textgen=False, ce_weight=None):
        # batch may come as ((x, y, y_len, raw_y), meta) to support temporal VM
        x, y, y_len, raw_y, batch_meta = self._split_batch(batch)

        ret = {'batch_size': x.size(0)}

        # reuse image_features, image_feature_map and text_outputs if possible
        image_features, image_feature_map, text_outputs = None, None, None

        if self.lambda_mm or not self.optimize_unused:
            infonce_loss, image_accuracy, text_accuracy, \
                image_entropy, text_entropy, logits_per_image, logits_per_text, \
                image_features, image_feature_map, text_outputs = \
                self.calculate_contrastive_loss_ddp(x, y, y_len)

            # log (DDP-safe if caller sets sync_dist=True)
            log(f"{stage}/infonce_loss", infonce_loss, batch_size=ret['batch_size'])
            log(f"{stage}/image_accuracy", image_accuracy, batch_size=ret['batch_size'])
            log(f"{stage}/text_accuracy", text_accuracy, batch_size=ret['batch_size'])
            log(f"{stage}/image_entropy", image_entropy, batch_size=ret['batch_size'])
            log(f"{stage}/text_entropy", text_entropy, batch_size=ret['batch_size'])
            log("temperature",
                (-self.model.logit_neg_log_temperature).exp().item())

            ret.update({
                'infonce_loss': infonce_loss.detach(),
                'image_accuracy': image_accuracy.detach() if torch.is_tensor(image_accuracy) else image_accuracy,
                'text_accuracy': text_accuracy.detach() if torch.is_tensor(text_accuracy) else text_accuracy,
                'image_entropy': image_entropy.detach() if torch.is_tensor(image_entropy) else image_entropy,
                'text_entropy': text_entropy.detach() if torch.is_tensor(text_entropy) else text_entropy,
            })
        else:
            infonce_loss = torch.tensor(0.0, device=x.device)

        # -------- Visual Memory: object-centric VQ loss (train-only) --------
        vm_total = torch.tensor(0.0, device=x.device)
        vm_q = torch.tensor(0.0, device=x.device)
        vm_temp = torch.tensor(0.0, device=x.device)
        vm_sep = torch.tensor(0.0, device=x.device)
        vm_lw_reg = torch.tensor(0.0, device=x.device)

        vm_cid_min = torch.tensor(0.0, device=x.device)
        vm_cid_max = torch.tensor(0.0, device=x.device)
        vm_num_active_protos = torch.tensor(0.0, device=x.device)
        vm_frac_active_protos = torch.tensor(0.0, device=x.device)

        use_vm = (
            stage == "train"
            and self.vm is not None
            and self.vm_lambda > 0.0
        )

        if stage == "train" and use_vm:
            if batch_meta is not None and isinstance(batch_meta, dict):
                sam_mask = batch_meta.get("sam_mask", None)              # (B,K,1,H,W)
                sam_concept_id = batch_meta.get("sam_mask_concept_id", None)  # (B,K)

                if sam_mask is not None and sam_concept_id is not None:
                    B, C, H, W = x.shape
                    Bm, K, _, Hm, Wm = sam_mask.shape
                    if Bm != B or Hm != H or Wm != W:
                        raise ValueError(
                            f"sam_mask shape mismatch: images {x.shape}, masks {sam_mask.shape}"
                        )

                    sam_mask_flat = sam_mask.view(B * K, 1, H, W)
                    cid_flat = sam_concept_id.view(B * K)

                    total_objs = sam_mask_flat.size(0)
                    if total_objs > 0:
                        vm_cid_min = cid_flat.min().float()
                        vm_cid_max = cid_flat.max().float()

                    valid = cid_flat >= 0

                    if total_objs > 0:
                        num_valid = valid.sum().float()

                    if valid.any():
                        cid_flat = cid_flat[valid]
                        masks_valid = sam_mask_flat[valid]  # (N_obj,1,H,W)

                        # optional verbose print for early steps on rank 0
                        if (
                            self.vm_debug_verbose
                            and self._get_global_rank() == 0
                            and int(getattr(self, "global_step", 0)) < 10
                        ):
                            print(
                                f"[VM-debug] step {int(self.global_step)}: "
                                f"B={B}, K={K}, total_objs={total_objs}, "
                                f"num_valid={int(num_valid.item())}"
                            )
                            print(
                                f"[VM-debug]   cid range: "
                                f"[{int(vm_cid_min.item())}, {int(vm_cid_max.item())}]"
                            )

                        # optional mask overlays to W&B
                        log_every = int(self.vm_debug_log_images_every or 0)
                        if (
                            log_every > 0
                            and self._get_global_rank() == 0
                        ):
                            step_int = int(getattr(self, "global_step", 0))
                            if step_int % log_every == 0:
                                try:
                                    with torch.no_grad():
                                        imgs_viz = self._unnorm_for_viz(x.detach())
                                        sel_imgs, sel_masks, sel_concepts = sample_masks_per_concept_for_viz(
                                            imgs_viz,
                                            sam_mask,
                                            sam_concept_id,
                                            num_concepts=getattr(self.vm, "num_concepts", 1),
                                            max_per_concept=2,
                                        )
                                        if sel_imgs.numel() > 0:
                                            self._log_vm_prompt_panels_to_wandb(
                                                tag=f"{stage}/vm_prompt",
                                                imgs=sel_imgs,
                                                masks=sel_masks,
                                                concept_ids=sel_concepts,
                                                max_images=16,
                                                overlay_alpha=0.4,
                                            )
                                except Exception as e:
                                    if (
                                        self.vm_debug_verbose
                                        and self._get_global_rank() == 0
                                    ):
                                        print(
                                            f"[VM-debug] mask overlay logging failed: {e}"
                                        )

                        # broadcast x over K and pick valid ones
                        x_exp = x.unsqueeze(1).expand(B, K, C, H, W).reshape(B * K, C, H, W)
                        x_valid = x_exp[valid]          # (N_obj,3,H,W)
                        # --- blurred background fill to avoid hard cutout artifacts ---
                        blur_ksize = int(self.args.get("vm_bg_blur_kernel", 23))
                        blur_sigma = float(self.args.get("vm_bg_blur_sigma", 5.0))

                        # blur full images once (B,3,H,W), then broadcast to (B*K,3,H,W) and select valid
                        x_blur = gaussian_blur2d(x, kernel_size=blur_ksize, sigma=blur_sigma)  # (B,3,H,W)
                        x_blur_exp = x_blur.unsqueeze(1).expand(B, K, C, H, W).reshape(B * K, C, H, W)
                        x_bg_valid = x_blur_exp[valid]  # (N_obj,3,H,W)

                        ring_px = int(self.args.get("vm_ctx_ring_px", 0))
                        ring_strength = float(self.args.get("vm_ctx_ring_strength", 1.0))
                        feather_sigma = float(self.args.get("vm_mask_feather_sigma", 0.0))
                        feather_kernel = int(self.args.get("vm_mask_feather_kernel", 0))
                        feather_kernel = None if feather_kernel <= 0 else feather_kernel

                        alpha_mask = make_context_alpha(
                            masks_valid,                   # (N_obj,1,H,W)
                            ring_px=ring_px,
                            ring_strength=ring_strength,
                            feather_sigma=feather_sigma,
                            feather_kernel=feather_kernel,
                        )  # (N_obj,1,H,W) in [0,1]

                        masked_rgb = x_valid * alpha_mask + x_bg_valid * (1.0 - alpha_mask)

                        # encode masked RGB for global and fmap
                        obj_global_feat, obj_fmap = self.model.encode_image(masked_rgb)
                        # ensure fmap for full images is available for downstream modules
                        if image_feature_map is None:
                            _, image_feature_map = self.model.encode_image(x)

                        Cf, Hf, Wf = obj_fmap.shape[1:]
                        masks_ds = F.interpolate(masks_valid, size=(Hf, Wf), mode="nearest")
                        obj_local_feat = masked_spatial_pool(obj_fmap, masks_ds)  # (N_obj,Cf)

                        # lazy init of object appearance encoder once we know dims
                        if self.obj_encoder is None:
                            self.obj_encoder = ObjectAppearanceEncoder(
                                dim_global=obj_global_feat.size(-1),
                                dim_local=obj_local_feat.size(-1),
                                dim_out=self.vm_obj_out_dim,
                                use_local=True,
                                local_weight=float(self.args.get("vm_local_weight", 1.0)),
                                learn_local_weight=bool(self.args.get("vm_learn_local_weight", False)),
                            ).to(self.device)

                        z_obj = self.obj_encoder(
                            global_feat=obj_global_feat,
                            local_feat=obj_local_feat,
                        )  # (N_obj, Dvm)

                        reg_lambda = float(self.args.get("vm_local_weight_reg_lambda", 0.0))
                        if reg_lambda > 0.0 and self.obj_encoder is not None:
                            w = getattr(self.obj_encoder, "local_weight", None)
                            if w is not None and w.requires_grad:
                                # anchor to the initialization value (vm_local_weight)
                                w_init = float(self.args.get("vm_local_weight", 1.0))

                                decay_steps = int(self.args.get("vm_local_weight_reg_decay_steps", 0))
                                if decay_steps > 0:
                                    decay = max(0.0, 1.0 - float(self.global_step) / float(decay_steps))
                                else:
                                    decay = 1.0

                                vm_lw_reg = (w - w.new_tensor(w_init)).pow(2) * (reg_lambda * decay)

                                log(f"{stage}/vm_local_weight_value", w)
                                log(f"{stage}/vm_local_weight_reg", vm_lw_reg)

                                ret.update({
                                    "vm_local_weight_value": w.detach(),
                                    "vm_local_weight_reg": vm_lw_reg.detach(),
                                })

                        # build per-object temporal metadata
                        clip_ids = batch_meta.get("clip_id", None)
                        frame_idx = batch_meta.get("frame_idx", None)
                        obj_meta = None
                        if clip_ids is not None and frame_idx is not None:
                            # flatten to Python lists
                            if isinstance(clip_ids, torch.Tensor):
                                clip_ids_list = clip_ids.tolist()
                            else:
                                clip_ids_list = list(clip_ids)

                            if isinstance(frame_idx, torch.Tensor):
                                frame_idx_list = frame_idx.tolist()
                            else:
                                frame_idx_list = list(frame_idx)

                            assert len(clip_ids_list) == B and len(frame_idx_list) == B, \
                                "clip_id and frame_idx must have length B"

                            # convert clip IDs to numeric IDs so torch.as_tensor works
                            clip_ids_numeric = [
                                abs(hash(c)) % (1 << 31) if isinstance(c, str) else int(c)
                                for c in clip_ids_list
                            ]

                            obj_clip_ids = []
                            obj_frame_idx = []
                            for b in range(B):
                                for _k in range(K):
                                    obj_clip_ids.append(clip_ids_numeric[b])
                                    obj_frame_idx.append(frame_idx_list[b])

                            # valid is boolean mask over B*K objects
                            obj_clip_ids = torch.as_tensor(
                                obj_clip_ids, device=x.device, dtype=torch.long
                            )[valid]
                            obj_frame_idx = torch.as_tensor(
                                obj_frame_idx, device=x.device, dtype=torch.long
                            )[valid]

                            obj_meta = {
                                "clip_id": obj_clip_ids,
                                "frame_idx": obj_frame_idx,
                            }

                        # call VM loss (usage-based weighting happens inside VisualMemory)
                        vm_total, vm_q, vm_temp, vm_sep = self.vm.loss(
                            feats=z_obj,
                            concept_ids=cid_flat,
                            batch_meta=obj_meta,
                        )

                        # log histogram of mask-instance counts per concept to W&B
                        self._log_vm_histogram_to_wandb(stage)

                        # optional warmup: zero out VM loss contribution during initial steps
                        if int(getattr(self, "global_step", 0)) < self.vm_warmup_steps:
                            vm_total = vm_total.detach() * 0.0
                            vm_q = vm_q.detach() * 0.0
                            vm_temp = vm_temp.detach() * 0.0
                            vm_sep = vm_sep.detach() * 0.0

                        # prototype usage statistics (from VisualMemory.loss)
                        assign_hist = getattr(self.vm, "_last_assign_hist", None)
                        if assign_hist is not None and isinstance(assign_hist, torch.Tensor) and assign_hist.numel() > 0:
                            assign_hist = assign_hist.to(x.device)
                            vm_num_active_protos = (assign_hist > 0).float().sum()
                            vm_frac_active_protos = vm_num_active_protos / float(assign_hist.numel())

                            if (
                                self.vm_debug_verbose
                                and self._get_global_rank() == 0
                                and int(getattr(self, "global_step", 0)) < 10
                            ):
                                print(
                                    "[VM-debug] active prototypes this batch: "
                                    f"{int(vm_num_active_protos.item())}/{assign_hist.numel()} "
                                    f"({float(vm_frac_active_protos.item()) * 100:.1f} percent)"
                                )

                        if (
                            self.vm_debug_verbose
                            and self._get_global_rank() == 0
                            and int(getattr(self, "global_step", 0)) < 10
                        ):
                            print(
                                "[VM-debug] losses: "
                                f"L_total={float(vm_total.item()):.4f}, "
                                f"L_q={float(vm_q.item()):.4f}, "
                                f"L_temp={float(vm_temp.item()):.4f}, "
                                f"L_sep={float(vm_sep.item()):.4f}"
                            )

        # log and attach VM metrics for training (as zeros if VM enabled but no valid objects)
        if stage == "train" and use_vm:
            log(f"{stage}/vm_loss_total", vm_total)
            log(f"{stage}/vm_loss_quant", vm_q)
            log(f"{stage}/vm_loss_temp", vm_temp)
            log(f"{stage}/vm_loss_sep", vm_sep)

            ret.update({
                'vm_loss_total': vm_total.detach(),
                'vm_loss_quant': vm_q.detach(),
                'vm_loss_temp': vm_temp.detach(),
                'vm_loss_sep': vm_sep.detach(),
            })

        # ------------------------------------------------------------------
        # GradCAM debug logging (side forward pass)
        # ------------------------------------------------------------------
        self._log_gradcam_debug_images(
            images=x,
            y=y,
            y_len=y_len,
            stage=stage,
            batch_idx=batch_idx,
        )

        # -------- NeSy --------
        ns_exist = torch.tensor(0.0, device=x.device)
        ns_hier = torch.tensor(0.0, device=x.device)
        pos_mask_ns = None

        if self.args.get("neurosym", False) and self.concepts is not None:
            concept_logits = self.vision_encoder.concept_logits(
                image_features, image_feature_map=image_feature_map
            )  # (B,C)
            raw_y_tokens = [self.decode_ids_to_tokens(u) for u in raw_y]
            pos_mask_ns = build_targets(raw_y_tokens, self.concepts).to(concept_logits.device)  # (B,C)
            try:
                if self.global_step < 3 and getattr(self, "local_rank", 0) == 0:
                    hits = pos_mask_ns.nonzero(as_tuple=False)
                    print(
                        f"[NeSy] step {self.global_step} - coverage:",
                        float((pos_mask_ns.sum(dim=1) > 0).float().mean())
                    )
                    if hits.numel():
                        b = int(hits[0, 0])
                        on = [self.concepts[i] for i in pos_mask_ns[b].nonzero().flatten().tolist()]
                        print(f"[NeSy] sample tokens:", raw_y_tokens[b][:20])
                        print(f"[NeSy] sample mentions ->", on)
            except Exception:
                pass
            pos_mask_leaf = mask_hypernyms_when_hyponyms_present(pos_mask_ns, self.ns_edges)

            cw = None if self.ns_class_weights is None else self.ns_class_weights.to(concept_logits.device)
            ns_exist = existential_soft_or_loss(concept_logits, pos_mask_leaf, class_weights=cw)
            log(f"{stage}/ns_exist_loss", ns_exist, batch_size=ret['batch_size'])

            if self.args.get("ns_lambda_hier", 0.0) > 0 and len(self.ns_edges):
                ew = self.ns_edge_weights.to(concept_logits.device) if self.ns_edge_weights is not None else None
                ns_hier = implication_hinge_loss(concept_logits, self.ns_edges, edge_weights=ew)
                log(f"{stage}/ns_hier_loss", ns_hier, batch_size=ret['batch_size'])

            ns_cov = (pos_mask_ns.sum(dim=1) > 0).float().mean()
            ns_avg_mentions = (pos_mask_ns.sum() / max(1, pos_mask_ns.size(0))).item()
            log(f"{stage}/ns_coverage", ns_cov, prog_bar=True, batch_size=ret['batch_size'])
            log(f"{stage}/ns_avg_mentions", ns_avg_mentions, batch_size=ret['batch_size'])

            ret.update({
                'ns_exist_loss': ns_exist.detach(),
                'ns_hier_loss': ns_hier.detach() if torch.is_tensor(ns_hier) else ns_hier,
            })

        # -------- LM --------
        if self.lambda_lm or not self.optimize_unused:
            # calculate language model ce loss
            ce_loss, _, _, attns, labels, image_features, image_feature_map = \
                self.calculate_ce_loss(
                    y, y_len, x=x,
                    outputs=text_outputs,
                    image_features=image_features,
                    image_feature_map=image_feature_map,
                    return_image_features=True,
                    tokenwise=True,
                    weight=ce_weight,
                )

            # get all kinds of losses with/without special tokens
            # In torch.nn.CrossEntropyLoss the sum of loss should be
            # divided by the sum of mask weighted by the weight.
            # Here it ignores the weight for simplicity.

            # standard loss including all special tokens
            mask = (labels != PAD_TOKEN_ID)
            n_tokens = mask.sum()
            lm_ce_loss = ce_loss.sum() / n_tokens

            mask = mask & (labels != SOS_TOKEN_ID)
            n_tokens_wo_sos = mask.sum()
            lm_ce_loss_wo_sos = (ce_loss * mask).sum() / n_tokens_wo_sos

            mask = mask & (labels != EOS_TOKEN_ID)
            n_tokens_wo_sos_eos = mask.sum()
            lm_ce_loss_wo_sos_eos = (ce_loss * mask).sum() / n_tokens_wo_sos_eos

            # log
            log(f"{stage}/ce_loss", lm_ce_loss, batch_size=ret['batch_size'])
            log(f"{stage}/ce_loss_wo_sos", lm_ce_loss_wo_sos, batch_size=ret['batch_size'])
            log(f"{stage}/ce_loss_wo_sos_eos", lm_ce_loss_wo_sos_eos, batch_size=ret['batch_size'])

            ret.update({
                'ce_loss': lm_ce_loss.detach(),
                'ce_loss_wo_sos': lm_ce_loss_wo_sos.detach(),
                'ce_loss_wo_sos_eos': lm_ce_loss_wo_sos_eos.detach(),
                'n_tokens': n_tokens,
                'n_tokens_wo_sos': n_tokens_wo_sos,
                'n_tokens_wo_sos_eos': n_tokens_wo_sos_eos,
            })

            # attention regularization loss
            if self.language_model.text_encoder.has_attention:
                attn_reg_loss = calculate_attn_reg_loss(attns)

                # log
                log(f"{stage}/attn_reg_loss", attn_reg_loss, batch_size=ret['batch_size'])

                ret.update({
                    'attn_reg_loss': attn_reg_loss.detach(),
                })
            else:
                attn_reg_loss = torch.tensor(0.0, device=x.device)

            if eval_textgen:
                beam_seq, log_prob = self.language_model.beam_search_decode(
                    batch_size=ret['batch_size'],
                    beam_width=self.beam_width,
                    decode_length=self.decode_length,
                    length_penalty_alpha=self.length_penalty_alpha,
                    image_features=image_features
                        if self.language_model.text_encoder.captioning else None,
                    image_feature_map=image_feature_map
                        if self.language_model.text_encoder.has_attention else None,
                )

                def ids_to_sentence(y_ids):
                    y_list = y_ids.tolist()
                    y_len_eff = 0
                    while y_len_eff < len(y_list) and y_list[y_len_eff] != PAD_TOKEN_ID:
                        y_len_eff += 1
                    y_list = y_list[:y_len_eff]
                    if len(y_list) > 0 and y_list[-1] == EOS_TOKEN_ID:
                        y_list = y_list[:-1]
                    if len(y_list) > 0 and y_list[0] == SOS_TOKEN_ID:
                        y_list = y_list[1:]
                    return ' '.join(self.text_encoder.idx2word[idx] for idx in y_list)

                gen_text_ids = beam_seq[:, 0]
                gen_text = [ids_to_sentence(y_seq) for y_seq in gen_text_ids]

                ret.update({
                    'raw_y': raw_y,
                    'gen_text': gen_text,
                })

        else:
            lm_ce_loss = torch.tensor(0.0, device=x.device)
            attn_reg_loss = torch.tensor(0.0, device=x.device)

        # -------- total loss --------
        loss = (
            self.lambda_mm * infonce_loss
            + self.lambda_lm * lm_ce_loss
            + self.lambda_ar * attn_reg_loss
            + self.args.get("ns_lambda_exist", 0.0) * ns_exist
            + self.args.get("ns_lambda_hier", 0.0) * ns_hier
            + self.vm_lambda * vm_total
            + vm_lw_reg
        )
        log(f"{stage}/loss", loss, batch_size=ret['batch_size'])

        ret.update({
            'loss': loss,
        })

        return ret

    def joint_loss_epoch_end(self, outputs, stage, log, eval_textgen=False):
        """
        Epoch-end aggregation that is safe under DDP.

        Instead of logging per-rank means with sync_dist=True (which can be biased
        if ranks see different numbers of examples), we sum numerators and
        denominators and all-reduce the sums.
        """
        device = self.device if hasattr(self, "device") else torch.device("cpu")

        def _as_float_tensor(v):
            if v is None:
                return None
            if torch.is_tensor(v):
                return v.detach().to(device).float()
            return torch.tensor(float(v), device=device, dtype=torch.float32)

        def _sum_all(t: torch.Tensor) -> torch.Tensor:
            if _dist_is_initialized() and _dist_world_size() > 1:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return t

        def mean_over_examples(name):
            local_sum = torch.tensor(0.0, device=device)
            local_n = torch.tensor(0.0, device=device)

            for output in outputs:
                bs = output.get("batch_size", 0)
                bs_t = _as_float_tensor(bs) if bs is not None else torch.tensor(0.0, device=device)

                val = output.get(name, None)
                if val is None:
                    continue

                val_t = _as_float_tensor(val)

                # If metric is a vector (eg, per-example), reduce it to a scalar mean first
                if torch.is_tensor(val_t) and val_t.ndim > 0:
                    val_t = val_t.mean()

                local_sum += val_t * bs_t
                local_n += bs_t

            local_sum = _sum_all(local_sum)
            local_n = _sum_all(local_n)
            return local_sum / local_n.clamp(min=1.0)

        def mean_over_tokens(name, n_tokens_name):
            local_sum = torch.tensor(0.0, device=device)
            local_n = torch.tensor(0.0, device=device)

            for output in outputs:
                if name not in output or n_tokens_name not in output:
                    continue
                n_tok = _as_float_tensor(output[n_tokens_name])
                val = _as_float_tensor(output[name])
                local_sum += val * n_tok
                local_n += n_tok

            local_sum = _sum_all(local_sum)
            local_n = _sum_all(local_n)
            return local_sum / local_n.clamp(min=1.0)

        def mean_over_batches(name):
            local_sum = torch.tensor(0.0, device=device)
            local_n = torch.tensor(0.0, device=device)
            for output in outputs:
                if name in output:
                    local_sum += _as_float_tensor(output[name])
                    local_n += 1.0
            local_sum = _sum_all(local_sum)
            local_n = _sum_all(local_n)
            return local_sum / local_n.clamp(min=1.0)

        # Only log on global zero to avoid duplicate logger writes.
        is_global_zero = True
        trainer = getattr(self, "trainer", None)
        if trainer is not None and hasattr(trainer, "is_global_zero"):
            is_global_zero = bool(trainer.is_global_zero)
        else:
            is_global_zero = (self._get_global_rank() == 0)

        if self.lambda_mm or not self.optimize_unused:
            for name in (
                'infonce_loss', 'image_accuracy', 'text_accuracy',
                'image_entropy', 'text_entropy',):
                value = mean_over_examples(name)
                if is_global_zero:
                    log(f"{stage}/{name}", value)

        if stage == 'train' and self.vm is not None:
            for name in (
                'vm_loss_total', 'vm_loss_quant', 'vm_loss_temp', 'vm_loss_sep',
            ):
                value = mean_over_batches(name)
                if is_global_zero:
                    log(f"{stage}/{name}", value)

        if self.lambda_lm or not self.optimize_unused:
            for suffix in ('', '_wo_sos', '_wo_sos_eos'):
                value_mean = mean_over_tokens(
                    f'ce_loss{suffix}', f'n_tokens{suffix}')
                if is_global_zero:
                    log(f"{stage}/ce_loss{suffix}", value_mean)

                # perplexity
                perplexity = float(np.exp(float(value_mean.item())))
                if is_global_zero:
                    log(f"{stage}/perplexity{suffix}", perplexity)

            if self.language_model.text_encoder.has_attention:
                for name in ('attn_reg_loss',):
                    value = mean_over_examples(name)
                    if is_global_zero:
                        log(f"{stage}/{name}", value)

            if eval_textgen:
                # Gather references/hypotheses across ranks to compute global textgen metrics.
                list_of_references, hypotheses = [], []
                for output in outputs:
                    list_of_references += output.get('raw_y', [])
                    hypotheses += output.get('gen_text', [])

                if _dist_is_initialized() and _dist_world_size() > 1:
                    gathered_refs = [None for _ in range(_dist_world_size())]
                    gathered_hyps = [None for _ in range(_dist_world_size())]
                    dist.all_gather_object(gathered_refs, list_of_references)
                    dist.all_gather_object(gathered_hyps, hypotheses)

                    if is_global_zero:
                        all_refs = []
                        all_hyps = []
                        for r in gathered_refs:
                            if r:
                                all_refs.extend(r)
                        for h in gathered_hyps:
                            if h:
                                all_hyps.extend(h)
                        list_of_references = all_refs
                        hypotheses = all_hyps
                    else:
                        list_of_references = []
                        hypotheses = []

                score_dict = {}
                if is_global_zero:
                    for example_id in PRINT_EVAL_TEXTGEN_EXAMPLE_IDS:
                        if example_id >= len(hypotheses):
                            break
                        print(f"example #{example_id}:")
                        references = list_of_references[example_id]
                        hypothesis = hypotheses[example_id]
                        print("references:")
                        print("\n".join(references))
                        print("hypothesis:")
                        print(hypothesis)

                    score_dict = textgen_eval(list_of_references, hypotheses)

                # Broadcast score_dict from rank 0 so all ranks have it (avoids None issues)
                if _dist_is_initialized() and _dist_world_size() > 1:
                    obj_list = [score_dict]
                    dist.broadcast_object_list(obj_list, src=0)
                    score_dict = obj_list[0]

                if is_global_zero:
                    for metric, score in score_dict.items():
                        log(f"{stage}/{metric}", score)

        for name in ('loss',):
            value = mean_over_examples(name)
            if is_global_zero:
                log(f"{stage}/{name}", value)

    def training_step(self, batch, batch_idx):
        try:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("train/lr", lr, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        except Exception:
            pass

        step_log = functools.partial(self.log, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        return self.calculate_joint_loss(
            batch, 'train', step_log, batch_idx, eval_textgen=False)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self.trainer.is_global_zero:
            return
        if (self.global_step % 500) != 0:
            return
        if self.vm is None:
            return

        self.vm.wandb_log_concept_mean_weight_bar(
            step=self.global_step,
            prefix="train/vm",
        )
        self.vm.wandb_log_proto_utilization_per_concept(
            step=self.global_step,
            prefix="train/vm",
            use_batch_hist=False,
        )

    def training_epoch_end(self, outputs):
        # Epoch metrics are reduced inside joint_loss_epoch_end (DDP-safe), so do not use sync_dist=True here.
        log = functools.partial(self.log, on_step=False, on_epoch=True, sync_dist=False)
        return self.joint_loss_epoch_end(outputs, 'train', log, eval_textgen=False)

    def validation_test_step(self, stage, batch, batch_idx, dataloader_idx=0):
        ret = {}

        if dataloader_idx == 0:
            val_log = functools.partial(
                self.log,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                add_dataloader_idx=False,  # important: keeps the key as "val/loss"
            )
            ret.update(self.calculate_joint_loss(
                batch, stage, val_log, batch_idx, eval_textgen=self.eval_textgen
            ))

        elif dataloader_idx == 1:
            x, y, y_len, raw_y, _ = self._split_batch(batch)

            # resize x so images from the same trial are in the batch dim
            # [B, N, C, H, W] -> [B*N, C, H, W]  (with B = 1)
            x = x.view(-1, *x.shape[-3:])

            if self.lambda_mm:
                logits_per_image, logits_per_text = self.model(x, y, y_len)
                logits = logits_per_text[0]  # get logits per trial

            elif self.lambda_lm and (
                    self.language_model.text_encoder.captioning or
                    self.language_model.text_encoder.has_attention) \
                    and y[0, 0].item() == SOS_TOKEN_ID:
                # tile y to match the batch size
                y = y.expand(x.size(0), -1)
                y_len = y_len.expand(x.size(0))

                # calculate language model ce loss
                ce_loss, _, _, _, labels = self.calculate_ce_loss(
                    y, y_len, x=x, tokenwise=True)

                # use - ce_loss on the word as logits
                logits = - ce_loss[:, 0]

            else:
                logits = None

            if logits is not None:
                # calculate accuracy
                pred = torch.argmax(logits).item()
                label = 0  # correct answer is always the first item
                accuracy = float(pred == label)
                entropy = get_entropy(logits)

                # log evaluation accuracy and entropy
                self.log(f"{stage}/accuracy", accuracy, batch_size=1)
                self.log(f"{stage}/entropy", entropy, batch_size=1)

                # log category-level evaluation accuracies as a separate metric
                category_label = raw_y[0][0]
                self.log(f"{stage}/accuracy_{category_label}", accuracy, batch_size=1)

                ret.update({'accuracy': accuracy})

        return ret

    def validation_test_epoch_end(self, stage, outputs):
        # Epoch metrics are reduced inside joint_loss_epoch_end (DDP-safe), so do not use sync_dist=True here.
        log = functools.partial(self.log, on_step=False, on_epoch=True, sync_dist=False)
        return self.joint_loss_epoch_end(
            outputs[0], stage, log, eval_textgen=self.eval_textgen)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx < N_VAL_DATALOADERS_PER_SPLIT:  # as normal
            return self.validation_test_step(
                'val', batch, batch_idx, dataloader_idx=dataloader_idx)
        else:  # actually a test_step
            return self.test_step(
                batch, batch_idx,
                dataloader_idx=dataloader_idx - N_VAL_DATALOADERS_PER_SPLIT)

    def validation_epoch_end(self, outputs):
        self.validation_test_epoch_end(
            'val', outputs[:N_VAL_DATALOADERS_PER_SPLIT])
        if len(outputs) > N_VAL_DATALOADERS_PER_SPLIT:
            self.test_epoch_end(outputs[N_VAL_DATALOADERS_PER_SPLIT:])

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_test_step(
            'test', batch, batch_idx, dataloader_idx=dataloader_idx)

    def test_epoch_end(self, outputs):
        return self.validation_test_epoch_end(
            'test', outputs)

    def on_before_zero_grad(self, optimizer) -> None:
        """Runs right after optimizer.step() and before zero_grad().
        In PL 1.9 this is the safest spot to read (unscaled) grads."""
        # total L2 grad norm over all params that have grads
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if grads:
            total = torch.norm(torch.stack([g.detach().float().norm(2) for g in grads]), 2)
            self.log(
                "train/grad_norm",
                total,
                on_step=True, on_epoch=False, prog_bar=False, sync_dist=True
            )

        # (nice to have) log current LR as well
        try:
            lr = optimizer.param_groups[0]["lr"]
            self.log("train/lr", lr, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        except Exception:
            pass