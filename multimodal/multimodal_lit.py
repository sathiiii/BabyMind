from __future__ import annotations

import argparse
import functools
import json
import os
import re
import zlib
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import spacy
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from huggingface_hub import hf_hub_download
from torchvision import transforms

from multimodal.multimodal import MultiModalModel, LanguageModel, calculate_attn_reg_loss
from multimodal.textgen_eval import evaluate as textgen_eval
from multimodal.utils import get_entropy
from multimodal.multimodal_data_module import (
    N_VAL_DATALOADERS_PER_SPLIT,
    MAX_LEN_UTTERANCE,
    PAD_TOKEN_ID,
    SOS_TOKEN_ID,
    EOS_TOKEN_ID,
    UNK_TOKEN_ID
)

from multimodal.object_mil import (
    l2norm as l2norm_obj,
    masked_pool_k,
    context_ring_masks,
    TrackConfig,
    build_object_tracks_greedy,
    pack_candidates_with_null,
    mil_logsumexp_logits,
)
from multimodal.visual_memory import PrototypeMemory
from multimodal.viz_utils import make_track_grid

# Optional: SAM concept registry for prepacked SAM masks.
try:
    from multimodal.sam_concept_registry import SamConceptRegistry, build_sam_concept_registry  # type: ignore
except Exception:  # pragma: no cover
    SamConceptRegistry = None  # type: ignore
    build_sam_concept_registry = None  # type: ignore

try:
    import wandb  # noqa: F401
except Exception:  # pragma: no cover
    wandb = None


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _stable_str_hash31(s: str) -> int:
    """Stable 31-bit int hash for strings (useful for deterministic sharding)."""

    return int(zlib.crc32(s.encode("utf-8")) & 0x7FFFFFFF)


def _ddp_all_gather_object_list(local_list: list) -> list:
    """All-gather a Python list across ranks and concatenate."""

    if not (dist.is_available() and dist.is_initialized()):
        return list(local_list)
    world_size = dist.get_world_size()
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_list)
    out: list = []
    for part in gathered:
        if part:
            out.extend(part)
    return out


def _is_dist_active() -> bool:
    try:
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    except Exception:
        return False


def _dist_rank() -> int:
    if _is_dist_active():
        return dist.get_rank()
    return 0


def _dist_world() -> int:
    if _is_dist_active():
        return dist.get_world_size()
    return 1


def _dist_all_gather_no_grad(x: torch.Tensor) -> torch.Tensor:
    """All-gather a tensor across ranks without autograd.

    Notes:
      * NCCL does not support torch.bool collectives. We therefore cast
        bool tensors to uint8 for the gather and cast back.
    """

    if not _is_dist_active():
        return x

    orig_dtype = x.dtype
    x_g = x
    if orig_dtype == torch.bool:
        x_g = x.to(torch.uint8)

    ws = dist.get_world_size()
    outs = [torch.zeros_like(x_g) for _ in range(ws)]
    dist.all_gather(outs, x_g.contiguous())
    y = torch.cat(outs, dim=0)
    if orig_dtype == torch.bool:
        y = y.to(torch.bool)
    return y


class _AllGatherWithGrad(torch.autograd.Function):
    """AllGather that preserves autograd by slicing the incoming gradient.

    This is the standard CLIP-style trick and avoids relying on
    torch.distributed.nn.functional.all_gather availability.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if not _is_dist_active():
            ctx.world_size = 1
            ctx.rank = 0
            ctx.local_n = x.size(0)
            return x

        ws = dist.get_world_size()
        rk = dist.get_rank()
        ctx.world_size = ws
        ctx.rank = rk
        ctx.local_n = x.size(0)

        outs = [torch.zeros_like(x) for _ in range(ws)]
        dist.all_gather(outs, x.contiguous())
        return torch.cat(outs, dim=0)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # If not distributed, gradient is identity.
        if int(getattr(ctx, 'world_size', 1)) == 1:
            return grad_out

        rk = int(ctx.rank)
        n = int(ctx.local_n)
        start = rk * n
        end = start + n

        # Each rank's loss produces gradients for every gathered slice.
        # We need to SUM the slice corresponding to this rank's input across
        # all ranks so the input receives gradients from all other ranks' losses.
        grad_input = grad_out.contiguous()[start:end]
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM)
        return grad_input


def _dist_all_gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    """All-gather a tensor across ranks with autograd support.

    Assumes x has the same shape on every rank.
    """

    if not _is_dist_active():
        return x
    return _AllGatherWithGrad.apply(x)


# -----------------------------------------------------------------------------
# Training defaults
# -----------------------------------------------------------------------------

OPTIMIZER = torch.optim.AdamW
LR = 3e-4
WEIGHT_DECAY = 0.01

# text generation evaluation defaults
BEAM_WIDTH = 3
DECODE_LENGTH = MAX_LEN_UTTERANCE
LENGTH_PENALTY_ALPHA = 0.0

PRINT_EVAL_TEXTGEN_EXAMPLE_IDS = range(10)


class MultiModalLitModel(pl.LightningModule):
    """Lightning module for SAYCam CVCL with Object-File MIL + Prototype Memory (Plan A)."""

    def __init__(self, vision_encoder: nn.Module, text_encoder: nn.Module, args: Optional[argparse.Namespace]):
        super().__init__()
        self.args: Dict[str, Any] = vars(args) if args is not None else {}

        # -------------------------
        # Optimizer
        # -------------------------
        self.optimizer_class = self.args.get("optimizer", OPTIMIZER)
        self.lr = float(self.args.get("lr", LR))
        self.weight_decay = float(self.args.get("weight_decay", WEIGHT_DECAY))
        self.lr_scheduler = bool(self.args.get("lr_scheduler", False))
        self.factor = float(self.args.get("factor", 0.1))
        self.patience = int(self.args.get("patience", 20))

        # -------------------------
        # Objective weights
        # -------------------------
        self.lambda_mm = float(self.args.get("lambda_mm", 1.0))
        # Keep LM losses off unless explicitly enabled.
        self.lambda_lm = float(self.args.get("lambda_lm", 0.0))
        self.lambda_ar = float(self.args.get("lambda_ar", 0.0))
        self.optimize_unused = bool(self.args.get("optimize_unused", False))

        # Text generation eval
        self.eval_textgen = bool(self.args.get("eval_textgen", False))
        self.beam_width = int(self.args.get("beam_width", BEAM_WIDTH))
        self.decode_length = int(self.args.get("decode_length", DECODE_LENGTH))
        self.length_penalty_alpha = float(self.args.get("length_penalty_alpha", LENGTH_PENALTY_ALPHA))

        # -------------------------
        # Backbones
        # -------------------------
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        self.model = MultiModalModel(self.vision_encoder, self.text_encoder, args)
        self.language_model = LanguageModel(self.text_encoder, args)

        # -------------------------
        # Vocab / tokenizer
        # -------------------------
        self.vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab.json")
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.nlp = spacy.load("en_core_web_sm")

        # Caches
        # - noun_cache: raw string -> tuple(nouns)
        # - phrase_token_cache: phrase string -> (token_ids, token_len)
        self._noun_cache = {}
        self._noun_cache_max = 50000
        self._phrase_token_cache = {}

        self.id2tok: Optional[Dict[int, str]] = None
        if isinstance(self.vocab, dict):
            self.id2tok = {int(i): tok for tok, i in self.vocab.items()}
        elif hasattr(self.vocab, "itos"):
            itos = list(self.vocab.itos)
            self.id2tok = {i: tok for i, tok in enumerate(itos)}

        ignore_tokens = ["<pad>", "<unk>", "<sos>", "<eos>", ".", ",", "?", "!", "...", "..", "...."]
        self.ignore_ids = {self.vocab[t] for t in ignore_tokens if t in self.vocab}

        # ------------------------------------------------------------------
        # SAM concept registry (optional)
        #
        # Many SAM-prepacked datasets store masks in a *local* concept space
        # defined by <sam_prepacked_dir>/concept_vocab.json. The registry
        # provides:
        #   * local -> global ID remapping (optional global concept list)
        #   * optional frequency-based filtering (min_masks_per_concept)
        #   * inverse-frequency weights for long-tail balancing
        #
        # IMPORTANT: this does **not** itself load masks; it helps interpret
        # the concept IDs associated with masks that the DataModule provides.
        # ------------------------------------------------------------------
        self.sam_prepacked_dir = self.args.get("sam_prepacked_dir", None)
        self.sam_concept_frequency_json = self.args.get("sam_concept_frequency_json", None)
        self.sam_min_masks_per_concept = int(self.args.get("sam_min_masks_per_concept", 0))
        self.sam_concept_list_file = self.args.get("sam_concept_list_file", None)
        self.sam_weight_alpha = float(self.args.get("sam_weight_alpha", 0.0))
        self.sam_weight_clip_min = float(self.args.get("sam_weight_clip_min", 0.25))
        self.sam_weight_clip_max = float(self.args.get("sam_weight_clip_max", 4.0))
        self.sam_registry_verbose = bool(self.args.get("sam_registry_verbose", False))
        self.sam_use_concept_weights = bool(self.args.get("sam_use_concept_weights", False))

        self.sam_registry: Optional[SamConceptRegistry] = None
        if self.sam_prepacked_dir is not None and build_sam_concept_registry is not None:
            try:
                self.sam_registry = build_sam_concept_registry(
                    sam_prepacked_dir=self.sam_prepacked_dir,
                    concept_frequency_json=self.sam_concept_frequency_json,
                    min_masks_per_concept=self.sam_min_masks_per_concept,
                    concept_list_file=self.sam_concept_list_file,
                    alpha=self.sam_weight_alpha,
                    clip_min=self.sam_weight_clip_min,
                    clip_max=self.sam_weight_clip_max,
                    verbose=self.sam_registry_verbose,
                )
            except Exception as e:  # pragma: no cover
                # Fail soft: training can proceed (MIL will simply not use
                # concept-aware filtering/weights).
                if int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))) == 0:
                    print(f"[multimodal_lit] WARNING: failed to build SamConceptRegistry: {e}")
                self.sam_registry = None

        # One-time warning if MIL is enabled but the dataloader never supplies SAM masks.
        self._warned_missing_sam_mask = False

        # ------------------------------------------------------------------
        # Object-file MIL (Plan A)
        # ------------------------------------------------------------------
        self.mil_enable = bool(self.args.get("mil_enable", True))
        # If True, also compute MIL/track losses on val/test (for monitoring),
        # but **never** update prototype EMA with val/test data.
        self.mil_run_val = bool(self.args.get("mil_run_val", True))
        self.mil_lambda = float(self.args.get("mil_lambda", 0.10))
        # Temperature used for logsumexp over tracks and for contrastive logits.
        self.mil_tau = float(self.args.get("mil_tau", 0.05))
        self.mil_min_mask_area = float(self.args.get("mil_min_mask_area", 0.01))

        self.mil_track = bool(self.args.get("mil_track", True))
        self.mil_track_sim_thresh = float(self.args.get("mil_track_sim_thresh", 0.55))
        self.mil_track_max_tracks = int(self.args.get("mil_track_max_tracks", 16))

        # Mask->embedding pooling
        self.mil_obj_ring_weight = float(self.args.get("mil_obj_ring_weight", 0.05))
        self.mil_obj_ring_px_fmap = int(self.args.get("mil_obj_ring_px_fmap", 1))

        # Patch fallback when SAM masks vanish at feature-map resolution or are missing.
        #   * If a mask becomes empty after downsampling to the feature-map grid,
        #     we sample a patch embedding around the mask centroid.
        #   * If a frame has no valid masks, we sample top-k patch embeddings from the feature map.
        self.mil_patch_topk = int(self.args.get("mil_patch_topk", 4))
        self.mil_patch_radius = int(self.args.get("mil_patch_radius", 1))


        # MIL text query mode (sentence vs nouns)
        self.mil_text_mode = str(self.args.get("mil_text_mode", "sentence")).lower()
        self.mil_noun_max = int(self.args.get("mil_noun_max", 5))
        self.mil_noun_min_chars = int(self.args.get("mil_noun_min_chars", 2))
        self.mil_noun_use_lemma = bool(self.args.get("mil_noun_use_lemma", True))
        self.mil_noun_keep_propn = bool(self.args.get("mil_noun_keep_propn", True))
        self.mil_noun_vocab_only = bool(self.args.get("mil_noun_vocab_only", True))
        self.mil_noun_dedup = bool(self.args.get("mil_noun_dedup", True))

        # Pivot C: object<->concept alignment using SAM concept IDs (optional)
        self.sam_concept_align_enable = bool(self.args.get("sam_concept_align_enable", False))
        self.sam_concept_align_lambda = float(self.args.get("sam_concept_align_lambda", 0.05))
        self.sam_concept_align_tau = float(self.args.get("sam_concept_align_tau", 0.07))

        # Alignment gating weight w_align
        self.w_align_sim0 = float(self.args.get("w_align_sim0", 0.10))
        self.w_align_simscale = float(self.args.get("w_align_simscale", 0.05))
        self.w_align_warmup_steps = int(self.args.get("w_align_warmup_steps", 500))
        self.w_align_min = float(self.args.get("w_align_min", 0.05))

        # Track coherence loss
        self.track_coh_enable = bool(self.args.get("track_coh_enable", True))
        self.track_coh_lambda = float(self.args.get("track_coh_lambda", 0.05))
        self.track_coh_match_thresh = float(self.args.get("track_coh_match_thresh", 0.30))
        self.track_coh_min_frames = int(self.args.get("track_coh_min_frames", 2))

        # Global-object agreement loss
        self.go_enable = bool(self.args.get("go_enable", True))
        self.go_lambda = float(self.args.get("go_lambda", 0.05))

        # ------------------------------------------------------------------
        # Prototype memory (visual vocabulary)
        # ------------------------------------------------------------------
        self.embedding_dim = int(self.args.get("embedding_dim", 128))

        # Some experimental branches (e.g. noun-only alignment pivots) refer to
        # the shared image/text contrastive embedding dimensionality as
        # `proj_dim`. Historically this name came from an earlier projection-head
        # implementation. In the current codebase the correct dimension is the
        # same as `embedding_dim`.
        #
        # Keep this attribute to avoid crashes when those branches are enabled.
        self.proj_dim = int(self.embedding_dim)

        self.proto_enable = bool(self.args.get("proto_enable", True))
        self.proto_num = int(self.args.get("proto_num", 64))
        self.proto_tau = float(self.args.get("proto_tau", 0.07))

        self.proto_use_sinkhorn = bool(self.args.get("proto_use_sinkhorn", True))
        self.proto_sinkhorn_iters = int(self.args.get("proto_sinkhorn_iters", 3))
        self.proto_sinkhorn_epsilon = float(self.args.get("proto_sinkhorn_epsilon", 0.05))
        self.proto_sinkhorn_min_samples = int(self.args.get("proto_sinkhorn_min_samples", 32))

        self.proto_ema_decay = float(self.args.get("proto_ema_decay", 0.99))
        self.proto_ema_eps = float(self.args.get("proto_ema_eps", 1e-3))
        self.proto_ema_ddp_sync = bool(self.args.get("proto_ema_ddp_sync", True))

        # Warm-start (one-shot) for prototypes
        self.proto_warm_start = bool(self.args.get("proto_warm_start", True))
        self.proto_warm_min_local = int(self.args.get("proto_warm_min_local", 128))
        # Select warm-start examples by *embedding-space* text-track cosine.
        self.proto_warm_sim_thresh = float(self.args.get("proto_warm_sim_thresh", 0.25))
        self.proto_warm_max_total = int(self.args.get("proto_warm_max_total", 4096))
        self.proto_warm_kmeans_iters = int(self.args.get("proto_warm_kmeans_iters", 10))

        self._proto_warm_started = False
        self._proto_warm_buffer: List[torch.Tensor] = []  # CPU tensors (D,)

        self.proto_mem: Optional[PrototypeMemory] = None
        if self.proto_enable:
            self.proto_mem = PrototypeMemory(
                embedding_dim=self.embedding_dim,
                num_prototypes=self.proto_num,
                tau=self.proto_tau,
                use_sinkhorn=self.proto_use_sinkhorn,
                sinkhorn_iters=self.proto_sinkhorn_iters,
                sinkhorn_epsilon=self.proto_sinkhorn_epsilon,
                sinkhorn_min_samples=self.proto_sinkhorn_min_samples,
                ema_decay=self.proto_ema_decay,
                ema_eps=self.proto_ema_eps,
                ema_ddp_sync=self.proto_ema_ddp_sync,
            )

        # ------------------------------------------------------------------
        # Null track and optional image adapter
        # ------------------------------------------------------------------
        # Null is used when there are no valid tracks (or to let MIL ignore a sample).
        self.null_obj = nn.Parameter(torch.randn(self.embedding_dim, dtype=torch.float32))

        self.img_adapter_enable = bool(self.args.get("img_adapter_enable", True))
        if self.img_adapter_enable:
            self.img_adapter = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            with torch.no_grad():
                self.img_adapter.weight.copy_(torch.eye(self.embedding_dim))
        else:
            self.img_adapter = nn.Identity()

        # ------------------------------------------------------------------
        # Debug visualization
        # ------------------------------------------------------------------
        self.debug_save_tracks = bool(self.args.get("debug_save_tracks", False))
        self.debug_tracks_topk = int(self.args.get("debug_tracks_topk", 6))
        self.debug_tracks_every_n_epochs = int(self.args.get("debug_tracks_every_n_epochs", 10))
        self.debug_tracks_overlay_alpha = float(self.args.get("debug_tracks_overlay_alpha", 0.45))

        self._dbg_track_heap: List[Tuple[float, Dict[str, Any]]] = []

        # IMPORTANT: keep save_hyperparameters() unfiltered so load_from_checkpoint can reconstruct.
        self.save_hyperparameters()

    # ------------------------------------------------------------------
    # DDP tensor helpers
    # ------------------------------------------------------------------
    def _dist_all_gather_no_grad(self, x: torch.Tensor) -> torch.Tensor:
        return _dist_all_gather_no_grad(x)

    def _dist_all_gather_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        return _dist_all_gather_with_grad(x)

    def _dist_rank(self) -> int:
        return _dist_rank()

    def _dist_world(self) -> int:
        return _dist_world()

    # ------------------------------------------------------------------
    # Argparse
    # ------------------------------------------------------------------
    @staticmethod
    def add_to_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # NOTE: train.py wires multiple ArgumentParser groups together
        # (data module + lit module). Some flags (e.g., SAM dataset flags)
        # may already be registered by the data module. We therefore add a
        # small helper to avoid hard crashes on duplicate option strings.
        def _safe_add(*args, **kwargs):
            try:
                parser.add_argument(*args, **kwargs)
            except argparse.ArgumentError:
                return

        # Optimizer
        parser.add_argument("--optimizer", type=lambda o: getattr(torch.optim, o), default=OPTIMIZER)
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
        parser.add_argument("--lr_scheduler", action="store_true")
        parser.add_argument("--factor", type=float, default=0.1)
        parser.add_argument("--patience", type=int, default=20)

        # Main loss
        parser.add_argument("--lambda_mm", type=float, default=1.0)
        parser.add_argument("--lambda_lm", type=float, default=0.0)
        parser.add_argument("--lambda_ar", type=float, default=0.0)
        parser.add_argument("--optimize_unused", action="store_true")

        # Optional text generation eval
        parser.add_argument("--eval_textgen", action="store_true")
        parser.add_argument("--beam_width", type=int, default=BEAM_WIDTH)
        parser.add_argument("--decode_length", type=int, default=DECODE_LENGTH)
        parser.add_argument("--length_penalty_alpha", type=float, default=LENGTH_PENALTY_ALPHA)

        # Object-file MIL
        parser.add_argument("--mil_enable", dest="mil_enable", action="store_true")
        parser.add_argument("--no_mil", dest="mil_enable", action="store_false")
        parser.set_defaults(mil_enable=True)
        parser.add_argument(
            "--mil_run_val",
            dest="mil_run_val",
            action="store_true",
            help="Compute MIL/track losses on val/test (monitoring only; no EMA updates).",
        )
        parser.add_argument("--no_mil_run_val", dest="mil_run_val", action="store_false", help=argparse.SUPPRESS)
        parser.set_defaults(mil_run_val=True)
        parser.add_argument("--mil_lambda", type=float, default=0.10)
        parser.add_argument("--mil_tau", type=float, default=0.05)
        parser.add_argument("--mil_min_mask_area", type=float, default=0.01)

        # SAM concept registry (optional)
        # These flags are used by SAM-prepacked datasets and/or by the MIL
        # losses to interpret concept IDs and apply frequency filtering.
        _safe_add(
            "--sam_prepacked_dir",
            type=str,
            default=None,
            help="Path to SAM prepacked root (must contain concept_vocab.json).",
        )
        _safe_add(
            "--sam_concept_frequency_json",
            type=str,
            default=None,
            help="Optional path to concept_frequency.json (defaults to <sam_prepacked_dir>/concept_frequency.json).",
        )
        _safe_add(
            "--sam_min_masks_per_concept",
            type=int,
            default=0,
            help="If >0 and concept_frequency.json is available, drop SAM concepts with fewer than this many masks.",
        )
        _safe_add(
            "--sam_concept_list_file",
            type=str,
            default=None,
            help="Optional JSON file defining the global concept ID space (list or dict).",
        )
        _safe_add(
            "--sam_weight_alpha",
            type=float,
            default=0.0,
            help="Inverse-frequency reweighting exponent alpha (0 disables).",
        )
        _safe_add("--sam_weight_clip_min", type=float, default=0.25, help="Min clip for concept weights.")
        _safe_add("--sam_weight_clip_max", type=float, default=4.0, help="Max clip for concept weights.")
        _safe_add(
            "--sam_use_concept_weights",
            action="store_true",
            help="If set, reweight MIL/coherence/GO losses by SAM concept frequency weights (approx. via best track).",
        )
        _safe_add(
            "--sam_registry_verbose",
            action="store_true",
            help="Print SAM registry summary on rank0.",
        )

        parser.add_argument("--mil_track", dest="mil_track", action="store_true")
        parser.add_argument("--mil_no_track", dest="mil_track", action="store_false", help=argparse.SUPPRESS)
        parser.set_defaults(mil_track=True)
        parser.add_argument("--mil_track_sim_thresh", type=float, default=0.55)
        parser.add_argument("--mil_track_max_tracks", type=int, default=16)

        parser.add_argument("--mil_obj_ring_weight", type=float, default=0.05)
        parser.add_argument("--mil_obj_ring_px_fmap", type=int, default=1)

        # Patch fallback (when SAM masks vanish or are missing)
        parser.add_argument("--mil_patch_topk", type=int, default=4, help="When a frame has no valid SAM masks, sample this many top patches from the feature map.")
        parser.add_argument("--mil_patch_radius", type=int, default=1, help="Radius (in feature-map cells) for averaging around a sampled patch location.")


        # MIL text query mode (sentence vs nouns)
        parser.add_argument(
            "--mil_text_mode",
            type=str,
            default="sentence",
            choices=["sentence", "noun_avg", "noun_multi"],
            help="Text representation for MIL losses: sentence embedding, averaged nouns, or multi-noun queries (t2i only).",
        )
        parser.add_argument("--mil_noun_max", type=int, default=5, help="Max #nouns per sample for noun-based MIL.")
        parser.add_argument("--mil_noun_min_chars", type=int, default=2, help="Minimum #characters for a noun token to be kept.")
        parser.add_argument("--mil_noun_use_lemma", dest="mil_noun_use_lemma", action="store_true", help="Use spaCy lemma for noun tokens (default).")
        parser.add_argument("--mil_noun_no_lemma", dest="mil_noun_use_lemma", action="store_false", help=argparse.SUPPRESS)
        parser.set_defaults(mil_noun_use_lemma=True)
        parser.add_argument("--mil_noun_keep_propn", dest="mil_noun_keep_propn", action="store_true", help="Keep proper nouns (PROPN) as nouns (default).")
        parser.add_argument("--mil_noun_drop_propn", dest="mil_noun_keep_propn", action="store_false", help=argparse.SUPPRESS)
        parser.set_defaults(mil_noun_keep_propn=True)
        parser.add_argument(
            "--mil_noun_vocab_only",
            dest="mil_noun_vocab_only",
            action="store_true",
            help="Keep only nouns present in vocab (avoid collapsing to <unk>). (default)",
        )
        parser.add_argument("--mil_noun_allow_unk", dest="mil_noun_vocab_only", action="store_false", help=argparse.SUPPRESS)
        parser.set_defaults(mil_noun_vocab_only=True)
        parser.add_argument("--mil_noun_dedup", dest="mil_noun_dedup", action="store_true", help="Deduplicate nouns within a sentence (default).")
        parser.add_argument("--mil_noun_no_dedup", dest="mil_noun_dedup", action="store_false", help=argparse.SUPPRESS)
        parser.set_defaults(mil_noun_dedup=True)

        # Pivot C: align SAM-mask embeddings to their SAM concept names (optional)
        parser.add_argument(
            "--sam_concept_align_enable",
            action="store_true",
            help="Auxiliary loss: classify SAM-mask embeddings by their concept names (object->concept InfoNCE over batch-unique concepts).",
        )
        parser.add_argument("--sam_concept_align_lambda", type=float, default=0.05)
        parser.add_argument("--sam_concept_align_tau", type=float, default=0.07)

        # w_align gating
        parser.add_argument("--w_align_sim0", type=float, default=0.10)
        parser.add_argument("--w_align_simscale", type=float, default=0.05)
        parser.add_argument("--w_align_warmup_steps", type=int, default=500)
        parser.add_argument("--w_align_min", type=float, default=0.05)

        # Track coherence
        parser.add_argument("--track_coh_enable", dest="track_coh_enable", action="store_true")
        parser.add_argument("--no_track_coh", dest="track_coh_enable", action="store_false")
        parser.set_defaults(track_coh_enable=True)
        parser.add_argument("--track_coh_lambda", type=float, default=0.05)
        parser.add_argument("--track_coh_match_thresh", type=float, default=0.30)
        parser.add_argument("--track_coh_min_frames", type=int, default=2)

        # Global-object agreement
        parser.add_argument("--go_enable", dest="go_enable", action="store_true")
        parser.add_argument("--no_go", dest="go_enable", action="store_false")
        parser.set_defaults(go_enable=True)
        parser.add_argument("--go_lambda", type=float, default=0.05)

        # Prototype memory
        # NOTE: embedding_dim is typically defined elsewhere in the codebase (e.g. model args).
        # When train.py composes multiple argparse groups, re-adding the same option string
        # raises an argparse.ArgumentError. We therefore add it only if it doesn't already exist.
        try:
            parser.add_argument("--embedding_dim", type=int, default=128)
        except argparse.ArgumentError:  # already registered by another group
            pass
        parser.add_argument("--proto_enable", dest="proto_enable", action="store_true")
        parser.add_argument("--no_proto", dest="proto_enable", action="store_false")
        parser.set_defaults(proto_enable=True)
        parser.add_argument("--proto_num", type=int, default=64)
        parser.add_argument("--proto_tau", type=float, default=0.07)

        parser.add_argument("--proto_use_sinkhorn", dest="proto_use_sinkhorn", action="store_true")
        parser.add_argument("--proto_no_sinkhorn", dest="proto_use_sinkhorn", action="store_false")
        parser.set_defaults(proto_use_sinkhorn=True)
        parser.add_argument("--proto_sinkhorn_iters", type=int, default=3)
        parser.add_argument("--proto_sinkhorn_epsilon", type=float, default=0.05)
        parser.add_argument("--proto_sinkhorn_min_samples", type=int, default=32)
        parser.add_argument("--proto_ema_decay", type=float, default=0.99)
        parser.add_argument("--proto_ema_eps", type=float, default=1e-3)
        parser.add_argument("--proto_ema_ddp_sync", dest="proto_ema_ddp_sync", action="store_true")
        parser.add_argument("--proto_ema_no_ddp_sync", dest="proto_ema_ddp_sync", action="store_false")
        parser.set_defaults(proto_ema_ddp_sync=True)

        # Warm start
        parser.add_argument("--proto_warm_start", action="store_true")
        parser.add_argument("--proto_no_warm_start", dest="proto_warm_start", action="store_false")
        parser.set_defaults(proto_warm_start=True)
        parser.add_argument("--proto_warm_min_local", type=int, default=128)
        parser.add_argument("--proto_warm_sim_thresh", type=float, default=0.25)
        parser.add_argument("--proto_warm_max_total", type=int, default=4096)
        parser.add_argument("--proto_warm_kmeans_iters", type=int, default=10)

        # Adapter
        parser.add_argument("--img_adapter_enable", dest="img_adapter_enable", action="store_true")
        parser.add_argument("--no_img_adapter", dest="img_adapter_enable", action="store_false")
        parser.set_defaults(img_adapter_enable=True)

        # Debug
        parser.add_argument("--debug_save_tracks", action="store_true")
        parser.add_argument("--debug_tracks_topk", type=int, default=6)
        parser.add_argument("--debug_tracks_every_n_epochs", type=int, default=10)
        parser.add_argument("--debug_tracks_overlay_alpha", type=float, default=0.45)

        return parser

    # ------------------------------------------------------------------
    # Optim
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            name_l = name.lower()
            # Don't decay biases / normalization params.
            # Also keep small auxiliary params (null track, adapter) out of weight decay by default.
            if (
                name.endswith(".bias")
                or "bn" in name_l
                or "ln" in name_l
                or "norm" in name_l
                or name == "null_obj"
                or name_l.startswith("img_adapter")
            ):
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = self.optimizer_class(
            [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=self.lr,
        )

        if not self.lr_scheduler:
            return optimizer

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=self.factor, patience=self.patience
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val/infonce_loss"}}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(self, x, y, y_len):
        return self.model(x, y, y_len)

    @staticmethod
    def load_model(model_name: str = "cvcl"):
        if model_name == "cvcl":
            checkpoint_name = "cvcl_s_dino_resnext50_embedding"
            checkpoint = hf_hub_download(repo_id="wkvong/" + checkpoint_name, filename=checkpoint_name + ".ckpt")
            model = MultiModalLitModel.load_from_checkpoint(checkpoint_path=checkpoint)
        else:
            raise ValueError("Model name not found.")

        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return model, preprocess

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to obtain image features."""

        image_features, _ = self.model.encode_image(x)
        return image_features

    def encode_text(self, y: torch.Tensor, y_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode text to obtain text features."""

        text_features, _ = self.model.encode_text(y, y_len)
        return text_features

    def tokenize(self, texts):
        """Tokenize texts to obtain tokens and token lengths."""

        max_seq_len = 25
        if isinstance(texts, str):
            texts = [texts]
        all_tokens = []
        token_lengths = []
        for text in texts:
            doc = self.nlp(text)
            word_tokens = [token.text for token in doc]
            if len(word_tokens) > max_seq_len - 2:
                word_tokens = word_tokens[: max_seq_len - 2]
            token_length = len(word_tokens) + 2
            tokens = (
                [self.vocab["<sos>"]]
                + [self.vocab.get(token, self.vocab["<unk>"]) for token in word_tokens]
                + [self.vocab["<eos>"]]
                + [self.vocab["<pad>"]] * (max_seq_len - len(word_tokens) - 2)
            )
            all_tokens.append(tokens)
            token_lengths.append(token_length)
        tokens = torch.tensor(all_tokens, dtype=torch.long)
        token_lengths = torch.tensor(token_lengths, dtype=torch.long)
        return tokens, token_lengths


    @staticmethod
    def _raw_to_text(raw) -> str:
        """Best-effort conversion of dataset raw_y element to a single string."""
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, (list, tuple)):
            # common case: ["utterance"] or [ref1, ref2, ...]
            parts = []
            for r in raw:
                if isinstance(r, str):
                    parts.append(r)
                elif isinstance(r, (list, tuple)) and len(r) > 0 and isinstance(r[0], str):
                    parts.append(r[0])
                else:
                    try:
                        parts.append(str(r))
                    except Exception:
                        pass
            return " ".join([p for p in parts if p])
        try:
            return str(raw)
        except Exception:
            return ""

    def _extract_nouns(self, text: str) -> List[str]:
        """Extract a small list of noun-like tokens from a caption using spaCy POS tags."""
        if text is None:
            return []
        key = text.strip().lower()
        if not key:
            return []

        cached = self._noun_cache.get(key)
        if cached is not None:
            return list(cached)

        doc = self.nlp(key)
        out: List[str] = []
        seen = set()
        for tok in doc:
            if tok.is_space or tok.is_punct or tok.like_num:
                continue
            if tok.is_stop:
                continue
            pos = tok.pos_
            if pos == "NOUN" or (self.mil_noun_keep_propn and pos == "PROPN"):
                w = tok.lemma_.lower() if self.mil_noun_use_lemma else tok.text.lower()
                w = w.strip()
                if len(w) < self.mil_noun_min_chars:
                    continue
                if self.mil_noun_vocab_only and (w not in self.vocab):
                    continue
                if self.mil_noun_dedup:
                    if w in seen:
                        continue
                    seen.add(w)
                out.append(w)
                if len(out) >= self.mil_noun_max:
                    break

        # cache
        self._noun_cache[key] = tuple(out)
        if len(self._noun_cache) > self._noun_cache_max:
            self._noun_cache.clear()
        return out

    def _tokenize_phrases(self, phrases: List[str], *, max_seq_len: int = 25) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a list of short phrases using the loaded vocab (no spaCy tokenization).

        Returns:
            tokens: (N, max_seq_len) LongTensor
            lengths: (N,) LongTensor (includes <sos>, <eos>)
        """
        pad = int(self.vocab.get("<pad>", PAD_TOKEN_ID))
        sos = int(self.vocab.get("<sos>", SOS_TOKEN_ID))
        eos = int(self.vocab.get("<eos>", EOS_TOKEN_ID))
        unk = int(self.vocab.get("<unk>", UNK_TOKEN_ID))

        seqs: List[List[int]] = []
        lens: List[int] = []
        for phrase in phrases:
            key = (phrase or "").strip().lower()
            if key in self._phrase_token_cache:
                seq, ln = self._phrase_token_cache[key]
                seqs.append(list(seq))
                lens.append(int(ln))
                continue

            words = [w for w in re.split(r"\s+", key) if w]
            ids = [self.vocab.get(w, unk) for w in words][: max_seq_len - 2]
            seq = [sos] + ids + [eos]
            ln = len(seq)
            if ln < max_seq_len:
                seq = seq + [pad] * (max_seq_len - ln)
            else:
                seq = seq[:max_seq_len]
                ln = max_seq_len

            self._phrase_token_cache[key] = (tuple(seq), ln)
            seqs.append(seq)
            lens.append(ln)

        tokens = torch.tensor(seqs, dtype=torch.long)
        lengths = torch.tensor(lens, dtype=torch.long)
        return tokens, lengths

    def _encode_phrases(self, phrases: List[str], *, device: torch.device) -> torch.Tensor:
        """Encode short phrases to normalized text embeddings."""
        if len(phrases) == 0:
            # caller should handle empties
            return torch.empty((0, int(self.proj_dim)), device=device)
        tok, tok_len = self._tokenize_phrases(phrases, max_seq_len=25)
        tok = tok.to(device)
        tok_len = tok_len.to(device)
        feat, _ = self.model.encode_text(tok, tok_len)
        return l2norm_obj(feat.float())

    def _noun_features_from_raw_y(
        self,
        raw_y,
        *,
        B: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract up to N nouns per sample and return (B,N,D) features + validity mask."""
        Nn = max(1, int(self.mil_noun_max))
        feat = torch.zeros((B, Nn, int(self.proj_dim)), device=device)
        valid = torch.zeros((B, Nn), dtype=torch.bool, device=device)
        count = torch.zeros((B,), dtype=torch.long, device=device)

        if raw_y is None:
            return feat, valid, count

        # raw_y is typically a list (len B) of ["caption"]
        phrases = []
        ij = []
        for i in range(B):
            try:
                raw_i = raw_y[i]
            except Exception:
                raw_i = raw_y
            text = self._raw_to_text(raw_i)
            nouns = self._extract_nouns(text)
            if len(nouns) == 0:
                continue
            nouns = nouns[:Nn]
            count[i] = len(nouns)
            for j, w in enumerate(nouns):
                phrases.append(w)
                ij.append((i, j))

        if len(phrases) == 0:
            return feat, valid, count

        emb = self._encode_phrases(phrases, device=device)  # (L,D)
        i_idx = torch.tensor([p[0] for p in ij], device=device, dtype=torch.long)
        j_idx = torch.tensor([p[1] for p in ij], device=device, dtype=torch.long)

        feat = feat.index_put((i_idx, j_idx), emb)
        valid = valid.index_put((i_idx, j_idx), torch.ones(len(ij), device=device, dtype=torch.bool))
        return feat, valid, count

    # ------------------------------------------------------------------
    # Batch parsing
    # ------------------------------------------------------------------
    def _split_batch(self, batch):
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
            f"Unexpected batch structure: type={type(batch)} "
            f"len={len(batch) if isinstance(batch, (list, tuple)) else 'n/a'}"
        )

    # ------------------------------------------------------------------
    # DDP safety (avoid `find_unused_parameters` requirement)
    # ------------------------------------------------------------------
    def _ddp_is_active(self) -> bool:
        try:
            return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        except Exception:
            return False

    def _ddp_find_unused_parameters_enabled(self) -> bool:
        """Return True if the PL strategy is configured with find_unused_parameters=True."""
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return False
        strat = getattr(trainer, "strategy", None)
        if strat is None:
            return False
        fup = getattr(strat, "find_unused_parameters", None)
        return bool(fup) if fup is not None else False

    def _tie_params_if_needed(self, loss: torch.Tensor, params: List[torch.Tensor]) -> torch.Tensor:
        """If DDP is active and find_unused_parameters=False, tie params into the graph via 0-weight sums."""
        if self._ddp_is_active() and (not self._ddp_find_unused_parameters_enabled()):
            tie = loss.new_tensor(0.0)
            for p in params:
                if p is None:
                    continue
                if torch.is_tensor(p) and p.requires_grad:
                    tie = tie + 0.0 * p.sum()
            return loss + tie
        return loss

    def _tie_module_params_if_needed(self, loss: torch.Tensor, module: nn.Module) -> torch.Tensor:
        if not (self._ddp_is_active() and (not self._ddp_find_unused_parameters_enabled())):
            return loss
        tie = loss.new_tensor(0.0)
        for p in module.parameters():
            if p.requires_grad:
                tie = tie + 0.0 * p.sum()
        return loss + tie

    # ------------------------------------------------------------------
    # LM helpers
    # ------------------------------------------------------------------
    def calculate_ce_loss(
        self,
        y,
        y_len,
        x=None,
        outputs=None,
        image_features=None,
        image_feature_map=None,
        return_image_features=False,
        **kwargs,
    ):
        # If captioning or attention is enabled, we must provide image features/maps and cannot reuse outputs.
        if self.language_model.text_encoder.captioning or self.language_model.text_encoder.has_attention:
            if image_features is None:
                image_features, image_feature_map = self.model.encode_image(x)
            outputs = None
        else:
            image_features, image_feature_map = None, None

        ret = self.language_model.calculate_ce_loss(
            y,
            y_len,
            outputs=outputs,
            image_features=image_features if self.language_model.text_encoder.captioning else None,
            image_feature_map=image_feature_map if self.language_model.text_encoder.has_attention else None,
            **kwargs,
        )
        if return_image_features:
            ret = ret + (image_features, image_feature_map)
        return ret

    # ------------------------------------------------------------------
    # MIL helpers: keep BatchNorm stable during extra forwards
    # ------------------------------------------------------------------
    @contextmanager
    def _vision_backbone_eval_for_mil(self):
        """Temporarily set ONLY the vision backbone module to eval(), then restore."""

        ve = getattr(self.model, "image_embed", None)
        backbone = getattr(ve, "model", None) if ve is not None else None
        if backbone is None:
            yield
            return

        was_training = backbone.training
        backbone.eval()
        try:
            yield
        finally:
            backbone.train(was_training)

    def _get_layer4_fmap(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Extract a layer4 feature map (N,C4,H4,W4). Falls back to encode_image's fmap if needed."""

        ve = getattr(self.model, "image_embed", None)
        if ve is None:
            raise RuntimeError("Model is missing image_embed; cannot compute fmap.")

        # Preferred path: CNN ResNet/ResNeXt with forward_to_layer3.
        backbone = getattr(ve, "model", None)
        if hasattr(ve, "forward_to_layer3") and backbone is not None and hasattr(backbone, "layer4"):
            with self._vision_backbone_eval_for_mil():
                with torch.no_grad():
                    f3 = ve.forward_to_layer3(x_flat)
                    fmap4 = backbone.layer4(f3)
            return fmap4.detach()

        # Fallback path: use encode_image feature map.
        with torch.no_grad():
            _g, fmap = self.model.encode_image(x_flat)
        if fmap is None:
            raise RuntimeError("encode_image did not return a feature map; cannot pool masks.")
        return fmap.detach()

    def _pool_object_embeddings_from_layer4(
        self,
        *,
        fmap4: torch.Tensor,  # (N,C4,H4,W4)
        masks_img: torch.Tensor,  # (N,K,1,H,W) float
        valid: torch.Tensor,  # (N,K) bool
    ) -> torch.Tensor:
        """Pool mask embeddings on layer4 fmap and project with backbone.fc into embedding space."""

        device = fmap4.device
        N, C4, H4, W4 = fmap4.shape
        Nk, K, _, H, W = masks_img.shape
        if Nk != N:
            raise ValueError(f"masks_img N mismatch: fmap4 N={N}, masks N={Nk}")
        if K == 0 or N == 0:
            return torch.empty((N, 0, self.embedding_dim), device=device, dtype=torch.float32)

        # Downsample masks to fmap resolution using area to preserve silhouette mass.
        masks_ds = F.interpolate(
            masks_img.view(N * K, 1, H, W).float(),
            size=(H4, W4),
            mode="area",
        ).view(N, K, 1, H4, W4)
        masks_ds = masks_ds.clamp(0.0, 1.0)

        # Optional context ring in fmap space.
        if self.mil_obj_ring_weight > 0.0 and self.mil_obj_ring_px_fmap > 0:
            ring = context_ring_masks((masks_ds > 0.0).float(), ring_px=int(self.mil_obj_ring_px_fmap))
            masks_w = (masks_ds + float(self.mil_obj_ring_weight) * ring).clamp(0.0, 1.0)
        else:
            masks_w = masks_ds

        pooled = masked_pool_k(fmap4, masks_w, eps=1e-6)  # (N,K,C4)

        # Project pooled C4 vectors using the existing backbone head into embedding space.
        ve = getattr(self.model, "image_embed", None)
        backbone = getattr(ve, "model", None) if ve is not None else None
        proj = getattr(backbone, "fc", None) if backbone is not None else None
        if proj is None:
            raise RuntimeError("Backbone has no .fc; cannot project pooled layer4 vectors.")

        z = proj(pooled.reshape(N * K, C4)).view(N, K, -1).float()  # (N,K,D)
        z = l2norm_obj(z)
        return z

    def _get_bag_tensors(
        self,
        x: torch.Tensor,
        meta: Optional[dict],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return bagged images + optional SAM masks.

        Returns:
          x_bag: (B,M,3,H,W)
          sam_mask: optional (B,M,K,1,H,W)
          sam_concept: optional (B,M,K) concept IDs (local or global)
        """

        if x.ndim == 4:
            x_bag = x.unsqueeze(1)
        elif x.ndim == 5:
            x_bag = x
        else:
            raise ValueError(f"Expected x 4D or 5D, got {tuple(x.shape)}")

        sam_mask = None
        sam_concept = None
        if meta is not None and isinstance(meta, dict):
            sam_mask = meta.get("sam_mask", None)

            # Optional concept ids per mask slot.
            # We accept several key aliases since different dataset pipelines
            # use different names.
            for k in (
                "sam_concept",
                "sam_concept_ids",
                "sam_concept_id",
                "sam_mask_concept",
                "sam_mask_concept_ids",
                "sam_mask_concept_id",
                "sam_local_concept_ids",
                "sam_global_concept_ids",
            ):
                if k in meta:
                    sam_concept = meta.get(k)
                    break

        return x_bag, sam_mask, sam_concept



    def _compute_frame_candidates(
        self,
        *,
        x_bag: torch.Tensor,  # (B,M,3,H,W)
        sam_mask: Optional[torch.Tensor],  # (B,M,K,1,H,W)
        sam_count: Optional[torch.Tensor],
        sam_concept: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,  # z_bmkd (B,M,K,D)
        torch.Tensor,  # valid_bmk (B,M,K)
        Optional[torch.Tensor],  # masks_bmkhw (B,M,K,H,W) or None
        Optional[torch.Tensor],  # concept_gid_bmk (B,M,K) or None
        Dict[str, Any],  # cand_meta
    ]:
        """Compute per-frame candidate embeddings from SAM masks (with patch fallback).

        We pool candidate embeddings from SAM masks on a layer4 feature map.

        Fallbacks:
          1) If a mask is valid at image resolution but becomes empty after
             downsampling to feature-map resolution, sample a patch embedding
             at the mask centroid (avg-pooled over a small neighborhood).
          2) If a frame ends up with *no* valid candidates (no masks / all
             filtered), sample top-k patch embeddings from the feature map.

        Returns:
          z_bmkd: (B,M,K,D)
          valid_bmk: (B,M,K) bool
          masks_bmkhw: (B,M,K,H,W) float (for debugging/vis) or None
          concept_gid_bmk: optional (B,M,K) global concept IDs (or None)
          cand_meta: dict with:
              - is_patch: (B,M,K) bool (True if candidate is patch-derived)
              - is_vanish_patch: (B,M,K) bool (patch from vanished downsampled mask)
              - is_topk_patch: (B,M,K) bool (patch from top-k fallback due to no masks)
              - patch_py/patch_px: (B,M,K) long feature-map coords (or -1)
              - Hf/Wf: feature-map spatial size (ints)
        """

        device = x_bag.device
        B, M, C, H, W = x_bag.shape

        # Helper: get projection head (C4 -> D) used by the backbone
        ve = getattr(self.model, "image_embed", None)
        backbone = getattr(ve, "model", None) if ve is not None else None
        proj = getattr(backbone, "fc", None) if backbone is not None else None
        if proj is None:
            raise RuntimeError("Backbone has no .fc; cannot project patch vectors.")

        # Helper: build an avg-pooled fmap for patch sampling
        def _avgpool_fmap(fmap4: torch.Tensor) -> torch.Tensor:
            r = max(int(getattr(self, "mil_patch_radius", 1)), 0)
            if r <= 0:
                return fmap4
            k = 2 * r + 1
            return F.avg_pool2d(fmap4, kernel_size=k, stride=1, padding=r)

        # ------------------------------------------------------------------
        # If there are no SAM masks at all, fall back to patch-only candidates.
        # ------------------------------------------------------------------
        if sam_mask is None or (torch.is_tensor(sam_mask) and int(sam_mask.size(2)) == 0):
            if B == 0 or M == 0:
                z = torch.empty((B, M, 0, self.embedding_dim), device=device, dtype=torch.float32)
                v = torch.zeros((B, M, 0), device=device, dtype=torch.bool)
                meta = {"is_patch": v, "is_vanish_patch": v, "is_topk_patch": v, "patch_py": v.long(), "patch_px": v.long(), "Hf": 0, "Wf": 0}
                return z, v, None, None, meta

            N = B * M
            x_flat = x_bag.reshape(N, C, H, W)
            fmap4 = self._get_layer4_fmap(x_flat)  # (N,C4,H4,W4) detached
            Nf, C4, H4, W4 = fmap4.shape
            if Nf != N:
                raise ValueError(f"Unexpected fmap N={Nf} for N={N}")

            fmap_p = _avgpool_fmap(fmap4)
            P = int(H4 * W4)
            kfill = int(min(int(getattr(self, "mil_patch_topk", 4)), P))
            kfill = max(kfill, 0)

            if kfill == 0:
                z = torch.empty((B, M, 0, self.embedding_dim), device=device, dtype=torch.float32)
                v = torch.zeros((B, M, 0), device=device, dtype=torch.bool)
                meta = {"is_patch": v, "is_vanish_patch": v, "is_topk_patch": v, "patch_py": v.long(), "patch_px": v.long(), "Hf": int(H4), "Wf": int(W4)}
                return z, v, None, None, meta

            # Project all patch cells and pick top-k by embedding norm.
            cells = fmap_p.permute(0, 2, 3, 1).reshape(N, P, C4)
            z_cells = proj(cells.reshape(N * P, C4)).view(N, P, -1).float()  # (N,P,D)
            score = z_cells.norm(dim=2)  # (N,P)
            top_idx = torch.topk(score, k=kfill, dim=1, largest=True, sorted=True).indices  # (N,k)

            idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, z_cells.size(2))
            z_top = torch.gather(z_cells, 1, idx_exp)  # (N,k,D)
            z_top = l2norm_obj(z_top)
            if self.img_adapter_enable:
                z_top = l2norm_obj(self.img_adapter(z_top))

            py = (top_idx // int(W4)).long()
            px = (top_idx % int(W4)).long()

            valid_flat = torch.ones((N, kfill), device=device, dtype=torch.bool)
            is_topk = torch.ones((N, kfill), device=device, dtype=torch.bool)
            is_vanish = torch.zeros_like(is_topk)
            is_patch = is_topk

            z_bmkd = z_top.view(B, M, kfill, -1)
            valid_bmk = valid_flat.view(B, M, kfill)

            cand_meta: Dict[str, Any] = {
                "is_patch": is_patch.view(B, M, kfill),
                "is_vanish_patch": is_vanish.view(B, M, kfill),
                "is_topk_patch": is_topk.view(B, M, kfill),
                "patch_py": py.view(B, M, kfill),
                "patch_px": px.view(B, M, kfill),
                "Hf": int(H4),
                "Wf": int(W4),
            }
            return z_bmkd, valid_bmk, None, None, cand_meta

        # ------------------------------------------------------------------
        # SAM masks exist: pool them, then patch-fallback as needed.
        # ------------------------------------------------------------------
        if sam_mask.ndim != 6:
            raise ValueError(f"Expected sam_mask (B,M,K,1,H,W), got {tuple(sam_mask.shape)}")

        sam_mask = sam_mask.to(device=device, dtype=torch.float32)
        Bm, Mm, K, _, Hm, Wm = sam_mask.shape
        if Bm != B or Mm != M:
            raise ValueError(f"Mask shape mismatch: x {tuple(x_bag.shape)}, mask {tuple(sam_mask.shape)}")

        # Resize masks to image size if needed.
        masks = sam_mask
        if (Hm, Wm) != (H, W):
            masks = F.interpolate(
                masks.view(B * M * K, 1, Hm, Wm),
                size=(H, W),
                mode="area",
            ).view(B, M, K, 1, H, W)

        # Valid: non-empty AND above min area.
        area = masks.sum(dim=(3, 4, 5))  # (B,M,K)
        area_frac = area / float(H * W)
        valid = area_frac >= float(self.mil_min_mask_area)

        # Optional count mask: ignore padded slots k >= count.
        if sam_count is not None and torch.is_tensor(sam_count):
            cnt = sam_count.to(device=device, dtype=torch.long)
            while cnt.ndim > 2:
                cnt = cnt.squeeze(-1)
            if cnt.ndim == 2 and cnt.shape[0] == B and cnt.shape[1] == M:
                kk = torch.arange(K, device=device).view(1, 1, K)
                valid = valid & (kk < cnt.unsqueeze(-1))

        # Optional: map SAM concept IDs -> global IDs and apply registry filtering.
        concept_gid_bmk: Optional[torch.Tensor] = None
        if sam_concept is not None and torch.is_tensor(sam_concept) and K > 0:
            cid = sam_concept.to(device=device, dtype=torch.long)
            while cid.ndim > 3:
                cid = cid.squeeze(-1)

            if cid.ndim == 3 and cid.shape[0] == B and cid.shape[1] == M and cid.shape[2] == K:
                if self.sam_registry is not None:
                    ltg = self.sam_registry.local_to_global.to(device=device)
                    local_C = int(ltg.numel())

                    cid_nonneg = cid[cid >= 0]
                    max_id = int(cid_nonneg.max().item()) if cid_nonneg.numel() > 0 else -1

                    if max_id < local_C:
                        in_range = (cid >= 0) & (cid < local_C)
                        gid = torch.full_like(cid, -1)
                        gid[in_range] = ltg[cid[in_range]]
                        concept_gid_bmk = gid
                        valid = valid & (concept_gid_bmk >= 0)
                    else:
                        gid = cid
                        Cg = int(self.sam_registry.weights.numel())
                        in_range = (gid >= 0) & (gid < Cg)
                        gid = torch.where(in_range, gid, torch.full_like(gid, -1))
                        concept_gid_bmk = gid

                        do_freq_filter = (
                            int(getattr(self.sam_registry, "min_masks_per_concept", 0)) > 0
                            and float(self.sam_registry.counts_full.sum().item()) > 0.0
                        )
                        if do_freq_filter:
                            keep = (self.sam_registry.counts_eff.to(device=device) > 0.0)
                            keep_mask = (concept_gid_bmk >= 0) & keep[concept_gid_bmk.clamp(min=0)]
                            valid = valid & keep_mask
                else:
                    concept_gid_bmk = cid

        # Flatten and pool.
        N = B * M
        x_flat = x_bag.reshape(N, C, H, W)
        masks_flat = masks.reshape(N, K, 1, H, W)
        valid_flat = valid.reshape(N, K)

        if K == 0 or N == 0:
            z = torch.empty((B, M, 0, self.embedding_dim), device=device, dtype=torch.float32)
            v = torch.zeros((B, M, 0), device=device, dtype=torch.bool)
            meta = {"is_patch": v, "is_vanish_patch": v, "is_topk_patch": v, "patch_py": v.long(), "patch_px": v.long(), "Hf": 0, "Wf": 0}
            return z, v, masks.squeeze(3), concept_gid_bmk, meta

        fmap4 = self._get_layer4_fmap(x_flat)
        Nf, C4, H4, W4 = fmap4.shape
        if Nf != N:
            raise ValueError(f"Unexpected fmap N={Nf} for N={N}")

        # Pool mask embeddings on fmap and project to embedding space.
        z_nkd = self._pool_object_embeddings_from_layer4(fmap4=fmap4, masks_img=masks_flat, valid=valid_flat)  # (N,K,D)

        # Track which candidates are patch-derived.
        is_vanish_patch = torch.zeros((N, K), device=device, dtype=torch.bool)
        is_topk_patch = torch.zeros((N, K), device=device, dtype=torch.bool)
        patch_py = torch.full((N, K), -1, device=device, dtype=torch.long)
        patch_px = torch.full((N, K), -1, device=device, dtype=torch.long)

        # --------------------------------------------------------------
        # (1) Mask-vanish fallback: if downsampled mask becomes empty.
        # --------------------------------------------------------------
        # Downsample masks to fmap resolution and check mass.
        masks_ds = F.interpolate(
            masks_flat.view(N * K, 1, H, W).float(),
            size=(H4, W4),
            mode="area",
        ).view(N, K, 1, H4, W4)
        ds_mass = masks_ds.sum(dim=(2, 3, 4))  # (N,K)

        vanish = (ds_mass <= 1e-6) & valid_flat
        if bool(vanish.any().item()):
            idx = vanish.nonzero(as_tuple=False)  # (n_v,2) [n,k]
            n_ids = idx[:, 0]
            k_ids = idx[:, 1]

            # Gather the corresponding masks at image resolution to compute centroids.
            m_sel = masks_flat[n_ids, k_ids, 0]  # (n_v,H,W)
            m_sum = m_sel.sum(dim=(1, 2)).clamp(min=1e-6)

            ys = torch.arange(H, device=device, dtype=torch.float32)
            xs = torch.arange(W, device=device, dtype=torch.float32)
            yc = (m_sel.sum(dim=2) * ys.view(1, H)).sum(dim=1) / m_sum
            xc = (m_sel.sum(dim=1) * xs.view(1, W)).sum(dim=1) / m_sum

            py = torch.clamp((yc * float(H4) / float(H)).long(), 0, int(H4) - 1)
            px = torch.clamp((xc * float(W4) / float(W)).long(), 0, int(W4) - 1)

            fmap_p = _avgpool_fmap(fmap4)
            patch_vec = fmap_p[n_ids, :, py, px]  # (n_v,C4)
            z_patch = proj(patch_vec).float()  # (n_v,D)
            z_patch = l2norm_obj(z_patch)

            # Build a dense (N,K,D) patch tensor via differentiable index_put.
            z_patch_full = torch.zeros((N, K, z_patch.size(1)), device=device, dtype=z_patch.dtype)
            z_patch_full = z_patch_full.index_put((n_ids, k_ids), z_patch)

            z_nkd = torch.where(vanish.unsqueeze(-1), z_patch_full, z_nkd)

            is_vanish_patch = vanish
            patch_py[n_ids, k_ids] = py
            patch_px[n_ids, k_ids] = px

        # --------------------------------------------------------------
        # (2) No-mask fallback: if a frame has no valid candidates.
        # --------------------------------------------------------------
        frame_has_any = valid_flat.any(dim=1)  # (N,)
        need_topk = ~frame_has_any

        P = int(H4 * W4)
        kfill = int(min(int(getattr(self, "mil_patch_topk", 4)), int(K), P))
        kfill = max(kfill, 0)

        if bool(need_topk.any().item()) and kfill > 0:
            fmap_p = _avgpool_fmap(fmap4)
            cells = fmap_p.permute(0, 2, 3, 1).reshape(N, P, C4)
            z_cells = proj(cells.reshape(N * P, C4)).view(N, P, -1).float()  # (N,P,D)
            score = z_cells.norm(dim=2)  # (N,P)
            top_idx = torch.topk(score, k=kfill, dim=1, largest=True, sorted=True).indices  # (N,k)

            idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, z_cells.size(2))
            z_top = torch.gather(z_cells, 1, idx_exp)  # (N,k,D)
            z_top = l2norm_obj(z_top)

            # Only apply to frames that need it.
            n_fill = need_topk.nonzero(as_tuple=False).view(-1)
            # Build dense (N,K,D) tensor for the filled slots.
            n_ids = n_fill.repeat_interleave(kfill)
            k_ids = torch.arange(kfill, device=device, dtype=torch.long).repeat(int(n_fill.numel()))
            src = z_top[n_fill].reshape(-1, z_top.size(2))

            z_fill_full = torch.zeros((N, K, z_top.size(2)), device=device, dtype=z_top.dtype)
            z_fill_full = z_fill_full.index_put((n_ids, k_ids), src)

            kk = torch.arange(K, device=device).view(1, K)
            fill_mask = need_topk.view(N, 1) & (kk < int(kfill))  # (N,K)

            z_nkd = torch.where(fill_mask.unsqueeze(-1), z_fill_full, z_nkd)
            valid_flat = valid_flat | fill_mask

            is_topk_patch = is_topk_patch | fill_mask
            py_top = (top_idx // int(W4)).long()
            px_top = (top_idx % int(W4)).long()
            patch_py[n_fill, :kfill] = py_top[n_fill]
            patch_px[n_fill, :kfill] = px_top[n_fill]

            # For top-k fallback patches, concept IDs are undefined: force -1.
            if concept_gid_bmk is not None:
                concept_gid_flat = concept_gid_bmk.view(N, K)
                concept_gid_flat[n_fill, :kfill] = -1
                concept_gid_bmk = concept_gid_flat.view(B, M, K)

        is_patch = is_vanish_patch | is_topk_patch

        # Apply adapter (image-only) and re-normalize.
        if self.img_adapter_enable:
            z_nkd = l2norm_obj(self.img_adapter(z_nkd))

        z_bmkd = z_nkd.view(B, M, K, -1)
        valid_bmk = valid_flat.view(B, M, K)

        cand_meta = {
            "is_patch": is_patch.view(B, M, K),
            "is_vanish_patch": is_vanish_patch.view(B, M, K),
            "is_topk_patch": is_topk_patch.view(B, M, K),
            "patch_py": patch_py.view(B, M, K),
            "patch_px": patch_px.view(B, M, K),
            "Hf": int(H4),
            "Wf": int(W4),
        }

        return z_bmkd, valid_bmk, masks.squeeze(3), concept_gid_bmk, cand_meta

    def _build_tracks(self, z_bmkd: torch.Tensor, valid_bmk: torch.Tensor) -> List[torch.Tensor]:
        """Build per-sample tracks (variable length list of (R_i,D))."""

        B, M, K, D = z_bmkd.shape
        tracks: List[torch.Tensor] = []

        if K == 0 or B == 0:
            return [torch.empty((0, D), device=z_bmkd.device, dtype=z_bmkd.dtype) for _ in range(B)]

        if self.mil_track:
            cfg = TrackConfig(sim_thresh=self.mil_track_sim_thresh, max_tracks=self.mil_track_max_tracks)
            for i in range(B):
                tr = build_object_tracks_greedy(z_bmkd[i], valid_bmk[i], cfg)
                tracks.append(tr)
        else:
            # No tracking: just flatten all candidates.
            for i in range(B):
                z_i = z_bmkd[i].reshape(M * K, D)
                v_i = valid_bmk[i].reshape(M * K)
                tracks.append(z_i[v_i])

        return tracks

    def _proto_logits_text_to_bag(
        self,
        q_text_local: torch.Tensor,  # (B,Kp)
        q_track_global: torch.Tensor,  # (Bg,R,Kp)
        mask_global: torch.Tensor,  # (Bg,R)
        tau: float,
    ) -> torch.Tensor:
        """Compute logits for local texts (rows) vs global bags (cols)."""

        tau = max(float(tau), 1e-6)
        # sim: (B, Bg, R)
        sim = torch.einsum("ik,jrk->ijr", q_text_local, q_track_global)
        sim = sim.masked_fill(~mask_global.unsqueeze(0), float("-inf"))
        logits = torch.logsumexp(sim / tau, dim=-1) * tau  # (B,Bg)
        return logits

    def _proto_logits_bag_to_text(
        self,
        q_track_local: torch.Tensor,  # (B,R,Kp)
        mask_local: torch.Tensor,  # (B,R)
        q_text_global: torch.Tensor,  # (Bg,Kp)
        tau: float,
    ) -> torch.Tensor:
        """Compute logits for local bags (rows) vs global texts (cols)."""

        tau = max(float(tau), 1e-6)
        # sim: (B, Bg, R)
        sim = torch.einsum("jrk,ik->jir", q_track_local, q_text_global)
        sim = sim.masked_fill(~mask_local.unsqueeze(1), float("-inf"))
        logits = torch.logsumexp(sim / tau, dim=-1) * tau  # (B,Bg)
        return logits

    def _compute_w_align(
        self, q_text: torch.Tensor, q_track: torch.Tensor, cand_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute w_align and best track index per sample.

        Returns:
          w_align: (B,) in [0,1]
          best_sim: (B,) best proto-space dot(q_text, q_track)
          best_r: (B,) long (index into R dimension; -1 means no non-null track)
        """

        device = q_text.device
        B, R, Kp = q_track.shape
        if B == 0:
            return (
                torch.zeros((0,), device=device),
                torch.zeros((0,), device=device),
                torch.full((0,), -1, device=device, dtype=torch.long),
            )

        # sim_r: (B,R)
        sim_r = (q_track * q_text.unsqueeze(1)).sum(dim=2)

        counts = cand_mask.long().sum(dim=1)  # includes null
        null_idx = (counts - 1).clamp(min=0)

        non_null_mask = cand_mask.clone()
        non_null_mask[torch.arange(B, device=device), null_idx] = False

        sim_r = sim_r.masked_fill(~non_null_mask, float("-inf"))
        best_sim, best_r = sim_r.max(dim=1)

        has_non_null = non_null_mask.any(dim=1)
        best_r = torch.where(has_non_null, best_r, torch.full_like(best_r, -1))
        best_sim = torch.where(has_non_null, best_sim, torch.zeros_like(best_sim))

        # Warmup: keep w_align=1 early to avoid starving the MIL signal.
        if int(self.global_step) < int(self.w_align_warmup_steps):
            w = torch.ones_like(best_sim)
        else:
            s0 = float(self.w_align_sim0)
            ss = max(float(self.w_align_simscale), 1e-6)
            w = torch.sigmoid((best_sim - s0) / ss)

        w = w.clamp(min=float(self.w_align_min), max=1.0)
        # If no track: allow the sample to act as negative, but don't weigh it as positive.
        w = torch.where(has_non_null, w, torch.zeros_like(w))
        return w.detach(), best_sim.detach(), best_r.detach()

    def _maybe_warm_start_prototypes(
        self,
        *,
        track_emb: torch.Tensor,  # (B,R,D) incl null
        cand_mask: torch.Tensor,  # (B,R)
        text_feat: torch.Tensor,  # (B,D)
    ) -> None:
        """Collect confident examples and warm-start prototype memory once."""

        if (not self.proto_enable) or (self.proto_mem is None) or (not self.proto_warm_start) or self._proto_warm_started:
            return

        device = track_emb.device
        B, R, D = track_emb.shape
        if B == 0:
            return

        # Select best *non-null* track by embedding-space cosine with text.
        counts = cand_mask.long().sum(dim=1)
        null_idx = (counts - 1).clamp(min=0)
        non_null_mask = cand_mask.clone()
        non_null_mask[torch.arange(B, device=device), null_idx] = False

        # (B,R)
        sim = torch.einsum("brd,bd->br", track_emb, l2norm_obj(text_feat))
        sim = sim.masked_fill(~non_null_mask, float("-inf"))
        best_sim, best_r = sim.max(dim=1)

        good = best_sim >= float(self.proto_warm_sim_thresh)
        good = good & non_null_mask.any(dim=1)

        if bool(good.any().item()):
            idx = good.nonzero(as_tuple=False).view(-1)
            # Add at most a small number per step to avoid huge CPU lists.
            max_add = int(self.args.get("proto_warm_max_add_per_step", 64))
            idx = idx[:max_add]
            z_add = track_emb[idx, best_r[idx]].detach().cpu()  # (n,D)
            self._proto_warm_buffer.extend([z_add[i] for i in range(z_add.size(0))])

        # Every step, compute global readiness via all_reduce(min).
        local_ready = 1 if len(self._proto_warm_buffer) >= int(self.proto_warm_min_local) else 0
        if _is_dist_active():
            flag = torch.tensor(local_ready, device=device, dtype=torch.int64)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            global_ready = int(flag.item()) == 1
        else:
            global_ready = bool(local_ready)

        if not global_ready:
            return

        # Warm-start now.
        z_local = torch.stack(self._proto_warm_buffer, dim=0).to(device=device, dtype=torch.float32)
        # Optional subsample (warm_start_ddp() also subsamples globally, but local cap helps memory).
        max_local = int(self.args.get("proto_warm_max_local", 1024))
        if z_local.size(0) > max_local:
            perm = torch.randperm(z_local.size(0), device=device)
            z_local = z_local[perm[:max_local]]

        # Deterministic seed based on run id if provided.
        seed = int(self.args.get("seed", 0))
        # NOTE: warm_start_ddp() gathers examples across ranks and initializes prototypes once.
        self.proto_mem.warm_start_ddp(
            z_local,
            seed=seed,
            kmeans_iters=int(self.proto_warm_kmeans_iters),
            max_total=int(self.proto_warm_max_total),
            verbose=((self.trainer is None) or bool(getattr(self.trainer, "is_global_zero", False))),
        )
        self._proto_warm_started = True
        self._proto_warm_buffer = []

        if (self.trainer is None) or self.trainer.is_global_zero:
            print(
                f"[multimodal_lit] Prototype warm-start complete: "
                f"K={self.proto_num}, seed={seed}, world={_dist_world()}"
            )

    # ------------------------------------------------------------------
    # MIL core (Plan A)
    # ------------------------------------------------------------------
    def _mil_losses(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_len: torch.Tensor,
        raw_y: Optional[Any],
        meta: Optional[dict],
        *,
        update_memory: bool,
        stage: str = "train",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute Plan-A MIL losses + memory update.

        Returns:
          aux_loss: scalar tensor (weighted sum)
          stats: dict of tensors for logging
        """

        device = x.device
        zero = x.new_tensor(0.0)

        if not self.mil_enable:
            return zero, {
                "mil_loss": zero.detach(),
                "mil_acc": zero.detach(),
                "mil_acc_t2i": zero.detach(),
                "mil_acc_i2t": zero.detach(),
                "noun_count_mean": zero.detach(),
                "noun_sample_frac": zero.detach(),
                "noun_query_frac": zero.detach(),
                "concept_align_loss": zero.detach(),
                "concept_align_acc": zero.detach(),
                "concept_align_n": zero.detach(),
                "w_align_mean": zero.detach(),
                "w_align_eff_mean": zero.detach(),
                "sam_concept_weight_mean": zero.detach(),
                "track_coh_loss": zero.detach(),
                "go_loss": zero.detach(),
                "proto_update_loss": zero.detach(),
                "proto_usage_eff_k": zero.detach(),
                "sam_mask_frac": zero.detach(),
                "patch_frac": zero.detach(),
            }

        x_bag, sam_mask, sam_concept = self._get_bag_tensors(x, meta)

        # Helpful one-time warning: if you're expecting SAM masks but none show up,
        # MIL will effectively run on empty candidates.
        if (sam_mask is None) and (not getattr(self, "_warned_missing_sam_mask", False)) and (str(stage) == "train"):
            if self.sam_prepacked_dir is not None:
                if int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))) == 0:
                    print(
                        "[multimodal_lit] WARNING: MIL is enabled but batch meta has no 'sam_mask'. "
                        "If you expect SAM masks, make sure the dataset/dataloader populates meta['sam_mask'] "
                        "(and optional meta['sam_concept_ids'])."
                    )
            self._warned_missing_sam_mask = True
        B, M, _, H, W = x_bag.shape

        # Text embeddings (sentence + optional noun pivots)
        text_feat_sent, _ = self.model.encode_text(y, y_len)
        text_feat_sent = l2norm_obj(text_feat_sent.float())

        # Default: sentence embedding as the MIL text query.
        text_feat = text_feat_sent
        noun_count_mean = zero.detach()
        noun_sample_frac = zero.detach()
        noun_query_frac = zero.detach()
        noun_count_b = None
        text_query_bnd = None  # (B, Nn, D) for noun_multi
        query_valid_bn = None  # (B, Nn)

        mode = str(getattr(self, "mil_text_mode", "sentence")).lower()
        if mode in ("noun_avg", "noun_multi"):
            noun_feat_bnd, noun_valid_bn, noun_count_b = self._noun_features_from_raw_y(raw_y, B=B, device=device)
            if noun_count_b.numel() > 0:
                noun_count_mean = noun_count_b.float().mean().detach()
                noun_sample_frac = (noun_count_b > 0).float().mean().detach()
            # Average nouns -> one vector per sample (fallback to sentence if no nouns).
            denom = noun_valid_bn.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            noun_sum = (noun_feat_bnd * noun_valid_bn.float().unsqueeze(-1)).sum(dim=1)
            noun_avg = l2norm_obj(noun_sum / denom)
            has_noun = noun_valid_bn.any(dim=1)
            text_feat = torch.where(has_noun.unsqueeze(1), noun_avg, text_feat_sent)
            text_feat = l2norm_obj(text_feat)

            if mode == "noun_multi":
                # Multi-noun queries for t2i (keep i2t on the bag-level text_feat).
                text_query_bnd = noun_feat_bnd
                query_valid_bn = noun_valid_bn
                # Ensure every sample contributes at least one query.
                no_noun = ~has_noun
                if bool(no_noun.any().item()):
                    sent_first = text_feat_sent.unsqueeze(1)  # (B,1,D)
                    text_query_bnd = torch.where(
                        no_noun.view(B, 1, 1),
                        torch.cat([sent_first, text_query_bnd[:, 1:, :]], dim=1),
                        text_query_bnd,
                    )
                    query_valid_bn = torch.where(
                        no_noun.view(B, 1),
                        torch.cat([torch.ones(B, 1, device=device, dtype=torch.bool), query_valid_bn[:, 1:]], dim=1),
                        query_valid_bn,
                    )
                noun_query_frac = query_valid_bn.float().mean().detach()
            else:
                noun_query_frac = noun_valid_bn.float().mean().detach()

        sam_count = meta.get("sam_mask_count", None) if isinstance(meta, dict) else None

        # Per-frame candidates
        z_bmkd, valid_bmk, masks_bmkhw, sam_gid_bmk, cand_meta = self._compute_frame_candidates(
            x_bag=x_bag, sam_mask=sam_mask, sam_count=sam_count, sam_concept=sam_concept
        )

        # Fractions: how often we are using SAM masks vs patch fallbacks
        is_patch_bmk = None
        patch_py_bmk = None
        patch_px_bmk = None
        Hf = int(cand_meta.get("Hf", 0)) if isinstance(cand_meta, dict) else 0
        Wf = int(cand_meta.get("Wf", 0)) if isinstance(cand_meta, dict) else 0
        if isinstance(cand_meta, dict):
            is_patch_bmk = cand_meta.get("is_patch", None)
            patch_py_bmk = cand_meta.get("patch_py", None)
            patch_px_bmk = cand_meta.get("patch_px", None)
        if is_patch_bmk is None:
            is_patch_bmk = torch.zeros_like(valid_bmk, dtype=torch.bool)
        n_frames = int(B) * int(M)
        if n_frames > 0:
            sam_frame = (valid_bmk & (~is_patch_bmk)).any(dim=2)
            patch_frame = (valid_bmk & is_patch_bmk).any(dim=2)
            sam_mask_frac = sam_frame.float().mean().detach()
            patch_frac = patch_frame.float().mean().detach()
        else:
            sam_mask_frac = zero.detach()
            patch_frac = zero.detach()

        # Tracks
        tracks_per_sample = self._build_tracks(z_bmkd, valid_bmk)

        # Pack tracks with a learnable null
        null_emb = l2norm_obj(self.null_obj.to(device=device, dtype=torch.float32))
        track_emb, cand_mask = pack_candidates_with_null(tracks_per_sample, null_emb)
        track_emb = l2norm_obj(track_emb.float())

        # Warm-start prototypes (one-shot). IMPORTANT: training-only (avoid val/test leakage).
        if bool(update_memory):
            self._maybe_warm_start_prototypes(track_emb=track_emb, cand_mask=cand_mask, text_feat=text_feat)

        # If prototypes are disabled, fall back to embedding-space MIL (legacy behavior).
        if (not self.proto_enable) or (self.proto_mem is None):
            # Build a (B,B) MIL matrix using embedding-space similarities.
            logits = mil_logsumexp_logits(text_feat, track_emb, cand_mask, tau=float(self.mil_tau))
            tgt = torch.arange(B, device=device, dtype=torch.long)
            loss_row = F.cross_entropy(logits, tgt)
            loss_col = F.cross_entropy(logits.t(), tgt)
            mil_loss = 0.5 * (loss_row + loss_col)
            aux = float(self.mil_lambda) * mil_loss
            acc_t2i = (logits.argmax(dim=1) == tgt).float().mean()
            acc_i2t = (logits.argmax(dim=0) == tgt).float().mean()
            acc = 0.5 * (acc_t2i + acc_i2t)

            return aux, {
                "mil_loss": mil_loss.detach(),
                "mil_acc": acc.detach(),
                "mil_acc_t2i": acc_t2i.detach(),
                "mil_acc_i2t": acc_i2t.detach(),
                "noun_count_mean": noun_count_mean,
                "noun_sample_frac": noun_sample_frac,
                "noun_query_frac": noun_query_frac,
                "concept_align_loss": zero.detach(),
                "concept_align_acc": zero.detach(),
                "concept_align_n": zero.detach(),
                "w_sample_mean": zero.detach(),
                "best_sim_mean": zero.detach(),
                "best_r_mode": zero.detach(),
                "avg_tracks": zero.detach(),
                "avg_eff_tracks": zero.detach(),
                "sam_concept_weight_nonzero": zero.detach(),
                "w_align_mean": zero.detach(),
                "w_align_eff_mean": zero.detach(),
                "sam_concept_weight_mean": zero.detach(),
                "track_coh_loss": zero.detach(),
                "go_loss": zero.detach(),
                "proto_update_loss": zero.detach(),
                "proto_usage_eff_k": zero.detach(),
                "sam_mask_frac": sam_mask_frac,
                "patch_frac": patch_frac,
            }

        # Prototype assignments (with grad w.r.t. embeddings)
        q_text = self.proto_mem.assign_with_grad(text_feat)  # (B,Kp)

        # Pad R across ranks before gathering.
        R_local = int(track_emb.size(1))
        R = R_local
        if _is_dist_active():
            rmax = torch.tensor([R_local], device=device, dtype=torch.int64)
            dist.all_reduce(rmax, op=dist.ReduceOp.MAX)
            R = int(rmax.item())

        if R > R_local:
            pad = torch.zeros((B, R - R_local, self.embedding_dim), device=device, dtype=track_emb.dtype)
            track_emb = torch.cat([track_emb, pad], dim=1)
            padm = torch.zeros((B, R - R_local), device=device, dtype=torch.bool)
            cand_mask = torch.cat([cand_mask, padm], dim=1)

        q_track = self.proto_mem.assign_with_grad(track_emb.view(B * R, self.embedding_dim)).view(B, R, -1)

        # Alignment gating from best within-sample match in proto space.
        w_align, best_sim, best_r = self._compute_w_align(q_text, q_track, cand_mask)
        w_align_mean = w_align.mean().detach() if w_align.numel() > 0 else zero.detach()

        # Optional: concept-frequency reweighting (uses SamConceptRegistry).
        #
        # We approximate a "track concept" by taking the anchor-frame (m=0)
        # candidate whose embedding is closest to the selected best track, and
        # apply the corresponding global concept weight to the per-sample loss.
        w_sample = w_align
        sam_w_mean = zero.detach()
        w_sample_mean = w_align_mean
        if (
            self.sam_use_concept_weights
            and (self.sam_registry is not None)
            and (sam_gid_bmk is not None)
            and (B > 0)
        ):
            weights_g = self.sam_registry.weights.to(device=device, dtype=torch.float32)
            Cg = int(weights_g.numel())
            sam_w = torch.ones((B,), device=device, dtype=torch.float32)

            # Anchor-frame concept for best track.
            for i in range(B):
                r = int(best_r[i].item())
                if r < 0:
                    continue
                v0 = valid_bmk[i, 0]
                if not bool(v0.any().item()):
                    continue
                tr = track_emb[i, r].detach()
                sims = (z_bmkd[i, 0].detach() @ tr).masked_fill(~v0, float("-inf"))
                k = int(torch.argmax(sims).item())
                gid = int(sam_gid_bmk[i, 0, k].item())
                if 0 <= gid < Cg:
                    sam_w[i] = weights_g[gid]

            sam_w = sam_w.clamp(min=0.0)
            sam_w_mean = sam_w.mean().detach() if sam_w.numel() > 0 else zero.detach()
            w_sample = w_align * sam_w
            w_sample_mean = w_sample.mean().detach() if w_sample.numel() > 0 else zero.detach()

        # -------------------------
        # Prototype-space MIL loss (distributed)
        # -------------------------
        mil_loss = zero
        mil_acc = zero
        mil_acc_t2i = zero
        mil_acc_i2t = zero

        if self.mil_lambda > 0.0 and B > 1:
            rank = _dist_rank()
            # Contrast bags and texts across GPUs (global batch)
            q_text_all = self._dist_all_gather_with_grad(q_text)
            q_track_all = self._dist_all_gather_with_grad(q_track)
            mask_all = self._dist_all_gather_no_grad(cand_mask)

            # Text->bag (t2i)
            use_multi = (
                str(getattr(self, "mil_text_mode", "sentence")).lower() == "noun_multi"
                and text_query_bnd is not None
                and query_valid_bn is not None
            )

            if use_multi:
                Nn = int(text_query_bnd.shape[1])
                q_text_q = self.proto_mem.assign_with_grad(text_query_bnd.reshape(B * Nn, -1))  # (B*Nn,Kp)
                logits_t2i = self._proto_logits_text_to_bag(q_text_q, q_track_all, mask_all, tau=float(self.mil_tau))  # (B*Nn,Bg)
                tgt_t2i = (torch.arange(B, device=device) + rank * B).repeat_interleave(Nn)

                loss_t2i_row = F.cross_entropy(logits_t2i, tgt_t2i, reduction="none")
                valid_flat = query_valid_bn.reshape(B * Nn).float()

                # Weighting: keep *per-sample* weight roughly constant regardless of
                # how many nouns it has (so captions with 5 nouns don't dominate).
                q_count = query_valid_bn.float().sum(dim=1).clamp(min=1.0)  # (B,)
                w_per_query = (w_sample / q_count).repeat_interleave(Nn)
                w_query = w_per_query * valid_flat
                denom_q = w_query.sum().clamp(min=1e-12)
                loss_t2i = (w_query * loss_t2i_row).sum() / denom_q

                mil_acc_t2i = (
                    ((logits_t2i.argmax(dim=1) == tgt_t2i).float() * valid_flat).sum()
                    / valid_flat.sum().clamp(min=1e-12)
                )
            else:
                logits_t2i = self._proto_logits_text_to_bag(q_text, q_track_all, mask_all, tau=float(self.mil_tau))
                tgt = torch.arange(B, device=device) + rank * B

                loss_t2i_row = F.cross_entropy(logits_t2i, tgt, reduction="none")
                denom = w_sample.sum().clamp(min=1e-12)
                loss_t2i = (w_sample * loss_t2i_row).sum() / denom

                mil_acc_t2i = (logits_t2i.argmax(dim=1) == tgt).float().mean()

            # Bag->text (i2t): always uses one text vector per sample (sentence / noun_avg)
            logits_i2t = self._proto_logits_bag_to_text(q_track, cand_mask, q_text_all, tau=float(self.mil_tau))
            tgt_i2t = torch.arange(B, device=device) + rank * B

            loss_i2t_row = F.cross_entropy(logits_i2t, tgt_i2t, reduction="none")
            denom = w_sample.sum().clamp(min=1e-12)
            loss_i2t = (w_sample * loss_i2t_row).sum() / denom

            mil_acc_i2t = (logits_i2t.argmax(dim=1) == tgt_i2t).float().mean()

            mil_loss = 0.5 * (loss_t2i + loss_i2t)
            mil_acc = 0.5 * (mil_acc_t2i + mil_acc_i2t)

        # -------------------------
        # Track coherence (prototype space)
        # -------------------------
        coh_loss = zero
        coh_pairs = zero

        if self.track_coh_enable and (self.track_coh_lambda > 0.0):
            eps = 1e-8
            sum_loss = zero
            sum_w = zero
            pairs = zero

            # Candidate embeddings for selection in embedding space.
            # z_bmkd: (B,M,K,D), valid_bmk: (B,M,K)
            for i in range(B):
                w_i = w_sample[i].detach()
                if float(w_i.item()) <= 0.0:
                    continue

                # number of non-null tracks for sample i
                cnt = int(cand_mask[i].long().sum().item())
                if cnt <= 1:
                    continue
                null_i = cnt - 1

                loss_i = zero
                n_tracks_used = 0
                pairs_i = 0

                for r in range(null_i):
                    tr = track_emb[i, r]  # (D,)
                    # pick best instance per frame
                    chosen: List[torch.Tensor] = []
                    for m in range(M):
                        v = valid_bmk[i, m]
                        if not bool(v.any().item()):
                            continue
                        sims = (z_bmkd[i, m] @ tr).masked_fill(~v, float("-inf"))
                        k = int(torch.argmax(sims).item())
                        s = float(sims[k].item())
                        if np.isfinite(s) and (s >= float(self.track_coh_match_thresh)):
                            chosen.append(z_bmkd[i, m, k])

                    if len(chosen) < int(self.track_coh_min_frames):
                        continue

                    z_sel = torch.stack(chosen, dim=0)  # (L,D)
                    q_sel = self.proto_mem.assign_with_grad(z_sel)  # (L,Kp)

                    # Teacher = track proto distribution (stop-grad)
                    q_tr = q_track[i, r].detach().clamp(min=eps)
                    q_tr = q_tr / q_tr.sum().clamp(min=eps)

                    kl = F.kl_div(torch.log(q_sel.clamp(min=eps)), q_tr.expand_as(q_sel), reduction="none").sum(dim=1)
                    kl_mean = kl.mean()

                    loss_i = loss_i + kl_mean
                    n_tracks_used += 1
                    pairs_i += int(q_sel.size(0))

                if n_tracks_used > 0:
                    loss_i = loss_i / float(n_tracks_used)
                    sum_loss = sum_loss + w_i * loss_i
                    sum_w = sum_w + w_i
                    pairs = pairs + x.new_tensor(float(pairs_i))

            if float(sum_w.detach().item()) > 0.0:
                coh_loss = sum_loss / sum_w.clamp(min=1e-12)
                coh_pairs = pairs.detach()

        # -------------------------
        # Global-object agreement (global emb vs recalled best track)
        # -------------------------
        go_loss = zero
        go_sim = zero

        if self.go_enable and (self.go_lambda > 0.0):
            # Global embedding for anchor frame (frame 0)
            x_anchor = x_bag[:, 0]
            g, _ = self.model.encode_image(x_anchor)
            g = l2norm_obj(g.float())
            if self.img_adapter_enable:
                g = l2norm_obj(self.img_adapter(g))

            sims: List[torch.Tensor] = []
            ws: List[torch.Tensor] = []

            for i in range(B):
                if float(w_sample[i].item()) <= 0.0:
                    continue
                r = int(best_r[i].item())
                if r < 0:
                    continue

                tr = track_emb[i, r]
                _q, z_rec = self.proto_mem.recall_with_grad(tr.unsqueeze(0))
                z_rec = z_rec.squeeze(0)

                s = (g[i] * z_rec).sum()
                sims.append(s)
                ws.append(w_sample[i])

            if sims:
                sim_t = torch.stack(sims, dim=0)
                w_t = torch.stack(ws, dim=0).clamp(min=1e-12)
                go_sim = (w_t * sim_t).sum() / w_t.sum()
                go_loss = 1.0 - go_sim

        # -------------------------
        # Pivot C: SAM concept->object alignment (optional)
        # -------------------------
        concept_align_loss = zero
        concept_align_acc = zero
        concept_align_n = zero

        if (
            self.sam_concept_align_enable
            and (self.sam_concept_align_lambda > 0.0)
            and (self.sam_registry is not None)
            and (sam_gid_bmk is not None)
        ):
            # Only true SAM-mask candidates (exclude patch fallbacks) with a valid concept id
            mask_c = valid_bmk & (~is_patch_bmk) & (sam_gid_bmk >= 0)
            if bool(mask_c.any().item()):
                z_obj = z_bmkd[mask_c]
                z_obj = l2norm_obj(z_obj.float())
                gid = sam_gid_bmk[mask_c].long()

                uniq, inv = torch.unique(gid, sorted=True, return_inverse=True)
                uniq_list = uniq.detach().cpu().tolist()
                concept_strs = [self.sam_registry.idx2concept[int(g)] for g in uniq_list]

                # Encode concept strings
                tok, tok_len = self._tokenize_phrases(concept_strs)
                tok = tok.to(device)
                tok_len = tok_len.to(device)
                c_feat, _ = self.model.encode_text(tok, tok_len)
                c_feat = l2norm_obj(c_feat.float())

                logits = (z_obj @ c_feat.t()) / float(self.sam_concept_align_tau)
                concept_align_loss = F.cross_entropy(logits, inv)
                concept_align_acc = (logits.argmax(dim=1) == inv).float().mean().detach()
                concept_align_n = z_obj.new_tensor(float(z_obj.size(0))).detach()

        # -------------------------
        # Prototype memory update (EMA) from track embeddings
        # -------------------------
        proto_up_loss = zero
        proto_eff_k = zero

        # Compute the prototype quantization loss for monitoring on any stage,
        # but only update the EMA prototypes during training.
        if self.proto_mem is not None:
            # Exclude null tracks
            counts = cand_mask.long().sum(dim=1)
            null_idx = (counts - 1).clamp(min=0)
            non_null_mask = cand_mask.clone()
            non_null_mask[torch.arange(B, device=device), null_idx] = False

            z_update = track_emb[non_null_mask]
            if z_update.ndim == 1:
                z_update = z_update.unsqueeze(0)
            with torch.no_grad():
                proto_up_loss, pstats = self.proto_mem.proto_loss_ddp_gather(
                    z_update.detach(),
                    update_ema=bool(update_memory),
                )
            proto_eff_k = pstats.get("usage_eff_k", zero).detach()

        # -------------------------
        # Debug track grids (optional)
        # -------------------------
        if (
            self.debug_save_tracks
            and str(stage) == "train"
            and (self.trainer is None or self.trainer.is_global_zero)
        ):
            # Save a few best examples by (w_align * best_sim).
            score = (w_sample * best_sim).detach().float()
            topk = min(int(self.debug_tracks_topk), B)
            if topk > 0:
                idx = torch.topk(score, k=topk, largest=True, sorted=True).indices
                for ii in idx.tolist():
                    # Build per-frame visualization payload for the selected best track.
                    per_frame: List[Dict[str, Any]] = []
                    r_sel = int(best_r[ii].item()) if torch.is_tensor(best_r) else -1
                    tr_sel = None
                    if r_sel >= 0:
                        tr_sel = track_emb[ii, r_sel].detach()
                    for m in range(M):
                        v_m = valid_bmk[ii, m]
                        if not bool(v_m.any().item()):
                            per_frame.append({"kind": "none", "conf": 0.0, "sim": 0.0})
                            continue
                        # Pick a candidate for this frame (either by similarity to the chosen track, or first valid).
                        if tr_sel is not None:
                            sims = (z_bmkd[ii, m].detach() @ tr_sel).masked_fill(~v_m, float("-inf"))
                            k_sel = int(torch.argmax(sims).item())
                            sim_val = float(sims[k_sel].item()) if np.isfinite(float(sims[k_sel].item())) else 0.0
                        else:
                            k_sel = int(v_m.nonzero(as_tuple=False)[0].item())
                            sim_val = 0.0
                        is_p = bool(is_patch_bmk[ii, m, k_sel].item()) if torch.is_tensor(is_patch_bmk) else False
                        kind = "patch" if is_p else "sam"
                        mask_np = None
                        conf = 0.0
                        if masks_bmkhw is not None and torch.is_tensor(masks_bmkhw):
                            mk = masks_bmkhw[ii, m, k_sel].detach().cpu()
                            if float(mk.sum().item()) > 0.0:
                                mask_np = mk.numpy()
                                conf = float(mk.mean().item())
                        info: Dict[str, Any] = {"kind": kind, "conf": float(conf), "sim": float(sim_val)}
                        if mask_np is not None:
                            info["mask"] = mask_np
                        if patch_py_bmk is not None and patch_px_bmk is not None and Hf > 0 and Wf > 0:
                            py = int(patch_py_bmk[ii, m, k_sel].item()) if torch.is_tensor(patch_py_bmk) else -1
                            px = int(patch_px_bmk[ii, m, k_sel].item()) if torch.is_tensor(patch_px_bmk) else -1
                            if py >= 0 and px >= 0:
                                info.update({"py": py, "px": px, "Hf": int(Hf), "Wf": int(Wf)})
                        per_frame.append(info)
                    ex = {
                        "score": float(score[ii].item()),
                        "caption": "",
                        "frames": x_bag[ii].detach().cpu(),
                        "per_frame": per_frame,
                    }
                    self._push_dbg_track(ex)

        # -------------------------
        # Total aux loss
        # -------------------------
        aux = zero
        if self.mil_lambda > 0.0:
            aux = aux + float(self.mil_lambda) * mil_loss
        if self.track_coh_enable and (self.track_coh_lambda > 0.0):
            aux = aux + float(self.track_coh_lambda) * coh_loss
        if self.go_enable and (self.go_lambda > 0.0):
            aux = aux + float(self.go_lambda) * go_loss

        if self.sam_concept_align_enable and (self.sam_concept_align_lambda > 0.0):
            aux = aux + float(self.sam_concept_align_lambda) * concept_align_loss

        stats = {
            "mil_loss": mil_loss.detach(),
            "mil_acc": mil_acc.detach(),
            "mil_acc_t2i": mil_acc_t2i.detach(),
            "mil_acc_i2t": mil_acc_i2t.detach(),
            "noun_count_mean": noun_count_mean,
            "noun_sample_frac": noun_sample_frac,
            "noun_query_frac": noun_query_frac,
            "concept_align_loss": concept_align_loss.detach(),
            "concept_align_acc": concept_align_acc,
            "concept_align_n": concept_align_n,
            "w_align_mean": w_align_mean,
            "w_align_eff_mean": w_sample_mean,
            "sam_concept_weight_mean": sam_w_mean,
            "track_coh_loss": coh_loss.detach(),
            "track_coh_pairs": coh_pairs.detach(),
            "go_loss": go_loss.detach(),
            "go_sim": go_sim.detach(),
            "proto_update_loss": proto_up_loss.detach(),
            "proto_usage_eff_k": proto_eff_k.detach(),
            "sam_mask_frac": sam_mask_frac,
            "patch_frac": patch_frac,
        }
        return aux, stats

    # ------------------------------------------------------------------
    # Debug track saving
    # ------------------------------------------------------------------
    def _push_dbg_track(self, ex: Dict[str, Any]) -> None:
        # Keep a small heap of top examples.
        if not hasattr(self, "_dbg_track_heap"):
            self._dbg_track_heap = []

        heap = self._dbg_track_heap
        score = float(ex.get("score", 0.0))
        if len(heap) < int(self.debug_tracks_topk):
            heap.append((score, ex))
            heap.sort(key=lambda x: x[0])
        else:
            if score > heap[0][0]:
                heap[0] = (score, ex)
                heap.sort(key=lambda x: x[0])

    def _debug_save_top_tracks_epoch_end(self) -> None:
        if not self.debug_save_tracks:
            return
        if (self.trainer is not None) and (not self.trainer.is_global_zero):
            return
        if (int(self.current_epoch) % int(self.debug_tracks_every_n_epochs)) != 0:
            return
        if not getattr(self, "_dbg_track_heap", None):
            return

        # Determine output directory
        trainer = getattr(self, "trainer", None)
        ckpt_dir = os.getcwd()
        if trainer is not None:
            cb = getattr(trainer, "checkpoint_callback", None)
            if cb is not None and getattr(cb, "dirpath", None):
                ckpt_dir = str(cb.dirpath)
            lg = getattr(trainer, "logger", None)
            if lg is not None and getattr(lg, "log_dir", None):
                ckpt_dir = str(lg.log_dir)

        out_dir = os.path.join(ckpt_dir, "debug_tracks", f"epoch_{int(self.current_epoch):04d}")
        os.makedirs(out_dir, exist_ok=True)

        items = sorted(self._dbg_track_heap, key=lambda x: float(x[0]), reverse=True)
        alpha = float(self.debug_tracks_overlay_alpha)

        for j, (score, ex) in enumerate(items):
            frames = ex["frames"]
            masks = ex.get("masks", None)
            # Use precomputed per-frame vis payload if available (preferred).
            per_frame = ex.get("per_frame", None)

            if per_frame is None:
                # Backwards-compatible fallback: show the first mask per frame if available.
                per_frame = []
                if masks is not None and torch.is_tensor(masks) and masks.ndim == 4:
                    # masks: (M,K,H,W)
                    M = int(masks.size(0))
                    for m in range(M):
                        mask0 = masks[m, 0].numpy()
                        per_frame.append({"kind": "sam", "mask": mask0, "conf": 1.0, "sim": 0.0})
                else:
                    for _ in range(int(frames.size(0))):
                        per_frame.append({"kind": "none", "conf": 0.0, "sim": 0.0})

            grid = make_track_grid(
                frames_mchw=frames,
                per_frame=per_frame,
                title=f"score={score:.3f} step={int(self.global_step)}",
                caption="",
                alpha=alpha,
            )

            png_path = os.path.join(out_dir, f"track_{j:02d}_score_{score:.4f}.png")
            grid.save(png_path)

            if wandb is not None and hasattr(self.logger, "experiment"):
                try:
                    self.logger.experiment.log({"debug_tracks/top": wandb.Image(grid)}, step=int(self.global_step))
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Main loss
    # ------------------------------------------------------------------
    def calculate_joint_loss(self, batch, stage, log, batch_idx, eval_textgen: bool = False, ce_weight=None):
        x, y, y_len, raw_y, meta = self._split_batch(batch)

        # Global InfoNCE uses anchor frame only.
        x_anchor = x[:, 0] if x.ndim == 5 else x

        ret: Dict[str, Any] = {"batch_size": int(x_anchor.size(0))}
        image_features, image_feature_map, text_outputs = None, None, None

        # -------------------------
        # CVCL InfoNCE
        # -------------------------
        if self.lambda_mm or not self.optimize_unused:
            (
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
            ) = self.model.calculate_contrastive_loss(x_anchor, y, y_len)

            retrieval_acc = 0.5 * (image_accuracy + text_accuracy)

            log(f"{stage}/infonce_loss", infonce_loss)
            log(f"{stage}/retrieval_acc", retrieval_acc)

            ret.update({"infonce_loss": infonce_loss.detach(), "retrieval_acc": retrieval_acc.detach()})
        else:
            infonce_loss = x_anchor.new_tensor(0.0)
            # If we skip the main model forward (optimize_unused=True), ensure DDP doesn't error.
            infonce_loss = self._tie_module_params_if_needed(infonce_loss, self.model)

        # -------------------------
        # MIL + prototype memory (Plan A)
        #   - By default we also compute these on val/test (for monitoring),
        #     but we never update prototype EMA outside training.
        # -------------------------
        aux_mil = x_anchor.new_tensor(0.0)
        do_mil = bool(self.mil_enable) and (stage == "train" or bool(getattr(self, "mil_run_val", False)))
        if do_mil:
            aux_mil, mil_stats = self._mil_losses(
                x=x,
                y=y,
                y_len=y_len,
                raw_y=raw_y,
                meta=meta if isinstance(meta, dict) else None,
                update_memory=bool(stage == "train"),
                stage=str(stage),
            )

            for k, v in mil_stats.items():
                log(f"{stage}/{k}", v)
            ret.update({k: v for k, v in mil_stats.items()})
        else:
            mil_stats = {}

        # -------------------------
        # LM (optional)
        # -------------------------
        if (self.lambda_lm > 0.0) or (self.lambda_ar > 0.0) or (not self.optimize_unused):
            ce_loss, _, _, attns, labels, image_features, image_feature_map = self.calculate_ce_loss(
                y,
                y_len,
                x=x_anchor,
                outputs=text_outputs,
                image_features=image_features,
                image_feature_map=image_feature_map,
                return_image_features=True,
                tokenwise=True,
                weight=ce_weight,
            )

            mask = labels != PAD_TOKEN_ID
            n_tokens = mask.sum()
            lm_ce_loss = ce_loss.sum() / n_tokens

            log(f"{stage}/ce_loss", lm_ce_loss)
            ret.update({"ce_loss": lm_ce_loss.detach(), "n_tokens": n_tokens})

            if self.language_model.text_encoder.has_attention and (self.lambda_ar > 0.0):
                attn_reg_loss = calculate_attn_reg_loss(attns)
                log(f"{stage}/attn_reg_loss", attn_reg_loss)
                ret["attn_reg_loss"] = attn_reg_loss.detach()
            else:
                attn_reg_loss = x_anchor.new_tensor(0.0)

            if eval_textgen:
                beam_seq, log_prob = self.language_model.beam_search_decode(
                    batch_size=ret["batch_size"],
                    beam_width=self.beam_width,
                    decode_length=self.decode_length,
                    length_penalty_alpha=self.length_penalty_alpha,
                    image_features=image_features if self.language_model.text_encoder.captioning else None,
                    image_feature_map=image_feature_map if self.language_model.text_encoder.has_attention else None,
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
                    return " ".join(self.text_encoder.idx2word[idx] for idx in y_list)

                gen_text_ids = beam_seq[:, 0]
                gen_text = [ids_to_sentence(y_seq) for y_seq in gen_text_ids]
                ret.update({"raw_y": raw_y, "gen_text": gen_text})
        else:
            lm_ce_loss = x_anchor.new_tensor(0.0)
            attn_reg_loss = x_anchor.new_tensor(0.0)
            lm_ce_loss = self._tie_module_params_if_needed(lm_ce_loss, self.language_model)

        # -------------------------
        # Total
        # -------------------------
        loss = (
            float(self.lambda_mm) * infonce_loss
            + float(self.lambda_lm) * lm_ce_loss
            + float(self.lambda_ar) * attn_reg_loss
            + aux_mil
        )

        # DDP safety: these params are only used by the MIL/proto branch.
        # (Safe to tie even if they were used; it adds a 0-weight term.)
        if self.null_obj is not None:
            loss = self._tie_params_if_needed(loss, [self.null_obj])
        if self.img_adapter is not None:
            loss = self._tie_params_if_needed(loss, list(self.img_adapter.parameters()))

        log(f"{stage}/loss", loss)
        ret["loss"] = loss
        return ret

    # ------------------------------------------------------------------
    # Epoch-end aggregation
    # ------------------------------------------------------------------
    def joint_loss_epoch_end(self, outputs, stage, log, eval_textgen: bool = False):
        def mean_over_examples(name: str) -> float:
            n_examples = 0
            value_sum = 0.0
            for output in outputs:
                batch_size = int(output["batch_size"])
                value = output.get(name, None)
                if value is None:
                    continue
                value_f = float(value.item()) if torch.is_tensor(value) else float(value)
                n_examples += batch_size
                value_sum += value_f * batch_size
            return value_sum / n_examples if n_examples > 0 else 0.0

        def mean_over_tokens(name: str, n_tokens_name: str) -> float:
            n_tokens_sum = 0
            value_sum = 0.0
            for output in outputs:
                if name not in output or n_tokens_name not in output:
                    continue
                n_tokens = int(output[n_tokens_name].item())
                value = float(output[name].item())
                n_tokens_sum += n_tokens
                value_sum += value * n_tokens
            return value_sum / n_tokens_sum if n_tokens_sum > 0 else 0.0

        # Contrastive
        if self.lambda_mm or not self.optimize_unused:
            for name in ("infonce_loss", "retrieval_acc"):
                if any(name in o for o in outputs):
                    log(f"{stage}/{name}", mean_over_examples(name))

        # MIL/proto (train and optionally val/test)
        if self.mil_enable and (stage == "train" or self.mil_run_val):
            for name in (
                "mil_loss",
                "mil_acc",
                "mil_acc_t2i",
                "mil_acc_i2t",
                "w_align_mean",
                "w_align_eff_mean",
                "sam_concept_weight_mean",
                "track_coh_loss",
                "track_coh_pairs",
                "go_loss",
                "go_sim",
                "proto_update_loss",
                "proto_usage_eff_k",
                "sam_mask_frac",
                "patch_frac",
                "noun_count_mean",
                "noun_sample_frac",
                "noun_query_frac",
                "concept_align_loss",
                "concept_align_acc",
                "concept_align_n",
            ):
                if any(name in o for o in outputs):
                    log(f"{stage}/{name}", mean_over_examples(name))

        # LM
        if (self.lambda_lm > 0.0) or (self.lambda_ar > 0.0) or (not self.optimize_unused):
            if any("ce_loss" in o for o in outputs):
                ce_mean = mean_over_tokens("ce_loss", "n_tokens")
                log(f"{stage}/ce_loss", ce_mean)
                log(f"{stage}/perplexity", float(np.exp(ce_mean)))

            if any("attn_reg_loss" in o for o in outputs):
                log(f"{stage}/attn_reg_loss", mean_over_examples("attn_reg_loss"))

        if eval_textgen:
            list_of_references, hypotheses = [], []
            for output in outputs:
                list_of_references += output["raw_y"]
                hypotheses += output["gen_text"]

            list_of_references = _ddp_all_gather_object_list(list_of_references)
            hypotheses = _ddp_all_gather_object_list(hypotheses)

            if (self.trainer is None) or self.trainer.is_global_zero:
                for example_id in PRINT_EVAL_TEXTGEN_EXAMPLE_IDS:
                    if example_id >= len(hypotheses):
                        continue
                    print(f"example #{example_id}:")
                    references = list_of_references[example_id]
                    hypothesis = hypotheses[example_id]
                    print("references:")
                    print("\n".join(references))
                    print("hypothesis:")
                    print(hypothesis)

                score_dict = textgen_eval(list_of_references, hypotheses)
                for metric, score in score_dict.items():
                    self.log(f"{stage}/{metric}", score, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

        log(f"{stage}/loss", mean_over_examples("loss"))

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        try:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("train/lr", lr, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        except Exception:
            pass
        return self.calculate_joint_loss(batch, "train", self.log, batch_idx, eval_textgen=False)

    def training_epoch_end(self, outputs):
        log = functools.partial(self.log, on_step=False, on_epoch=True, sync_dist=True)
        ret = self.joint_loss_epoch_end(outputs, "train", log, eval_textgen=False)
        self._debug_save_top_tracks_epoch_end()
        return ret

    def validation_test_step(self, stage, batch, batch_idx, dataloader_idx: int = 0):
        log = functools.partial(self.log, on_step=False, on_epoch=True, sync_dist=True)
        ret: Dict[str, Any] = {}

        if dataloader_idx == 0:
            empty_log = lambda *args, **kwargs: None
            ret.update(self.calculate_joint_loss(batch, stage, empty_log, batch_idx, eval_textgen=self.eval_textgen))
        elif dataloader_idx == 1:
            x, y, y_len, raw_y, _ = self._split_batch(batch)
            x = x.view(-1, *x.shape[-3:])

            if self.lambda_mm:
                logits_per_image, logits_per_text = self.model(x, y, y_len)
                logits = logits_per_text[0]
            elif (
                (self.lambda_lm > 0.0)
                and (self.language_model.text_encoder.captioning or self.language_model.text_encoder.has_attention)
                and y[0, 0].item() == SOS_TOKEN_ID
            ):
                y = y.expand(x.size(0), -1)
                y_len = y_len.expand(x.size(0))
                ce_loss, _, _, _, labels = self.calculate_ce_loss(y, y_len, x=x, tokenwise=True)
                logits = -ce_loss[:, 0]
            else:
                logits = None

            if logits is not None:
                pred = torch.argmax(logits).item()
                label = 0
                accuracy = int(pred == label)
                entropy = get_entropy(logits)

                log(f"{stage}/accuracy", accuracy)
                log(f"{stage}/entropy", entropy)

                category_label = raw_y[0][0]
                log(f"{stage}/accuracy_{category_label}", accuracy)
                ret.update({"accuracy": accuracy})

        return ret

    def validation_test_epoch_end(self, stage, outputs):
        log = functools.partial(self.log, on_step=False, on_epoch=True, sync_dist=True)
        return self.joint_loss_epoch_end(outputs[0], stage, log, eval_textgen=self.eval_textgen)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx < N_VAL_DATALOADERS_PER_SPLIT:
            return self.validation_test_step("val", batch, batch_idx, dataloader_idx=dataloader_idx)
        return self.test_step(batch, batch_idx, dataloader_idx=dataloader_idx - N_VAL_DATALOADERS_PER_SPLIT)

    def validation_epoch_end(self, outputs):
        self.validation_test_epoch_end("val", outputs[:N_VAL_DATALOADERS_PER_SPLIT])
        if len(outputs) > N_VAL_DATALOADERS_PER_SPLIT:
            self.test_epoch_end(outputs[N_VAL_DATALOADERS_PER_SPLIT:])

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_test_step("test", batch, batch_idx, dataloader_idx=dataloader_idx)

    def test_epoch_end(self, outputs):
        return self.validation_test_epoch_end("test", outputs)

    def on_before_zero_grad(self, optimizer) -> None:
        """Runs right after optimizer.step() and before zero_grad()."""

        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if grads:
            total = torch.norm(torch.stack([g.detach().float().norm(2) for g in grads]), 2)
            self.log("train/grad_norm", total, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        try:
            lr = optimizer.param_groups[0]["lr"]
            self.log("train/lr", lr, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        except Exception:
            pass
