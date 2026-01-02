from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List, Union
import math

import torch
import torch.nn.functional as F
from torch import nn

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------


def l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=eps)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: (N, D), b: (M, D) -> (N, M)
    Assumes a and b are L2-normalized if you want true cosine similarity.
    """
    return a @ b.t()


def masked_spatial_pool(fmap: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    fmap: (B,E,Hf,Wf), mask: (B,1,Hf,Wf) boolean or float
    returns pooled (B,E)
    """
    mask = mask.float()
    num = (fmap * mask).sum(dim=(2, 3))
    denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
    return num / denom


def _ensure_odd(k: int) -> int:
    k = int(k)
    if k <= 1:
        return 1
    return k if (k % 2 == 1) else (k + 1)


def gaussian_blur2d(
    x: torch.Tensor,
    kernel_size: int = 23,
    sigma: float = 5.0,
    padding_mode: str = "reflect",  # "reflect" for images, "constant" for masks
) -> torch.Tensor:
    """
    Fixed Gaussian blur with depthwise conv2d.
    x: (B,C,H,W)
    """
    if sigma is None:
        return x
    sigma = float(sigma)
    if sigma <= 0.0:
        return x

    k = _ensure_odd(int(kernel_size))
    if k <= 1:
        return x

    device = x.device
    dtype_in = x.dtype
    x_f = x.float()

    # build 1D gaussian
    radius = (k - 1) / 2.0
    coords = torch.arange(k, device=device, dtype=torch.float32) - radius
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum().clamp_min(1e-12)

    # outer product -> 2D kernel
    kernel2d = torch.outer(g, g)
    kernel2d = kernel2d / kernel2d.sum().clamp_min(1e-12)

    # depthwise conv kernel: (C,1,k,k)
    C = x_f.size(1)
    weight = kernel2d.view(1, 1, k, k).to(device=device, dtype=torch.float32)
    weight = weight.expand(C, 1, k, k).contiguous()

    pad = k // 2
    if pad > 0:
        if padding_mode == "reflect":
            x_pad = F.pad(x_f, (pad, pad, pad, pad), mode="reflect")
        elif padding_mode == "replicate":
            x_pad = F.pad(x_f, (pad, pad, pad, pad), mode="replicate")
        elif padding_mode == "constant":
            x_pad = F.pad(x_f, (pad, pad, pad, pad), mode="constant", value=0.0)
        else:
            raise ValueError(f"Unsupported padding_mode: {padding_mode}")
    else:
        x_pad = x_f

    y = F.conv2d(x_pad, weight, bias=None, stride=1, padding=0, groups=C)
    return y.to(dtype_in)


def dilate_mask2d(mask: torch.Tensor, radius_px: int) -> torch.Tensor:
    """
    Binary/soft dilation via max pooling.
    mask: (B,1,H,W) float in [0,1]
    """
    r = int(radius_px)
    if r <= 0:
        return mask
    k = 2 * r + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=r)


def make_context_alpha(
    mask: torch.Tensor,
    ring_px: int = 0,
    ring_strength: float = 1.0,
    feather_sigma: float = 0.0,
    feather_kernel: int | None = None,
) -> torch.Tensor:
    """
    Build a soft alpha mask used for blur-fill prompting.

    mask: (B,1,H,W) or (B,H,W) in {0,1} (or soft), foreground=1
    ring_px: dilation radius in pixels (at input resolution, e.g. 224x224)
    ring_strength: alpha value for the ring region (0..1)
    feather_sigma: Gaussian sigma for feathering the alpha boundaries
    feather_kernel: optional kernel size; if None, derived from sigma

    Returns:
      alpha: (B,1,H,W) in [0,1]
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4 or mask.size(1) != 1:
        raise ValueError(f"mask must have shape (B,1,H,W) or (B,H,W), got {tuple(mask.shape)}")

    m = mask.float().clamp(0.0, 1.0)

    rp = int(ring_px)
    rs = float(ring_strength)
    rs = max(0.0, min(1.0, rs))

    # ring: dilate outward and optionally keep some surrounding context
    if rp > 0:
        d = dilate_mask2d(m, rp)                 # object + ring
        ring = (d - m).clamp(0.0, 1.0)           # ring only
        alpha = (m + rs * ring).clamp(0.0, 1.0)  # object=1, ring=rs
    else:
        alpha = m

    fs = float(feather_sigma)
    if fs > 0.0:
        if feather_kernel is None or int(feather_kernel) <= 0:
            feather_kernel = int(2 * math.ceil(3.0 * fs) + 1)
        feather_kernel = _ensure_odd(int(feather_kernel))

        # for masks, use constant padding (outside image should be 0)
        alpha = gaussian_blur2d(alpha, kernel_size=feather_kernel, sigma=fs, padding_mode="constant")
        alpha = alpha.clamp(0.0, 1.0)

        # keep object interior fully visible
        alpha = torch.where(m > 0.5, torch.ones_like(alpha), alpha)

    return alpha


# ----------------------------------------------------------------------
# Object appearance encoder
# ----------------------------------------------------------------------


class ObjectAppearanceEncoder(nn.Module):
    """
    Fuse global and local features for an object into a single embedding.

    NOTE:
    - This is unchanged structurally from your current version.
    - It returns L2-normalized embeddings (good for cosine-based VM).
    """

    def __init__(
        self,
        dim_global: int,
        dim_local: int,
        dim_out: int,
        use_local: bool = True,
        local_weight: float = 0.5,
        learn_local_weight: bool = False,
    ):
        super().__init__()
        self.use_local = use_local
        w = torch.tensor(float(local_weight))
        self.local_weight = nn.Parameter(w, requires_grad=learn_local_weight)
        self.proj = nn.Linear(dim_global + dim_local, dim_out)

    def forward(self, global_feat: torch.Tensor, local_feat: torch.Tensor) -> torch.Tensor:
        if not self.use_local:
            local_feat = torch.zeros_like(local_feat)
        fused = torch.cat([global_feat, self.local_weight * local_feat], dim=-1)
        z = self.proj(fused)
        z = l2norm(z)
        return z


# ----------------------------------------------------------------------
# Sinkhorn (SwAV-style) for balanced assignments
# ----------------------------------------------------------------------


@torch.no_grad()
def sinkhorn_knopp(
    scores: torch.Tensor,
    n_iters: int = 3,
    epsilon: float = 0.05,
) -> torch.Tensor:
    """
    Compute balanced soft assignments using Sinkhorn-Knopp.

    Args:
      scores: (B, K) similarity scores (higher is better).
      n_iters: number of normalization iterations.
      epsilon: entropy regularization / sharpness. Smaller => sharper.

    Returns:
      q: (B, K) row-stochastic (each row sums to 1) with approximately uniform
         marginal over prototypes across the batch (each prototype gets ~B/K mass).
    """
    if scores.dim() != 2:
        raise ValueError(f"scores must be (B,K), got {tuple(scores.shape)}")
    B, K = scores.shape
    if B == 0 or K == 0:
        return scores.new_empty((B, K))

    eps = float(max(epsilon, 1e-6))

    # logits in a stable range
    x = scores.float() / eps
    x = x - x.max(dim=1, keepdim=True).values  # stabilize exp per row

    Q = torch.exp(x).t()  # (K, B)
    Q = Q / Q.sum().clamp_min(1e-12)

    # desired marginals: rows -> 1/K, cols -> 1/B
    r = torch.full((K, 1), 1.0 / K, device=Q.device, dtype=Q.dtype)
    c = torch.full((1, B), 1.0 / B, device=Q.device, dtype=Q.dtype)

    for _ in range(int(n_iters)):
        # normalize rows
        Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        Q = Q * r
        # normalize cols
        Q = Q / Q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        Q = Q * c

    Q = Q * B  # make columns sum to 1
    return Q.t().contiguous()  # (B, K)


# ----------------------------------------------------------------------
# Prototype visual memory (concept conditioned)
# ----------------------------------------------------------------------


@dataclass
class VMConfig:
    embedding_dim: int = 128
    num_concepts: int = 0
    protos_per_concept: int = 0
    bank_capacity: int = 512

    # assignment / clustering
    tau: float = 0.5
    use_soft_assignment: bool = True

    # Sinkhorn balancing (Tool D)
    use_sinkhorn: bool = True
    sinkhorn_iters: int = 3
    sinkhorn_epsilon: float = 0.05
    sinkhorn_min_samples: int = 2  # if < this, fall back to plain softmax

    # optional commitment term (cosine distance to quantized)
    beta: float = 0.25

    # temporal consistency (Fix D + best companions)
    enable_temporal: bool = True
    t_window: int = 2
    temporal_weight: float = 1.0
    temporal_same_concept_only: bool = True  # Fix A
    temporal_sim_thresh: float = 0.3         # Fix B (cosine on z)
    temporal_detach_target: bool = True      # Fix C

    # prototype separation (still important with SwAV-style)
    # sep_margin retains the old meaning approximately: "minimum Euclidean distance"
    # but we implement it via a cosine threshold derived from unit vectors.
    sep_margin: float = 0.5
    sep_weight: float = 0.1

    # concept-frequency reweighting (kept)
    enable_usage_reweighting: bool = True
    use_dataset_concept_counts: bool = True
    usage_smoothing: float = 1.0
    usage_power: float = 1.0
    usage_min: float = 0.33
    usage_max: float = 3.0
    usage_count_floor_quantile: float = 0.0

    # always keep prototypes on the unit sphere
    normalize_prototypes: bool = True


class VisualMemory(nn.Module):
    """
    Concept-conditioned prototype memory with:
      - cosine similarity (features + prototypes L2-normalized)
      - Sinkhorn-balanced soft assignments (SwAV-style) to prevent dead prototypes
      - temporal consistency via assignment-distribution distillation (Fix D)
        with same-concept gating + similarity threshold + detach-target

    Public API kept:
      - loss(feats, concept_ids, batch_meta) -> (total, L_q, L_temp, L_sep)
      - assign(feats, concept_ids, batch_meta) -> dict
      - W&B helpers
      - concept totals/names
    """

    def __init__(self, **kwargs):
        super().__init__()
        cfg = VMConfig(**kwargs)
        self.cfg = cfg

        D = int(cfg.embedding_dim)

        # concept-conditioned or global bank
        if cfg.num_concepts > 0 and cfg.protos_per_concept > 0:
            self.num_concepts = int(cfg.num_concepts)
            self.protos_per_concept = int(cfg.protos_per_concept)
            K = self.num_concepts * self.protos_per_concept
            proto_concept_ids = torch.arange(self.num_concepts).repeat_interleave(self.protos_per_concept)
        else:
            # global bank
            self.num_concepts = 1
            self.protos_per_concept = int(cfg.bank_capacity)
            K = self.protos_per_concept
            proto_concept_ids = torch.zeros(K, dtype=torch.long)

        self.K = int(K)

        self.prototypes = nn.Parameter(torch.randn(self.K, D))
        nn.init.normal_(self.prototypes, std=0.02)

        self.register_buffer("proto_concept_ids", proto_concept_ids)

        # usage tracking (diagnostics)
        self.register_buffer("usage_counts", torch.zeros(self.K, dtype=torch.float32))
        self._last_assign_hist: Optional[torch.Tensor] = None
        self._last_bank_util: Optional[torch.Tensor] = None

        # concept counts for reweighting
        self.register_buffer("concept_mask_counts", torch.zeros(self.num_concepts, dtype=torch.float32))
        self.register_buffer("concept_totals", torch.zeros(self.num_concepts, dtype=torch.float32))
        self._last_concept_weights: Optional[torch.Tensor] = None
        self._last_concept_weights_raw: Optional[torch.Tensor] = None
        self._last_clip_frac_at_max: Optional[torch.Tensor] = None
        self._last_slot_weights: Optional[torch.Tensor] = None

        # concept id -> name mapping (python side)
        self._concept_names: Optional[List[str]] = None

        # wandb step guard
        self._wandb_last_step: Optional[int] = None

    # ------------------------------------------------------------------
    # public setters
    # ------------------------------------------------------------------

    @torch.no_grad()
    def set_concept_totals(self, concept_totals: torch.Tensor) -> None:
        ct = concept_totals.detach().float().to(self.concept_totals.device)
        if ct.numel() != self.num_concepts:
            raise ValueError(f"concept_totals must have shape ({self.num_concepts},), got {tuple(ct.shape)}")
        self.concept_totals.copy_(ct)

    @torch.no_grad()
    def set_concept_names(self, names: Union[List[str], Dict[int, str]]) -> None:
        """
        Provide readable labels for concept ids used in W&B plots.
        """
        if isinstance(names, dict):
            out = [f"c{c}" for c in range(self.num_concepts)]
            for k, v in names.items():
                k_int = int(k)
                if 0 <= k_int < self.num_concepts:
                    out[k_int] = str(v)
            self._concept_names = out
        else:
            out = [str(x) for x in names]
            if len(out) != self.num_concepts:
                raise ValueError(f"names must have length {self.num_concepts}, got {len(out)}")
            self._concept_names = out

    def _concept_label(self, c: int) -> str:
        if self.num_concepts <= 1:
            return "all"
        if self._concept_names is None:
            return f"c{c}"
        if 0 <= c < len(self._concept_names):
            return str(self._concept_names[c])
        return f"c{c}"

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _renorm_prototypes_(self) -> None:
        if not bool(self.cfg.normalize_prototypes):
            return
        self.prototypes.copy_(l2norm(self.prototypes))

    @staticmethod
    def _filter_batch_meta(batch_meta: Optional[Dict[str, Any]], valid_mask: torch.Tensor) -> Optional[Dict[str, Any]]:
        if batch_meta is None:
            return None
        if not torch.is_tensor(valid_mask):
            return batch_meta
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()

        out: Dict[str, Any] = {}
        for k, v in batch_meta.items():
            if isinstance(v, (list, tuple)):
                if len(v) != int(valid_mask.numel()):
                    out[k] = v
                else:
                    idxs = torch.nonzero(valid_mask, as_tuple=False).flatten().tolist()
                    out[k] = [v[i] for i in idxs]
            elif torch.is_tensor(v):
                if v.numel() != valid_mask.numel():
                    out[k] = v
                else:
                    out[k] = v[valid_mask]
            else:
                out[k] = v
        return out

    def _compute_concept_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          w_clipped: (C,)
          w_raw:     (C,)
          frac_at_max: scalar
        """
        device = self.concept_mask_counts.device

        if (not bool(self.cfg.enable_usage_reweighting)) or self.num_concepts <= 1:
            w = torch.ones(self.num_concepts, device=device, dtype=torch.float32)
            return w, w.clone(), torch.tensor(0.0, device=device, dtype=torch.float32)

        use_totals = bool(self.cfg.use_dataset_concept_counts) and (self.concept_totals.sum() > 0)
        if use_totals:
            base_counts = torch.where(self.concept_totals > 0, self.concept_totals, self.concept_mask_counts)
        else:
            base_counts = self.concept_mask_counts

        counts = base_counts + float(self.cfg.usage_smoothing)

        q = float(self.cfg.usage_count_floor_quantile)
        if q > 0.0 and counts.numel() > 1:
            q = min(max(q, 0.0), 0.49)
            floor_val = torch.quantile(counts, q)
            counts = torch.clamp(counts, min=float(floor_val.item()))

        med_val = float(torch.median(counts).item())
        if med_val <= 0.0:
            med_val = float(counts.mean().clamp_min(1.0).item())

        w_raw = (med_val / counts).pow(float(self.cfg.usage_power))
        w_clipped = torch.clamp(w_raw, float(self.cfg.usage_min), float(self.cfg.usage_max))
        frac_at_max = (w_clipped >= (float(self.cfg.usage_max) - 1e-6)).float().mean()

        return w_clipped.to(device=device), w_raw.to(device=device), frac_at_max.to(device=device)

    # ------------------------------------------------------------------
    # per-concept prototype utilization (diagnostics)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_per_concept_proto_stats(self, use_batch_hist: bool = False) -> Dict[str, torch.Tensor]:
        if use_batch_hist:
            counts = (
                self._last_assign_hist.to(self.usage_counts.device)
                if self._last_assign_hist is not None
                else torch.zeros_like(self.usage_counts)
            )
        else:
            counts = self.usage_counts

        if self.num_concepts > 1 and self.protos_per_concept > 0:
            C = self.num_concepts
            P = self.protos_per_concept
            counts_cp = counts.view(C, P)
        else:
            C = 1
            P = counts.numel()
            counts_cp = counts.view(1, P)

        total = counts_cp.sum(dim=1)
        used_frac = (counts_cp > 1e-6).float().mean(dim=1)
        top1_frac = counts_cp.max(dim=1).values / total.clamp_min(1.0)

        return {"total": total, "used_frac": used_frac, "top1_frac": top1_frac}

    # ------------------------------------------------------------------
    # loss
    # ------------------------------------------------------------------

    def loss(
        self,
        feats: torch.Tensor,
        concept_ids: Optional[torch.Tensor] = None,
        batch_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          total, L_q, L_temp, L_sep
        """
        if feats is None:
            zero = self.prototypes.new_tensor(0.0)
            return zero, zero, zero, zero

        if feats.dim() != 2:
            feats = feats.view(feats.size(0), -1)

        device = feats.device
        z = feats.float()

        # filter invalid concept ids
        cid = concept_ids
        if cid is not None:
            cid = cid.to(device)
            valid_mask = cid >= 0
            if not valid_mask.any():
                zero = z.new_tensor(0.0)
                return zero, zero, zero, zero
            z = z[valid_mask]
            cid = cid[valid_mask]
            batch_meta = self._filter_batch_meta(batch_meta, valid_mask)

        if z.size(0) == 0:
            zero = feats.new_tensor(0.0)
            return zero, zero, zero, zero

        # cosine space: normalize features
        z = l2norm(z)

        # keep prototypes on the unit sphere
        self._renorm_prototypes_()
        p_all = l2norm(self.prototypes)  # (K, D)

        N = z.size(0)

        # update concept mask counts + compute concept weights
        with torch.no_grad():
            if cid is not None and self.num_concepts > 1:
                cid_long = cid.detach().long().clamp(min=0, max=self.num_concepts - 1)
                chist = torch.bincount(cid_long, minlength=self.num_concepts).float().to(self.concept_mask_counts.device)
                self.concept_mask_counts.add_(chist)

            w_concept, w_raw, frac_at_max = self._compute_concept_weights()
            self._last_concept_weights = w_concept
            self._last_concept_weights_raw = w_raw
            self._last_clip_frac_at_max = frac_at_max

            self._last_slot_weights = w_concept[self.proto_concept_ids.to(w_concept.device).long()]

        # per-instance weights (for rare concepts)
        if cid is None or self.num_concepts <= 1:
            w_inst = torch.ones(N, device=device, dtype=z.dtype)
        else:
            w_inst = self._last_concept_weights.to(device=device, dtype=z.dtype)[
                cid.long().clamp(min=0, max=self.num_concepts - 1)
            ].detach()

        w_inst = w_inst / w_inst.mean().clamp_min(1e-6)

        # group by concept (when concept-conditioned)
        if cid is None or self.num_concepts <= 1 or self.protos_per_concept <= 0:
            concept_groups = [(0, torch.arange(N, device=device))]
        else:
            # only process concepts present in this batch
            uniq = torch.unique(cid.detach().long().clamp(min=0, max=self.num_concepts - 1))
            concept_groups = []
            for c in uniq.tolist():
                idx_c = torch.nonzero(cid == int(c), as_tuple=False).flatten()
                if idx_c.numel() > 0:
                    concept_groups.append((int(c), idx_c))

        # accumulators
        Lq_sum = z.new_tensor(0.0)
        Lq_wsum = z.new_tensor(0.0)

        Lt_sum = z.new_tensor(0.0)
        Lt_wsum = z.new_tensor(0.0)

        # for usage statistics (soft histogram)
        hist_full = torch.zeros(self.K, device=self.usage_counts.device, dtype=torch.float32)

        # temporal meta (optional)
        clip_ids_all = None
        frame_idx_all = None
        if bool(self.cfg.enable_temporal) and batch_meta is not None:
            if "clip_id" in batch_meta and "frame_idx" in batch_meta:
                clip_ids_all = batch_meta["clip_id"]
                frame_idx_all = batch_meta["frame_idx"]
                if torch.is_tensor(clip_ids_all):
                    clip_ids_all = clip_ids_all.to(device=device, dtype=torch.long)
                else:
                    clip_ids_all = None
                if torch.is_tensor(frame_idx_all):
                    frame_idx_all = frame_idx_all.to(device=device, dtype=torch.long)
                else:
                    frame_idx_all = None

        tau = float(max(self.cfg.tau, 1e-6))
        beta = float(max(self.cfg.beta, 0.0))

        # -------- per-concept assignment + loss --------
        for c, idx_c in concept_groups:
            z_c = z[idx_c]               # (Nc, D)
            w_c = w_inst[idx_c]          # (Nc,)

            # select prototype slice for this concept
            if self.num_concepts > 1 and self.protos_per_concept > 0:
                p0 = int(c) * int(self.protos_per_concept)
                p1 = p0 + int(self.protos_per_concept)
            else:
                p0, p1 = 0, self.K

            p_c = p_all[p0:p1]  # (Pc, D)
            if p_c.numel() == 0:
                continue

            # cosine similarity
            sim = cosine_sim(z_c, p_c)  # (Nc, Pc)

            # differentiable prediction distribution
            log_probs = F.log_softmax(sim / tau, dim=1)  # (Nc, Pc)
            q_pred = log_probs.exp()                     # (Nc, Pc)

            # balanced target distribution (Sinkhorn) or fallback
            use_sinkhorn = bool(self.cfg.use_sinkhorn) and bool(self.cfg.use_soft_assignment)
            if use_sinkhorn and z_c.size(0) >= int(self.cfg.sinkhorn_min_samples) and p_c.size(0) > 1:
                q_tgt = sinkhorn_knopp(
                    scores=sim.detach(),
                    n_iters=int(self.cfg.sinkhorn_iters),
                    epsilon=float(self.cfg.sinkhorn_epsilon),
                ).to(device=device, dtype=z.dtype)
            else:
                # fallback: use current softmax as target
                q_tgt = q_pred.detach()

            # update usage histogram (soft mass) in full prototype space
            with torch.no_grad():
                hist_c = q_tgt.detach().float().sum(dim=0)  # (Pc,)
                # move to usage buffer device if needed
                if hist_full.device != self.usage_counts.device:
                    hist_c = hist_c.to(self.usage_counts.device)
                hist_full[p0:p1] += hist_c

            # quantization / clustering loss:
            #   CE(q_tgt || softmax(sim/tau)) + beta * commitment_cos(z, z_hat)
            per_ex_ce = -(q_tgt * log_probs).sum(dim=1)  # (Nc,)

            if beta > 0.0:
                # quantized embedding (mixture of prototypes), then unit-normalize
                z_hat = l2norm(q_tgt @ p_c)  # (Nc, D), q_tgt treated as constant
                # commitment: encourage z to align with z_hat (detach target)
                per_ex_commit = 1.0 - (z_c * z_hat.detach()).sum(dim=1)
                per_ex = per_ex_ce + beta * per_ex_commit
            else:
                per_ex = per_ex_ce

            Lq_sum = Lq_sum + (per_ex * w_c).sum()
            Lq_wsum = Lq_wsum + w_c.sum().clamp_min(1e-12)

            # -------- temporal loss (Fix D + A/B/C) --------
            if (
                bool(self.cfg.enable_temporal)
                and float(self.cfg.temporal_weight) > 0.0
                and clip_ids_all is not None
                and frame_idx_all is not None
                and z_c.size(0) >= 2
            ):
                # same-concept gating: already in this group
                clip_c = clip_ids_all[idx_c]
                t_c = frame_idx_all[idx_c]

                same_clip = clip_c.unsqueeze(0) == clip_c.unsqueeze(1)
                close_time = (t_c.unsqueeze(0) - t_c.unsqueeze(1)).abs() <= int(self.cfg.t_window)

                P = same_clip & close_time
                P.fill_diagonal_(False)

                # only count each unordered pair once
                P = torch.triu(P, diagonal=1)

                if P.any():
                    ii, jj = torch.nonzero(P, as_tuple=True)
                    if ii.numel() > 0:
                        # similarity gate (Fix B)
                        sim_thresh = float(self.cfg.temporal_sim_thresh)
                        if sim_thresh > -1.0:
                            cos_pairs = (z_c[ii] * z_c[jj]).sum(dim=1)
                            keep = cos_pairs > sim_thresh
                            if keep.any():
                                ii = ii[keep]
                                jj = jj[keep]
                            else:
                                ii = ii[:0]
                                jj = jj[:0]

                        if ii.numel() > 0:
                            ti = t_c[ii]
                            tj = t_c[jj]

                            # skip equal timestamps (no direction)
                            neq = ti != tj
                            if neq.any():
                                ii = ii[neq]
                                jj = jj[neq]
                                ti = ti[neq]
                                tj = tj[neq]

                            if ii.numel() > 0:
                                # direction: earlier is target, later is student
                                i_earlier = torch.where(ti < tj, ii, jj)
                                i_later = torch.where(ti < tj, jj, ii)

                                # target distribution (detach for stability, Fix C)
                                if bool(self.cfg.temporal_detach_target):
                                    q_t = q_pred[i_earlier].detach()
                                else:
                                    q_t = q_pred[i_earlier]

                                # student log-probs
                                logp_s = log_probs[i_later]  # (Np, Pc)

                                # cross-entropy (equiv to KL up to constant):
                                #   H(q_target, q_student) = -sum q_target * log q_student
                                per_pair = -(q_t * logp_s).sum(dim=1)  # (Np,)

                                # weight by the "student" instance weight
                                w_pair = w_c[i_later]
                                Lt_sum = Lt_sum + (per_pair * w_pair).sum()
                                Lt_wsum = Lt_wsum + w_pair.sum().clamp_min(1e-12)

        # finalize losses
        L_q = Lq_sum / Lq_wsum.clamp_min(1e-12)

        if Lt_wsum.item() > 0:
            L_temp = (Lt_sum / Lt_wsum.clamp_min(1e-12)) * float(self.cfg.temporal_weight)
        else:
            L_temp = z.new_tensor(0.0)

        # prototype separation (cosine form derived from sep_margin distance)
        L_sep = z.new_tensor(0.0)
        if float(self.cfg.sep_weight) > 0.0 and self.K > 1:
            # convert distance margin to cosine threshold for unit vectors:
            # ||u - v||^2 = 2 - 2cos  => cos = 1 - d^2/2
            d = float(self.cfg.sep_margin)
            cos_max = 1.0 - (d * d) / 2.0
            cos_max = float(max(-1.0, min(1.0, cos_max)))

            per_concept_losses: List[torch.Tensor] = []
            if self.num_concepts > 1 and self.protos_per_concept > 0:
                C = self.num_concepts
                Pp = self.protos_per_concept
                for c in range(C):
                    p0 = c * Pp
                    p1 = p0 + Pp
                    if p1 - p0 <= 1:
                        continue
                    pc = p_all[p0:p1]  # (Pp, D), unit
                    cosm = pc @ pc.t()  # (Pp, Pp)
                    # off-diagonal
                    mask = ~torch.eye(Pp, device=cosm.device, dtype=torch.bool)
                    vals = cosm[mask]
                    if vals.numel() == 0:
                        continue
                    loss_c = torch.relu(vals - cos_max).pow(2).mean()
                    per_concept_losses.append(loss_c)
            else:
                # global bank
                K = p_all.size(0)
                cosm = p_all @ p_all.t()
                mask = ~torch.eye(K, device=cosm.device, dtype=torch.bool)
                vals = cosm[mask]
                if vals.numel() > 0:
                    per_concept_losses.append(torch.relu(vals - cos_max).pow(2).mean())

            if per_concept_losses:
                L_sep = torch.stack(per_concept_losses).mean() * float(self.cfg.sep_weight)

        total = L_q + L_temp + L_sep

        # update cumulative usage counts (soft histogram)
        with torch.no_grad():
            self.usage_counts.add_(hist_full.to(self.usage_counts.device))
            self._last_assign_hist = hist_full.detach().clone()

            used = (self.usage_counts > 1e-6).float()
            self._last_bank_util = used.mean() if used.numel() > 0 else torch.tensor(0.0, device=self.usage_counts.device)

        return total, L_q, L_temp, L_sep

    # ------------------------------------------------------------------
    # assign (for analysis / visualization)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def assign(
        self,
        feats: torch.Tensor,
        concept_ids: torch.Tensor,
        batch_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        device = feats.device
        concept_ids = concept_ids.to(device)

        valid = concept_ids >= 0
        if valid.sum() == 0:
            D = feats.size(-1)
            return {
                "valid_mask": valid,
                "embeds": feats.new_empty((0, D)),
                "concept_ids": concept_ids.new_empty((0,)),
                "proto_indices": concept_ids.new_empty((0,), dtype=torch.long),
                "quantized": feats.new_empty((0, D)),
            }

        x = feats[valid].float()
        cids = concept_ids[valid].long()

        x = l2norm(x)

        self._renorm_prototypes_()
        p_all = l2norm(self.prototypes)

        tau = float(max(self.cfg.tau, 1e-6))

        proto_indices_list: List[torch.Tensor] = []
        quantized_list: List[torch.Tensor] = []

        if self.num_concepts > 1 and self.protos_per_concept > 0:
            uniq = torch.unique(cids.clamp(min=0, max=self.num_concepts - 1))
            for c in uniq.tolist():
                idx = torch.nonzero(cids == int(c), as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                p0 = int(c) * int(self.protos_per_concept)
                p1 = p0 + int(self.protos_per_concept)
                pc = p_all[p0:p1]
                sim = cosine_sim(x[idx], pc)
                if bool(self.cfg.use_soft_assignment):
                    q = torch.softmax(sim / tau, dim=1)
                    zq = l2norm(q @ pc)
                    pi_local = q.argmax(dim=1) + p0
                else:
                    pi_local = sim.argmax(dim=1) + p0
                    zq = p_all[pi_local]
                proto_indices_list.append(pi_local)
                quantized_list.append(zq)

            proto_idx = torch.cat(proto_indices_list, dim=0)
            z_q = torch.cat(quantized_list, dim=0)

            # NOTE: concatenation order follows uniq-concept order, not original order.
            # If you need original ordering, you can re-sort using gathered indices.
            return {
                "valid_mask": valid,
                "embeds": x,
                "concept_ids": cids,
                "proto_indices": proto_idx,
                "quantized": z_q,
            }

        else:
            sim = cosine_sim(x, p_all)
            if bool(self.cfg.use_soft_assignment):
                q = torch.softmax(sim / tau, dim=1)
                z_q = l2norm(q @ p_all)
                proto_idx = q.argmax(dim=1)
            else:
                proto_idx = sim.argmax(dim=1)
                z_q = p_all[proto_idx]

            return {
                "valid_mask": valid,
                "embeds": x,
                "concept_ids": cids,
                "proto_indices": proto_idx,
                "quantized": z_q,
            }

    # ------------------------------------------------------------------
    # W&B visualization (tables + wandb.plot.bar)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def wandb_log_concept_mean_weight_bar(self, step: int, prefix: str = "vm") -> None:
        try:
            import wandb  # type: ignore
        except Exception:
            return
        run = getattr(wandb, "run", None)
        if run is None or self._last_concept_weights is None:
            return

        w = self._last_concept_weights.detach().float().cpu()
        table = wandb.Table(columns=["concept", "mean_weight"])
        for c in range(int(w.numel())):
            table.add_data(self._concept_label(c), float(w[c].item()))

        run.log(
            {f"{prefix}/concept_mean_weight": wandb.plot.bar(
                table, "concept", "mean_weight", title="Mean weights"
            )},
            commit=True
        )

    @torch.no_grad()
    def wandb_log_proto_utilization_per_concept(
        self,
        step: int,
        prefix: str = "vm",
        use_batch_hist: bool = False
    ) -> None:
        """
        Bar charts for per-concept prototype utilization.
        """
        try:
            import wandb  # type: ignore
        except Exception:
            return

        stats = self.get_per_concept_proto_stats(use_batch_hist=use_batch_hist)
        used_frac = stats["used_frac"].detach().float().cpu()
        top1_frac = stats["top1_frac"].detach().float().cpu()
        total = stats["total"].detach().float().cpu()

        C = int(used_frac.numel())
        table = wandb.Table(columns=["concept", "used_frac", "top1_frac", "total"])
        for c in range(C):
            table.add_data(
                self._concept_label(c),
                float(used_frac[c].item()),
                float(top1_frac[c].item()),
                float(total[c].item()),
            )

        wandb.log(
            {
                f"{prefix}/proto_used_frac": wandb.plot.bar(
                    table, "concept", "used_frac", title="Prototype used fraction"
                ),
                f"{prefix}/proto_top1_frac": wandb.plot.bar(
                    table, "concept", "top1_frac", title="Top-1 prototype dominance"
                )
            },
            commit=True
        )


# ----------------------------------------------------------------------
# Visualization helpers for object masks
# ----------------------------------------------------------------------


def sample_masks_per_concept_for_viz(
    imgs: torch.Tensor,
    masks: torch.Tensor,
    mask_concepts: torch.Tensor,
    num_concepts: int,
    max_per_concept: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = imgs.device
    B, K, _, H, W = masks.shape

    imgs_flat = imgs.unsqueeze(1).expand(B, K, 3, H, W).reshape(-1, 3, H, W)
    masks_flat = masks.reshape(-1, 1, H, W)
    concepts_flat = mask_concepts.reshape(-1).to(device).long()

    sel_imgs = []
    sel_masks = []
    sel_concepts = []

    for c in range(num_concepts):
        idx_c = torch.nonzero(concepts_flat == c, as_tuple=False).flatten()
        if idx_c.numel() == 0:
            continue
        if idx_c.numel() > max_per_concept:
            perm = torch.randperm(idx_c.numel(), device=device)[:max_per_concept]
            idx_c = idx_c[perm]
        sel_imgs.append(imgs_flat[idx_c])
        sel_masks.append(masks_flat[idx_c])
        sel_concepts.append(concepts_flat[idx_c])

    if not sel_imgs:
        return (
            torch.empty(0, 3, H, W, device=device),
            torch.empty(0, 1, H, W, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    sel_imgs = torch.cat(sel_imgs, dim=0)
    sel_masks = torch.cat(sel_masks, dim=0)
    sel_concepts = torch.cat(sel_concepts, dim=0)
    return sel_imgs, sel_masks, sel_concepts


def overlay_masks_on_images(imgs: torch.Tensor, masks: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    m = masks.clamp(0.0, 1.0)
    base = imgs.clone()

    overlay = torch.zeros_like(base)
    overlay[:, 0:1, :, :] = 1.0

    overlays = base * (1.0 - alpha * m) + overlay * (alpha * m)
    return overlays
