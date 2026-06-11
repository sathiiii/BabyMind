from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def masked_pool_k(
    fmap: torch.Tensor,
    masks: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Vectorized masked average pooling over K masks per feature map.

    fmap:  (N, C, Hf, Wf)
    masks: (N, K, 1, H, W) or (N, K, 1, Hf, Wf), float in [0,1]
    returns: (N, K, C)

    Key detail:
      - If masks need resizing to fmap resolution, we use area downsampling when shrinking
        (preserves silhouette mass and reduces aliasing).
    """
    if fmap.ndim != 4:
        raise ValueError(f"masked_pool_k expects fmap (N,C,Hf,Wf), got {fmap.shape}")
    if masks.ndim != 5:
        raise ValueError(f"masked_pool_k expects masks (N,K,1,H,W), got {masks.shape}")

    N, C, Hf, Wf = fmap.shape
    Hm, Wm = masks.shape[-2], masks.shape[-1]

    m = masks.float().clamp(0.0, 1.0)

    if (Hm, Wm) != (Hf, Wf):
        # Downsample with area when shrinking; bilinear when expanding.
        if Hm >= Hf and Wm >= Wf:
            m = F.interpolate(m.view(N * m.size(1), 1, Hm, Wm), size=(Hf, Wf), mode="area").view(N, m.size(1), 1, Hf, Wf)
        else:
            m = F.interpolate(
                m.view(N * m.size(1), 1, Hm, Wm),
                size=(Hf, Wf),
                mode="bilinear",
                align_corners=False,
            ).view(N, m.size(1), 1, Hf, Wf)

    w = m  # (N,K,1,Hf,Wf) soft weights

    num = (fmap.unsqueeze(1) * w).sum(dim=(3, 4))                 # (N,K,C)
    den = w.sum(dim=(3, 4)).clamp(min=eps)                        # (N,K,1)
    return num / den


def context_ring_masks(
    masks_ds: torch.Tensor,
    ring_px: int,
) -> torch.Tensor:
    """
    Build context-ring masks at feature-map resolution via dilation.

    masks_ds: (N, K, 1, Hf, Wf) float in [0,1]
    returns:  (N, K, 1, Hf, Wf) float in [0,1]
    """
    if ring_px <= 0:
        return torch.zeros_like(masks_ds)

    k = 2 * int(ring_px) + 1
    m = masks_ds.float().clamp(0.0, 1.0)
    m_flat = m.view(-1, 1, m.size(-2), m.size(-1))
    dil = F.max_pool2d(m_flat, kernel_size=k, stride=1, padding=ring_px)
    dil = dil.view_as(m)
    ring = (dil - m).clamp(0.0, 1.0)
    return ring


@dataclass
class TrackConfig:
    sim_thresh: float = 0.3
    max_tracks: int = 32


def build_object_tracks_greedy(
    z_mkd: torch.Tensor,
    valid_mk: torch.Tensor,
    cfg: TrackConfig,
) -> torch.Tensor:
    """
    Greedy object-file tracking across frames.

    z_mkd:   (M, K, D) L2-normalized embeddings
    valid_mk:(M, K) boolean

    Returns:
      tracks: (R, D) L2-normalized track embeddings (R <= cfg.max_tracks)
    """
    if z_mkd.ndim != 3:
        raise ValueError(f"build_object_tracks_greedy expects (M,K,D), got {z_mkd.shape}")
    if valid_mk.ndim != 2:
        raise ValueError(f"build_object_tracks_greedy expects (M,K), got {valid_mk.shape}")

    M, K, D = z_mkd.shape
    device = z_mkd.device

    tracks: List[torch.Tensor] = []
    counts: List[int] = []

    for t in range(M):
        vt = valid_mk[t]
        if not bool(vt.any()):
            continue
        objs = z_mkd[t][vt]  # (n_obj, D)
        n_obj = int(objs.size(0))
        if n_obj == 0:
            continue

        if len(tracks) == 0:
            for i in range(n_obj):
                tracks.append(objs[i].clone())
                counts.append(1)
            continue

        track_tensor = torch.stack(tracks, dim=0)  # (T, D)
        sims = objs @ track_tensor.t()  # (n_obj, T)
        best_sim, best_idx = sims.max(dim=1)  # (n_obj,)

        order = torch.argsort(best_sim, descending=True)
        assigned_obj = torch.zeros(n_obj, device=device, dtype=torch.bool)
        used_tracks = set()

        for oi in order.tolist():
            sim_val = float(best_sim[oi].item())
            if sim_val < float(cfg.sim_thresh):
                break
            tj = int(best_idx[oi].item())
            if tj in used_tracks:
                continue
            used_tracks.add(tj)
            assigned_obj[oi] = True

            cnt = counts[tj]
            new_emb = (tracks[tj] * float(cnt) + objs[oi]) / float(cnt + 1)
            tracks[tj] = l2norm(new_emb)
            counts[tj] = cnt + 1

        for oi in range(n_obj):
            if not bool(assigned_obj[oi].item()):
                tracks.append(objs[oi].clone())
                counts.append(1)

        if len(tracks) > int(cfg.max_tracks):
            keep = sorted(range(len(tracks)), key=lambda j: counts[j], reverse=True)[
                : int(cfg.max_tracks)
            ]
            tracks = [tracks[j] for j in keep]
            counts = [counts[j] for j in keep]

    if len(tracks) == 0:
        return torch.empty(0, D, device=device, dtype=z_mkd.dtype)
    return l2norm(torch.stack(tracks, dim=0))


def pack_candidates_with_null(
    tracks_per_sample: List[torch.Tensor],
    null_emb: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable packing:
      - builds per-sample candidate tensors via torch.cat (keeps grad)
      - pads with F.pad (keeps grad)
      - stacks
    """
    if null_emb.ndim != 1:
        raise ValueError(f"null_emb must be (D,), got {null_emb.shape}")

    device = null_emb.device
    dtype = null_emb.dtype

    cand_list: List[torch.Tensor] = []
    mask_list: List[torch.Tensor] = []
    Rmax = 1

    for tr in tracks_per_sample:
        tr = tr.to(device=device, dtype=dtype)
        if tr.ndim == 1:
            tr = tr.unsqueeze(0)  # (1,D) safety

        # Append null at end of this sample's list
        cand_i = torch.cat([tr, null_emb.unsqueeze(0)], dim=0)  # (Ri+1, D)
        mask_i = torch.ones((cand_i.size(0),), device=device, dtype=torch.bool)

        cand_list.append(cand_i)
        mask_list.append(mask_i)
        Rmax = max(Rmax, int(cand_i.size(0)))

    # Pad to Rmax
    cand_pad: List[torch.Tensor] = []
    mask_pad: List[torch.Tensor] = []
    for cand_i, mask_i in zip(cand_list, mask_list):
        pad_len = Rmax - int(cand_i.size(0))
        if pad_len > 0:
            # Pad rows at bottom: (left,right, top,bottom) for 2D is (0,0,0,pad_len)
            cand_i = F.pad(cand_i, (0, 0, 0, pad_len))
            mask_i = torch.cat(
                [mask_i, torch.zeros((pad_len,), device=device, dtype=torch.bool)],
                dim=0
            )
        cand_pad.append(cand_i)
        mask_pad.append(mask_i)

    cand = torch.stack(cand_pad, dim=0)  # (B,Rmax,D)
    mask = torch.stack(mask_pad, dim=0)  # (B,Rmax)
    return cand, mask


def mil_logsumexp_logits(
    text_emb: torch.Tensor,
    cand_emb: torch.Tensor,
    cand_mask: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """
    Compute MIL logits between query texts and candidate bags.

    text_emb:  (B, D) normalized
    cand_emb:  (B, R, D) normalized
    cand_mask: (B, R) boolean
    tau: temperature for log-sum-exp pooling

    Returns:
      logits: (B, B) where logits[i,j] = tau * logsumexp_r ( (t_i dot c_{j,r}) / tau )
    """
    if text_emb.ndim != 2:
        raise ValueError(f"text_emb must be (B,D), got {text_emb.shape}")
    if cand_emb.ndim != 3:
        raise ValueError(f"cand_emb must be (B,R,D), got {cand_emb.shape}")
    if cand_mask.ndim != 2:
        raise ValueError(f"cand_mask must be (B,R), got {cand_mask.shape}")

    B, D = text_emb.shape
    Bb, R, Db = cand_emb.shape
    if Bb != B or Db != D:
        raise ValueError(
            f"Shape mismatch: text {text_emb.shape}, cand {cand_emb.shape}"
        )

    tau_f = float(tau)
    tau_f = max(tau_f, 1e-6)

    cand_flat = cand_emb.reshape(B * R, D)  # (B*R,D)
    sim = text_emb @ cand_flat.t()  # (B, B*R)
    sim = sim.view(B, B, R)  # (B_query, B_bag, R)

    # mask invalid candidates per bag
    bag_mask = cand_mask.unsqueeze(0)  # (1, B_bag, R)
    sim = sim.masked_fill(~bag_mask, float("-inf"))

    logits = tau_f * torch.logsumexp(sim / tau_f, dim=2)  # (B,B)
    return logits


def per_sample_candidate_weights(
    text_emb: torch.Tensor,
    cand_emb: torch.Tensor,
    cand_mask: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """
    For each sample i, produce weights over its own candidates.

    Returns:
      w: (B, R) with sum=1 over valid candidates per row.
    """
    if text_emb.ndim != 2 or cand_emb.ndim != 3 or cand_mask.ndim != 2:
        raise ValueError("Bad shapes for per_sample_candidate_weights")

    B, D = text_emb.shape
    _, R, _ = cand_emb.shape
    tau_f = max(float(tau), 1e-6)

    # sim_i,r = t_i dot c_i,r
    sim = (cand_emb * text_emb.unsqueeze(1)).sum(dim=-1)  # (B,R)
    sim = sim.masked_fill(~cand_mask, float("-inf"))
    w = torch.softmax(sim / tau_f, dim=1)
    w = w.masked_fill(~cand_mask, 0.0)
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return w
