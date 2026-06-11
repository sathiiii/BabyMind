#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

# Headless-safe matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

# Project imports (match your train.py wiring)
from multimodal.multimodal_data_module import MultiModalDataModule
from multimodal.multimodal_saycam_data_module import MultiModalSAYCamDataModule
from multimodal.coco_captions_data_module import COCOCaptionsDataModule
from multimodal.multimodal import VisionEncoder, TextEncoder, MultiModalModel, LanguageModel
from multimodal.multimodal_lit import MultiModalLitModel

from multimodal.object_mil import l2norm as l2norm_obj, pack_candidates_with_null


# -----------------------------
# Small stats helpers
# -----------------------------
def gini_coefficient(x: np.ndarray) -> float:
    """Gini coefficient for a nonnegative vector x (not necessarily normalized)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    s = float(x.sum())
    if s <= 0.0:
        return 0.0
    x = np.sort(x)  # ascending
    n = x.size
    i = np.arange(1, n + 1, dtype=np.float64)  # 1..n
    g = (2.0 * np.sum(i * x)) / (n * s) - (n + 1.0) / n
    return float(g)


def _pick_first_dataloader(dl):
    """Lightning DMs sometimes return list-of-dataloaders; we want the main one (idx 0)."""
    if isinstance(dl, (list, tuple)):
        return dl[0]
    return dl


# -----------------------------
# Visualization helpers
# -----------------------------
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _frame_to_uint8_hwc(frame_chw: torch.Tensor) -> np.ndarray:
    """
    Convert a CHW tensor to uint8 HWC for visualization.
    If it looks normalized (values outside [0,1]), unnormalize with ImageNet stats.
    """
    x = frame_chw.detach().cpu().float()
    if float(x.min()) < 0.0 or float(x.max()) > 1.0:
        x = x * _IMAGENET_STD + _IMAGENET_MEAN
    x = x.clamp(0.0, 1.0)
    hwc = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return hwc


def _resize_mask_to_image(mask_hw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a mask to image resolution using bilinear interpolation (keeps soft masks)."""
    if mask_hw.shape[0] == out_h and mask_hw.shape[1] == out_w:
        return mask_hw
    m = mask_hw.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0
    m = np.clip(m, 0.0, 1.0)
    pil = Image.fromarray((m * 255.0).astype(np.uint8))
    pil = pil.resize((out_w, out_h), resample=Image.BILINEAR)
    out = (np.asarray(pil).astype(np.float32) / 255.0)
    return out


def _overlay_mask_red(img_hwc_u8: np.ndarray, mask_hw: np.ndarray, alpha: float) -> np.ndarray:
    """Overlay a binary/soft mask in red."""
    img = img_hwc_u8.astype(np.float32) / 255.0
    H, W = img.shape[:2]
    m = _resize_mask_to_image(mask_hw, H, W)

    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)[None, None, :]
    a = float(alpha) * m[..., None]
    out = img * (1.0 - a) + red * a
    out = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    return out


def _draw_patch_box(pil_img: Image.Image, *, py: int, px: int, Hf: int, Wf: int) -> None:
    """
    Draw a green square around a patch center given in feature-map coords (py, px) on (Hf, Wf).
    """
    if Hf <= 0 or Wf <= 0 or py < 0 or px < 0:
        return

    W, H = pil_img.size
    cx = (float(px) + 0.5) / float(Wf) * float(W)
    cy = (float(py) + 0.5) / float(Hf) * float(H)

    half = max(6, int(round(min(W, H) * 0.06)))
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1 = int(round(cx + half))
    y1 = int(round(cy + half))

    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))

    draw = ImageDraw.Draw(pil_img)
    lw = max(2, int(round(min(W, H) * 0.01)))
    draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=lw)


# -----------------------------
# Track frame picking + rendering
# -----------------------------
@torch.no_grad()
def _select_top_frames_for_track(
    *,
    track_emb: torch.Tensor,   # (D,) device
    z_mkd: torch.Tensor,       # (M,K,D) device
    valid_mk: torch.Tensor,    # (M,K) bool device
    num_frames: int,
) -> List[Tuple[int, int]]:
    """
    For each frame m, pick the best candidate k by similarity to track_emb.
    Then select the top `num_frames` frames by that similarity, and sort them chronologically.
    Returns list of (m, k_sel) length exactly num_frames (pads if needed).
    """
    M = int(z_mkd.size(0))
    if M <= 0:
        return []

    sims_mk = torch.einsum("mkd,d->mk", z_mkd, track_emb)  # (M,K)
    sims_mk = sims_mk.masked_fill(~valid_mk, float("-inf"))

    best_sim_m, best_k_m = sims_mk.max(dim=1)  # (M,), (M,)
    best_sim = best_sim_m.clone()
    best_sim[~torch.isfinite(best_sim)] = -1e9

    good = best_sim > -1e8
    if not bool(good.any().item()):
        m0 = 0
        k0 = int(best_k_m[m0].item()) if int(best_k_m.numel()) > 0 else 0
        return [(m0, k0)] * int(num_frames)

    m_good = good.nonzero(as_tuple=False).view(-1)
    k_take = min(int(num_frames), int(m_good.numel()))
    sims_good = best_sim[m_good]
    top_rel = torch.topk(sims_good, k=k_take, largest=True, sorted=True).indices
    m_sel = m_good[top_rel]

    if int(m_sel.numel()) < int(num_frames):
        pad = m_sel[:1].repeat(int(num_frames) - int(m_sel.numel()))
        m_sel = torch.cat([m_sel, pad], dim=0)

    m_sel, _ = torch.sort(m_sel)
    out: List[Tuple[int, int]] = []
    for m in m_sel.tolist():
        k = int(best_k_m[m].item())
        out.append((int(m), int(k)))
    return out


@torch.no_grad()
def render_track_strip(
    *,
    frames_mchw_cpu: torch.Tensor,          # (M,3,H,W) CPU
    track_emb: torch.Tensor,                # (D,) device
    z_mkd: torch.Tensor,                    # (M,K,D) device
    valid_mk: torch.Tensor,                 # (M,K) bool device
    masks_mkhw: Optional[torch.Tensor],     # (M,K,H,W) device or None
    is_patch_mk: Optional[torch.Tensor],    # (M,K) bool device or None
    patch_py_mk: Optional[torch.Tensor],    # (M,K) long device or None
    patch_px_mk: Optional[torch.Tensor],    # (M,K) long device or None
    Hf: int,
    Wf: int,
    overlay_alpha: float,
    num_frames: int = 4,
    pad_px: int = 8,
) -> Image.Image:
    """Render a single track as a 1x4 strip (no black borders)."""
    picks = _select_top_frames_for_track(
        track_emb=track_emb,
        z_mkd=z_mkd,
        valid_mk=valid_mk,
        num_frames=int(num_frames),
    )

    rendered: List[Image.Image] = []
    for (m, k) in picks:
        frame_u8 = _frame_to_uint8_hwc(frames_mchw_cpu[m])
        pil = Image.fromarray(frame_u8)

        is_patch = False
        if is_patch_mk is not None:
            is_patch = bool(is_patch_mk[m, k].item())

        if is_patch:
            if patch_py_mk is not None and patch_px_mk is not None:
                py = int(patch_py_mk[m, k].item())
                px = int(patch_px_mk[m, k].item())
                _draw_patch_box(pil, py=py, px=px, Hf=int(Hf), Wf=int(Wf))
        else:
            if masks_mkhw is not None and torch.is_tensor(masks_mkhw):
                mk = masks_mkhw[m, k].detach().cpu().numpy()
                if mk.size > 0 and float(mk.sum()) > 0.0:
                    over = _overlay_mask_red(np.asarray(pil), mk, alpha=float(overlay_alpha))
                    pil = Image.fromarray(over)

        rendered.append(pil)

    W, H = rendered[0].size
    total_w = int(num_frames) * W + (int(num_frames) - 1) * int(pad_px)
    strip = Image.new("RGB", (total_w, H), (255, 255, 255))

    x = 0
    for im in rendered:
        strip.paste(im, (x, 0))
        x += W + int(pad_px)

    return strip


# -----------------------------
# Mining: keep TOP-N track strips per prototype
# -----------------------------
Candidate = Dict[str, Any]  # {"score": float, "strip": PIL.Image, "emb": np.ndarray}


@torch.no_grad()
def mine_candidates_per_prototype(
    *,
    lit_model: MultiModalLitModel,
    dataloader,
    device: torch.device,
    proto_ids: List[int],
    num_batches: int,
    frames_per_track: int,
    overlay_alpha: float,
    top_tracks_per_proto: int,
) -> Dict[int, List[Candidate]]:
    """
    Mine top-N candidate tracks per prototype (from the scanned batches).
    Returns: proto_id -> list[Candidate] sorted by descending score.
    """

    proto_mem = lit_model.proto_mem
    if proto_mem is None:
        raise RuntimeError("proto_mem is None (prototype memory disabled).")

    # per-proto min-heaps: store (score, tie_breaker, Candidate)
    heaps: Dict[int, List[Tuple[float, int, Candidate]]] = {int(k): [] for k in proto_ids}
    tie = 0

    dl_iter = iter(dataloader)
    for b_idx in range(int(num_batches)):
        try:
            batch = next(dl_iter)
        except StopIteration:
            break

        x, y, y_len, raw_y, meta = lit_model._split_batch(batch)
        x = x.to(device)

        meta_d = meta if isinstance(meta, dict) else None
        x_bag, sam_mask, sam_concept = lit_model._get_bag_tensors(x, meta_d)
        B, M, _, H, W = x_bag.shape

        # Candidate embeddings and masks
        sam_count = meta_d.get("sam_mask_count", None) if isinstance(meta_d, dict) else None
        z_bmkd, valid_bmk, masks_bmkhw, _sam_gid_bmk, cand_meta = lit_model._compute_frame_candidates(
            x_bag=x_bag,
            sam_mask=sam_mask,
            sam_count=sam_count,
            sam_concept=sam_concept,
        )

        # Tracks
        tracks_per_sample = lit_model._build_tracks(z_bmkd, valid_bmk)

        # Pack with null track
        null_emb = l2norm_obj(lit_model.null_obj.to(device=device, dtype=torch.float32))
        track_emb, cand_mask = pack_candidates_with_null(tracks_per_sample, null_emb)
        track_emb = l2norm_obj(track_emb.float())  # (B,R,D)
        cand_mask = cand_mask.to(device)

        B2, R, D = track_emb.shape
        if B2 == 0 or R == 0:
            continue

        # Prototype assignment for each track
        q_track = proto_mem.soft_assign(track_emb.view(B * R, D)).view(B, R, -1)  # (B,R,Kp)

        # Exclude null track per sample (null = last valid entry)
        counts = cand_mask.long().sum(dim=1)  # includes null
        null_idx = (counts - 1).clamp(min=0)  # (B,)
        non_null_mask = cand_mask.clone()
        ar = torch.arange(B, device=device)
        non_null_mask[ar, null_idx] = False

        # Overlay meta
        Hf = int(cand_meta.get("Hf", 0)) if isinstance(cand_meta, dict) else 0
        Wf = int(cand_meta.get("Wf", 0)) if isinstance(cand_meta, dict) else 0
        is_patch_bmk = cand_meta.get("is_patch", None) if isinstance(cand_meta, dict) else None
        patch_py_bmk = cand_meta.get("patch_py", None) if isinstance(cand_meta, dict) else None
        patch_px_bmk = cand_meta.get("patch_px", None) if isinstance(cand_meta, dict) else None

        # Cache strips per (i,r) so if one track improves multiple protos we render once
        strip_cache: Dict[Tuple[int, int], Tuple[Image.Image, np.ndarray]] = {}

        for i in range(B):
            frames_cpu = x_bag[i].detach().cpu()  # (M,3,H,W)

            z_mkd = z_bmkd[i]
            v_mk = valid_bmk[i]
            masks_mkhw = masks_bmkhw[i] if (masks_bmkhw is not None and torch.is_tensor(masks_bmkhw)) else None
            is_patch_mk = is_patch_bmk[i] if (is_patch_bmk is not None and torch.is_tensor(is_patch_bmk)) else None
            py_mk = patch_py_bmk[i] if (patch_py_bmk is not None and torch.is_tensor(patch_py_bmk)) else None
            px_mk = patch_px_bmk[i] if (patch_px_bmk is not None and torch.is_tensor(patch_px_bmk)) else None

            for r in range(R):
                if not bool(non_null_mask[i, r].item()):
                    continue

                # Determine which prototypes this track could improve (quick reject by heap worst)
                needs: List[Tuple[int, float]] = []
                for k in proto_ids:
                    k = int(k)
                    score = float(q_track[i, r, k].item())
                    heap = heaps[k]
                    if len(heap) < int(top_tracks_per_proto) or score > float(heap[0][0]):
                        needs.append((k, score))

                if not needs:
                    continue

                key = (i, r)
                if key not in strip_cache:
                    tr = track_emb[i, r]  # (D,) on device
                    strip = render_track_strip(
                        frames_mchw_cpu=frames_cpu,
                        track_emb=tr,
                        z_mkd=z_mkd,
                        valid_mk=v_mk,
                        masks_mkhw=masks_mkhw,
                        is_patch_mk=is_patch_mk,
                        patch_py_mk=py_mk,
                        patch_px_mk=px_mk,
                        Hf=Hf,
                        Wf=Wf,
                        overlay_alpha=float(overlay_alpha),
                        num_frames=int(frames_per_track),
                        pad_px=8,
                    )
                    emb_cpu = tr.detach().cpu().float().numpy()
                    # normalize just in case
                    n = float(np.linalg.norm(emb_cpu) + 1e-12)
                    emb_cpu = emb_cpu / n
                    strip_cache[key] = (strip, emb_cpu)

                strip, emb_cpu = strip_cache[key]

                for k, score in needs:
                    cand: Candidate = {"score": float(score), "strip": strip, "emb": emb_cpu}
                    heapq.heappush(heaps[int(k)], (float(score), tie, cand))
                    tie += 1
                    if len(heaps[int(k)]) > int(top_tracks_per_proto):
                        heapq.heappop(heaps[int(k)])

    # Convert heaps to sorted lists
    out: Dict[int, List[Candidate]] = {}
    for k in proto_ids:
        items = sorted(heaps[int(k)], key=lambda t: float(t[0]), reverse=True)
        out[int(k)] = [cand for (_s, _tie, cand) in items]
    return out


# -----------------------------
# Diversity selection
# -----------------------------
def _max_cos_sim(emb: np.ndarray, selected: List[np.ndarray]) -> float:
    if not selected:
        return 0.0
    sims = [float(np.dot(emb, e)) for e in selected]
    return float(max(sims))


def select_diverse_tracks(
    *,
    proto_rank_by_usage: np.ndarray,
    candidates: Dict[int, List[Candidate]],
    num_rows: int,
    max_sim: float,
    sim_penalty: float,
) -> List[Candidate]:
    """
    Greedy selection:
    - iterate prototypes in usage order
    - for each prototype, pick the candidate track that best trades off score vs similarity
    - skip prototypes whose candidates are all too-similar (under current threshold)
    - relax threshold if needed to fill `num_rows`
    """
    selected: List[Candidate] = []
    selected_embs: List[np.ndarray] = []
    used_protos: set[int] = set()

    # Multi-pass with relaxed similarity thresholds to ensure we fill rows
    thresholds = [float(max_sim), min(0.98, float(max_sim) + 0.03), 1.0]

    for thr in thresholds:
        for k in proto_rank_by_usage.tolist():
            k = int(k)
            if k in used_protos:
                continue
            cand_list = candidates.get(k, [])
            if not cand_list:
                continue

            best: Optional[Candidate] = None
            best_util = -1e18

            for cand in cand_list:
                emb = cand["emb"]
                sim = _max_cos_sim(emb, selected_embs)
                if sim > thr:
                    continue
                util = float(cand["score"]) - float(sim_penalty) * float(sim)
                if util > best_util:
                    best_util = util
                    best = cand

            if best is None:
                continue

            selected.append(best)
            selected_embs.append(best["emb"])
            used_protos.add(k)

            if len(selected) >= int(num_rows):
                return selected

    # If still not enough rows, fill ignoring similarity (pick best remaining by score)
    if len(selected) < int(num_rows):
        remaining: List[Candidate] = []
        for k in proto_rank_by_usage.tolist():
            k = int(k)
            if k in used_protos:
                continue
            for cand in candidates.get(k, []):
                remaining.append(cand)
        remaining.sort(key=lambda c: float(c["score"]), reverse=True)
        for cand in remaining:
            selected.append(cand)
            if len(selected) >= int(num_rows):
                break

    return selected[: int(num_rows)]


# -----------------------------
# Argparse
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, choices=["saycam", "coco"], default="saycam")
    parser.add_argument("--exp_name", type=str, default="multimodal_test")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=None,
        help="Path to a Lightning checkpoint (.ckpt). If omitted, uses checkpoints/<exp_name>/last.ckpt",
    )

    parser.add_argument("--out_pdf", type=Path, default=Path("figures/prototype_diagnostics.pdf"))
    parser.add_argument("--out_png", type=Path, default=None)

    # Use train by default (best chance of multi-frame windows)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="train")
    parser.add_argument("--num_batches", type=int, default=50)

    parser.add_argument("--num_protos_viz", type=int, default=4)
    parser.add_argument("--frames_per_track", type=int, default=4)
    parser.add_argument("--overlay_alpha", type=float, default=0.45)

    # NEW: mine more + select diverse tracks
    parser.add_argument("--candidate_pool", type=int, default=32,
                        help="Mine candidates from the top-N most-used prototypes (larger => more diversity options).")
    parser.add_argument("--top_tracks_per_proto", type=int, default=5,
                        help="How many candidate tracks to keep per prototype for diversity selection.")
    parser.add_argument("--diversity_max_sim", type=float, default=0.92,
                        help="Max cosine similarity allowed between selected track embeddings (lower => more different).")
    parser.add_argument("--diversity_penalty", type=float, default=0.35,
                        help="Penalty weight for similarity when choosing tracks within a prototype.")

    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")

    # Include the same arg groups as train.py so your dataset/model config is available.
    data_group = parser.add_argument_group("Data Args")
    MultiModalDataModule.add_to_argparse(data_group)
    MultiModalSAYCamDataModule.add_additional_to_argparse(data_group)
    COCOCaptionsDataModule.add_additional_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    VisionEncoder.add_to_argparse(model_group)
    TextEncoder.add_to_argparse(model_group)
    MultiModalModel.add_to_argparse(model_group)
    LanguageModel.add_to_argparse(model_group)

    lit_group = parser.add_argument_group("LitModel Args")
    MultiModalLitModel.add_to_argparse(lit_group)

    return parser


def _resolve_dataloader(data, split: str):
    if split == "train":
        return _pick_first_dataloader(data.train_dataloader())
    if split == "val":
        return _pick_first_dataloader(data.val_dataloader())
    return _pick_first_dataloader(data.test_dataloader())


def main() -> None:
    os.environ["PYTHONHASHSEED"] = "0"

    parser = build_parser()
    args = parser.parse_args()

    # Resolve checkpoint path
    if args.ckpt_path is None:
        ckpt_dir = Path("checkpoints") / args.exp_name
        args.ckpt_path = ckpt_dir / "last.ckpt"
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Device
    device = torch.device("cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda")

    # Seed
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    # Data module
    DataModuleClass = {"saycam": MultiModalSAYCamDataModule, "coco": COCOCaptionsDataModule}[args.dataset]
    data = DataModuleClass(args)

    if args.split in ("train", "val"):
        data.setup("fit")
    else:
        data.setup("test")

    # Build model (matches train.py wiring)
    vocab = data.read_vocab()
    vision_encoder = VisionEncoder(args=args)
    text_encoder = TextEncoder(vocab, image_feature_map_dim=vision_encoder.last_cnn_out_dim, args=args)
    lit_model = MultiModalLitModel(vision_encoder, text_encoder, args)

    # Load checkpoint weights
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    incomp = lit_model.load_state_dict(sd, strict=False)
    missing = getattr(incomp, "missing_keys", [])
    unexpected = getattr(incomp, "unexpected_keys", [])
    if missing:
        print(f"[proto_diag] WARNING: missing keys ({len(missing)}). Showing up to 10:\n  {missing[:10]}")
    if unexpected:
        print(f"[proto_diag] WARNING: unexpected keys ({len(unexpected)}). Showing up to 10:\n  {unexpected[:10]}")

    lit_model.to(device)
    lit_model.eval()

    if lit_model.proto_mem is None:
        raise RuntimeError("proto_enable=False or proto_mem missing; cannot build prototype diagnostics.")

    # -------------------------
    # Left panel: usage histogram
    # -------------------------
    ema_sz = lit_model.proto_mem.ema_cluster_size.detach().cpu().numpy().astype(np.float64)
    Kp = int(ema_sz.size)
    if float(ema_sz.sum()) <= 0.0:
        ema_sz = np.ones((Kp,), dtype=np.float64)

    usage = ema_sz / float(ema_sz.sum())
    eff_k = float(lit_model.proto_mem.usage_eff_k().detach().cpu().item())
    gini = gini_coefficient(ema_sz)

    proto_rank = np.argsort(-usage)  # descending by usage

    # Candidate prototypes to mine from (top usage prototypes)
    pool = int(max(args.candidate_pool, args.num_protos_viz * 8))
    pool = int(min(Kp, pool))
    proto_pool = [int(i) for i in proto_rank[:pool]]

    # -------------------------
    # Dataloader for mining
    # -------------------------
    dl = _resolve_dataloader(data, str(args.split))

    # Fallback if split yields single frames (x is 4D)
    try:
        batch0 = next(iter(dl))
        x0, *_rest = lit_model._split_batch(batch0)
        if getattr(x0, "ndim", 0) == 4 and str(args.split) != "train":
            print(f"[proto_diag] NOTE: split='{args.split}' yields single frames (x is 4D). Falling back to train_dataloader.")
            dl = _resolve_dataloader(data, "train")
    except Exception:
        pass

    # -------------------------
    # Mine top candidates per prototype
    # -------------------------
    cand = mine_candidates_per_prototype(
        lit_model=lit_model,
        dataloader=dl,
        device=device,
        proto_ids=proto_pool,
        num_batches=int(args.num_batches),
        frames_per_track=int(args.frames_per_track),
        overlay_alpha=float(args.overlay_alpha),
        top_tracks_per_proto=int(args.top_tracks_per_proto),
    )

    # -------------------------
    # Select diverse tracks (this is the key change)
    # -------------------------
    selected = select_diverse_tracks(
        proto_rank_by_usage=proto_rank,
        candidates=cand,
        num_rows=int(args.num_protos_viz),
        max_sim=float(args.diversity_max_sim),
        sim_penalty=float(args.diversity_penalty),
    )

    # -------------------------
    # Render figure
    # -------------------------
    usage_sorted = usage[proto_rank]  # descending

    P = int(args.num_protos_viz)
    fig_w = 10.5
    fig_h = max(4.0, 1.55 * P)

    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 2.55], wspace=0.10)

    # Left: histogram
    ax0 = fig.add_subplot(outer[0])
    ax0.bar(np.arange(Kp), usage_sorted)
    ax0.set_title("Prototype usage")
    ax0.set_xlabel("prototype (sorted)")
    ax0.set_ylabel("fraction")

    txt = f"Effective K = {eff_k:.2f}/{Kp}\nGini = {gini:.3f}"
    ax0.text(
        0.98, 0.98, txt,
        transform=ax0.transAxes,
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, linewidth=0.9),
    )

    # Right: one strip per row, no labels
    sub = outer[1].subgridspec(P, 1, hspace=0.22)

    for i in range(P):
        ax = fig.add_subplot(sub[i, 0])
        ax.set_axis_off()

        if i < len(selected):
            pil_img = selected[i]["strip"]
            ax.imshow(np.asarray(pil_img))
            ax.set_aspect("auto")
        else:
            ax.text(0.5, 0.5, "no example found", ha="center", va="center", fontsize=11)

    # Save outputs
    out_pdf: Path = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    out_png: Path
    if args.out_png is None:
        out_png = out_pdf.with_suffix(".png")
    else:
        out_png = Path(args.out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)

    print(f"[proto_diag] wrote: {out_pdf}")
    print(f"[proto_diag] wrote: {out_png}")
    print("[proto_diag] diversity settings:",
          f"candidate_pool={pool}, top_tracks_per_proto={args.top_tracks_per_proto}, "
          f"max_sim={args.diversity_max_sim}, penalty={args.diversity_penalty}")


if __name__ == "__main__":
    main()
