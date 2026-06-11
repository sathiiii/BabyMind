#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Polished temporal-misalignment strip for ONE chosen candidate (publication-ready).

Special panel behavior:
  - Save uncropped image+mask from panel k==2.
  - For panel k==3, reuse saved k==2 uncropped image+mask,
    apply (1) motion blur to the copied IMAGE only,
    apply (2) random in-bounds crop shift,
    then crop 224x224 for BOTH image and mask.
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from multimodal.multimodal_data_module import SamPrepackedIndex


# -----------------------------
# Optional OpenCV (best motion blur)
# -----------------------------
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False


# -----------------------------
# Cropping utilities
# -----------------------------
def _pad_to_at_least(arr: np.ndarray, size: int) -> np.ndarray:
    if arr.ndim == 2:
        H, W = arr.shape
    else:
        H, W = arr.shape[:2]

    if H >= size and W >= size:
        return arr

    pad_y = max(0, size - H)
    pad_x = max(0, size - W)
    top = pad_y // 2
    bot = pad_y - top
    left = pad_x // 2
    right = pad_x - left

    if arr.ndim == 2:
        return np.pad(arr, ((top, bot), (left, right)), mode="constant", constant_values=0)
    return np.pad(arr, ((top, bot), (left, right), (0, 0)), mode="constant", constant_values=0)


def crop_with_box(arr: np.ndarray, x0: int, y0: int, size: int = 224) -> np.ndarray:
    if arr.ndim == 2:
        return arr[y0:y0 + size, x0:x0 + size]
    return arr[y0:y0 + size, x0:x0 + size, :]


def center_crop_np(arr: np.ndarray, size: int = 224) -> np.ndarray:
    arr = _pad_to_at_least(arr, size)
    if arr.ndim == 2:
        H, W = arr.shape
    else:
        H, W = arr.shape[:2]
    y0 = (H - size) // 2
    x0 = (W - size) // 2
    return crop_with_box(arr, x0, y0, size=size)


def shifted_center_crop_box(
    H: int,
    W: int,
    size: int,
    rng: np.random.Generator,
    max_shift_px: Optional[int] = None,
) -> Tuple[int, int]:
    base_y0 = (H - size) // 2
    base_x0 = (W - size) // 2

    max_x0 = W - size
    max_y0 = H - size

    allow_neg_x = base_x0
    allow_pos_x = max_x0 - base_x0
    allow_neg_y = base_y0
    allow_pos_y = max_y0 - base_y0

    if max_shift_px is not None:
        allow_neg_x = min(allow_neg_x, int(max_shift_px))
        allow_pos_x = min(allow_pos_x, int(max_shift_px))
        allow_neg_y = min(allow_neg_y, int(max_shift_px))
        allow_pos_y = min(allow_pos_y, int(max_shift_px))

    dx = int(rng.integers(-allow_neg_x, allow_pos_x + 1)) if (allow_neg_x + allow_pos_x) > 0 else 0
    dy = int(rng.integers(-allow_neg_y, allow_pos_y + 1)) if (allow_neg_y + allow_pos_y) > 0 else 0

    x0 = int(np.clip(base_x0 + dx, 0, max_x0))
    y0 = int(np.clip(base_y0 + dy, 0, max_y0))
    return x0, y0


# -----------------------------
# Motion blur
# -----------------------------
def _make_motion_kernel_cv2(length: int, angle_deg: float) -> np.ndarray:
    """
    length: odd int recommended
    angle_deg: 0 = horizontal, 90 = vertical
    """
    L = int(max(1, length))
    if L % 2 == 0:
        L += 1

    kernel = np.zeros((L, L), dtype=np.float32)
    c = L // 2

    # draw a horizontal line through center
    kernel[c, :] = 1.0

    # rotate around center
    M = cv2.getRotationMatrix2D((c, c), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (L, L), flags=cv2.INTER_LINEAR)

    s = float(kernel.sum())
    if s > 0:
        kernel /= s
    return kernel


def apply_motion_blur(
    img_rgb: np.ndarray,
    rng: np.random.Generator,
    length: int,
    angle_deg: Optional[float] = None,
) -> np.ndarray:
    """
    Apply motion blur to RGB uint8 image.
    Uses cv2 if available; else falls back to horizontal blur.
    """
    L = int(length)
    if L <= 1:
        return img_rgb

    if angle_deg is None:
        angle_deg = float(rng.uniform(0.0, 180.0))

    if _HAS_CV2:
        # cv2 expects BGR
        bgr = img_rgb[..., ::-1].copy()
        kernel = _make_motion_kernel_cv2(L, float(angle_deg))
        out = cv2.filter2D(bgr, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT101)
        rgb = out[..., ::-1]
        return rgb.astype(np.uint8)

    # fallback: simple horizontal average blur of length L
    if L % 2 == 0:
        L += 1
    pad = L // 2
    x = img_rgb.astype(np.float32)
    xpad = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode="reflect")
    out = np.zeros_like(x)
    for t in range(L):
        out += xpad[:, t:t + x.shape[1], :]
    out /= float(L)
    return np.clip(out, 0, 255).astype(np.uint8)


# -----------------------------
# Misc
# -----------------------------
def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def fmt_dt(dt: float) -> str:
    if abs(dt) < 1e-9:
        return "t (paired)"
    return f"{dt:+.1f}s"


def _load_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    if font_path is not None and Path(font_path).is_file():
        return ImageFont.truetype(font_path, size=size)
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if Path(p).is_file():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def draw_rounded_rect(draw: ImageDraw.ImageDraw, xy: Tuple[int, int, int, int], r: int, fill):
    x0, y0, x1, y1 = xy
    r = max(0, min(r, (x1 - x0) // 2, (y1 - y0) // 2))
    if r == 0:
        draw.rectangle(xy, fill=fill)
        return
    draw.rounded_rectangle(xy, radius=r, fill=fill)


# -----------------------------
# SAM resolver
# -----------------------------
class SamResolver:
    def __init__(self, sam_prepacked_dir: Path, cache_size: int = 2048):
        self.root = Path(sam_prepacked_dir)
        self.index = SamPrepackedIndex.load(self.root, concept2idx=None, cache_size=0)
        self.keys = list(self.index.frame_to_file.keys())

        self.concepts = [str(c).lower() for c in self.index.concepts]
        self.name2local = {n: i for i, n in enumerate(self.concepts)}

        base_counts: Dict[str, int] = {}
        suf2_counts: Dict[str, int] = {}
        for k in self.keys:
            base = Path(k).name
            base_counts[base] = base_counts.get(base, 0) + 1
            parts = k.replace("\\", "/").split("/")
            suf2 = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            suf2_counts[suf2] = suf2_counts.get(suf2, 0) + 1

        self.base2key: Dict[str, str] = {}
        self.suf2key: Dict[str, str] = {}
        for k in self.keys:
            base = Path(k).name
            if base_counts[base] == 1:
                self.base2key[base] = k
            parts = k.replace("\\", "/").split("/")
            suf2 = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            if suf2_counts[suf2] == 1:
                self.suf2key[suf2] = k

        self.cache_size = int(cache_size)
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def resolve_key(self, frame_rel: str) -> Optional[str]:
        if frame_rel in self.index.frame_to_file:
            return frame_rel
        fr = frame_rel.replace("\\", "/")
        base = Path(fr).name
        if base in self.base2key:
            return self.base2key[base]
        parts = fr.split("/")
        suf2 = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        if suf2 in self.suf2key:
            return self.suf2key[suf2]
        return None

    def get_masks(self, frame_rel: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        rk = self.resolve_key(frame_rel)
        if rk is None:
            return None

        if rk in self._cache:
            m, c = self._cache[rk]
            return m.copy(), c.copy()

        out = self.index.get_masks_for_relpath(rk)
        if out is None:
            return None

        masks_t, cids_t = out
        masks = masks_t.detach().cpu().numpy().astype(np.float32)
        cids = cids_t.detach().cpu().numpy().astype(np.int64).reshape(-1)

        if self.cache_size > 0:
            self._cache[rk] = (masks.copy(), cids.copy())
            if len(self._cache) > self.cache_size:
                self._cache.pop(next(iter(self._cache)))

        return masks, cids

    def largest_mask_for_concept_name(self, frame_rel: str, concept_name: str) -> Optional[np.ndarray]:
        cname = str(concept_name).lower().strip()
        if cname not in self.name2local:
            return None
        cid_local = int(self.name2local[cname])

        out = self.get_masks(frame_rel)
        if out is None:
            return None
        masks, cids = out
        idxs = np.where(cids == cid_local)[0]
        if idxs.size == 0:
            return None

        best_i = None
        best_area = -1
        for i in idxs.tolist():
            m = (masks[i] > 0.5).astype(np.uint8)
            a = int(m.sum())
            if a > best_area:
                best_area = a
                best_i = i

        if best_i is None or best_area <= 0:
            return None
        return (masks[best_i] > 0.5).astype(np.uint8)


# -----------------------------
# Mask overlay (PIL)
# -----------------------------
def mask_edge(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return m

    p = np.pad(m, 1, mode="constant", constant_values=0)
    nb = (
        p[0:-2, 0:-2] + p[0:-2, 1:-1] + p[0:-2, 2:] +
        p[1:-1, 0:-2] +               0 + p[1:-1, 2:] +
        p[2:,   0:-2] + p[2:,   1:-1] + p[2:,   2:]
    )
    edge = ((m == 1) & (nb < 8)).astype(np.uint8)

    for _ in range(max(0, int(thickness) - 1)):
        ep = np.pad(edge, 1, mode="constant", constant_values=0)
        edge = (
            (ep[0:-2, 0:-2] | ep[0:-2, 1:-1] | ep[0:-2, 2:] |
             ep[1:-1, 0:-2] | ep[1:-1, 1:-1] | ep[1:-1, 2:] |
             ep[2:,   0:-2] | ep[2:,   1:-1] | ep[2:,   2:])
        ).astype(np.uint8)
    return edge


def overlay_mask_red(
    img_rgb: np.ndarray,
    mask01: Optional[np.ndarray],
    alpha: float = 0.40,
    outline: bool = True
) -> Image.Image:
    base = Image.fromarray(img_rgb.astype(np.uint8), mode="RGB")
    if mask01 is None:
        return base

    m = (mask01 > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return base

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ov_np = np.array(overlay, dtype=np.uint8)
    ov_np[..., 0] = 255
    ov_np[..., 3] = (alpha * 255.0 * m).astype(np.uint8)
    overlay = Image.fromarray(ov_np, mode="RGBA")
    out = Image.alpha_composite(base.convert("RGBA"), overlay)

    if outline:
        e = mask_edge(m, thickness=2)
        if int(e.sum()) > 0:
            out_np = np.array(out, dtype=np.uint8)
            out_np[e > 0, 0:3] = 255
            out = Image.fromarray(out_np, mode="RGBA")

    return out.convert("RGB")


# -----------------------------
# Utterance drawing with highlight
# -----------------------------
def split_for_highlight(text: str, concept: str) -> List[Tuple[str, bool]]:
    t = str(text)
    c = re.escape(str(concept).strip())
    if not c:
        return [(t, False)]
    pat = re.compile(rf"\b({c})\b", flags=re.IGNORECASE)

    out: List[Tuple[str, bool]] = []
    last = 0
    for m in pat.finditer(t):
        if m.start() > last:
            out.append((t[last:m.start()], False))
        out.append((t[m.start():m.end()], True))
        last = m.end()
    if last < len(t):
        out.append((t[last:], False))
    if not out:
        out = [(t, False)]
    return out


def draw_utterance_with_chip(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    prefix: str,
    utterance: str,
    concept: str,
    font: ImageFont.FreeTypeFont,
    chip_font: ImageFont.FreeTypeFont,
    color_text=(0, 0, 0),
    chip_bg=(190, 0, 0),
    chip_fg=(255, 255, 255),
) -> None:
    draw.text((x, y), prefix, font=font, fill=color_text)
    xb = draw.textbbox((x, y), prefix, font=font)
    cx = xb[2] + 6

    q1 = '"'
    draw.text((cx, y), q1, font=font, fill=color_text)
    cx = draw.textbbox((cx, y), q1, font=font)[2]

    parts = split_for_highlight(utterance, concept)
    for seg, is_c in parts:
        if not seg:
            continue
        if not is_c:
            draw.text((cx, y), seg, font=font, fill=color_text)
            cx = draw.textbbox((cx, y), seg, font=font)[2]
        else:
            bb = draw.textbbox((cx, y), seg, font=chip_font)
            pad_x = 10
            pad_y = 6
            rect = (bb[0] - pad_x, bb[1] - pad_y, bb[2] + pad_x, bb[3] + pad_y)
            draw_rounded_rect(draw, rect, r=10, fill=chip_bg)
            draw.text((cx, y), seg, font=chip_font, fill=chip_fg)
            cx = rect[2] + 2

    q2 = '"'
    draw.text((cx, y), q2, font=font, fill=color_text)


# -----------------------------
# Offsets: future only
# -----------------------------
def choose_future_offsets(
    paired_i: int,
    n_frames: int,
    n_panels: int,
    fps: float,
    timestamps: Optional[List[float]],
    visible_offsets: Optional[List[int]],
) -> List[int]:
    max_future = max(0, (n_frames - 1) - paired_i)
    base = [0] + [o for o in range(1, max_future + 1)]
    if len(base) == 1:
        return [0]

    offs = base[: max(1, min(int(n_panels), len(base)))]

    best_vis = None
    if visible_offsets:
        vis_pos = sorted({int(o) for o in visible_offsets if int(o) > 0})
        if vis_pos:
            best_vis = vis_pos[0]

    if best_vis is not None and best_vis <= max_future and best_vis not in offs:
        if len(offs) < int(n_panels):
            offs.append(best_vis)
        else:
            offs[-1] = best_vis

    offs = sorted(set(offs))
    while len(offs) < int(n_panels) and len(offs) < len(base):
        cand = max(offs) + 1
        if cand <= max_future:
            offs.append(cand)
        else:
            break

    return offs[: int(n_panels)]


# -----------------------------
# Main rendering
# -----------------------------
def render_polished_strip(
    utterance: str,
    concept: str,
    frames: List[str],
    paired_i: int,
    frames_root: Path,
    sam: SamResolver,
    fps: float,
    timestamps: Optional[List[float]],
    visible_offsets: Optional[List[int]],
    n_panels: int,
    out_png: Path,
    out_pdf: Optional[Path],
    panel_px: int = 224,
    gap_px: int = 24,
    outer_px: int = 28,
    shift_seed: int = 0,
    shift_max_px: Optional[int] = 64,
    motion_blur_len: int = 0,
    motion_blur_angle: Optional[float] = None,
) -> None:
    rng = np.random.default_rng(int(shift_seed))

    font_main = _load_font(None, 40)
    font_chip = _load_font(None, 40)
    font_time = _load_font(None, 44)

    offs = choose_future_offsets(
        paired_i=paired_i,
        n_frames=len(frames),
        n_panels=int(n_panels),
        fps=float(fps),
        timestamps=timestamps,
        visible_offsets=visible_offsets,
    )

    header_h = 96
    time_h = 70

    W = outer_px * 2 + len(offs) * panel_px + (len(offs) - 1) * gap_px
    H = outer_px * 2 + header_h + time_h + panel_px

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw_utterance_with_chip(
        draw=draw,
        x=outer_px,
        y=outer_px,
        prefix="Utterance:",
        utterance=utterance,
        concept=concept,
        font=font_main,
        chip_font=font_chip,
        chip_bg=(185, 0, 0),
        chip_fg=(255, 255, 255),
        color_text=(0, 0, 0),
    )

    y_time = outer_px + header_h
    y_img = y_time + time_h
    x0 = outer_px

    saved_full_img_k2: Optional[np.ndarray] = None
    saved_full_mask_k2: Optional[np.ndarray] = None

    for k, off in enumerate(offs):
        i = paired_i + int(off)
        if i < 0 or i >= len(frames):
            continue

        frame_rel = frames[i]
        img_path = frames_root / frame_rel
        if not img_path.is_file():
            rk = sam.resolve_key(frame_rel)
            if rk is not None and (frames_root / rk).is_file():
                img_path = frames_root / rk

        img_full = np.array(Image.open(img_path).convert("RGB"))
        mask_full = sam.largest_mask_for_concept_name(frame_rel, concept)
        mask_full = mask_full.astype(np.uint8) if mask_full is not None else None

        if k == 2:
            saved_full_img_k2 = img_full.copy()
            saved_full_mask_k2 = mask_full.copy() if mask_full is not None else None

        if k == 3 and saved_full_img_k2 is not None:
            img_full = saved_full_img_k2.copy()
            mask_full = saved_full_mask_k2.copy() if saved_full_mask_k2 is not None else None

            # apply motion blur to the copied IMAGE only
            if int(motion_blur_len) > 1:
                img_full = apply_motion_blur(
                    img_rgb=img_full,
                    rng=rng,
                    length=int(motion_blur_len),
                    angle_deg=motion_blur_angle,
                )

        img_full = _pad_to_at_least(img_full, size=panel_px)
        if mask_full is not None:
            mask_full = _pad_to_at_least(mask_full, size=panel_px)

        if k == 3:
            Hi, Wi = img_full.shape[:2]
            x_crop, y_crop = shifted_center_crop_box(
                H=Hi,
                W=Wi,
                size=panel_px,
                rng=rng,
                max_shift_px=shift_max_px,
            )
            img = crop_with_box(img_full, x_crop, y_crop, size=panel_px)
            m = crop_with_box(mask_full, x_crop, y_crop, size=panel_px) if mask_full is not None else None
        else:
            img = center_crop_np(img_full, size=panel_px)
            m = center_crop_np(mask_full, size=panel_px) if mask_full is not None else None

        pil_panel = overlay_mask_red(img, m, alpha=0.40, outline=True)

        if timestamps is not None and 0 <= paired_i < len(timestamps) and 0 <= i < len(timestamps):
            dt = safe_float(timestamps[i])
            dt0 = safe_float(timestamps[paired_i])
            title = fmt_dt(dt - dt0) if (dt is not None and dt0 is not None) else fmt_dt(off / float(fps))
        else:
            title = fmt_dt(off / float(fps))

        tx = x0 + k * (panel_px + gap_px)
        tbb = draw.textbbox((0, 0), title, font=font_time)
        tw = tbb[2] - tbb[0]
        th = tbb[3] - tbb[1]
        draw.text(
            (tx + (panel_px - tw) // 2, y_time + (time_h - th) // 2),
            title,
            font=font_time,
            fill=(0, 0, 0),
        )

        canvas.paste(pil_panel, (tx, y_img))

        is_paired = (off == 0)
        is_visible = (m is not None and int(np.sum(m)) > 0)

        border_col = (200, 200, 200)
        border_w = 4
        if is_paired or is_visible:
            border_col = (0, 0, 0)
            border_w = 6

        for bw in range(border_w):
            draw.rectangle(
                [tx - bw, y_img - bw, tx + panel_px + bw - 1, y_img + panel_px + bw - 1],
                outline=border_col,
                width=1,
            )

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)

    if out_pdf is not None:
        out_pdf = Path(out_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_pdf, "PDF", resolution=300.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate_json", type=str, required=True)
    ap.add_argument("--out_png", type=str, required=True)
    ap.add_argument("--out_pdf", type=str, default=None)

    ap.add_argument("--frames_root", type=str, default=None)
    ap.add_argument("--sam_prepacked_dir", type=str, default=None)
    ap.add_argument("--concept", type=str, default=None)
    ap.add_argument("--paired_i", type=int, default=None)

    ap.add_argument("--n_panels", type=int, default=5)
    ap.add_argument("--panel_px", type=int, default=224)

    ap.add_argument("--shift_seed", type=int, default=0)
    ap.add_argument("--shift_max_px", type=int, default=64)

    # motion blur (applied only for k==3 copied panel)
    ap.add_argument("--motion_blur_len", type=int, default=7,
                    help="0/1 disables; >=3 enables motion blur length (odd recommended).")
    ap.add_argument("--motion_blur_angle", type=float, default=None,
                    help="If set, fixed angle in degrees; otherwise random in [0,180).")

    args = ap.parse_args()

    cand = json.loads(Path(args.candidate_json).read_text())

    utterance = str(cand.get("utterance", "")).strip()
    concept = str(args.concept if args.concept is not None else cand.get("concept", "")).strip()
    frames = list(cand.get("frames", []))
    paired_i = int(args.paired_i if args.paired_i is not None else cand.get("paired_i", 0))

    fps = float(cand.get("fps", 5.0))
    timestamps = cand.get("timestamps", None)
    timestamps = timestamps if isinstance(timestamps, list) else None
    visible_offsets = cand.get("visible_offsets", None)
    visible_offsets = visible_offsets if isinstance(visible_offsets, list) else None

    frames_root = Path(args.frames_root if args.frames_root is not None else cand.get("frames_root", ""))
    sam_dir = Path(args.sam_prepacked_dir if args.sam_prepacked_dir is not None else cand.get("sam_prepacked_dir", ""))

    if not frames_root.exists():
        raise FileNotFoundError(f"frames_root not found: {frames_root}")
    if not sam_dir.exists():
        raise FileNotFoundError(f"sam_prepacked_dir not found: {sam_dir}")
    if not utterance:
        raise ValueError("Missing utterance in candidate json.")
    if not concept:
        raise ValueError("Missing concept (use --concept to override).")
    if not frames:
        raise ValueError("Missing frames list in candidate json.")
    if paired_i < 0 or paired_i >= len(frames):
        raise ValueError(f"paired_i out of range: {paired_i} (len(frames)={len(frames)})")

    sam = SamResolver(sam_dir, cache_size=2048)

    render_polished_strip(
        utterance=utterance,
        concept=concept,
        frames=frames,
        paired_i=paired_i,
        frames_root=frames_root,
        sam=sam,
        fps=fps,
        timestamps=timestamps,
        visible_offsets=visible_offsets,
        n_panels=int(args.n_panels),
        out_png=Path(args.out_png),
        out_pdf=(Path(args.out_pdf) if args.out_pdf else None),
        panel_px=int(args.panel_px),
        gap_px=24,
        outer_px=28,
        shift_seed=int(args.shift_seed),
        shift_max_px=int(args.shift_max_px) if args.shift_max_px is not None else None,
        motion_blur_len=int(args.motion_blur_len),
        motion_blur_angle=args.motion_blur_angle,
    )

    print(f"[Saved] {args.out_png}")
    if args.out_pdf:
        print(f"[Saved] {args.out_pdf}")
    if int(args.motion_blur_len) > 1 and not _HAS_CV2:
        print("[Note] cv2 not found, used fallback horizontal motion blur.")


if __name__ == "__main__":
    main()
