from __future__ import annotations

"""Visualization utilities.

This file is intentionally free of training logic.

It provides:
  * Image tensor to uint8 conversion (ImageNet normalization assumed by default).
  * Simple overlays (binary mask, heatmap) for interpretability.
  * Track grid rendering for object-file tracking debugging.
  * Prototype-to-text decoding helpers (top-k nearest text labels).

All functions are safe to call under DDP as long as only rank0 persists outputs.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def chw_tensor_to_uint8(
    x_chw: "torch.Tensor",
    *,
    mean: Tuple[float, float, float] = _IMAGENET_MEAN,
    std: Tuple[float, float, float] = _IMAGENET_STD,
) -> np.ndarray:
    """Convert a normalized CHW tensor to an HxWx3 uint8 numpy image."""
    if torch is None:
        raise RuntimeError("torch is required for chw_tensor_to_uint8")

    if x_chw.ndim != 3 or int(x_chw.size(0)) != 3:
        raise ValueError(f"Expected x_chw (3,H,W), got {tuple(x_chw.shape)}")

    x = x_chw.detach().float().cpu()
    mean_t = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)
    x = (x * std_t) + mean_t
    x = x.clamp(0.0, 1.0)
    img = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return img


def overlay_heatmap_red(
    img_uint8: np.ndarray,
    heat_hw: Any,
    *,
    alpha: float = 0.45,
    gamma: float = 2.0,
    thr: float = 0.10,
) -> np.ndarray:
    """Overlay a heatmap (0..1) on an uint8 image as red tint."""
    if heat_hw is None:
        return img_uint8

    if torch is not None and isinstance(heat_hw, torch.Tensor):
        h = heat_hw.detach().float().cpu().numpy()
    else:
        h = np.asarray(heat_hw)

    h = np.squeeze(h)
    if h.ndim != 2:
        raise ValueError(f"heat_hw must be 2D after squeeze, got {h.shape}")

    h = h.astype(np.float32)
    h = np.clip(h, 0.0, 1.0)

    # Contrast shaping
    if gamma != 1.0:
        h = np.power(h, gamma)
    if thr > 0.0:
        h = np.clip((h - thr) / (1.0 - thr + 1e-6), 0.0, 1.0)

    H, W = img_uint8.shape[0], img_uint8.shape[1]
    if h.shape != (H, W):
        hm = Image.fromarray((h * 255.0).astype(np.uint8)).resize((W, H), resample=Image.BILINEAR)
        h = (np.array(hm).astype(np.float32) / 255.0).clip(0.0, 1.0)

    a = float(alpha)
    a_map = (a * h)[..., None]

    out = img_uint8.astype(np.float32).copy()
    out[..., 0] = out[..., 0] * (1.0 - a_map[..., 0]) + 255.0 * a_map[..., 0]
    return out.clip(0, 255).astype(np.uint8)


def _binary_mask_boundary(m_bool: np.ndarray, *, width: int = 2) -> np.ndarray:
    """Return a boolean boundary map for a binary mask.

    Uses morphological gradient (dilate - erode) via PIL filters.
    """
    w = int(max(width, 1))
    m_u8 = (m_bool.astype(np.uint8) * 255)
    pil = Image.fromarray(m_u8, mode="L")

    # PIL's Min/MaxFilter sizes must be odd.
    k = 2 * w + 1
    dil = pil.filter(ImageFilter.MaxFilter(size=k))
    ero = pil.filter(ImageFilter.MinFilter(size=k))

    dil_a = np.array(dil, dtype=np.int16)
    ero_a = np.array(ero, dtype=np.int16)
    grad = (dil_a - ero_a) > 0
    return grad


def overlay_binary_mask_red(
    img_uint8: np.ndarray,
    mask_hw: Any,
    *,
    alpha: float = 0.45,
    border: bool = True,
    border_width: int = 2,
    border_rgb: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Overlay a binary mask on an uint8 image as a red region (with an optional border)."""
    if mask_hw is None:
        return img_uint8

    if torch is not None and isinstance(mask_hw, torch.Tensor):
        m = mask_hw.detach().cpu().numpy()
    else:
        m = np.asarray(mask_hw)

    m = np.squeeze(m)
    if m.ndim != 2:
        raise ValueError(f"mask_hw must be 2D after squeeze, got {m.shape}")

    m = (m > 0).astype(np.float32)

    H, W = img_uint8.shape[0], img_uint8.shape[1]
    if m.shape != (H, W):
        mm = Image.fromarray((m * 255.0).astype(np.uint8)).resize((W, H), resample=Image.NEAREST)
        m = (np.array(mm).astype(np.float32) / 255.0) > 0.5
        m = m.astype(np.float32)

    m3 = np.repeat(m[:, :, None], 3, axis=2)

    color = np.zeros_like(img_uint8, dtype=np.float32)
    color[:, :, 0] = 255.0

    out = img_uint8.astype(np.float32) * (1.0 - float(alpha) * m3) + color * (float(alpha) * m3)
    out = out.clip(0, 255).astype(np.uint8)

    if bool(border):
        b = _binary_mask_boundary(m > 0.5, width=int(border_width))
        if b.shape == (H, W):
            out[b] = np.array(border_rgb, dtype=np.uint8)

    return out


def draw_patch_box(
    img_uint8: np.ndarray,
    *,
    py: int,
    px: int,
    Hf: int,
    Wf: int,
    width: int = 3,
    rgb: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw a feature-map cell bounding box (py,px) over an image."""
    pil = Image.fromarray(img_uint8)
    draw = ImageDraw.Draw(pil)

    H, W = img_uint8.shape[0], img_uint8.shape[1]
    if Hf <= 0 or Wf <= 0:
        return img_uint8

    y0 = int(round(py * (H / float(Hf))))
    y1 = int(round((py + 1) * (H / float(Hf))))
    x0 = int(round(px * (W / float(Wf))))
    x1 = int(round((px + 1) * (W / float(Wf))))

    draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=rgb, width=int(width))
    return np.array(pil)


# -----------------------------------------------------------------------------
# Text drawing
# -----------------------------------------------------------------------------


def _get_font(size: int) -> ImageFont.ImageFont:
    size = int(size)
    font = None
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            font = ImageFont.truetype(p, size=size)
            break
        except Exception:
            font = None

    if font is None:
        font = ImageFont.load_default()
    return font


def draw_text_with_outline(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int] = (255, 255, 255),
    outline: Tuple[int, int, int] = (0, 0, 0),
    outline_width: int = 2,
) -> None:
    x, y = int(xy[0]), int(xy[1])
    ow = int(outline_width)
    if ow > 0:
        for dx in range(-ow, ow + 1):
            for dy in range(-ow, ow + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=fill)


def wrap_text_simple(text: str, *, max_chars: int = 90, max_lines: int = 3) -> List[str]:
    """Simple and deterministic caption wrapping."""
    txt = (text or "").replace("\n", " ").strip()
    if not txt:
        return [""]

    words = txt.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        if len(test) <= int(max_chars):
            cur = test
            continue
        if cur:
            lines.append(cur)
        cur = w
        if len(lines) >= int(max_lines):
            break
    if cur and len(lines) < int(max_lines):
        lines.append(cur)

    return lines


# -----------------------------------------------------------------------------
# Track grid rendering
# -----------------------------------------------------------------------------


@dataclass
class TrackFrameVis:
    """Per-frame visualization payload for a chosen track."""

    kind: str  # "sam" | "patch" | "none"
    conf: float
    sim: float

    # SAM
    mask: Optional[Any] = None

    # PATCH
    py: Optional[int] = None
    px: Optional[int] = None
    Hf: Optional[int] = None
    Wf: Optional[int] = None
    heatmap: Optional[Any] = None


def render_track_grid(
    *,
    frames_mchw: "torch.Tensor",
    per_frame: Sequence[TrackFrameVis],
    caption: str,
    title: str,
    alpha: float = 0.45,
    font_tile: int = 18,
    font_title: int = 22,
    font_caption: int = 18,
    caption_wrap: int = 90,
    heat_gamma: float = 2.0,
    heat_thr: float = 0.10,
) -> Image.Image:
    """Render a row of frames with overlays and a header."""
    if torch is None:
        raise RuntimeError("torch is required for render_track_grid")

    if frames_mchw.ndim != 4:
        raise ValueError(f"Expected frames_mchw (M,3,H,W), got {tuple(frames_mchw.shape)}")

    M = int(frames_mchw.size(0))
    if len(per_frame) != M:
        raise ValueError(f"per_frame length {len(per_frame)} must match M={M}")

    ftile = _get_font(int(font_tile))
    ftitle = _get_font(int(font_title))
    fcap = _get_font(int(font_caption))

    imgs: List[Image.Image] = []

    for m in range(M):
        img = chw_tensor_to_uint8(frames_mchw[m])
        info = per_frame[m]

        # Overlay SAM mask (if any) regardless of kind.
        if info.mask is not None:
            img = overlay_binary_mask_red(img, info.mask, alpha=float(alpha), border=True, border_width=2)

        # Overlay heatmap (if any).
        if info.heatmap is not None:
            img = overlay_heatmap_red(
                img,
                info.heatmap,
                alpha=float(alpha),
                gamma=float(heat_gamma),
                thr=float(heat_thr),
            )

        # Draw patch box (if provided).
        if info.py is not None and info.px is not None and info.Hf is not None and info.Wf is not None:
            img = draw_patch_box(img, py=int(info.py), px=int(info.px), Hf=int(info.Hf), Wf=int(info.Wf))

        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        txt = f"t={m} {info.kind} conf={info.conf:.2f} sim={info.sim:.2f}"
        draw_text_with_outline(d, (6, 6), txt, font=ftile, outline_width=2)
        imgs.append(pil)

    W = imgs[0].size[0]
    H = imgs[0].size[1]
    pad = 6

    total_w = M * W + (M - 1) * pad
    header_pad = 10

    # Header height
    title_h = ftitle.getbbox("Ag")[3] - ftitle.getbbox("Ag")[1]
    cap_h = fcap.getbbox("Ag")[3] - fcap.getbbox("Ag")[1]

    lines = wrap_text_simple(caption, max_chars=int(caption_wrap), max_lines=3)
    header_h = header_pad + title_h + 6 + len(lines) * (cap_h + 4) + header_pad

    canvas = Image.new("RGB", (total_w, H + header_h), (0, 0, 0))
    d0 = ImageDraw.Draw(canvas)

    # Title
    draw_text_with_outline(d0, (10, 8), title, font=ftitle, outline_width=2)

    # Caption
    y0 = 8 + title_h + 6
    for li in lines:
        draw_text_with_outline(d0, (10, y0), li, font=fcap, outline_width=2)
        y0 += cap_h + 4

    # Frames
    x0 = 0
    for pil in imgs:
        canvas.paste(pil, (x0, header_h))
        x0 += W + pad

    return canvas


# -----------------------------------------------------------------------------
# Backwards-compatible wrapper
# -----------------------------------------------------------------------------


def make_track_grid(
    *,
    frames_mchw: "torch.Tensor",
    per_frame: Sequence[Any],
    title: str,
    caption: str,
    alpha: float = 0.45,
    font_tile: int = 18,
    font_title: int = 22,
    font_caption: int = 18,
    caption_wrap: int = 90,
    heat_gamma: float = 2.0,
    heat_thr: float = 0.10,
) -> Image.Image:
    """Compatibility wrapper used by the Lightning module.

    The training code historically called a `make_track_grid()` function and
    passed a list of *dicts* (not `TrackFrameVis`).

    This wrapper accepts either:
      * `TrackFrameVis` objects, or
      * dictionaries with keys like {kind, conf, sim, mask, py, px, Hf/Wf, heatmap}

    and forwards to `render_track_grid()`.
    """

    if torch is None:
        raise RuntimeError("torch is required for make_track_grid")

    per_frame_vis: List[TrackFrameVis] = []
    for info in per_frame:
        if isinstance(info, TrackFrameVis):
            per_frame_vis.append(info)
            continue

        if isinstance(info, dict):
            kind = str(info.get("kind", "none")).lower()
            # Some callers used "mask" instead of "sam".
            if kind == "mask":
                kind = "sam"

            conf = float(info.get("conf", info.get("confidence", 0.0)))
            sim = float(info.get("sim", info.get("similarity", 0.0)))

            # Support both Hf/Wf and legacy H4/W4 naming.
            Hf = info.get("Hf", info.get("H4", None))
            Wf = info.get("Wf", info.get("W4", None))

            per_frame_vis.append(
                TrackFrameVis(
                    kind=kind,
                    conf=conf,
                    sim=sim,
                    mask=info.get("mask", None),
                    py=info.get("py", None),
                    px=info.get("px", None),
                    Hf=Hf,
                    Wf=Wf,
                    heatmap=info.get("heatmap", None),
                )
            )
            continue

        # Unknown type
        per_frame_vis.append(TrackFrameVis(kind="none", conf=0.0, sim=0.0))

    return render_track_grid(
        frames_mchw=frames_mchw,
        per_frame=per_frame_vis,
        caption=caption,
        title=title,
        alpha=float(alpha),
        font_tile=int(font_tile),
        font_title=int(font_title),
        font_caption=int(font_caption),
        caption_wrap=int(caption_wrap),
        heat_gamma=float(heat_gamma),
        heat_thr=float(heat_thr),
    )


# -----------------------------------------------------------------------------
# Prototype decoding helpers
# -----------------------------------------------------------------------------


def topk_texts_for_prototypes(
    prototypes_kd: "torch.Tensor",
    text_emb_nd: "torch.Tensor",
    texts: Sequence[str],
    *,
    topk: int = 5,
) -> List[List[Tuple[str, float]]]:
    """Return top-k nearest text labels for each prototype.

    Inputs are expected to be L2-normalized.
    """
    if torch is None:
        raise RuntimeError("torch is required for topk_texts_for_prototypes")

    if prototypes_kd.ndim != 2:
        raise ValueError(f"prototypes_kd must be (K,D), got {tuple(prototypes_kd.shape)}")
    if text_emb_nd.ndim != 2:
        raise ValueError(f"text_emb_nd must be (N,D), got {tuple(text_emb_nd.shape)}")

    if len(texts) != int(text_emb_nd.size(0)):
        raise ValueError("texts length must match text_emb_nd rows")

    sims = prototypes_kd.float() @ text_emb_nd.float().t()  # (K,N)
    k = min(int(topk), int(sims.size(1)))
    vals, idx = torch.topk(sims, k=k, dim=1)

    out: List[List[Tuple[str, float]]] = []
    for i in range(int(sims.size(0))):
        row: List[Tuple[str, float]] = []
        for j in range(k):
            jj = int(idx[i, j].item())
            row.append((str(texts[jj]), float(vals[i, j].item())))
        out.append(row)
    return out
