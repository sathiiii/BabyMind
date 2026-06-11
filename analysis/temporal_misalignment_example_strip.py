#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temporal misalignment qualitative strip (save top-K candidates).

Changes vs your current version:
  - NO bbox cropping. Always show a 224x224 center crop (image and mask).
  - Mask overlay is solid RED (RGBA), not a colormap.
  - Layout uses explicit subplots_adjust (no tight_layout). Cleaner spacing.

Outputs:
  outdir/candidates/
    rankXX_concept_frame.png
    rankXX_concept_frame.pdf
    rankXX_concept_frame.json
    candidates_summary.json
"""

import argparse
import json
import math
import re
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from multimodal.multimodal_data_module import SamPrepackedIndex


DEFAULT_22 = [
    "ball", "basket", "car", "cat", "chair", "computer", "crib", "door", "floor", "foot",
    "ground", "hand", "kitchen", "paper", "puzzle", "road", "room", "sand", "stairs",
    "table", "toy", "window",
]

DEFAULT_AVOID = {"sand", "floor", "ground", "road", "room", "kitchen"}


def set_pub_style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def tokenize(text: str) -> List[str]:
    return [w for w in re.split(r"[^a-zA-Z]+", text.lower()) if w]


def slug(s: str, max_len: int = 80) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s or "x"


def load_concepts(concept_list_file: Optional[str]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    if concept_list_file is None:
        names = DEFAULT_22
        c2i = {n: i for i, n in enumerate(names)}
        i2c = {i: n for n, i in c2i.items()}
        return names, c2i, i2c

    p = Path(concept_list_file)
    raw = json.loads(p.read_text())
    if isinstance(raw, list):
        names = [str(x).lower() for x in raw]
        c2i = {n: i for i, n in enumerate(names)}
    elif isinstance(raw, dict):
        if all(isinstance(v, int) for v in raw.values()):
            c2i = {str(k).lower(): int(v) for k, v in raw.items()}
            names = [None] * (max(c2i.values()) + 1)
            for k, v in c2i.items():
                names[v] = k
            names = [n for n in names if n is not None]
        else:
            names = [str(k).lower() for k in raw.keys()]
            c2i = {n: i for i, n in enumerate(names)}
    else:
        raise ValueError("Unsupported concept_list_file format (expected list or dict).")

    i2c = {i: n for n, i in c2i.items()}
    return names, c2i, i2c


def center_crop_np(arr: np.ndarray, size: int = 224) -> np.ndarray:
    """
    Center crop for:
      - image: (H,W,3)
      - mask : (H,W)
    If arr is already 224x224, this is a no-op.
    """
    if arr.ndim == 2:
        H, W = arr.shape
    else:
        H, W = arr.shape[:2]

    if H < size or W < size:
        # If something is smaller than 224 (unlikely in your pipeline),
        # fall back to a centered pad then crop.
        pad_y = max(0, size - H)
        pad_x = max(0, size - W)
        top = pad_y // 2
        bot = pad_y - top
        left = pad_x // 2
        right = pad_x - left
        if arr.ndim == 2:
            arr = np.pad(arr, ((top, bot), (left, right)), mode="constant", constant_values=0)
        else:
            arr = np.pad(arr, ((top, bot), (left, right), (0, 0)), mode="constant", constant_values=0)
        if arr.ndim == 2:
            H, W = arr.shape
        else:
            H, W = arr.shape[:2]

    y0 = (H - size) // 2
    x0 = (W - size) // 2
    if arr.ndim == 2:
        return arr[y0:y0 + size, x0:x0 + size]
    return arr[y0:y0 + size, x0:x0 + size, :]


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return x0, y0, x1, y1


class SamResolver:
    """
    Wrap SamPrepackedIndex but add robust key resolution:
      - direct key match
      - unique basename match
      - unique suffix (last 2 components) match
    """

    def __init__(self, sam_prepacked_dir: Path, concept2idx: Dict[str, int], cache_size: int = 4096):
        self.index = SamPrepackedIndex.load(sam_prepacked_dir, concept2idx=concept2idx, cache_size=0)
        self.keys = list(self.index.frame_to_file.keys())

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

        self._mask_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.cache_size = int(cache_size)

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

        if rk in self._mask_cache:
            m, c = self._mask_cache[rk]
            return m.copy(), c.copy()

        out = self.index.get_masks_for_relpath(rk)
        if out is None:
            return None

        masks_t, cids_t = out
        masks = masks_t.detach().cpu().numpy().astype(np.float32)  # (K,H,W)
        cids = cids_t.detach().cpu().numpy().astype(np.int64).reshape(-1)

        if self.cache_size > 0:
            self._mask_cache[rk] = (masks.copy(), cids.copy())
            if len(self._mask_cache) > self.cache_size:
                self._mask_cache.pop(next(iter(self._mask_cache)))

        return masks, cids

    def masks_for_concept(self, frame_rel: str, cid: int) -> List[np.ndarray]:
        out = self.get_masks(frame_rel)
        if out is None:
            return []
        masks, cids = out
        sel = np.where(cids == int(cid))[0]
        if sel.size == 0:
            return []
        ms = []
        for i in sel.tolist():
            m = (masks[i] > 0.5).astype(np.uint8)
            if int(m.sum()) > 0:
                ms.append(m)
        return ms

    def largest_mask_for_concept(self, frame_rel: str, cid: int) -> Optional[np.ndarray]:
        ms = self.masks_for_concept(frame_rel, cid)
        if not ms:
            return None
        areas = [int(m.sum()) for m in ms]
        return ms[int(np.argmax(areas))]

    def any_mentioned_visible(self, frame_rel: str, mentioned_cids: List[int]) -> bool:
        out = self.get_masks(frame_rel)
        if out is None:
            return False
        _, cids = out
        s = set(int(x) for x in cids.tolist())
        return any(int(cid) in s for cid in mentioned_cids)


def overlay_mask_red(ax, mask01: np.ndarray, alpha: float = 0.38, outline: bool = True) -> None:
    """
    Solid red overlay using RGBA (not a cmap).
    mask01: (H,W) in {0,1}
    """
    if mask01 is None or int(mask01.sum()) == 0:
        return
    H, W = mask01.shape
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[..., 0] = 1.0  # red
    rgba[..., 3] = alpha * mask01.astype(np.float32)
    ax.imshow(rgba, interpolation="nearest")
    if outline:
        try:
            ax.contour(mask01.astype(float), levels=[0.5], linewidths=1.2, colors="white")
        except Exception:
            pass


def choose_shown_offsets(visible_offsets: List[int], n_panels: int) -> List[int]:
    vis = sorted({o for o in visible_offsets if o != 0}, key=lambda x: abs(x))
    out = [0]
    for o in vis[: max(1, n_panels - 1)]:
        out.append(o)

    k = 1
    while len(out) < n_panels:
        for sgn in (-1, 1):
            cand = sgn * k
            if cand not in out:
                out.append(cand)
                if len(out) == n_panels:
                    break
        k += 1

    return sorted(out, key=lambda x: x)


@dataclass
class Candidate:
    score: float
    utterance: str
    concept: str
    cid: int
    frames: List[str]
    paired_i: int
    visible_offsets: List[int]
    shown_offsets: List[int]
    timestamps: Optional[List[float]]
    best_area_frac: float
    visible_frac: float
    compactness: float
    paired_frame: str


def gaussian(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z)


def render_strip(
    cand: Candidate,
    frames_root: Path,
    sam: SamResolver,
    fps: float,
    out_pdf: Path,
    out_png: Path,
) -> None:
    set_pub_style()

    offs = cand.shown_offsets
    n = len(offs)

    # Better default sizing: slightly taller header; consistent canvas.
    fig_w = 8.4 if n >= 5 else 7.4
    fig_h = 2.55
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        nrows=2,
        ncols=n,
        height_ratios=[0.62, 1.0],
        wspace=0.035,
        hspace=0.03,
    )

    # Explicit margins, no tight_layout.
    fig.subplots_adjust(
        left=0.015,
        right=0.995,
        top=0.985,
        bottom=0.055,
    )

    # Header area
    ax_t = fig.add_subplot(gs[0, :])
    ax_t.axis("off")

    utt_wrapped = textwrap.fill(cand.utterance, width=95)
    ax_t.text(
        0.0, 0.72,
        f'Utterance: "{utt_wrapped}"',
        ha="left", va="center",
        transform=ax_t.transAxes,
        fontsize=12,
    )
    ax_t.text(
        0.0, 0.15,
        f"Target: {cand.concept}",
        ha="left", va="center",
        transform=ax_t.transAxes,
        fontsize=14,
        fontweight="bold",
    )

    # Panels
    for j, off in enumerate(offs):
        ax = fig.add_subplot(gs[1, j])
        ax.set_xticks([])
        ax.set_yticks([])

        i = cand.paired_i + off
        if i < 0 or i >= len(cand.frames):
            ax.axis("off")
            continue

        frame_rel = cand.frames[i]
        img_path = frames_root / frame_rel
        if not img_path.is_file():
            rk = sam.resolve_key(frame_rel)
            if rk is not None and (frames_root / rk).is_file():
                img_path = frames_root / rk

        img = np.array(Image.open(img_path).convert("RGB"))
        img = center_crop_np(img, size=224)

        mask = sam.largest_mask_for_concept(frame_rel, cand.cid)
        if mask is not None:
            mask = center_crop_np(mask.astype(np.uint8), size=224)
        else:
            mask = None

        ax.imshow(img, interpolation="nearest")

        visible = (mask is not None) and (int(mask.sum()) > 0)
        if visible and mask is not None:
            overlay_mask_red(ax, (mask > 0).astype(np.uint8), alpha=0.38, outline=True)

        # Time label
        if (
            cand.timestamps is not None
            and 0 <= cand.paired_i < len(cand.timestamps)
            and 0 <= i < len(cand.timestamps)
        ):
            dt = float(cand.timestamps[i]) - float(cand.timestamps[cand.paired_i])
            title = "t (paired)" if off == 0 else f"{dt:+.1f}s"
        else:
            dt = off / float(fps)
            title = "t (paired)" if off == 0 else f"{dt:+.1f}s"

        ax.set_title(title, pad=4, fontsize=16)

        # Borders: paired and visible get stronger emphasis.
        for sp in ax.spines.values():
            sp.set_linewidth(2.0)
            sp.set_alpha(0.20)

        if off == 0:
            for sp in ax.spines.values():
                sp.set_alpha(0.95)

        if visible:
            for sp in ax.spines.values():
                sp.set_alpha(0.95)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    # Keep tight bounding box for clean inclusion in latex, but layout is controlled above.
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=str, required=True)
    ap.add_argument("--frames_root", type=str, required=True)
    ap.add_argument("--sam_prepacked_dir", type=str, required=True)
    ap.add_argument("--concept_list_file", type=str, default=None)

    ap.add_argument("--outdir", type=str, default="paper_figs")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--window_sec", type=float, default=2.0)
    ap.add_argument("--n_panels", type=int, default=5)
    ap.add_argument("--max_utterances", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug_every", type=int, default=5000)

    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--per_concept_k", type=int, default=4)
    ap.add_argument("--candidates_subdir", type=str, default="candidates")

    ap.add_argument("--require_no_referents_in_paired", type=int, default=1)

    ap.add_argument("--min_best_area_frac", type=float, default=0.01)
    ap.add_argument("--max_best_area_frac", type=float, default=0.45)
    ap.add_argument("--max_visible_frac", type=float, default=0.55)
    ap.add_argument("--min_compactness", type=float, default=0.10)

    ap.add_argument("--area_target", type=float, default=0.12)
    ap.add_argument("--area_sigma", type=float, default=0.10)
    ap.add_argument("--avoid_concepts", type=str, default=",".join(sorted(DEFAULT_AVOID)))
    ap.add_argument("--avoid_penalty", type=float, default=2.0)

    args = ap.parse_args()
    rng = np.random.default_rng(int(args.seed))

    concepts, concept2idx, _ = load_concepts(args.concept_list_file)
    avoid = {s.strip().lower() for s in str(args.avoid_concepts).split(",") if s.strip()}

    sam = SamResolver(Path(args.sam_prepacked_dir), concept2idx=concept2idx, cache_size=4096)
    data = json.loads(Path(args.metadata).read_text())["data"]
    frames_root = Path(args.frames_root)

    window_frames = int(round(float(args.window_sec) * float(args.fps)))

    pool: List[Candidate] = []
    per_concept_counts: Dict[str, int] = {}

    def maybe_add(c: Candidate) -> None:
        nonlocal pool
        if per_concept_counts.get(c.concept, 0) >= int(args.per_concept_k):
            return
        pool.append(c)
        per_concept_counts[c.concept] = per_concept_counts.get(c.concept, 0) + 1

        if len(pool) > int(args.top_k) * 6:
            pool.sort(key=lambda x: x.score, reverse=True)
            pool = pool[: int(args.top_k) * 3]
            per_concept_counts.clear()
            for cc in pool:
                per_concept_counts[cc.concept] = per_concept_counts.get(cc.concept, 0) + 1

    for u_i, ex in enumerate(data[: int(args.max_utterances)]):
        utt = str(ex.get("utterance", "")).strip()
        if not utt:
            continue

        toks = tokenize(utt)
        mentioned = [c for c in concepts if c in toks]
        if not mentioned:
            continue

        frames = list(ex.get("frame_filenames", []))
        if not frames:
            continue

        timestamps = ex.get("timestamps", None)
        timestamps = timestamps if isinstance(timestamps, list) else None

        mentioned_cids = [concept2idx[m] for m in mentioned if m in concept2idx]

        cand_is = {0, len(frames) // 2, max(0, len(frames) - 1)}
        if len(frames) > 6:
            cand_is.add(int(rng.integers(0, len(frames))))
            cand_is.add(int(rng.integers(0, len(frames))))
        cand_is = sorted({i for i in cand_is if 0 <= i < len(frames)})

        for paired_i in cand_is:
            paired_frame = frames[paired_i]

            if int(args.require_no_referents_in_paired) == 1:
                if sam.any_mentioned_visible(paired_frame, mentioned_cids):
                    continue

            for concept in mentioned:
                cid = concept2idx.get(concept, None)
                if cid is None:
                    continue

                # target must be absent in paired frame
                if sam.largest_mask_for_concept(paired_frame, cid) is not None:
                    continue

                visible_offsets: List[int] = []
                best_area = 0.0
                best_off = None
                best_mask = None

                valid_offsets: List[int] = []
                for off in range(-window_frames, window_frames + 1):
                    if off == 0:
                        continue
                    j = paired_i + off
                    if j < 0 or j >= len(frames):
                        continue
                    valid_offsets.append(off)
                    f = frames[j]
                    m = sam.largest_mask_for_concept(f, cid)
                    if m is None:
                        continue
                    area = float(m.sum())
                    if area <= 0:
                        continue
                    visible_offsets.append(off)
                    if area > best_area:
                        best_area = area
                        best_off = off
                        best_mask = m

                if not visible_offsets or best_off is None or best_mask is None:
                    continue

                Hm, Wm = best_mask.shape[:2]
                img_area = float(Hm * Wm)
                best_area_frac = float(best_area / max(1.0, img_area))
                visible_frac = float(len(set(visible_offsets)) / max(1, len(valid_offsets)))

                # compactness = mask_area / bbox_area (computed on full 224x224 mask)
                x0, y0, x1, y1 = bbox_from_mask(best_mask)
                bbox_area = float((x1 - x0 + 1) * (y1 - y0 + 1))
                compactness = float(best_area / max(1.0, bbox_area))

                if best_area_frac < float(args.min_best_area_frac):
                    continue
                if best_area_frac > float(args.max_best_area_frac):
                    continue
                if visible_frac > float(args.max_visible_frac):
                    continue
                if compactness < float(args.min_compactness):
                    continue

                nearest = min(visible_offsets, key=lambda x: abs(x))
                dt = abs(nearest) / float(args.fps)

                area_q = gaussian(best_area_frac, float(args.area_target), float(args.area_sigma))
                time_q = gaussian(dt, 0.4, 0.45)
                sparsity_q = gaussian(visible_frac, 0.25, 0.18)
                compact_q = gaussian(compactness, 0.55, 0.25)

                score = (
                    100.0 * area_q
                    + 40.0 * time_q
                    + 45.0 * sparsity_q
                    + 25.0 * compact_q
                )

                if concept in avoid:
                    score = score / float(args.avoid_penalty)

                shown = choose_shown_offsets(visible_offsets, n_panels=int(args.n_panels))

                cand = Candidate(
                    score=float(score),
                    utterance=utt,
                    concept=concept,
                    cid=int(cid),
                    frames=frames,
                    paired_i=int(paired_i),
                    visible_offsets=sorted({int(o) for o in visible_offsets}),
                    shown_offsets=shown,
                    timestamps=timestamps,
                    best_area_frac=float(best_area_frac),
                    visible_frac=float(visible_frac),
                    compactness=float(compactness),
                    paired_frame=str(paired_frame),
                )
                maybe_add(cand)

        if int(args.debug_every) > 0 and (u_i + 1) % int(args.debug_every) == 0:
            top = max(pool, key=lambda x: x.score) if pool else None
            print(f"[Debug] utterances_seen={u_i+1} pool={len(pool)} top_score={(top.score if top else -1):.2f}")

    if not pool:
        raise RuntimeError(
            "No candidates found. Try:\n"
            "  1) --require_no_referents_in_paired 0\n"
            "  2) increase --window_sec (e.g., 3.0)\n"
            "  3) relax filters: --max_best_area_frac 0.60 --max_visible_frac 0.80"
        )

    pool.sort(key=lambda x: x.score, reverse=True)
    selected: List[Candidate] = []
    cc: Dict[str, int] = {}
    for c in pool:
        if cc.get(c.concept, 0) >= int(args.per_concept_k):
            continue
        selected.append(c)
        cc[c.concept] = cc.get(c.concept, 0) + 1
        if len(selected) >= int(args.top_k):
            break

    outdir = Path(args.outdir)
    cand_dir = outdir / str(args.candidates_subdir)
    cand_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for r, c in enumerate(selected, start=1):
        frame_tag = slug(Path(c.paired_frame).stem, 60)
        name = f"rank{r:02d}_{slug(c.concept,20)}_{frame_tag}"

        out_png = cand_dir / f"{name}.png"
        out_pdf = cand_dir / f"{name}.pdf"
        out_json = cand_dir / f"{name}.json"

        render_strip(
            c,
            frames_root=frames_root,
            sam=sam,
            fps=float(args.fps),
            out_pdf=out_pdf,
            out_png=out_png,
        )

        meta = asdict(c)
        meta.update({
            "rank": r,
            "metadata": str(args.metadata),
            "frames_root": str(frames_root),
            "sam_prepacked_dir": str(args.sam_prepacked_dir),
            "window_frames": int(window_frames),
            "fps": float(args.fps),
            "window_sec": float(args.window_sec),
        })
        out_json.write_text(json.dumps(meta, indent=2))
        summary.append({
            "rank": r,
            "score": c.score,
            "concept": c.concept,
            "paired_frame": c.paired_frame,
            "best_area_frac": c.best_area_frac,
            "visible_frac": c.visible_frac,
            "compactness": c.compactness,
            "png": str(out_png),
            "pdf": str(out_pdf),
            "json": str(out_json),
            "utterance": c.utterance,
            "shown_offsets": c.shown_offsets,
        })

    (cand_dir / "candidates_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Saved] {len(selected)} candidates to {cand_dir}")
    print(f"[Saved] {cand_dir / 'candidates_summary.json'}")


if __name__ == "__main__":
    main()
