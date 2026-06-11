#!/usr/bin/env python3
# tools/temporal_misalignment_figB.py
# -*- coding: utf-8 -*-

"""
Figure B: Temporal misalignment diagnostic for CVCL-style frame–utterance pairing.

For utterances that mention any target concept, estimate:
  (1) P(referent visible at paired frame t)
  (2) P(referent visible within a temporal window [t-Δ, t+Δ])

Visibility is computed using frozen SAM instance outputs.

Supports two SAM output formats:
  A) Prepacked:
     <sam_masks_dir>/sam_prepacked/sam_prepacked_index.json
     <sam_masks_dir>/sam_prepacked/concept_vocab.json
     <sam_masks_dir>/sam_prepacked/*.pt
  B) Per-frame JSON files:
     <sam_masks_dir>/**/<frame>.json

All args after `--` are forwarded to train._setup_parser, like your other tools.
"""

import argparse
import collections
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- repo imports (same pipeline as training) ---
from multimodal.multimodal_saycam_data_module import MultiModalSAYCamDataModule
from multimodal.nesy_constraints import build_targets  # canonicalization + targets
from train import _setup_parser as build_repo_parser


DEFAULT_22 = [
    "ball", "basket", "car", "cat", "chair", "computer", "crib", "door", "floor", "foot",
    "ground", "hand", "kitchen", "paper", "puzzle", "road", "room", "sand", "stairs",
    "table", "toy", "window",
]


# ----------------------------
# shared plotting style
# ----------------------------

def set_pub_style() -> None:
    """Consistent, compact paper style (good for subfigures)."""
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.2,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,  # editable text in illustrator
        "ps.fonttype": 42,
    })


def save_both(fig: plt.Figure, out_pdf: Path, out_png: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ----------------------------
# utterance decoding
# ----------------------------

def build_id_maps_and_ignores(dm):
    vocab = dm.read_vocab()
    if isinstance(vocab, dict):
        id2tok = {int(i): tok for tok, i in vocab.items()}
    elif hasattr(vocab, "itos"):
        itos = list(vocab.itos)
        id2tok = {i: tok for i, tok in enumerate(itos)}
    else:
        raise ValueError("Unsupported vocab type returned by DataModule.read_vocab()")

    ignore_tokens = {"<pad>", "<unk>", "<sos>", "<eos>", ".", ",", "?", "!", "...", "..", "...."}
    ignore_ids = set()
    if isinstance(vocab, dict):
        for t in ignore_tokens:
            if t in vocab:
                ignore_ids.add(int(vocab[t]))
    return vocab, id2tok, ignore_ids


def token_to_words(tok: str) -> List[str]:
    tok = tok.replace("▁", " ")
    tok = tok.replace("Ġ", " ")
    tok = tok.replace("##", "")
    return [w for w in re.split(r"[^a-zA-Z]+", tok.lower()) if w]


def decode_any(obj, id2tok, ignore_ids):
    import torch

    def _flatten(x):
        if torch.is_tensor(x):
            for v in x.reshape(-1).tolist():
                yield v
        elif isinstance(x, (list, tuple)):
            for y in x:
                yield from _flatten(y)
        else:
            yield x

    toks: List[str] = []
    for item in _flatten(obj):
        if isinstance(item, int):
            if item in ignore_ids:
                continue
            tok = id2tok.get(int(item), "<unk>")
            toks.extend(token_to_words(tok))
        elif isinstance(item, str):
            toks.extend(token_to_words(item))
        else:
            toks.extend(token_to_words(str(item)))
    return toks


def extract_utterances_from_batch(batch, id2tok, ignore_ids) -> List[List[str]]:
    """Return List[List[str]] of length B for this batch."""
    import torch

    B = None
    if isinstance(batch, (list, tuple)) and len(batch) > 0 and torch.is_tensor(batch[0]) and batch[0].ndim >= 1:
        B = int(batch[0].shape[0])

    # 1) raw utterance list (len B)
    if isinstance(batch, (list, tuple)):
        for item in batch:
            if isinstance(item, (list, tuple)) and B is not None and len(item) == B:
                return [decode_any(u, id2tok, ignore_ids) for u in item]

    # 2) y token IDs tensor at batch[1] with optional lengths at batch[2]
    y = None
    y_len = None
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        cand = batch[1]
        if hasattr(cand, "ndim") and cand.ndim == 2:
            y = cand
    if y is not None and isinstance(batch, (list, tuple)) and len(batch) >= 3:
        cand = batch[2]
        if hasattr(cand, "ndim") and cand.ndim == 1 and int(cand.shape[0]) == int(y.shape[0]):
            y_len = cand

    if y is not None:
        out = []
        for i in range(int(y.shape[0])):
            L = int(y_len[i].item()) if y_len is not None else None
            row = y[i][:L] if L is not None else y[i]
            out.append(decode_any(row, id2tok, ignore_ids))
        return out

    # 3) fallback
    if isinstance(batch, (list, tuple)) and len(batch) > 0:
        return [decode_any(batch[0], id2tok, ignore_ids)]
    return []


# ----------------------------
# frame key extraction + neighbors
# ----------------------------

def find_strings(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(find_strings(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(find_strings(v))
    return out


def extract_frame_keys_from_batch(batch: Any, frames_root: Optional[Path]) -> List[str]:
    """Return per-sample frame keys (length B) that match SAM keys."""
    import torch

    cand_paths: List[str] = []
    if isinstance(batch, dict):
        cand_paths = find_strings(batch)
    elif isinstance(batch, (list, tuple)):
        for item in batch:
            if isinstance(item, dict):
                cand_paths.extend(find_strings(item))

    img_paths = [p for p in cand_paths if re.search(r"\.(jpg|jpeg|png)$", p.lower())]
    img_paths = [p for p in img_paths if "http" not in p.lower()]
    img_paths = list(dict.fromkeys(img_paths))  # de-dupe

    B = None
    if isinstance(batch, (list, tuple)) and len(batch) > 0 and torch.is_tensor(batch[0]) and batch[0].ndim >= 1:
        B = int(batch[0].shape[0])
    if B is None:
        B = len(img_paths)

    if len(img_paths) != B:
        raise RuntimeError(
            f"Could not extract per-sample image paths from batch (found {len(img_paths)}, expected {B}). "
            "Update extract_frame_keys_from_batch() for your batch structure."
        )

    keys: List[str] = []
    for p in img_paths:
        pp = Path(p)
        if frames_root is not None:
            try:
                rel = pp.relative_to(frames_root)
                key = str(rel.with_suffix(""))
            except Exception:
                key = str(Path(*pp.parts[-2:]).with_suffix(""))
        else:
            key = str(Path(*pp.parts[-2:]).with_suffix(""))
        keys.append(key.replace("\\", "/"))
    return keys


def parse_numeric_suffix(stem: str) -> Optional[Tuple[str, int, int]]:
    base = Path(stem)
    name = base.name
    m = re.match(r"^(.*?)(\d+)$", name)
    if not m:
        return None
    prefix_name, digits = m.group(1), m.group(2)
    pad = len(digits)
    idx = int(digits)
    parent = str(base.parent).replace("\\", "/")
    parent = "" if parent == "." else parent + "/"
    return parent + prefix_name, idx, pad


def neighbor_frame_key(frame_key: str, offset: int) -> Optional[str]:
    parsed = parse_numeric_suffix(frame_key)
    if parsed is None:
        return None
    parent_plus_prefix, idx, pad = parsed
    new_idx = idx + offset
    if new_idx < 0:
        return None
    base_prefix = Path(parent_plus_prefix).name
    parent = str(Path(frame_key).parent).replace("\\", "/")
    parent = "" if parent == "." else parent + "/"
    new_name = f"{base_prefix}{new_idx:0{pad}d}"
    return (parent + new_name).strip("/")


# ----------------------------
# SAM access
# ----------------------------

class SamAccessor:
    def get_labels(self, frame_key: str) -> Tuple[str, ...]:
        raise NotImplementedError


class JsonSamAccessor(SamAccessor):
    def __init__(self, sam_masks_dir: Path, max_files: Optional[int] = None):
        self.sam_masks_dir = sam_masks_dir
        self.index = self._build_index(max_files=max_files)

    def _load_labels_from_json(self, path: Path) -> Tuple[str, ...]:
        data = json.loads(path.read_text())
        inst_list = None
        if isinstance(data, dict):
            for k in ["instances", "objects", "predictions", "masks", "annotations"]:
                if k in data and isinstance(data[k], list):
                    inst_list = data[k]
                    break
        elif isinstance(data, list):
            inst_list = data
        else:
            inst_list = []

        labs: List[str] = []
        for it in inst_list or []:
            if not isinstance(it, dict):
                continue
            lab = it.get("label") or it.get("concept") or it.get("name")
            if lab is None:
                continue
            labs.append(str(lab).lower())
        return tuple(labs)

    def _build_index(self, max_files: Optional[int]) -> Dict[str, Tuple[str, ...]]:
        files = sorted(self.sam_masks_dir.rglob("*.json"))
        if max_files is not None:
            files = files[: int(max_files)]
        idx: Dict[str, Tuple[str, ...]] = {}
        for fp in files:
            rel = fp.relative_to(self.sam_masks_dir)
            key = str(rel.with_suffix("")).replace("\\", "/")
            try:
                idx[key] = self._load_labels_from_json(fp)
            except Exception:
                idx[key] = tuple()
        return idx

    def get_labels(self, frame_key: str) -> Tuple[str, ...]:
        return self.index.get(frame_key, tuple())


class PrepackedSamAccessor(SamAccessor):
    def __init__(self, prepacked_dir: Path, cache_size: int = 8192):
        self.prepacked_dir = prepacked_dir
        self.frame_to_pt = self._load_frame_index()
        self.id_to_label = self._load_vocab()
        self._labels_cached = lru_cache(maxsize=int(cache_size))(self._labels_uncached)

    def _load_vocab(self) -> List[str]:
        vocab_path = self.prepacked_dir / "concept_vocab.json"
        data = json.loads(vocab_path.read_text())
        concepts = data.get("concepts", [])
        return [str(x).lower() for x in concepts]

    def _load_frame_index(self) -> Dict[str, Path]:
        index_path = self.prepacked_dir / "sam_prepacked_index.json"
        raw = json.loads(index_path.read_text())
        out: Dict[str, Path] = {}
        for frame_relpath, pt_name in raw.items():
            k = str(Path(frame_relpath).with_suffix("")).replace("\\", "/")
            out[k] = self.prepacked_dir / str(pt_name)
        return out

    def _labels_uncached(self, frame_key: str) -> Tuple[str, ...]:
        import torch
        pt_path = self.frame_to_pt.get(frame_key, None)
        if pt_path is None or not pt_path.is_file():
            return tuple()
        try:
            data = torch.load(pt_path, map_location="cpu")
        except Exception:
            return tuple()
        cids = data.get("concept_ids", None)
        if cids is None:
            return tuple()
        labs: List[str] = []
        for cid in cids.reshape(-1).tolist():
            cid_int = int(cid)
            if 0 <= cid_int < len(self.id_to_label):
                labs.append(self.id_to_label[cid_int])
        return tuple(labs)

    def get_labels(self, frame_key: str) -> Tuple[str, ...]:
        return self._labels_cached(frame_key)


def build_sam_accessor(
    sam_masks_dir: Path,
    sam_mode: str,
    max_json_files: Optional[int],
    prepacked_cache_size: int,
) -> SamAccessor:
    if sam_mode not in {"auto", "prepacked", "json"}:
        raise ValueError("--sam_mode must be one of: auto, prepacked, json")

    prepacked_dir = sam_masks_dir / "sam_prepacked"
    has_prepacked = (prepacked_dir / "sam_prepacked_index.json").is_file()

    if sam_mode in {"auto", "prepacked"} and has_prepacked:
        return PrepackedSamAccessor(prepacked_dir=prepacked_dir, cache_size=prepacked_cache_size)

    return JsonSamAccessor(sam_masks_dir=sam_masks_dir, max_files=max_json_files)


# ----------------------------
# visibility check
# ----------------------------

def label_matches(label: str, concept: str, mode: str) -> bool:
    if mode == "exact":
        return label == concept
    if mode == "substring":
        return concept in label
    return concept in set(token_to_words(label))


def frame_visible_for_concepts(
    sam_accessor: SamAccessor,
    frame_key: str,
    concepts: Iterable[str],
    label_match: str,
) -> bool:
    labs = sam_accessor.get_labels(frame_key)
    if not labs:
        return False
    wanted = [c.lower() for c in concepts]
    for lab in labs:
        for c in wanted:
            if label_matches(lab, c, mode=label_match):
                return True
    return False


# ----------------------------
# main computation
# ----------------------------

def compute_temporal_misalignment(
    loader,
    id2tok,
    ignore_ids,
    concepts: List[str],
    frames_root: Optional[Path],
    sam_accessor: SamAccessor,
    window_frames: int,
    label_match: str,
) -> Dict[str, Any]:
    names = [c.lower() for c in concepts]

    total_utt_with_any = 0
    paired_visible_any = 0
    window_visible_any = 0

    per_c_total = collections.Counter()
    per_c_paired = collections.Counter()
    per_c_window = collections.Counter()

    for batch in loader:
        utts = extract_utterances_from_batch(batch, id2tok, ignore_ids)
        mention_mask = build_targets(utts, names)  # (B,C)
        frame_keys = extract_frame_keys_from_batch(batch, frames_root=frames_root)

        B = int(mention_mask.shape[0])
        for i in range(B):
            mentioned = [names[j] for j in range(len(names)) if int(mention_mask[i, j].item()) == 1]
            if not mentioned:
                continue

            total_utt_with_any += 1
            fk = frame_keys[i]

            vis_paired = frame_visible_for_concepts(sam_accessor, fk, mentioned, label_match=label_match)
            if vis_paired:
                paired_visible_any += 1

            vis_window = vis_paired
            if not vis_window:
                for off in range(-window_frames, window_frames + 1):
                    if off == 0:
                        continue
                    nk = neighbor_frame_key(fk, off)
                    if nk is None:
                        continue
                    if frame_visible_for_concepts(sam_accessor, nk, mentioned, label_match=label_match):
                        vis_window = True
                        break
            if vis_window:
                window_visible_any += 1

            for c in mentioned:
                per_c_total[c] += 1
                vis_c = frame_visible_for_concepts(sam_accessor, fk, [c], label_match=label_match)
                if vis_c:
                    per_c_paired[c] += 1
                    per_c_window[c] += 1
                    continue

                for off in range(-window_frames, window_frames + 1):
                    if off == 0:
                        continue
                    nk = neighbor_frame_key(fk, off)
                    if nk is None:
                        continue
                    if frame_visible_for_concepts(sam_accessor, nk, [c], label_match=label_match):
                        per_c_window[c] += 1
                        break

    agg = {
        "utterances_with_any_target_mention": int(total_utt_with_any),
        "paired_visible_any": int(paired_visible_any),
        "window_visible_any": int(window_visible_any),
        "paired_visible_rate_any": float(paired_visible_any / max(1, total_utt_with_any)),
        "window_visible_rate_any": float(window_visible_any / max(1, total_utt_with_any)),
        "window_frames": int(window_frames),
        "label_match": str(label_match),
        "num_concepts": int(len(names)),
    }

    rows = []
    for c in names:
        tot = int(per_c_total[c])
        rows.append({
            "concept": c,
            "num_utterances_mentioning": tot,
            "paired_visible": int(per_c_paired[c]),
            "window_visible": int(per_c_window[c]),
            "paired_visible_rate": float(per_c_paired[c] / max(1, tot)),
            "window_visible_rate": float(per_c_window[c] / max(1, tot)),
        })
    per_df = pd.DataFrame(rows).sort_values("num_utterances_mentioning", ascending=False)
    return {"aggregate": agg, "per_concept": per_df}


# ----------------------------
# plotting
# ----------------------------

def plot_figB(agg: Dict[str, Any], out_pdf: Path, out_png: Path) -> None:
    paired = 100.0 * float(agg["paired_visible_rate_any"])
    window = 100.0 * float(agg["window_visible_rate_any"])

    # Compact, subfigure-friendly aspect ratio
    fig, ax = plt.subplots(figsize=(3.8, 2.5))
    labels = ["Paired frame", f"Within ±{agg['window_frames']} frames"]
    vals = [paired, window]
    ax.bar(labels, vals)
    ax.set_ylabel("Referent visible (%)")

    # Dynamic y-limit to avoid huge empty space
    ymax = max(vals) * 1.6 + 2.0
    ymax = min(35.0, max(12.0, ymax))
    ax.set_ylim(0, ymax)

    for i, v in enumerate(vals):
        ax.text(i, v + 0.04 * ymax, f"{v:.1f}", ha="center", va="bottom")

    fig.tight_layout(pad=0.2)
    save_both(fig, out_pdf, out_png)


def plot_per_concept_bars(per_df: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    df = per_df.copy()
    x = np.arange(len(df))
    w = 0.42

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.bar(x - w / 2, 100.0 * df["paired_visible_rate"].to_numpy(), width=w, label="Paired")
    ax.bar(x + w / 2, 100.0 * df["window_visible_rate"].to_numpy(), width=w, label="Window")

    ax.set_xticks(x)
    ax.set_xticklabels(df["concept"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("Visible (%)")
    ax.legend(frameon=False, ncol=2, handlelength=1.4)

    ymax = float(np.nanmax(100.0 * df["window_visible_rate"].to_numpy())) * 1.3 + 2.0
    ax.set_ylim(0, min(60.0, max(8.0, ymax)))

    fig.tight_layout(pad=0.2)
    save_both(fig, out_pdf, out_png)


# ----------------------------
# main
# ----------------------------

def main():
    p = argparse.ArgumentParser("Temporal misalignment Figure B (paired vs window visibility)")
    p.add_argument("--concepts", type=str, default=None,
                   help="Path to JSON list of concept names (default: Labeled-S 22).")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--outdir", type=str, default="temporal_misalignment_out")
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--window_sec", type=float, default=2.0)
    p.add_argument("--sam_masks_dir", type=str, default="expt_saycam/train_sam_masks")
    p.add_argument("--sam_mode", type=str, default="auto", choices=["auto", "prepacked", "json"])
    p.add_argument("--prepacked_cache_size", type=int, default=8192)
    p.add_argument("--frames_root", type=str, default=None)
    p.add_argument("--label_match", type=str, default="exact", choices=["exact", "token", "substring"])
    p.add_argument("--plot_per_concept", type=int, default=0)
    p.add_argument("--max_json_files", type=int, default=0)

    known, unknown = p.parse_known_args()
    args = known

    repo_parser = build_repo_parser()
    data_args = repo_parser.parse_args(unknown if unknown is not None else [])
    data_args.dataset = "saycam"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.concepts:
        with open(args.concepts, "r") as f:
            concepts = json.load(f)
        if not isinstance(concepts, list):
            raise ValueError("--concepts must be a JSON list of strings")
    else:
        concepts = DEFAULT_22

    dm = MultiModalSAYCamDataModule(data_args)
    dm.prepare_data()
    dm.setup()

    if args.split == "train":
        loader = dm.train_dataloader()
    elif args.split == "val":
        loader = dm.val_dataloader()[0]
    else:
        loader = dm.test_dataloader()[0]

    _, id2tok, ignore_ids = build_id_maps_and_ignores(dm)

    sam_dir = Path(args.sam_masks_dir)
    max_json = None if int(args.max_json_files) <= 0 else int(args.max_json_files)
    sam_accessor = build_sam_accessor(
        sam_masks_dir=sam_dir,
        sam_mode=str(args.sam_mode),
        max_json_files=max_json,
        prepacked_cache_size=int(args.prepacked_cache_size),
    )

    frames_root = Path(args.frames_root) if args.frames_root else None
    window_frames = int(round(float(args.window_sec) * float(args.fps)))

    stats = compute_temporal_misalignment(
        loader=loader,
        id2tok=id2tok,
        ignore_ids=ignore_ids,
        concepts=concepts,
        frames_root=frames_root,
        sam_accessor=sam_accessor,
        window_frames=window_frames,
        label_match=str(args.label_match),
    )

    summary_path = outdir / f"{args.split}_temporal_misalignment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(stats["aggregate"], f, indent=2)

    per_csv = outdir / f"{args.split}_per_concept_temporal_misalignment.csv"
    stats["per_concept"].to_csv(per_csv, index=False)

    set_pub_style()
    fig_pdf = outdir / f"{args.split}_figB_paired_vs_window.pdf"
    fig_png = outdir / f"{args.split}_figB_paired_vs_window.png"
    plot_figB(stats["aggregate"], fig_pdf, fig_png)

    if int(args.plot_per_concept) == 1:
        pc_pdf = outdir / f"{args.split}_per_concept_paired_vs_window.pdf"
        pc_png = outdir / f"{args.split}_per_concept_paired_vs_window.png"
        plot_per_concept_bars(stats["per_concept"], pc_pdf, pc_png)

    print("[Done]")
    print(json.dumps(stats["aggregate"], indent=2))
    print(f"[Saved] {summary_path}")
    print(f"[Saved] {per_csv}")
    print(f"[Saved] {fig_pdf} / {fig_png}")


if __name__ == "__main__":
    main()
