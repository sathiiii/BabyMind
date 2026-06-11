#!/usr/bin/env python3
"""
Prepack SAM mask NPZ files into per-frame .pt tensors (filename-driven, no JSONL).

Input:
  <output_dir>/npz/obj/*.npz

Expected filename pattern (both are supported):
  <frame_base>_obj_<label>_<NNN>.npz
  <frame_base>_obj_<label>_<NNN>_<run_tag...>.npz

Outputs (default):
  <output_dir>/sam_prepacked/
    - <frame_base>.pt                      # dict: {"masks": uint8[M,H,W], "concept_ids": int16[M]}
    - concept_vocab.json                   # {"concepts": [id->label]}
    - sam_prepacked_index.json             # {frame_base: "<frame_base>.pt"}
    - concept_frequency.json               # parsed and packed label counts + debug stats

Notes:
- No aliasing. Labels come directly from filenames.
- Optional: --canonical-only will only pack the 15 canonical Labeled-S object labels (still reports all parsed labels).
- Resume-friendly:
    * If concept_vocab.json exists, it is loaded and IDs are kept stable.
    * Existing <frame_base>.pt files are skipped unless --overwrite is set.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Any

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

# Try to import image size from your data module
try:
    from multimodal.multimodal_data_module import IMAGE_H, IMAGE_W  # type: ignore
except Exception:
    IMAGE_H = None
    IMAGE_W = None


# Canonical 15 (ordering only; NOT aliasing)
CANONICAL_15: List[str] = [
    "ball", "basket", "car", "cat", "chair", "computer", "crib", "door",
    "foot", "hand", "paper", "puzzle", "stairs", "table", "window",
]
CANONICAL_15_SET = set(CANONICAL_15)

# Robust parse:
#   frame_base = group(1)
#   label      = group(2)
#   inst       = group(3) (unused)
# Works for both "..._000.npz" and "..._000_<tag>.npz"
_STEM_RE = re.compile(r"^(?P<frame>.+?)_obj_(?P<label>.+?)_(?P<inst>\d{3})(?:_.*)?$")


@dataclass(frozen=True)
class MaskFile:
    path: Path
    frame_base: str
    label: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepack SAM mask npz files into per-frame .pt tensors (from filenames, no JSONL)."
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root output directory that contains npz/obj (same as miner --output-dir).",
    )
    p.add_argument(
        "--npz-obj-dir",
        type=str,
        default=None,
        help="Optional override for NPZ folder. Default: <output-dir>/npz/obj",
    )
    p.add_argument(
        "--npz-glob",
        type=str,
        default="*.npz",
        help="Glob under npz-obj-dir (default: *.npz). Use '**/*.npz' if nested.",
    )
    p.add_argument(
        "--prepacked-dir",
        type=str,
        default=None,
        help="Where to write prepacked outputs. Default: <output-dir>/sam_prepacked",
    )
    p.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="Target image height. If not set, falls back to IMAGE_H from data module.",
    )
    p.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="Target image width. If not set, falls back to IMAGE_W from data module.",
    )
    p.add_argument(
        "--canonical-only",
        action="store_true",
        help="Only pack masks whose filename label is one of the 15 canonical Labeled-S object labels.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing <frame_base>.pt files (default: skip if exists).",
    )
    p.add_argument(
        "--path-prefix",
        type=str,
        default=None,
        help="Optional prefix to try if a mask path is not found (rare when scanning a directory).",
    )
    return p.parse_args()


def resolve_image_size(args: argparse.Namespace) -> Tuple[int, int]:
    h = args.image_height if args.image_height is not None else IMAGE_H
    w = args.image_width if args.image_width is not None else IMAGE_W
    if h is None or w is None:
        raise RuntimeError(
            "Image size not specified. Pass --image-height/--image-width or ensure IMAGE_H/IMAGE_W can be imported."
        )
    return int(h), int(w)


def safe_load_mask_npz(
    mask_path: Path,
    target_h: int,
    target_w: int,
    path_prefix: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """
    Loads 'mask' from a .npz and returns uint8 array (H,W) in {0,1}, resized to (target_h,target_w).
    """
    p = mask_path
    if not p.is_file():
        if path_prefix is not None:
            p2 = path_prefix / p
            if p2.is_file():
                p = p2
        if not p.is_file():
            return None

    try:
        data = np.load(p)
    except Exception:
        return None

    if "mask" not in data:
        return None

    m = np.asarray(data["mask"])

    # Normalize to uint8 {0,1}
    if m.dtype == np.bool_:
        m_u8 = m.astype(np.uint8)
    else:
        m2 = m
        if m2.dtype != np.uint8:
            # handle float/other
            m2 = m2.astype(np.float32)
            # if looks like 0..255, normalize
            if float(np.max(m2)) > 1.5:
                m2 = m2 / 255.0
            m_u8 = (m2 > 0.5).astype(np.uint8)
        else:
            # uint8: could be {0,1} or {0,255}
            mx = int(m2.max()) if m2.size else 0
            if mx > 1:
                m_u8 = (m2 > 127).astype(np.uint8)
            else:
                m_u8 = (m2 > 0).astype(np.uint8)

    # Resize if needed
    if m_u8.shape != (target_h, target_w):
        img = Image.fromarray(m_u8 * 255)
        img = img.resize((target_w, target_h), resample=Image.NEAREST)
        m_u8 = (np.array(img, dtype=np.uint8) > 127).astype(np.uint8)

    return m_u8


def parse_mask_file(p: Path) -> Optional[MaskFile]:
    stem = p.stem
    m = _STEM_RE.match(stem)
    if not m:
        return None
    frame_base = m.group("frame")
    label = m.group("label").strip().lower()
    if not frame_base or not label:
        return None
    return MaskFile(path=p, frame_base=frame_base, label=label)


def load_existing_vocab(prepacked_dir: Path) -> Optional[List[str]]:
    vocab_path = prepacked_dir / "concept_vocab.json"
    if not vocab_path.is_file():
        return None
    try:
        obj = json.loads(vocab_path.read_text())
    except Exception:
        return None
    concepts = obj.get("concepts", None)
    if not isinstance(concepts, list) or not all(isinstance(x, str) for x in concepts):
        return None
    return [str(x) for x in concepts]


def load_existing_frame_index(prepacked_dir: Path) -> Dict[str, str]:
    idx_path = prepacked_dir / "sam_prepacked_index.json"
    if not idx_path.is_file():
        return {}
    try:
        obj = json.loads(idx_path.read_text())
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in obj.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out


def build_vocab_from_labels(labels: Iterable[str]) -> List[str]:
    labels_set = {str(x).strip().lower() for x in labels if str(x).strip()}
    ordered: List[str] = []
    for c in CANONICAL_15:
        if c in labels_set:
            ordered.append(c)
    for c in sorted(labels_set - set(ordered)):
        ordered.append(c)
    return ordered


def save_concept_vocab(prepacked_dir: Path, id_to_concept: List[str]) -> None:
    vocab_path = prepacked_dir / "concept_vocab.json"
    payload = {"concepts": list(id_to_concept)}
    vocab_path.write_text(json.dumps(payload, indent=2))
    print(f"[INFO] Wrote: {vocab_path}")


def save_frame_index(prepacked_dir: Path, frame_index: Dict[str, str]) -> None:
    idx_path = prepacked_dir / "sam_prepacked_index.json"
    idx_path.write_text(json.dumps(frame_index, indent=2))
    print(f"[INFO] Wrote: {idx_path}")


def save_concept_frequency(
    prepacked_dir: Path,
    *,
    npz_dir: Path,
    npz_glob: str,
    total_npz_files: int,
    matched_pattern: int,
    bad_pattern_files: List[str],
    parsed_counts: Dict[str, int],
    packed_counts: Dict[str, int],
    total_packed_masks: int,
    failed_npz_loads: int,
    canonical_only: bool,
) -> None:
    out_path = prepacked_dir / "concept_frequency.json"

    def _sorted_counts(d: Dict[str, int]) -> Dict[str, int]:
        items = sorted(d.items(), key=lambda kv: (-int(kv[1]), kv[0]))
        return {k: int(v) for k, v in items}

    payload: Dict[str, Any] = {
        "npz_dir": str(npz_dir),
        "npz_glob": str(npz_glob),
        "total_npz_files": int(total_npz_files),
        "matched_filename_pattern": int(matched_pattern),
        "bad_pattern_files": int(len(bad_pattern_files)),
        "example_bad_pattern_files": bad_pattern_files[:50],
        "canonical_only": bool(canonical_only),
        "total_masks_packed": int(total_packed_masks),
        "failed_npz_loads": int(failed_npz_loads),
        "counts_parsed_from_filenames": _sorted_counts(parsed_counts),
        "counts_packed_successfully": _sorted_counts(packed_counts),
    }

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[INFO] Wrote: {out_path}")


def group_by_frame(records: List[MaskFile]) -> Iterable[Tuple[str, List[MaskFile]]]:
    if not records:
        return
    # records are expected sorted by frame_base already
    cur_frame = records[0].frame_base
    buf: List[MaskFile] = []
    for r in records:
        if r.frame_base != cur_frame:
            yield cur_frame, buf
            buf = []
            cur_frame = r.frame_base
        buf.append(r)
    if buf:
        yield cur_frame, buf


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    if args.npz_obj_dir is None:
        npz_dir = output_dir / "npz" / "obj"
    else:
        npz_dir = Path(args.npz_obj_dir)

    if args.prepacked_dir is None:
        prepacked_dir = output_dir / "sam_prepacked"
    else:
        prepacked_dir = Path(args.prepacked_dir)

    prepacked_dir.mkdir(parents=True, exist_ok=True)

    target_h, target_w = resolve_image_size(args)
    path_prefix = Path(args.path_prefix) if args.path_prefix is not None else None

    if not npz_dir.is_dir():
        raise FileNotFoundError(f"NPZ dir not found: {npz_dir}")

    npz_files = sorted(npz_dir.glob(args.npz_glob))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found under {npz_dir} with glob '{args.npz_glob}'")

    # Resume safety:
    existing_pt = list(prepacked_dir.glob("*.pt"))
    existing_vocab = load_existing_vocab(prepacked_dir)
    if existing_pt and existing_vocab is None and not args.overwrite:
        raise RuntimeError(
            f"Found existing .pt files in {prepacked_dir} but concept_vocab.json is missing.\n"
            f"To avoid ID mismatches, either delete {prepacked_dir} and re-run, or run with --overwrite."
        )

    # Parse all filenames
    records: List[MaskFile] = []
    bad_pattern: List[str] = []
    parsed_counts: Dict[str, int] = {}

    matched = 0
    for p in tqdm(npz_files, desc="Parsing NPZ filenames"):
        mf = parse_mask_file(p)
        if mf is None:
            bad_pattern.append(str(p))
            continue
        matched += 1
        parsed_counts[mf.label] = parsed_counts.get(mf.label, 0) + 1
        records.append(mf)

    if not records:
        raise RuntimeError("No NPZ files matched the expected '*_obj_<label>_<NNN>*' filename pattern.")

    # Optional filter for packing
    if args.canonical_only:
        records_pack = [r for r in records if r.label in CANONICAL_15_SET]
    else:
        records_pack = records

    # Build / load vocab
    if existing_vocab is not None and not args.overwrite:
        id_to_concept = existing_vocab
        concept2id = {c: i for i, c in enumerate(id_to_concept)}
        print(f"[INFO] Loaded existing vocab with {len(id_to_concept)} concepts from {prepacked_dir / 'concept_vocab.json'}")
    else:
        # Deterministic vocab (canonical first, then sorted extras)
        labels_for_vocab = {r.label for r in records_pack} if records_pack else set()
        id_to_concept = build_vocab_from_labels(labels_for_vocab)
        concept2id = {c: i for i, c in enumerate(id_to_concept)}
        print(f"[INFO] Built new vocab with {len(id_to_concept)} concepts")

    # Load existing index (resume)
    frame_index = {} if args.overwrite else load_existing_frame_index(prepacked_dir)

    # Sort by frame_base so grouping is contiguous
    records_pack.sort(key=lambda r: (r.frame_base, r.label, r.path.name))

    packed_counts: Dict[str, int] = {}
    total_packed_masks = 0
    failed_npz_loads = 0

    # Pack per frame
    frame_groups = list(group_by_frame(records_pack))
    for frame_base, items in tqdm(frame_groups, desc="Packing frames"):
        out_path = prepacked_dir / f"{frame_base}.pt"

        if out_path.is_file() and not args.overwrite:
            frame_index[frame_base] = out_path.name
            continue

        masks: List[np.ndarray] = []
        cids: List[int] = []

        for mf in items:
            m = safe_load_mask_npz(mf.path, target_h, target_w, path_prefix=path_prefix)
            if m is None:
                failed_npz_loads += 1
                continue

            # ensure label in vocab (if overwriting + new label appears)
            if mf.label not in concept2id:
                # append at end deterministically (rare; happens if vocab loaded but new labels exist)
                concept2id[mf.label] = len(id_to_concept)
                id_to_concept.append(mf.label)

            cid = int(concept2id[mf.label])
            masks.append(m)
            cids.append(cid)

            packed_counts[mf.label] = packed_counts.get(mf.label, 0) + 1
            total_packed_masks += 1

        if not masks:
            # nothing valid for this frame
            continue

        masks_arr = np.stack(masks, axis=0).astype(np.uint8)  # [M,H,W]
        cids_arr = np.asarray(cids, dtype=np.int16)           # [M]

        tensors = {
            "masks": torch.from_numpy(masks_arr),          # uint8
            "concept_ids": torch.from_numpy(cids_arr),     # int16
        }
        torch.save(tensors, out_path)
        frame_index[frame_base] = out_path.name

    # Save metadata
    save_concept_vocab(prepacked_dir, id_to_concept)
    save_frame_index(prepacked_dir, frame_index)

    save_concept_frequency(
        prepacked_dir,
        npz_dir=npz_dir,
        npz_glob=args.npz_glob,
        total_npz_files=len(npz_files),
        matched_pattern=matched,
        bad_pattern_files=bad_pattern,
        parsed_counts=parsed_counts,
        packed_counts=packed_counts,
        total_packed_masks=total_packed_masks,
        failed_npz_loads=failed_npz_loads,
        canonical_only=bool(args.canonical_only),
    )

    print(f"[INFO] Done. Prepacked outputs in: {prepacked_dir}")


if __name__ == "__main__":
    main()
