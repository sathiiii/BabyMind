#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepack SAM mask npz files into per-frame .pt tensors "
            "for fast loading at training time."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help=(
            "Root output directory used by the GDINO+SAM+CLIP script "
            "(same as its --output-dir). This is where npz and index JSONL live."
        ),
    )
    parser.add_argument(
        "--index-jsonl",
        type=str,
        action="append",
        default=None,
        help=(
            "Path to a gdino_sam_clip_multi_index_rank*.jsonl file. "
            "Can be given multiple times. If omitted, the script will "
            "search under --output-dir for files matching that pattern."
        ),
    )
    parser.add_argument(
        "--prepacked-dir",
        type=str,
        default=None,
        help=(
            "Directory where prepacked .pt files and vocab/index JSON will be saved. "
            "Default: <output-dir>/sam_prepacked"
        ),
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="Target image height. If not set, falls back to IMAGE_H from data module.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="Target image width. If not set, falls back to IMAGE_W from data module.",
    )
    parser.add_argument(
        "--path-prefix",
        type=str,
        default=None,
        help=(
            "Optional prefix to prepend to mask_path if the stored path "
            "is not found on disk."
        ),
    )

    args = parser.parse_args()
    return args


def resolve_image_size(args: argparse.Namespace) -> Tuple[int, int]:
    h = args.image_height if args.image_height is not None else IMAGE_H
    w = args.image_width if args.image_width is not None else IMAGE_W

    if h is None or w is None:
        raise RuntimeError(
            "Image size is not specified. Either pass --image-height/--image-width "
            "or ensure IMAGE_H/IMAGE_W can be imported from multimodal.multimodal_data_module."
        )
    return int(h), int(w)


def find_index_files(output_dir: Path, explicit: Optional[List[str]]) -> List[Path]:
    if explicit:
        return [Path(p) for p in explicit]

    # Default: glob for rank JSONLs
    pattern = "gdino_sam_clip_multi_index_rank*.jsonl"
    files = sorted(output_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No index JSONL files found in {output_dir} matching {pattern}. "
            "Pass them explicitly with --index-jsonl."
        )
    return files


def safe_load_mask_npz(
    mask_path: Path,
    target_h: int,
    target_w: int,
    path_prefix: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """
    Load a single mask from a .npz file and resize to (target_h, target_w).
    Returns (H, W) uint8 array in {0, 1}, or None on failure.
    """
    if not mask_path.is_file():
        if path_prefix is not None:
            mask_path = path_prefix / mask_path
        if not mask_path.is_file():
            print(f"[WARN] mask npz not found: {mask_path}")
            return None

    try:
        data = np.load(mask_path)
    except Exception as e:
        print(f"[WARN] failed to load npz {mask_path}: {e}")
        return None

    if "mask" not in data:
        print(f"[WARN] 'mask' key not found in {mask_path}")
        return None

    m = np.asarray(data["mask"])

    # Convert to float and normalize if needed
    m = m.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0

    # Resize if shape mismatch
    if m.shape != (target_h, target_w):
        img = Image.fromarray((m * 255.0).astype(np.uint8))
        img = img.resize((target_w, target_h), resample=Image.NEAREST)
        m = np.array(img, dtype=np.uint8)
        m = (m > 127).astype(np.uint8)
    else:
        m = (m > 0.5).astype(np.uint8)

    return m


def frame_to_safe_name(frame_relpath: str) -> str:
    """
    Convert a frame_relpath like 'sub1/train/frame_0001.jpg'
    to a safe base name such as 'sub1_train_frame_0001'.
    """
    base = frame_relpath.replace(os.sep, "_")
    base = base.split(".")[0]
    return base


def process_frame_entries(
    frame_relpath: str,
    entries: List[Dict[str, Any]],
    prepacked_dir: Path,
    concept2id: Dict[str, int],
    frame_index: Dict[str, str],
    target_h: int,
    target_w: int,
    path_prefix: Optional[Path],
) -> None:
    """
    Given all index entries for a single frame, load masks, map labels to IDs,
    and write a single .pt file for that frame.
    """
    if not entries:
        return

    safe_name = frame_to_safe_name(frame_relpath)
    out_path = prepacked_dir / f"{safe_name}.pt"

    # Skip if already done (useful for resume).
    if out_path.is_file():
        frame_index[frame_relpath] = out_path.name
        return

    masks: List[np.ndarray] = []
    concept_ids: List[int] = []

    for entry in entries:
        clip_label = entry.get("concept_clip")
        mask_path_str = entry.get("mask_path")

        if not clip_label or not mask_path_str:
            continue

        mask_path = Path(mask_path_str)
        m = safe_load_mask_npz(mask_path, target_h, target_w, path_prefix=path_prefix)
        if m is None:
            continue

        # Assign concept ID
        if clip_label not in concept2id:
            concept2id[clip_label] = len(concept2id)
        cid = concept2id[clip_label]

        masks.append(m)
        concept_ids.append(cid)

    if not masks:
        # No valid masks for this frame
        return

    masks_arr = np.stack(masks, axis=0)  # (M, H, W)
    cids_arr = np.array(concept_ids, dtype=np.int16)

    tensors = {
        "masks": torch.from_numpy(masks_arr),        # uint8 [M, H, W]
        "concept_ids": torch.from_numpy(cids_arr),   # int16 [M]
    }

    torch.save(tensors, out_path)
    frame_index[frame_relpath] = out_path.name


def process_index_file(
    index_path: Path,
    prepacked_dir: Path,
    concept2id: Dict[str, int],
    frame_index: Dict[str, str],
    target_h: int,
    target_w: int,
    path_prefix: Optional[Path],
) -> None:
    """
    Stream through one JSONL index file and prepack masks frame by frame.
    Assumes that all entries for a given frame_relpath are contiguous
    (which is true for the GDINO+SAM+CLIP script).
    """
    print(f"[INFO] Processing index JSONL: {index_path}")

    current_frame: Optional[str] = None
    buffer: List[Dict[str, Any]] = []

    with index_path.open("r") as f:
        for line in tqdm(f, desc=index_path.name):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            frame_relpath = entry.get("frame_relpath")
            if not frame_relpath:
                continue

            # If we are still on the same frame, just accumulate
            if current_frame is None:
                current_frame = frame_relpath

            if frame_relpath != current_frame:
                # Flush previous frame
                process_frame_entries(
                    current_frame,
                    buffer,
                    prepacked_dir,
                    concept2id,
                    frame_index,
                    target_h,
                    target_w,
                    path_prefix,
                )
                buffer = []
                current_frame = frame_relpath

            buffer.append(entry)

    # Flush last frame
    if current_frame is not None and buffer:
        process_frame_entries(
            current_frame,
            buffer,
            prepacked_dir,
            concept2id,
            frame_index,
            target_h,
            target_w,
            path_prefix,
        )


def save_concept_vocab(prepacked_dir: Path, concept2id: Dict[str, int]) -> None:
    if not concept2id:
        print("[WARN] No concepts encountered, not writing concept_vocab.json")
        return

    # Build id_to_concept in index order
    id_to_concept = [None] * len(concept2id)
    for name, idx in concept2id.items():
        id_to_concept[idx] = name

    vocab = {
        "concepts": id_to_concept,
    }

    vocab_path = prepacked_dir / "concept_vocab.json"
    with vocab_path.open("w") as f:
        json.dump(vocab, f, indent=2)
    print(f"[INFO] Wrote concept vocabulary to {vocab_path}")


def save_frame_index(prepacked_dir: Path, frame_index: Dict[str, str]) -> None:
    if not frame_index:
        print("[WARN] No frames were packed, not writing sam_prepacked_index.json")
        return

    index_path = prepacked_dir / "sam_prepacked_index.json"
    with index_path.open("w") as f:
        json.dump(frame_index, f, indent=2)
    print(f"[INFO] Wrote frame index to {index_path}")


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.prepacked_dir is None:
        prepacked_dir = output_dir / "sam_prepacked"
    else:
        prepacked_dir = Path(args.prepacked_dir)
    prepacked_dir.mkdir(parents=True, exist_ok=True)

    target_h, target_w = resolve_image_size(args)

    index_files = find_index_files(output_dir, args.index_jsonl)
    path_prefix = Path(args.path_prefix) if args.path_prefix is not None else None

    # Global mappings
    concept2id: Dict[str, int] = {}
    frame_index: Dict[str, str] = {}

    for index_path in index_files:
        process_index_file(
            index_path=index_path,
            prepacked_dir=prepacked_dir,
            concept2id=concept2id,
            frame_index=frame_index,
            target_h=target_h,
            target_w=target_w,
            path_prefix=path_prefix,
        )

    save_concept_vocab(prepacked_dir, concept2id)
    save_frame_index(prepacked_dir, frame_index)

    print(f"[INFO] Done. Prepacked .pt files are in {prepacked_dir}")


if __name__ == "__main__":
    main()
