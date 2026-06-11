from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import os

import torch


def _rank0() -> bool:
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))) == 0


def _load_frequency_counts(freq_path: Path) -> Dict[str, int]:
    """
    Returns dict: {concept_lower: count_int}.
    Reads keys in priority:
      counts_packed_successfully
      counts_parsed_from_filenames
      counts
    """
    if not freq_path.is_file():
        return {}
    try:
        obj = json.loads(freq_path.read_text())
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}

    counts_obj = (
        obj.get("counts_packed_successfully")
        or obj.get("counts_parsed_from_filenames")
        or obj.get("counts")
        or {}
    )
    if not isinstance(counts_obj, dict):
        return {}

    out: Dict[str, int] = {}
    for k, v in counts_obj.items():
        try:
            out[str(k).strip().lower()] = int(v)
        except Exception:
            continue
    return out


def _load_concept_vocab(prepacked_root: Path) -> List[str]:
    vocab_path = prepacked_root / "concept_vocab.json"
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Missing {vocab_path}")
    obj = json.loads(vocab_path.read_text())
    concepts = obj.get("concepts", [])
    if not isinstance(concepts, list):
        raise ValueError(f"{vocab_path} has no 'concepts' list")
    return [str(x).strip().lower() for x in concepts]


def _load_concept_list_file(path: Union[str, Path]) -> Tuple[Dict[str, int], List[Optional[str]]]:
    """
    concept_list_file can be:
      - JSON list: ["cat","dog",...] defines contiguous IDs in order
      - JSON dict: {"cat": 7, "dog": 2} defines explicit IDs
      - JSON dict without int values: keys are used in insertion order
    Returns:
      concept2idx (lowercase)
      idx2concept (list, may contain None gaps if IDs are non-contiguous)
    """
    p = Path(path)
    raw = json.loads(p.read_text())

    if isinstance(raw, list):
        names = [str(x).strip().lower() for x in raw]
        concept2idx = {n: i for i, n in enumerate(names)}
        idx2concept = list(names)
        return concept2idx, idx2concept

    if isinstance(raw, dict):
        if all(isinstance(v, int) for v in raw.values()):
            concept2idx = {str(k).strip().lower(): int(v) for k, v in raw.items()}
            max_id = max(concept2idx.values()) if concept2idx else -1
            idx2concept: List[Optional[str]] = [None] * (max_id + 1)
            for n, i in concept2idx.items():
                if 0 <= i < len(idx2concept):
                    idx2concept[i] = n
            return concept2idx, idx2concept

        names = [str(k).strip().lower() for k in raw.keys()]
        concept2idx = {n: i for i, n in enumerate(names)}
        idx2concept = list(names)
        return concept2idx, idx2concept

    raise ValueError(f"Unsupported concept_list_file JSON type: {type(raw)}")


@dataclass
class SamConceptRegistry:
    # Global concept space
    concept2idx: Dict[str, int]
    idx2concept: List[Optional[str]]  # global_id -> name (or None if gap)

    # Local SAM concept space (concept_vocab.json)
    local_concepts: Tuple[str, ...]   # local_id -> name
    local_to_global: torch.Tensor     # (local_C,) long, -1 means drop

    # Counts/weights in GLOBAL space
    counts_full: torch.Tensor         # (C,) float
    counts_eff: torch.Tensor          # (C,) float, filtered concepts set to 0
    weights: torch.Tensor             # (C,) float, filtered concepts weight=1

    # Book-keeping
    freq_path: str
    min_masks_per_concept: int
    dropped_local: Dict[str, str]     # local_name -> reason


def build_sam_concept_registry(
    *,
    sam_prepacked_dir: Union[str, Path],
    concept_frequency_json: Optional[Union[str, Path]],
    min_masks_per_concept: int,
    concept_list_file: Optional[Union[str, Path]],
    alpha: float,
    clip_min: float,
    clip_max: float,
    verbose: bool = False,
) -> SamConceptRegistry:
    root = Path(sam_prepacked_dir)

    # 1) local concept list from concept_vocab.json
    local = _load_concept_vocab(root)
    local_concepts = tuple(local)
    local_C = len(local_concepts)

    # 2) frequency counts
    if concept_frequency_json is not None:
        freq_path = Path(concept_frequency_json)
    else:
        freq_path = root / "concept_frequency.json"
    freq_counts = _load_frequency_counts(freq_path)

    # 3) global ID space
    if concept_list_file is not None:
        concept2idx, idx2concept = _load_concept_list_file(concept_list_file)
    else:
        # global IDs equal local IDs in concept_vocab.json order
        idx2concept = list(local_concepts)
        concept2idx = {n: i for i, n in enumerate(idx2concept)}

    C = len(idx2concept)

    # 4) local -> global remap with filtering
    dropped: Dict[str, str] = {}
    local_to_global = torch.full((local_C,), -1, dtype=torch.long)

    do_freq_filter = int(min_masks_per_concept) > 0 and len(freq_counts) > 0

    for lid, name in enumerate(local_concepts):
        gid = concept2idx.get(name, -1)
        if gid < 0:
            dropped[name] = "not_in_concept_list"
            continue    

        if do_freq_filter:
            c = int(freq_counts.get(name, 0))
            if c < int(min_masks_per_concept):
                dropped[name] = f"freq<{int(min_masks_per_concept)} (count={c})"
                continue

        local_to_global[lid] = int(gid)

    # 5) counts in GLOBAL space (by name)
    counts_full = torch.zeros(C, dtype=torch.float32)
    for gid, gname in enumerate(idx2concept):
        if gname is None:
            continue
        counts_full[gid] = float(freq_counts.get(str(gname).lower(), 0))

    # 6) counts_eff + weights
    counts_eff = counts_full.clone()
    if do_freq_filter:
        counts_eff[counts_eff < float(min_masks_per_concept)] = 0.0

    keep = counts_eff > 0.0

    weights = torch.ones_like(counts_full)
    if bool(keep.any().item()):
        ref = counts_eff[keep].median()
        a = max(float(alpha), 0.0)
        w = (ref / counts_eff[keep].clamp(min=1.0)).pow(a)
        w = w.clamp(min=float(clip_min), max=float(clip_max))
        w = w / w.mean().clamp(min=1e-12)
        weights[keep] = w
    # filtered or missing counts stay at 1.0

    reg = SamConceptRegistry(
        concept2idx=concept2idx,
        idx2concept=idx2concept,
        local_concepts=local_concepts,
        local_to_global=local_to_global,
        counts_full=counts_full,
        counts_eff=counts_eff,
        weights=weights,
        freq_path=str(freq_path),
        min_masks_per_concept=int(min_masks_per_concept),
        dropped_local=dropped,
    )

    if verbose and _rank0():
        kept_n = int((counts_eff > 0).sum().item())
        print("[sam-reg] Built SAM concept registry")
        print(f"  sam_prepacked_dir: {root}")
        print(f"  freq_path: {freq_path}")
        print(f"  local concepts: {local_C}")
        print(f"  global concepts (C): {C}")
        print(f"  min_masks_per_concept: {int(min_masks_per_concept)}")
        print(f"  kept (counts_eff>0): {kept_n}")
        print(f"  dropped local concepts: {len(dropped)}")
        if dropped:
            preview = list(dropped.items())[:30]
            print("  dropped preview (up to 30):")
            for n, r in preview:
                print(f"    - {n}: {r}")

        # Print kept weights
        keep_ids = torch.where(counts_eff > 0)[0].tolist()
        rows = []
        for gid in keep_ids:
            name = idx2concept[gid] if 0 <= gid < len(idx2concept) else None
            rows.append((gid, name, float(counts_full[gid].item()), float(weights[gid].item())))
        rows.sort(key=lambda x: (-x[3], x[0]))

        print("[sam-reg] ---- KEPT concepts (sorted by weight desc) ----")
        for gid, name, c, w in rows:
            print(f"[sam-reg] KEEP gid={gid:>4} name={str(name):>20} count={c:>10.0f} weight={w:.6f}")

        # Print filtered concepts too
        filt_ids = torch.where(counts_eff == 0)[0].tolist()
        rows_f = []
        for gid in filt_ids:
            name = idx2concept[gid] if 0 <= gid < len(idx2concept) else None
            rows_f.append((gid, name, float(counts_full[gid].item()), float(weights[gid].item())))
        rows_f.sort(key=lambda x: (-x[2], x[0]))

        print("[sam-reg] ---- FILTERED/NOCOUNT concepts (counts_eff==0) ----")
        for gid, name, c, w in rows_f:
            print(f"[sam-reg] FILT gid={gid:>4} name={str(name):>20} count={c:>10.0f} weight={w:.6f}")

    return reg
