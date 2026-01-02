# multimodal/sam_prepacked_loader.py
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class SamPrepackedIndex:
    """
    Index for prepacked SAM masks.

    Directory layout (produced by prepack_sam_masks.py):

      sam_prepacked/
        concept_vocab.json        # {"concepts": [name0, name1, ...]}
        sam_prepacked_index.json  # {"sub1/vid1/frame_0001.jpg": "sub1_vid1_frame_0001.pt", ...}
        *.pt                      # each: {"masks": (M,H,W) uint8, "concept_ids": (M,) int16}
    """
    root: Path
    frame_to_file: Dict[str, str]
    concepts: Tuple[str, ...]  # id -> name

    @classmethod
    def load(cls, root: str | Path) -> "SamPrepackedIndex":
        root = Path(root)
        index_path = root / "sam_prepacked_index.json"
        vocab_path = root / "concept_vocab.json"

        if not index_path.is_file():
            raise FileNotFoundError(f"Missing {index_path}")
        if not vocab_path.is_file():
            raise FileNotFoundError(f"Missing {vocab_path}")

        with index_path.open("r") as f:
            frame_to_file = json.load(f)

        with vocab_path.open("r") as f:
            vocab = json.load(f)
        concepts = tuple(vocab["concepts"])

        return cls(root=root, frame_to_file=frame_to_file, concepts=concepts)

    def get_masks_for_relpath(
        self,
        frame_relpath: str,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """
        frame_relpath must match the 'frame_relpath' used when generating masks,
        e.g. 'sub1/train_5fps/vid1/frame_000123.jpg'.

        Returns:
          masks: (M, H, W) float32 in {0., 1.}
          concept_ids: (M,) long
        or None if no masks.
        """
        fname = self.frame_to_file.get(frame_relpath)
        if fname is None:
            return None

        pt_path = self.root / fname
        if not pt_path.is_file():
            return None

        data = torch.load(pt_path, map_location="cpu")
        masks = data["masks"].float()  # uint8 -> float
        masks = (masks > 0.5).float()
        concept_ids = data["concept_ids"].long()
        return masks, concept_ids
