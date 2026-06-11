from pathlib import Path
from typing import Any, Tuple
from collections import Counter
import os
import glob
import itertools
import json
import random
import re
import shutil
import time
import hashlib
import argparse

import cv2 as cv
import imageio
from PIL import Image
import numpy as np
import pandas as pd
from gsheets import Sheets
import torch
from torch.utils.data import get_worker_info
import spacy
import clip
from torchvision import transforms as tvt
from torchvision.transforms import functional as TF, InterpolationMode

from multimodal.multimodal_data_module import (
    MultiModalDataset,
    MultiModalDataModule,
    read_vocab,
    load_data,
    load_and_print_info,
    PAD_TOKEN,
    UNK_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN_ID,
    UNK_TOKEN_ID,
    SOS_TOKEN_ID,
    EOS_TOKEN_ID,
    IMAGE_H,
    IMAGE_W,
)
from multimodal.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------
# Directories and filenames
# --------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("BABYMIND_DATA_DIR", "./expt_saycam"))
GSHEETS_CREDENTIALS_FILENAME = DATA_DIR / "credentials.json"
TRANSCRIPT_LINKS_FILENAME = DATA_DIR / "converted_SAYCam_transcript_links.csv"
TRANSCRIPTS_DIRNAME = DATA_DIR / "transcripts"
PREPROCESSED_TRANSCRIPTS_DIRNAME = DATA_DIR / "preprocessed_transcripts_5fps"
RAW_VIDEO_DIRNAME = DATA_DIR / "S"
LABELED_S_DIRNAME = DATA_DIR / "eval"
FILTERED_LABELED_S_DIRNAME = DATA_DIR / "eval_filtered"
EXTRACTED_FRAMES_DIRNAME = DATA_DIR / "train_5fps"
EVAL_FRAMES_DIRNAME = DATA_DIR / "eval"
FILTERED_EVAL_FRAMES_DIRNAME = DATA_DIR / "eval_filtered"
MANUAL_FILTERED_EVAL_FRAMES_DIRNAME = DATA_DIR / "eval_manual_filtered"
ANIMATED_FRAMES_DIRNAME = DATA_DIR / "train_animated_5fps"

TRAIN_METADATA_FILENAME = DATA_DIR / "train.json"
TRAIN_SHUFFLED_METADATA_FILENAME = DATA_DIR / "train_shuffled.json"
VAL_METADATA_FILENAME = DATA_DIR / "val.json"
TEST_METADATA_FILENAME = DATA_DIR / "test.json"

EVAL_DEV_METADATA_FILENAME = DATA_DIR / "eval_dev.json"
EVAL_TEST_METADATA_FILENAME = DATA_DIR / "eval_test.json"
FILTERED_EVAL_DEV_METADATA_FILENAME = DATA_DIR / "eval_filtered_dev.json"
FILTERED_EVAL_TEST_METADATA_FILENAME = DATA_DIR / "eval_filtered_test.json"
MANUAL_FILTERED_EVAL_TEST_METADATA_FILENAME = DATA_DIR / "eval_manual_filtered_test.json"

VOCAB_FILENAME = DATA_DIR / "vocab.json"

TRAIN_SAM_MASKS_DIRNAME = DATA_DIR / "train_sam_masks"
MAX_SAM_INSTANCES_PER_FRAME = 8

# default arguments
TRAIN_FRAC = 0.9
VAL_FRAC = 0.05

# sampling arguments
MAX_FRAMES_PER_UTTERANCE = 32

# training arguments
MULTIPLE_FRAMES = False
SHUFFLE_UTTERANCES = False

BAG_NUM_FRAMES = 1  # new default
BAG_CONTIGUOUS = False  # NEW
BAG_SORT_BY_TIME = True  # NEW


def _clip_to_uid64(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="little", signed=False)


class JointImageMaskTransform:
    """
    Jointly apply the same random geometric transforms to an image and
    a stack of binary masks with shape (K, 1, H, W).
    """

    def __init__(
        self,
        size: Tuple[int, int],
        scale: Tuple[float, float] = (1.0, 1.0),
        ratio: Tuple[float, float] = (1.0, 1.0),
        hflip_prob: float = 0.5,
    ) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.hflip_prob = hflip_prob

    def __call__(self, img: Image.Image, masks: torch.Tensor | None):
        if self.scale != (1.0, 1.0) or self.ratio != (1.0, 1.0):
            i, j, h, w = tvt.RandomResizedCrop.get_params(img, scale=self.scale, ratio=self.ratio)
            img = TF.resized_crop(
                img,
                top=i,
                left=j,
                height=h,
                width=w,
                size=self.size,
                interpolation=InterpolationMode.BILINEAR,
            )
            if masks is not None and masks.numel() > 0:
                k, _, _, _ = masks.shape
                masks_out = []
                for idx in range(k):
                    m = masks[idx]
                    m = TF.resized_crop(
                        m,
                        top=i,
                        left=j,
                        height=h,
                        width=w,
                        size=self.size,
                        interpolation=InterpolationMode.NEAREST,
                    )
                    masks_out.append(m)
                masks = torch.stack(masks_out, dim=0)

        if self.hflip_prob > 0.0 and torch.rand(()) < self.hflip_prob:
            img = TF.hflip(img)
            if masks is not None and masks.numel() > 0:
                masks = TF.hflip(masks)

        return img, masks


class MultiModalSAYCamDataset(MultiModalDataset):
    """
    Returns:
      - img: (3,H,W) if bag_num_frames==1 else (M,3,H,W)
      - utterance_idxs: (T,)
      - utterance_length: int
      - raw_utterance: [str]
      - meta: dict with optional SAM fields.
    """

    _TAIL_RE = re.compile(r"^(?P<clip>.+)_(?P<utt>\d{3})_(?P<frm>\d{2})$")

    def __init__(
        self,
        data,
        vocab,
        multiple_frames,
        transform,
        use_sam_masks: bool = False,
        sam_prepacked_dir: str | None = None,
        sam_frames_root: str | None = None,
        max_sam_instances_per_frame: int = MAX_SAM_INSTANCES_PER_FRAME,
        sam_cache_size: int = 0,
        bag_num_frames: int = 1,
        bag_contiguous: bool = BAG_CONTIGUOUS,
        bag_sort_by_time: bool = BAG_SORT_BY_TIME,
        sam_verbose_stats: bool = False,
        sam_cid_remap: torch.Tensor | None = None,
        sam_dropped_concepts: dict[str, str] | None = None,
    ):
        super().__init__(
            sam_prepacked_dir=sam_prepacked_dir if use_sam_masks else None,
            sam_frames_root=sam_frames_root if use_sam_masks else None,

            # IMPORTANT: when remap is provided, SamPrepackedIndex.load will use it
            sam_cid_remap=sam_cid_remap,
            sam_dropped_concepts=sam_dropped_concepts,

            # These should stay disabled to avoid double filtering
            sam_concept2idx=None,
            sam_min_masks_per_concept=0,
            sam_concept_frequency_json=None,

            sam_cache_size=sam_cache_size if use_sam_masks else 0,
            sam_verbose_stats=sam_verbose_stats,
        )
        self.data = data
        self.vocab = vocab
        self.multiple_frames = multiple_frames
        self.transform = transform
        self.use_sam_masks = bool(use_sam_masks)
        self.max_sam_instances = int(max_sam_instances_per_frame)
        self.bag_num_frames = int(bag_num_frames)
        self.bag_contiguous = bool(bag_contiguous)
        self.bag_sort_by_time = bool(bag_sort_by_time)

        self._mask_rng = None
        self.joint_img_mask_transform: JointImageMaskTransform | None = None

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def _safe_float(x) -> float | None:
        try:
            if x is None:
                return None
            xf = float(x)
            if np.isnan(xf):
                return None
            return xf
        except Exception:
            return None

    def _get_mask_rng(self) -> torch.Generator:
        if self._mask_rng is None:
            wi = get_worker_info()
            base_seed = wi.seed if wi is not None else torch.initial_seed()
            g = torch.Generator()
            g.manual_seed((base_seed + 1337) % (2**63 - 1))
            self._mask_rng = g
        return self._mask_rng

    def _parse_temporal_from_relname(self, rel_name: str, example: dict[str, Any]) -> tuple[str, int, int]:
        stem = Path(rel_name).stem
        m = self._TAIL_RE.match(stem)
        if m is not None:
            clip_id = m.group("clip")
            utt_idx = int(m.group("utt"))
            frm_local = int(m.group("frm"))
            return clip_id, utt_idx, frm_local

        vid = example.get("video_filename", None)
        trn = example.get("transcript_filename", None)
        if isinstance(vid, str) and len(vid) > 0:
            clip_id = Path(vid).stem
        elif isinstance(trn, str) and len(trn) > 0:
            clip_id = Path(trn).stem
        else:
            clip_id = stem
        return clip_id, 0, 0

    def _frame_idx_from_timestamps(
        self,
        timestamps: Any,
        frm_local: int,
        utt_idx: int,
        fps: float = 5.0,
    ) -> tuple[int, float | None]:
        ts_sec: float | None = None
        if isinstance(timestamps, (list, tuple)) and frm_local < len(timestamps):
            ts_sec = self._safe_float(timestamps[frm_local])
        if ts_sec is not None:
            frame_idx = int(np.floor(ts_sec * fps + 1e-6))
            return frame_idx, ts_sec
        frame_idx = int(utt_idx) * 1000 + int(frm_local)
        return frame_idx, None

    def _load_and_pad_sam(self, img_path: Path) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
          sam_mask: (Kmax,1,H,W) float32
          sam_cid:  (Kmax,) long (padded with -1)
          sam_count: int (real count)
        """
        Kmax = int(self.max_sam_instances)
        sam_mask = None
        sam_mask_concept_id = None
        sam_mask_count = 0

        if self.use_sam_masks and getattr(self, "sam_index", None) is not None:
            sam_meta = self.get_sam_meta_for_image_path(img_path)
            if sam_meta is not None:
                sam_mask = sam_meta["sam_mask"]  # (K,1,H,W)
                sam_mask_concept_id = sam_meta["sam_mask_concept_id"]  # (K,)
                sam_mask_count = int(sam_meta["sam_mask_count"].item())

                if sam_mask.shape[0] > Kmax > 0:
                    K = sam_mask.shape[0]
                    idxs = torch.randperm(K, generator=self._get_mask_rng())[:Kmax]
                    sam_mask = sam_mask[idxs]
                    sam_mask_concept_id = sam_mask_concept_id[idxs]
                    sam_mask_count = min(sam_mask_count, Kmax)

        if sam_mask is None or sam_mask.numel() == 0:
            sam_mask = torch.zeros(Kmax, 1, IMAGE_H, IMAGE_W, dtype=torch.float32)
            sam_mask_concept_id = torch.full((Kmax,), -1, dtype=torch.long)
            sam_mask_count = 0
        else:
            K = sam_mask.shape[0]
            if K < Kmax:
                pad_k = Kmax - K
                H, W = sam_mask.shape[-2], sam_mask.shape[-1]
                sam_mask = torch.cat(
                    [sam_mask, torch.zeros(pad_k, 1, H, W, dtype=sam_mask.dtype)],
                    dim=0,
                )
                sam_mask_concept_id = torch.cat(
                    [sam_mask_concept_id, torch.full((pad_k,), -1, dtype=sam_mask_concept_id.dtype)],
                    dim=0,
                )

        return sam_mask, sam_mask_concept_id, int(sam_mask_count)

    def __getitem__(self, idx: int):
        example = self.data[idx]

        # --- text ---
        utterance = example["utterance"]
        utterance_words = [SOS_TOKEN] + utterance.split() + [EOS_TOKEN]
        utterance_length = len(utterance_words)
        utterance_idxs = torch.tensor(
            [self.vocab.get(w, UNK_TOKEN_ID) for w in utterance_words],
            dtype=torch.long,
        )

        # --- frame selection (single or bag) ---
        img_filenames = example["frame_filenames"]
        M = int(self.bag_num_frames)

        if M <= 1:
            if self.multiple_frames and len(img_filenames) > 1:
                rel_name = img_filenames[int(torch.randint(0, len(img_filenames), (1,)).item())]
            else:
                rel_name = img_filenames[0]

            img_path = Path(EXTRACTED_FRAMES_DIRNAME, rel_name)
            img = Image.open(img_path).convert("RGB")

            sam_mask, sam_cid, sam_count = self._load_and_pad_sam(img_path) if self.use_sam_masks else (None, None, 0)

            if self.joint_img_mask_transform is not None and sam_mask is not None:
                img, sam_mask = self.joint_img_mask_transform(img, sam_mask)

            if self.transform is not None:
                img = self.transform(img)

            clip_id, utt_idx, frm_local = self._parse_temporal_from_relname(rel_name, example)
            clip_uid = _clip_to_uid64(clip_id)
            timestamps = example.get("timestamps", None)
            frame_idx, ts_sec = self._frame_idx_from_timestamps(timestamps, frm_local, utt_idx, fps=5.0)

            meta: dict[str, Any] = {
                "clip_id": clip_id,
                "clip_uid": clip_uid,
                "frame_idx": frame_idx,
                "frame_filename": rel_name,
                "utt_idx": utt_idx,
                "frm_local": frm_local,
            }
            if ts_sec is not None:
                meta["timestamp_sec"] = ts_sec

            if self.use_sam_masks and sam_mask is not None:
                meta["sam_mask"] = sam_mask
                meta["sam_mask_concept_id"] = sam_cid if sam_cid is not None else torch.full((sam_mask.shape[0],), -1, dtype=torch.long)
                meta["sam_mask_count"] = torch.tensor(sam_count, dtype=torch.long)
                meta["vm_concept_id"] = meta["sam_mask_concept_id"]

            return img, utterance_idxs, utterance_length, [utterance], meta

        # -------------------------
        # Bagged frames (M > 1)
        # -------------------------
        n_frames = len(img_filenames)
        if n_frames <= 0:
            raise ValueError("Example has no frame_filenames")

        # choose indices
        if n_frames >= M:
            if self.bag_contiguous:
                start_max = n_frames - M
                start = int(torch.randint(0, start_max + 1, (1,)).item()) if start_max > 0 else 0
                chosen = list(range(start, start + M))
            else:
                chosen = torch.randperm(n_frames)[:M].tolist()
        else:
            chosen = torch.randint(0, n_frames, (M,)).tolist()

        # build sortable items with temporal indices first
        items = []
        timestamps = example.get("timestamps", None)

        clip_id_final = None
        utt_idx_final = 0
        clip_uid_final = 0

        for j in chosen:
            rel_name = img_filenames[int(j)]
            clip_id, utt_idx, frm_local = self._parse_temporal_from_relname(rel_name, example)
            frame_idx, ts_sec = self._frame_idx_from_timestamps(timestamps, frm_local, utt_idx, fps=5.0)

            if clip_id_final is None:
                clip_id_final = clip_id
                utt_idx_final = int(utt_idx)
                clip_uid_final = _clip_to_uid64(clip_id_final)

            items.append(
                {
                    "rel_name": rel_name,
                    "clip_id": clip_id,
                    "utt_idx": int(utt_idx),
                    "frm_local": int(frm_local),
                    "frame_idx": int(frame_idx),
                    "timestamp_sec": float(ts_sec) if ts_sec is not None else float("nan"),
                }
            )

        if self.bag_sort_by_time:
            items.sort(key=lambda d: d["frame_idx"])

        imgs, masks, cids, counts = [], [], [], []
        frame_idxs, rel_names, frm_locals, ts_list = [], [], [], []

        for it in items:
            rel_name = it["rel_name"]
            rel_names.append(rel_name)
            frame_idxs.append(it["frame_idx"])
            frm_locals.append(it["frm_local"])
            ts_list.append(it["timestamp_sec"])

            img_path = Path(EXTRACTED_FRAMES_DIRNAME, rel_name)
            img = Image.open(img_path).convert("RGB")

            sam_mask, sam_cid, sam_count = self._load_and_pad_sam(img_path) if self.use_sam_masks else (None, None, 0)

            if self.joint_img_mask_transform is not None and sam_mask is not None:
                img, sam_mask = self.joint_img_mask_transform(img, sam_mask)

            if self.transform is not None:
                img = self.transform(img)

            imgs.append(img)

            if self.use_sam_masks and sam_mask is not None:
                masks.append(sam_mask)
                cids.append(
                    sam_cid if sam_cid is not None else torch.full((sam_mask.shape[0],), -1, dtype=torch.long)
                )
                counts.append(int(sam_count))

        img_tensor = torch.stack(imgs, dim=0)  # (M,3,H,W)

        meta: dict[str, Any] = {
            "clip_id": clip_id_final if clip_id_final is not None else "",
            "clip_uid": int(clip_uid_final),
            "utt_idx": int(utt_idx_final),
            "frame_idx": torch.tensor(frame_idxs, dtype=torch.long),  # (M,)
            "frm_local": torch.tensor(frm_locals, dtype=torch.long),  # (M,)
            "timestamp_sec": torch.tensor(ts_list, dtype=torch.float32),  # (M,)
            "frame_filename": list(rel_names),  # (M,)
        }

        if self.use_sam_masks and len(masks) > 0:
            meta["sam_mask"] = torch.stack(masks, dim=0)  # (M,K,1,H,W)
            meta["sam_mask_concept_id"] = torch.stack(cids, dim=0)  # (M,K)
            meta["sam_mask_count"] = torch.tensor(counts, dtype=torch.long)  # (M,)
            meta["vm_concept_id"] = meta["sam_mask_concept_id"]

        return img_tensor, utterance_idxs, utterance_length, [utterance], meta


class MultiModalSAYCamDataModule(MultiModalDataModule):
    """
    Adds:
      --use_sam_masks
      --bag_num_frames  (train only, when SAM is active)
    """

    def __init__(self, args=None) -> None:
        super().__init__(args)
        self.multiple_frames = self.args.get("multiple_frames", MULTIPLE_FRAMES)
        self.shuffle_utterances = self.args.get("shuffle_utterances", SHUFFLE_UTTERANCES)

        self.use_sam_masks = bool(self.args.get("use_sam_masks", False))
        # If MIL wants saliency/none masks, do not load SAM in the dataset even if --use_sam_masks was passed.
        self.mil_mask_source = str(self.args.get("mil_mask_source", "sam")).lower()
        if self.mil_mask_source != "sam":
            self.use_sam_masks = False
        self.bag_num_frames = int(self.args.get("bag_num_frames", BAG_NUM_FRAMES))
        self.bag_contiguous = bool(self.args.get("bag_contiguous", False))
        self.bag_sort_by_time = bool(self.args.get("bag_sort_by_time", True))

        if not hasattr(self, "sam_prepacked_dir"):
            self.sam_prepacked_dir = self.args.get("sam_prepacked_dir", None)
        if not hasattr(self, "sam_frames_root"):
            self.sam_frames_root = self.args.get("sam_frames_root", None)

        sam_dir_arg = self.args.get("sam_masks_dir", str(TRAIN_SAM_MASKS_DIRNAME))
        self.sam_masks_dir = Path(sam_dir_arg) if sam_dir_arg is not None else None

        self.max_sam_instances_per_frame = int(
            self.args.get("max_sam_instances_per_frame", MAX_SAM_INSTANCES_PER_FRAME)
        )

        self.sam_prepacked_cache_size = int(self.args.get("sam_prepacked_cache_size", 0))

        if self.args.get("sam_frames_root", None) is None:
            self.sam_frames_root = str(EXTRACTED_FRAMES_DIRNAME)
        if (self.args.get("sam_prepacked_dir", None) is None) and (self.sam_masks_dir is not None):
            self.sam_prepacked_dir = str(self.sam_masks_dir / "sam_prepacked")

        self.joint_img_mask_transform_train: JointImageMaskTransform | None = None
        if self.use_sam_masks:

            def _split_geom_and_other(tfm):
                if not isinstance(tfm, tvt.Compose):
                    return None, tfm
                geom = []
                other = []
                for t in tfm.transforms:
                    if isinstance(
                        t,
                        (
                            tvt.RandomResizedCrop,
                            tvt.RandomHorizontalFlip,
                            tvt.RandomRotation,
                            tvt.RandomAffine,
                            tvt.RandomCrop,
                        ),
                    ):
                        geom.append(t)
                    else:
                        other.append(t)
                other_comp = tvt.Compose(other) if other else None
                return geom, other_comp

            geom_tfms, non_geom_tfms = _split_geom_and_other(self.transform)
            if geom_tfms:
                rrc = None
                hflip = None
                for t in geom_tfms:
                    if isinstance(t, tvt.RandomResizedCrop):
                        rrc = t
                    elif isinstance(t, tvt.RandomHorizontalFlip):
                        hflip = t

                size = getattr(rrc, "size", (IMAGE_H, IMAGE_W)) if rrc is not None else (IMAGE_H, IMAGE_W)
                if isinstance(size, int):
                    size = (size, size)
                scale = getattr(rrc, "scale", (1.0, 1.0)) if rrc is not None else (1.0, 1.0)
                ratio = getattr(rrc, "ratio", (1.0, 1.0)) if rrc is not None else (1.0, 1.0)
                hflip_prob = getattr(hflip, "p", 0.0) if hflip is not None else 0.0

                self.joint_img_mask_transform_train = JointImageMaskTransform(
                    size=size,
                    scale=scale,
                    ratio=ratio,
                    hflip_prob=hflip_prob,
                )
                self.transform = non_geom_tfms

    @staticmethod
    def add_additional_to_argparse(parser):
        parser.add_argument("--multiple_frames", action="store_true")
        parser.add_argument("--shuffle_utterances", action="store_true")

        parser.add_argument("--use_sam_masks", action="store_true")
        parser.add_argument("--sam_masks_dir", type=str, default=str(TRAIN_SAM_MASKS_DIRNAME))
        parser.add_argument("--max_sam_instances_per_frame", type=int, default=MAX_SAM_INSTANCES_PER_FRAME)
        parser.add_argument("--sam_prepacked_cache_size", type=int, default=0)
        parser.add_argument("--sam_min_masks_per_concept", type=int, default=10)
        parser.add_argument("--sam_concept_frequency_json", type=str, default='expt_saycam/train_sam_masks/sam_prepacked/concept_frequency.json')
        parser.add_argument("--sam_verbose_stats", action="store_true")

        # NEW: how many frames per utterance to return (train split only when SAM is on)
        parser.add_argument("--bag_num_frames", type=int, default=BAG_NUM_FRAMES)
        parser.add_argument("--bag_contiguous", action="store_true")
        parser.add_argument("--bag_sort_by_time", action="store_true")
        parser.add_argument("--bag_no_sort_by_time", dest="bag_sort_by_time", action="store_false", help=argparse.SUPPRESS)
        parser.set_defaults(bag_sort_by_time=True)

        return parser

    @staticmethod
    def add_to_argparse(parser):
        parser = super(MultiModalSAYCamDataModule, MultiModalSAYCamDataModule).add_to_argparse(parser)
        parser = MultiModalSAYCamDataModule.add_additional_to_argparse(parser)
        return parser

    def prepare_data(self, *args, **kwargs) -> None:
        super().prepare_data(*args, **kwargs)
        _download_transcripts()
        _rename_transcripts()
        _preprocess_transcripts()
        _extract_train_frames()
        _create_train_metadata()
        _create_train_shuffled_metadata()
        _filter_eval_frames()
        _extract_eval_frames()
        _extract_filtered_eval_frames()
        _create_eval_metadata()
        _create_filtered_eval_metadata()
        _create_manual_filtered_eval_metadata()
        _create_extra_eval_metadata()
        _create_extra_filtered_eval_metadata()
        _create_vocab()

    def read_vocab(self):
        return read_vocab(VOCAB_FILENAME)

    def create_datasets(self, vocab):
        datasets = {}

        if self.shuffle_utterances:
            # use shuffled training data
            print("Training using shuffled utterances!")
            stage_splits = [
                ("train", TRAIN_SHUFFLED_METADATA_FILENAME, self.multiple_frames, self.transform),
                ("val", VAL_METADATA_FILENAME, False, self.base_transform),
                ("test", TEST_METADATA_FILENAME, False, self.base_transform),
            ]
        else:
            # use matched training data
            print("Training using matched utterances!")
            stage_splits = [
                ("train", TRAIN_METADATA_FILENAME, self.multiple_frames, self.transform),
                ("val", VAL_METADATA_FILENAME, False, self.base_transform),
                ("test", TEST_METADATA_FILENAME, False, self.base_transform),
            ]

        reg = getattr(self, "sam_registry", None)

        for split, filename, multiple_frames, transform in stage_splits:
            data = load_data(filename)

            use_sam = self.use_sam_masks if split == "train" else False
            bag_num_frames = self.bag_num_frames if (split == "train") else 1

            sam_cid_remap = None
            sam_dropped = None
            if split == "train" and use_sam and reg is not None:
                sam_cid_remap = reg.local_to_global
                sam_dropped = reg.dropped_local

            dataset = MultiModalSAYCamDataset(
                data=data,
                vocab=vocab,
                multiple_frames=multiple_frames,
                transform=transform,
                use_sam_masks=use_sam,
                sam_prepacked_dir=self.sam_prepacked_dir if use_sam else None,
                sam_frames_root=self.sam_frames_root if use_sam else None,
                sam_cid_remap=sam_cid_remap,
                sam_dropped_concepts=sam_dropped,
                max_sam_instances_per_frame=self.max_sam_instances_per_frame,
                sam_cache_size=self.sam_prepacked_cache_size if use_sam else 0,
                bag_num_frames=bag_num_frames,
                bag_contiguous=self.bag_contiguous,
                bag_sort_by_time=self.bag_sort_by_time,
                sam_verbose_stats=bool(self.args.get("sam_verbose_stats", False)),
            )

            if split == "train" and use_sam and self.joint_img_mask_transform_train is not None:
                dataset.joint_img_mask_transform = self.joint_img_mask_transform_train

            datasets[split] = dataset
            print(f"=== Loaded {split} dataset with {len(dataset)} samples. ===")

        return datasets


# ---------------------------------------------------------------------------------
# The rest of the file (data preparation / eval prep) remains unchanged below.
# ---------------------------------------------------------------------------------
# (No changes made to the preprocessing helpers.)
# ---------------------------------------------------------------------------------

def _download_transcripts():
    if os.path.exists(TRANSCRIPTS_DIRNAME):
        print("SAYCam transcripts have already been downloaded. Skipping this step.")
    else:
        print("Downloading SAYCam transcripts from Google Sheets")

        # create transcript folder
        if not os.path.exists(TRANSCRIPTS_DIRNAME):
            os.makedirs(TRANSCRIPTS_DIRNAME)

        # set up google sheets object
        sheets = Sheets.from_files(GSHEETS_CREDENTIALS_FILENAME)

        # get urls of saycam files to download
        df = pd.read_csv(TRANSCRIPT_LINKS_FILENAME)
        urls = df["GoogleSheets Link"].unique()

        for i, url in enumerate(urls):
            print(f"Downloading SAYCam transcript {i+1}/{len(urls)}: {url}")
            s = sheets.get(url)
            title = s.title.split("_")
            title = "_".join(title[:3])

            # read all sheets (skipping the first one since it is blank)
            for j in range(1, len(s.sheets)):
                try:
                    # try and parse this sheet as a data frame
                    # convert worksheet to data frame
                    df = s.sheets[j].to_frame()
                    # get filename of dataframe
                    filename = f"{TRANSCRIPTS_DIRNAME}/{title}_{s.sheets[j].title}.csv"
                    df.to_csv(filename, index=False)
                except pd.errors.ParserError:
                    continue
            time.sleep(30)


def _rename_transcripts():
    """Manually rename a few of the transcripts that don't match naming scheme."""

    if os.path.exists(TRANSCRIPTS_DIRNAME / "S_20141029_2412_part 2.csv"):
        print("Renaming transcripts")
        os.rename(
            TRANSCRIPTS_DIRNAME / "S_20141029_2412_part 2.csv",
            TRANSCRIPTS_DIRNAME / "S_20141029_2412_02.csv",
        )
        os.rename(
            TRANSCRIPTS_DIRNAME / "S_20141029_2412_part 3.csv",
            TRANSCRIPTS_DIRNAME / "S_20141029_2412_03.csv",
        )
        os.rename(
            TRANSCRIPTS_DIRNAME / "S_20141029_2412_part 4.csv",
            TRANSCRIPTS_DIRNAME / "S_20141029_2412_04.csv",
        )
        os.rename(
            TRANSCRIPTS_DIRNAME / "S_20141029_2412_part 5.csv",
            TRANSCRIPTS_DIRNAME / "S_20141029_2412_05.csv",
        )
        os.rename(
            TRANSCRIPTS_DIRNAME / "S_20141029_2412_part 6.csv",
            TRANSCRIPTS_DIRNAME / "S_20141029_2412_06.csv",
        )
    if os.path.exists(TRANSCRIPTS_DIRNAME / "S_20141122_2505_part 1.csv"):
        print("Renaming transcripts")
        os.rename(
            TRANSCRIPTS_DIRNAME / "S_20141122_2505_part 1.csv",
            TRANSCRIPTS_DIRNAME / "S_20141122_2505_01.csv",
        )
        os.rename(
            TRANSCRIPTS_DIRNAME / "S_20141122_2505_part 2.csv",
            TRANSCRIPTS_DIRNAME / "S_20141122_2505_02.csv",
        )
    else:
        print("Transcripts have already been renamed. Skipping this step.")


def _preprocess_transcripts():
    """Preprocess transcripts by cleaning the text and extracting frame timings."""

    # check if transcripts have already been downloaded
    if os.path.exists(PREPROCESSED_TRANSCRIPTS_DIRNAME):
        print("Transcripts have already been preprocessed. Skipping this step.")
    else:
        print("Preprocessing transcripts")

        # create preprocessed transcripts folder
        if not os.path.exists(PREPROCESSED_TRANSCRIPTS_DIRNAME):
            os.makedirs(PREPROCESSED_TRANSCRIPTS_DIRNAME)

        # get all transcripts and allowed speakers
        transcripts = sorted(Path(TRANSCRIPTS_DIRNAME).glob("*.csv"))
        allowed_speakers = ["M", "Mom", "mom", "m", "mother", "Mother", "papa", "the mom"]

        # build spacy model
        nlp = spacy.load("en_core_web_sm")

        # preprocess each transcript
        for transcript_idx, transcript_filename in enumerate(transcripts):
            # empty list to store processed transcript information
            preprocessed_transcript = []
            preprocessed_transcript_filename = PREPROCESSED_TRANSCRIPTS_DIRNAME / transcript_filename.name

            # read transcript CSV
            print(
                f"Preprocessing transcript: {transcript_filename.name} "
                f"({transcript_idx+1}/{len(transcripts)})"
            )
            transcript = pd.read_csv(transcript_filename)

            # skip empty transcripts
            if len(transcript) <= 1:
                continue

            # create new column of timestamps converted to seconds
            new_timestamps = convert_timestamps_to_seconds(transcript["Time"])
            transcript["Time (Seconds)"] = new_timestamps

            # reset utterance count
            utterance_num = 1

            # extract unique video filename from transcript
            video_filename = pd.unique(transcript["Video Name"])

            # drop any missing filenames, or any filenames with "part" in them
            video_filename = [x for x in video_filename if not pd.isnull(x)]
            video_filename = [x for x in video_filename if "part" not in x]

            # skip if video filename is not unique
            if len(video_filename) != 1:
                continue

            # extract video filename and replace suffix
            video_filename = video_filename[0]
            video_filename = Path(video_filename).with_suffix(".mp4")

            # check video and transcript filenames match
            assert video_filename.stem == transcript_filename.stem

            for transcript_row_idx, row in transcript.iterrows():
                # get information from current utterance
                utterance = str(row["Utterance"])  # convert to string
                speaker = str(row["Speaker"])
                start_timestamp = row["Time (Seconds)"]

                # get end timestamp
                # hack: if last timestamp, just set end timestamp to be start time
                # this means we don't have to read the video file for this to work
                if transcript_row_idx < len(transcript) - 1:
                    end_timestamp = transcript["Time (Seconds)"][transcript_row_idx + 1]
                else:
                    # this will sample a single frame for the last utterance
                    end_timestamp = start_timestamp

                # skip processing utterance if start or end timestamps are null,
                # or if speaker is not allowed
                if (
                    pd.isnull(start_timestamp)
                    or pd.isnull(end_timestamp)
                    or speaker not in allowed_speakers
                ):
                    continue

                # preprocess utterance to extract sub utterances and timestamps
                utterances, timestamps, num_frames = _preprocess_utterance(
                    nlp,
                    utterance,
                    start_timestamp,
                    end_timestamp,
                )

                if len(utterances) == 0:
                    continue

                # create dataset based on preprocessed utterances
                for curr_utterance, curr_timestamps, curr_num_frames in zip(
                    utterances, timestamps, num_frames
                ):
                    for frame_num, curr_timestamp in enumerate(curr_timestamps):
                        frame_filename = f"{video_filename.stem}_{utterance_num:03}_{frame_num:02}.jpg"
                        preprocessed_transcript.append(
                            [
                                transcript_filename.name,
                                video_filename.name,
                                curr_utterance,
                                curr_timestamp,
                                utterance_num,
                                frame_num,
                                frame_filename,
                            ]
                        )

                    utterance_num += 1

            # save preprocessed transcript as CSV
            if len(preprocessed_transcript) > 0:
                preprocessed_transcript_columns = [
                    "transcript_filename",
                    "video_filename",
                    "utterance",
                    "timestamp",
                    "utterance_num",
                    "frame_num",
                    "frame_filename",
                ]
                preprocessed_transcript_df = pd.DataFrame(
                    preprocessed_transcript,
                    columns=preprocessed_transcript_columns,
                )
                preprocessed_transcript_df.to_csv(
                    preprocessed_transcript_filename,
                    index=False,
                )


def _preprocess_utterance(nlp, utterance, start_timestamp, end_timestamp):
    """
    Preprocess a single utterance, splitting it into multiple clean utterances
    with separate timestamps.
    """

    # check start timestamp is before end timestamp
    assert start_timestamp <= end_timestamp

    # remove anything in asterisks or parentheses
    inaudible = "INAUDIBLE"

    def repl(matchobj):
        return inaudible if "inaudible" in matchobj.group(0) else ""

    utterance = re.sub(r"\*[^)]*\*", repl, utterance)
    utterance = re.sub(r"\[[^)]*\]", repl, utterance)
    utterance = re.sub(r"\([^)]*\)", repl, utterance)
    utterance = re.sub(r"\binaudible\b", repl, utterance)
    utterance = utterance.replace(r"*", "")

    # process utterance
    doc = nlp(utterance)
    utterances = [
        " ".join(
            map(
                lambda token: UNK_TOKEN if token == inaudible else token.lower(),
                map(str, sent),
            )
        )
        for sent in doc.sents
    ]

    if len(utterances) > 0:
        # get interpolated timestamps, including end timestamp (which we remove later)
        timestamps = np.linspace(
            start_timestamp,
            end_timestamp,
            len(utterances) + 1,
            endpoint=True,
        )
        timestamps = [int(timestamp) for timestamp in timestamps]
        all_timestamps = []
        num_frames = []

        # calculate number of frames to extract per utterance (max 32 frames at 5 fps)
        for i in range(len(timestamps) - 1):
            curr_num_frames = max(
                min(
                    int((timestamps[i + 1] - timestamps[i]) / 0.2),
                    MAX_FRAMES_PER_UTTERANCE,
                ),
                1,
            )
            curr_timestamps = np.linspace(
                timestamps[i],
                timestamps[i] + (curr_num_frames / 5),
                curr_num_frames,
                endpoint=False,
            )
            assert len(curr_timestamps) == curr_num_frames

            # append information
            num_frames.append(curr_num_frames)
            all_timestamps.append(curr_timestamps)

        timestamps = timestamps[:-1]  # remove end timestamp
    else:
        all_timestamps = []
        num_frames = []

    # check everything is the same length
    assert len(utterances) == len(all_timestamps)
    assert len(all_timestamps) == len(num_frames)

    return utterances, all_timestamps, num_frames


def _extract_train_frames():
    """Extract aligned frames from SAYCam videos."""

    if os.path.exists(EXTRACTED_FRAMES_DIRNAME):
        print("Training frames have already been extracted. Skipping this step.")
    else:
        print("Extracting training frames")

        if not os.path.exists(EXTRACTED_FRAMES_DIRNAME):
            os.makedirs(EXTRACTED_FRAMES_DIRNAME)

        transcripts = sorted(Path(PREPROCESSED_TRANSCRIPTS_DIRNAME).glob("*.csv"))

        for idx, transcript in enumerate(transcripts):
            transcript_df = pd.read_csv(transcript)
            video_filename = Path(
                RAW_VIDEO_DIRNAME,
                pd.unique(transcript_df["video_filename"]).item(),
            )

            if not video_filename.exists():
                print(f"{video_filename} missing! Skipping")
                continue

            print(
                f"Extracting frames: {video_filename.name} "
                f"({idx+1}/{len(transcripts)})"
            )

            # read in video and get information
            cap = cv.VideoCapture(str(video_filename))
            video_info = _get_video_info(cap)
            frame_count, frame_width, frame_height, frame_rate, frame_length = video_info

            for transcript_row_idx, row in transcript_df.iterrows():
                frame_filename = Path(EXTRACTED_FRAMES_DIRNAME, str(row["frame_filename"]))
                timestamp = float(row["timestamp"])
                framestamp = int(timestamp * frame_rate)

                cap.set(1, framestamp)
                ret, frame = cap.read()
                frame = _extract_frame(frame, frame_height, frame_width)

                # save frame
                if frame is not None:
                    cv.imwrite(str(frame_filename), frame)


def _get_video_info(cap):
    """Return video information."""
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv.CAP_PROP_FPS)  # leave this as a float
    frame_length = frame_count // frame_rate
    return frame_count, frame_width, frame_height, frame_rate, frame_length


def _extract_frame(frame, frame_height, frame_width):
    """Extract a single frame."""

    # settings for frame extraction
    resized_minor_length = 256
    new_height = frame_height * resized_minor_length // min(frame_height, frame_width)
    new_width = frame_width * resized_minor_length // min(frame_height, frame_width)

    # function to resize frame and recolor
    try:
        resized_frame = cv.resize(
            frame,
            (new_width, new_height),
            interpolation=cv.INTER_CUBIC,
        )
    except Exception as e:
        print(str(e))
        return None

    # crop
    height, width, _ = resized_frame.shape
    startx = width // 2 - (IMAGE_W // 2)
    starty = height // 2 - (IMAGE_H // 2) - 16
    cropped_frame = resized_frame[starty : starty + IMAGE_H, startx : startx + IMAGE_W]
    assert cropped_frame.shape[0] == IMAGE_H and cropped_frame.shape[1] == IMAGE_W, (
        cropped_frame.shape,
        height,
        width,
    )

    cropped_frame = np.array(cropped_frame)
    cropped_frame = cropped_frame[::-1, ::-1, :]
    # cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    return cropped_frame


def _filter_eval_frames():
    """Use CLIP to create a filtered evaluation set."""

    if os.path.exists(FILTERED_LABELED_S_DIRNAME):
        print("Evaluation frames have already been filtered. Skipping this step.")
    else:
        print("Filtering evaluation frames using CLIP")

        eval_categories = sorted(os.listdir(LABELED_S_DIRNAME))
        eval_categories.remove("carseat")
        eval_categories.remove("couch")
        eval_categories.remove("greenery")
        eval_categories.remove("plushanimal")

        os.makedirs(FILTERED_LABELED_S_DIRNAME, exist_ok=True)
        for eval_category in eval_categories:
            os.makedirs(Path(FILTERED_LABELED_S_DIRNAME) / eval_category, exist_ok=True)

        # load CLIP model
        model, preprocess = clip.load("ViT-B/16", device=device)
        model.eval()

        texts = clip.tokenize([f"{category}" for category in eval_categories]).to(device)
        text_features = model.encode_text(texts).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        for i, eval_category in enumerate(eval_categories):
            # get frames for each evaluation category
            eval_category_dir = os.path.join(LABELED_S_DIRNAME, eval_category)
            frames = glob.glob(f"{eval_category_dir}/*.jpeg")
            print(f"Filtering {len(frames)} from the category: {eval_category}")

            for frame in frames:
                I = Image.open(frame).convert("RGB")
                image = preprocess(I).unsqueeze(0).to(device)
                image_features = model.encode_image(image).float()
                image_features /= image_features.norm(dim=-1, keepdim=True)

                logits_per_text = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                pred = torch.argmax(logits_per_text, dim=-1).item()

                # copy over image frame if prediction is correct
                if pred == eval_categories.index(eval_category):
                    frame_filename = frame.split("/")[-1]
                    print(f"Copying {frame_filename} to filtered set")
                    new_frame = os.path.join(FILTERED_LABELED_S_DIRNAME, eval_category, frame_filename)
                    shutil.copyfile(frame, new_frame)


def _extract_eval_frames():
    """
    Extract evaluation frames from labeled S dataset,
    splitting evenly for dev and test.
    """

    if os.path.exists(EVAL_FRAMES_DIRNAME):
        print("Evaluation frames have already been extracted. Skipping this step.")
    else:
        print("Extracting evaluation frames")

        if not os.path.exists(EVAL_FRAMES_DIRNAME):
            os.makedirs(EVAL_FRAMES_DIRNAME)
            os.makedirs(EVAL_FRAMES_DIRNAME / "dev")
            os.makedirs(EVAL_FRAMES_DIRNAME / "test")

        eval_categories = os.listdir(LABELED_S_DIRNAME)
        for eval_category in eval_categories:
            eval_category_dirname = os.path.join(LABELED_S_DIRNAME, eval_category)
            eval_category_frames = sorted(os.listdir(eval_category_dirname))

            split_idxs = np.arange(len(eval_category_frames))
            np.random.shuffle(split_idxs)
            dev_idxs = split_idxs[: int(len(eval_category_frames) * 0.5)]
            test_idxs = split_idxs[int(len(eval_category_frames) * 0.5) :]

            assert len(dev_idxs) + len(test_idxs) == len(split_idxs)

            print(f"Copying {eval_category} frames for dev set")

            if not os.path.exists(os.path.join(EVAL_FRAMES_DIRNAME, "dev", eval_category)):
                os.makedirs(os.path.join(EVAL_FRAMES_DIRNAME, "dev", eval_category))

            for dev_idx in dev_idxs:
                original_filename = os.path.join(
                    LABELED_S_DIRNAME, eval_category, eval_category_frames[dev_idx]
                )
                shutil.copyfile(
                    original_filename,
                    os.path.join(EVAL_FRAMES_DIRNAME, "dev", eval_category, eval_category_frames[dev_idx]),
                )

            print(f"Copying {eval_category} frames for test set")

            if not os.path.exists(os.path.join(EVAL_FRAMES_DIRNAME, "test", eval_category)):
                os.makedirs(os.path.join(EVAL_FRAMES_DIRNAME, "test", eval_category))

            for test_idx in test_idxs:
                original_filename = os.path.join(
                    LABELED_S_DIRNAME, eval_category, eval_category_frames[test_idx]
                )
                shutil.copyfile(
                    original_filename,
                    os.path.join(EVAL_FRAMES_DIRNAME, "test", eval_category, eval_category_frames[test_idx]),
                )


def _extract_filtered_eval_frames():
    """
    Extract evaluation frames from CLIP filtered labeled S dataset,
    splitting evenly for dev and test.
    """

    if os.path.exists(FILTERED_EVAL_FRAMES_DIRNAME):
        print("Filtered evaluation frames have already been extracted. Skipping this step.")
    else:
        print("Extracting filtered evaluation frames")

        if not os.path.exists(FILTERED_EVAL_FRAMES_DIRNAME):
            os.makedirs(FILTERED_EVAL_FRAMES_DIRNAME)
            os.makedirs(FILTERED_EVAL_FRAMES_DIRNAME / "dev")
            os.makedirs(FILTERED_EVAL_FRAMES_DIRNAME / "test")

        eval_categories = sorted(os.listdir(FILTERED_LABELED_S_DIRNAME))
        for eval_category in eval_categories:
            eval_category_dirname = os.path.join(FILTERED_LABELED_S_DIRNAME, eval_category)
            eval_category_frames = sorted(os.listdir(eval_category_dirname))

            split_idxs = np.arange(len(eval_category_frames))
            np.random.shuffle(split_idxs)
            dev_idxs = split_idxs[: int(len(eval_category_frames) * 0.5)]
            test_idxs = split_idxs[int(len(eval_category_frames) * 0.5) :]

            assert len(dev_idxs) + len(test_idxs) == len(split_idxs)

            print(f"Copying filtered {eval_category} frames for dev set")

            if not os.path.exists(os.path.join(FILTERED_EVAL_FRAMES_DIRNAME, "dev", eval_category)):
                os.makedirs(os.path.join(FILTERED_EVAL_FRAMES_DIRNAME, "dev", eval_category))

            for dev_idx in dev_idxs:
                original_filename = os.path.join(
                    FILTERED_LABELED_S_DIRNAME, eval_category, eval_category_frames[dev_idx]
                )
                shutil.copyfile(
                    original_filename,
                    os.path.join(FILTERED_EVAL_FRAMES_DIRNAME, "dev", eval_category, eval_category_frames[dev_idx]),
                )

            print(f"Copying filtered {eval_category} frames for test set")

            if not os.path.exists(os.path.join(FILTERED_EVAL_FRAMES_DIRNAME, "test", eval_category)):
                os.makedirs(os.path.join(FILTERED_EVAL_FRAMES_DIRNAME, "test", eval_category))

            for test_idx in test_idxs:
                original_filename = os.path.join(
                    FILTERED_LABELED_S_DIRNAME, eval_category, eval_category_frames[test_idx]
                )
                shutil.copyfile(
                    original_filename,
                    os.path.join(FILTERED_EVAL_FRAMES_DIRNAME, "test", eval_category, eval_category_frames[test_idx]),
                )


def _create_train_metadata():
    """Create JSON files with image utterance information."""

    if (
        os.path.exists(TRAIN_METADATA_FILENAME)
        and os.path.exists(VAL_METADATA_FILENAME)
        and os.path.exists(TEST_METADATA_FILENAME)
    ):
        print("Training metadata files have already been created. Skipping this step.")
    else:
        print("Creating metadata files for train, validation and test.")

        transcripts = sorted(Path(PREPROCESSED_TRANSCRIPTS_DIRNAME).glob("*.csv"))

        utterances = []

        for idx, transcript in enumerate(transcripts):
            transcript_df = pd.read_csv(transcript)

            utterance_groups = transcript_df.groupby("utterance_num")
            for utterance, utterance_group in utterance_groups:
                curr_utterance = {}
                curr_utterance["utterance"] = pd.unique(utterance_group["utterance"]).item()
                curr_utterance["transcript_filename"] = pd.unique(
                    utterance_group["transcript_filename"]
                ).item()
                curr_utterance["video_filename"] = pd.unique(utterance_group["video_filename"]).item()
                curr_utterance["utterance_num"] = pd.unique(utterance_group["utterance_num"]).item()
                curr_utterance["num_frames"] = len(utterance_group)
                curr_utterance["timestamps"] = list(utterance_group["timestamp"])

                curr_utterance["frame_filenames"] = []
                curr_utterance_filenames = sorted(list(utterance_group["frame_filename"]))

                if not isinstance(curr_utterance["utterance"], str):
                    continue

                for frame_filename in curr_utterance_filenames:
                    if (EXTRACTED_FRAMES_DIRNAME / frame_filename).exists():
                        curr_utterance["frame_filenames"].append(frame_filename)
                    else:
                        print(f"{frame_filename} does not exist, removing it from this list")

                if len(curr_utterance["frame_filenames"]) == 0:
                    print("No corresponding frames found, skipping this utterance")
                    continue

                utterances.append(curr_utterance)

        random.shuffle(utterances)

        train_n = int(len(utterances) * TRAIN_FRAC)
        val_n = int(len(utterances) * VAL_FRAC)
        test_n = int(len(utterances) - train_n - val_n)
        idxs = np.arange(len(utterances))
        train_idxs = idxs[:train_n]
        val_idxs = idxs[train_n : train_n + val_n]
        test_idxs = idxs[train_n + val_n :]
        train_utterances = [utterances[i] for i in train_idxs]
        val_utterances = [utterances[i] for i in val_idxs]
        test_utterances = [utterances[i] for i in test_idxs]

        train_dict = {"data": train_utterances}
        val_dict = {"data": val_utterances}
        test_dict = {"data": test_utterances}

        with open(TRAIN_METADATA_FILENAME, "w") as f:
            json.dump(train_dict, f)

        with open(VAL_METADATA_FILENAME, "w") as f:
            json.dump(val_dict, f)

        with open(TEST_METADATA_FILENAME, "w") as f:
            json.dump(test_dict, f)


def _create_train_shuffled_metadata():
    """
    Create a JSON containing a shuffled version of the training data with
    image utterance pairs randomly paired.
    """

    if os.path.exists(TRAIN_SHUFFLED_METADATA_FILENAME):
        print("Shuffled training metadata file has already been created. Skipping this step.")
    else:
        print("Creating metadata for shuffled train.")

        with open(TRAIN_METADATA_FILENAME) as f:
            train_metadata = json.load(f)
            train_metadata = train_metadata["data"]

        utterances = [trial["utterance"] for trial in train_metadata]
        random.shuffle(utterances)

        for i, trial in enumerate(train_metadata):
            trial["utterance"] = utterances[i]

        train_shuffled_dict = {"data": train_metadata}

        with open(TRAIN_SHUFFLED_METADATA_FILENAME, "w") as f:
            json.dump(train_shuffled_dict, f)


def _create_eval_metadata():
    """Create files for evaluating multimodal SAYCam model."""

    if os.path.exists(EVAL_DEV_METADATA_FILENAME) and os.path.exists(EVAL_TEST_METADATA_FILENAME):
        print("Evaluation metadata files have already been created. Skipping this step.")
    else:
        print("Creating metadata files for evaluation.")

        n_foils = 3
        n_evaluations = 100
        eval_dev_dataset = []
        eval_test_dataset = []

        eval_dev_dirname = EVAL_FRAMES_DIRNAME / "dev"
        eval_test_dirname = EVAL_FRAMES_DIRNAME / "test"
        eval_categories = sorted(os.listdir(eval_dev_dirname))
        eval_categories.remove("carseat")
        eval_categories.remove("couch")
        eval_categories.remove("greenery")
        eval_categories.remove("plushanimal")

        for target_category in eval_categories:
            for i in range(n_evaluations):
                target_category_dirname = os.path.join(eval_dev_dirname, target_category)
                target_img_filename = os.path.join(
                    target_category_dirname,
                    np.random.choice(os.listdir(target_category_dirname)),
                )

                foil_categories = eval_categories.copy()
                foil_categories.remove(target_category)
                foil_categories = np.random.choice(foil_categories, size=n_foils, replace=False)
                foil_img_filenames = []

                for j in range(n_foils):
                    foil_category_dirname = os.path.join(eval_dev_dirname, foil_categories[j])
                    foil_img_filename = os.path.join(
                        foil_category_dirname,
                        np.random.choice(os.listdir(foil_category_dirname)),
                    )
                    foil_img_filenames.append(foil_img_filename)

                eval_trial = {}
                eval_trial["trial_num"] = i
                eval_trial["target_category"] = target_category
                eval_trial["target_img_filename"] = target_img_filename
                eval_trial["foil_categories"] = list(foil_categories)
                eval_trial["foil_img_filenames"] = foil_img_filenames
                eval_dev_dataset.append(eval_trial)

        for target_category in eval_categories:
            for i in range(n_evaluations):
                target_category_dirname = os.path.join(eval_test_dirname, target_category)
                target_img_filename = os.path.join(
                    target_category_dirname,
                    np.random.choice(os.listdir(target_category_dirname)),
                )

                foil_categories = eval_categories.copy()
                foil_categories.remove(target_category)
                foil_categories = np.random.choice(foil_categories, size=n_foils, replace=False)
                foil_img_filenames = []

                for j in range(n_foils):
                    foil_category_dirname = os.path.join(eval_test_dirname, foil_categories[j])
                    foil_img_filename = os.path.join(
                        foil_category_dirname,
                        np.random.choice(os.listdir(foil_category_dirname)),
                    )
                    foil_img_filenames.append(foil_img_filename)

                eval_trial = {}
                eval_trial["trial_num"] = i
                eval_trial["target_category"] = target_category
                eval_trial["target_img_filename"] = target_img_filename
                eval_trial["foil_categories"] = list(foil_categories)
                eval_trial["foil_img_filenames"] = foil_img_filenames
                eval_test_dataset.append(eval_trial)

        eval_dev_dict = {"data": eval_dev_dataset}
        eval_test_dict = {"data": eval_test_dataset}

        with open(EVAL_DEV_METADATA_FILENAME, "w") as f:
            json.dump(eval_dev_dict, f)

        with open(EVAL_TEST_METADATA_FILENAME, "w") as f:
            json.dump(eval_test_dict, f)


def _create_filtered_eval_metadata():
    """
    Create files for evaluating multimodal SAYCam model
    using filtered evaluation frames.
    """

    if os.path.exists(FILTERED_EVAL_DEV_METADATA_FILENAME) and os.path.exists(
        FILTERED_EVAL_TEST_METADATA_FILENAME
    ):
        print("Evaluation metadata files have already been created. Skipping this step.")
    else:
        print("Creating metadata files for evaluation using filtered evaluation frames.")

        n_foils = 3
        n_evaluations = 100
        eval_dev_dataset = []
        eval_test_dataset = []

        eval_dev_dirname = FILTERED_EVAL_FRAMES_DIRNAME / "dev"
        eval_test_dirname = FILTERED_EVAL_FRAMES_DIRNAME / "test"
        eval_categories = sorted(os.listdir(eval_dev_dirname))

        for target_category in eval_categories:
            for i in range(n_evaluations):
                target_category_dirname = os.path.join(eval_dev_dirname, target_category)
                target_img_filename = os.path.join(
                    target_category_dirname,
                    np.random.choice(os.listdir(target_category_dirname)),
                )

                foil_categories = eval_categories.copy()
                foil_categories.remove(target_category)
                foil_categories = np.random.choice(foil_categories, size=n_foils, replace=False)
                foil_img_filenames = []

                for j in range(n_foils):
                    foil_category_dirname = os.path.join(eval_dev_dirname, foil_categories[j])
                    foil_img_filename = os.path.join(
                        foil_category_dirname,
                        np.random.choice(os.listdir(foil_category_dirname)),
                    )
                    foil_img_filenames.append(foil_img_filename)

                eval_trial = {}
                eval_trial["trial_num"] = i
                eval_trial["target_category"] = target_category
                eval_trial["target_img_filename"] = target_img_filename
                eval_trial["foil_categories"] = list(foil_categories)
                eval_trial["foil_img_filenames"] = foil_img_filenames
                eval_dev_dataset.append(eval_trial)

        for target_category in eval_categories:
            for i in range(n_evaluations):
                target_category_dirname = os.path.join(eval_test_dirname, target_category)
                target_img_filename = os.path.join(
                    target_category_dirname,
                    np.random.choice(os.listdir(target_category_dirname)),
                )

                foil_categories = eval_categories.copy()
                foil_categories.remove(target_category)
                foil_categories = np.random.choice(foil_categories, size=n_foils, replace=False)
                foil_img_filenames = []

                for j in range(n_foils):
                    foil_category_dirname = os.path.join(eval_test_dirname, foil_categories[j])
                    foil_img_filename = os.path.join(
                        foil_category_dirname,
                        np.random.choice(os.listdir(foil_category_dirname)),
                    )
                    foil_img_filenames.append(foil_img_filename)

                eval_trial = {}
                eval_trial["trial_num"] = i
                eval_trial["target_category"] = target_category
                eval_trial["target_img_filename"] = target_img_filename
                eval_trial["foil_categories"] = list(foil_categories)
                eval_trial["foil_img_filenames"] = foil_img_filenames
                eval_test_dataset.append(eval_trial)

        eval_dev_dict = {"data": eval_dev_dataset}
        eval_test_dict = {"data": eval_test_dataset}

        with open(FILTERED_EVAL_DEV_METADATA_FILENAME, "w") as f:
            json.dump(eval_dev_dict, f)

        with open(FILTERED_EVAL_TEST_METADATA_FILENAME, "w") as f:
            json.dump(eval_test_dict, f)


def _create_manual_filtered_eval_metadata():
    """
    Create files for evaluating multimodal SAYCam model using manually
    filtered evaluation frames.

    This evaluation only contains 15 rather than 22 categories since scene
    and overlapping categories are removed, and only contains a test folder.
    """

    if os.path.exists(MANUAL_FILTERED_EVAL_TEST_METADATA_FILENAME):
        print(
            "Manual filtered evaluation metadata files have already been created. "
            "Skipping this step."
        )
    else:
        print("Creating metadata files for evaluation using manually filtered evaluation frames.")

        n_foils = 3
        n_evaluations = 100
        eval_test_dataset = []

        eval_test_dirname = MANUAL_FILTERED_EVAL_FRAMES_DIRNAME / "test"
        eval_categories = sorted(os.listdir(eval_test_dirname))
        print(eval_categories)

        for target_category in eval_categories:
            for i in range(n_evaluations):
                target_category_dirname = os.path.join(eval_test_dirname, target_category)
                target_img_filename = os.path.join(
                    target_category_dirname,
                    np.random.choice(os.listdir(target_category_dirname)),
                )

                foil_categories = eval_categories.copy()
                foil_categories.remove(target_category)
                foil_categories = np.random.choice(foil_categories, size=n_foils, replace=False)
                foil_img_filenames = []

                for j in range(n_foils):
                    foil_category_dirname = os.path.join(eval_test_dirname, foil_categories[j])
                    foil_img_filename = os.path.join(
                        foil_category_dirname,
                        np.random.choice(os.listdir(foil_category_dirname)),
                    )
                    foil_img_filenames.append(foil_img_filename)

                eval_trial = {}
                eval_trial["trial_num"] = i
                eval_trial["target_category"] = target_category
                eval_trial["target_img_filename"] = target_img_filename
                eval_trial["foil_categories"] = list(foil_categories)
                eval_trial["foil_img_filenames"] = foil_img_filenames
                eval_test_dataset.append(eval_trial)

        eval_test_dict = {"data": eval_test_dataset}

        with open(MANUAL_FILTERED_EVAL_TEST_METADATA_FILENAME, "w") as f:
            json.dump(eval_test_dict, f)


def _generate_eval_trial(idx, stage, target_category, n_foils, eval_categories, eval_root):
    """Generate a single evaluation trial with one category label and N images."""
    eval_dirname = eval_root / f"{stage}"

    target_category_dirname = os.path.join(eval_dirname, target_category)
    target_img_filename = os.path.join(
        target_category_dirname,
        np.random.choice(os.listdir(target_category_dirname)),
    )

    foil_categories = eval_categories.copy()
    foil_categories.remove(target_category)
    foil_categories = np.random.choice(foil_categories, size=n_foils, replace=False)
    foil_img_filenames = []

    for i in range(n_foils):
        foil_category_dirname = os.path.join(eval_dirname, foil_categories[i])
        foil_img_filename = os.path.join(
            foil_category_dirname,
            np.random.choice(os.listdir(foil_category_dirname)),
        )
        foil_img_filenames.append(foil_img_filename)

    eval_trial = {}
    eval_trial["trial_num"] = idx
    eval_trial["target_category"] = target_category
    eval_trial["target_img_filename"] = target_img_filename
    eval_trial["foil_categories"] = list(foil_categories)
    eval_trial["foil_img_filenames"] = foil_img_filenames
    return eval_trial


def _create_extra_eval_metadata():
    """
    Create extra splits for evaluating Multimodal SAYCam models
    using 10 or 22 possible images per trial.
    """
    extra_files = [
        DATA_DIR / "eval_dev_9_foils.json",
        DATA_DIR / "eval_dev_21_foils.json",
        DATA_DIR / "eval_test_9_foils.json",
        DATA_DIR / "eval_test_21_foils.json",
    ]
    if all(p.exists() for p in extra_files):
        print("Extra evaluation metadata files have already been created. Skipping this step.")
        return

    print("Creating extra metadata files for evaluation.")

    stages = ["dev", "test"]
    n_foils_list = [9, 21]
    conds = itertools.product(stages, n_foils_list)
    n_evaluations = 100

    eval_dev_dirname = EVAL_FRAMES_DIRNAME / "dev"
    eval_categories = sorted(os.listdir(eval_dev_dirname))
    eval_categories.remove("carseat")
    eval_categories.remove("couch")
    eval_categories.remove("greenery")
    eval_categories.remove("plushanimal")

    for stage, n_foil in conds:
        eval_dataset = []
        for target_category in eval_categories:
            for i in range(n_evaluations):
                eval_trial = _generate_eval_trial(
                    i,
                    stage,
                    target_category,
                    n_foil,
                    eval_categories,
                    EVAL_FRAMES_DIRNAME,
                )
                eval_dataset.append(eval_trial)

        eval_dict = {"data": eval_dataset}

        with open(DATA_DIR / f"eval_{stage}_{n_foil}_foils.json", "w") as f:
            json.dump(eval_dict, f)


def _create_extra_filtered_eval_metadata():
    """
    Create extra splits for evaluating Multimodal SAYCam models with filtered frames,
    using 10 or 22 possible images per trial.
    """
    extra_files = [
        DATA_DIR / "eval_filtered_dev_9_foils.json",
        DATA_DIR / "eval_filtered_dev_21_foils.json",
        DATA_DIR / "eval_filtered_test_9_foils.json",
        DATA_DIR / "eval_filtered_test_21_foils.json",
    ]
    if all(p.exists() for p in extra_files):
        print("Extra filtered evaluation metadata files have already been created. Skipping this step.")
        return

    print("Creating extra metadata files for evaluation using filtered evaluation frames.")

    stages = ["dev", "test"]
    n_foils_list = [9, 21]
    conds = itertools.product(stages, n_foils_list)
    n_evaluations = 100

    eval_dev_dirname = FILTERED_EVAL_FRAMES_DIRNAME / "dev"
    eval_categories = sorted(os.listdir(eval_dev_dirname))

    for stage, n_foil in conds:
        eval_dataset = []
        for target_category in eval_categories:
            for i in range(n_evaluations):
                eval_trial = _generate_eval_trial(
                    i,
                    stage,
                    target_category,
                    n_foil,
                    eval_categories,
                    FILTERED_EVAL_FRAMES_DIRNAME,
                )
                eval_dataset.append(eval_trial)

        eval_dict = {"data": eval_dataset}

        with open(DATA_DIR / f"eval_filtered_{stage}_{n_foil}_foils.json", "w") as f:
            json.dump(eval_dict, f)


def _create_vocab(freq_threshold=3):
    """Create vocabulary object and save to file."""

    if VOCAB_FILENAME.exists():
        print("Vocabulary file already exists. Skipping this step.")
    else:
        print("Creating vocab.json file!")

        counter = Counter()

        with open(TRAIN_METADATA_FILENAME) as f:
            train_dict = json.load(f)

        for example in train_dict["data"]:
            utterance = example["utterance"]
            tokens = utterance.split()
            counter.update(tokens)

        vocab = sorted(counter.most_common(), key=lambda item: (-item[1], item[0]))

        special_token_and_ids = [
            (PAD_TOKEN, PAD_TOKEN_ID),
            (UNK_TOKEN, UNK_TOKEN_ID),
            (SOS_TOKEN, SOS_TOKEN_ID),
            (EOS_TOKEN, EOS_TOKEN_ID),
        ]
        special_tokens = [token for token, token_id in special_token_and_ids]
        vocab = special_tokens + [
            token
            for token, freq in vocab
            if token not in special_tokens and freq >= freq_threshold
        ]
        for token, token_id in special_token_and_ids:
            assert vocab[token_id] == token

        vocab_dict = {token: idx for idx, token in enumerate(vocab)}

        with open(VOCAB_FILENAME, "w") as f:
            json.dump(vocab_dict, f)


def _create_animations():
    """Create animated GIFs of extracted frames paired with each utterance."""

    if os.path.exists(ANIMATED_FRAMES_DIRNAME):
        print("Animated gifs have already been created. Skipping this step.")
    else:
        print("Creating animated gifs")

        if not os.path.exists(ANIMATED_FRAMES_DIRNAME):
            os.makedirs(ANIMATED_FRAMES_DIRNAME)

        transcripts = sorted(Path(PREPROCESSED_TRANSCRIPTS_DIRNAME).glob("*.csv"))[:5]

        for idx, transcript in enumerate(transcripts):
            print(f"Creating animated gifs: {transcript} ({idx+1}/{len(transcripts)})")

            transcript_df = pd.read_csv(transcript)
            utterance_groups = transcript_df.groupby("utterance_num")

            for utterance, utterance_group in utterance_groups:
                utterance_num = pd.unique(utterance_group["utterance_num"]).item()
                gif_filename = (
                    f"{pd.unique(utterance_group['transcript_filename']).item()[:-4]}_"
                    f"{utterance_num:03}.gif"
                )
                gif_filepath = Path(ANIMATED_FRAMES_DIRNAME, gif_filename)
                frame_filenames = utterance_group["frame_filename"]

                frames = []
                for frame_filename in frame_filenames:
                    frame_filepath = EXTRACTED_FRAMES_DIRNAME / frame_filename
                    try:
                        img = imageio.imread(frame_filepath)
                    except FileNotFoundError:
                        continue
                    frames.append(img)

                if len(frames) > 0:
                    print(f"Saving {gif_filepath}, with {len(frames)} frames")
                    imageio.mimsave(gif_filepath, frames, fps=10)


if __name__ == "__main__":
    load_and_print_info(MultiModalSAYCamDataModule)