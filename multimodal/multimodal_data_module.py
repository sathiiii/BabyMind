from pathlib import Path
from typing import Any, Tuple, Optional, Dict, List, Union
import json
import argparse
import os
import random
from dataclasses import dataclass, field

from collections import defaultdict
import numpy as np
from PIL import Image
from torchvision import transforms

import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import torch
import pytorch_lightning as pl

from multimodal.utils import GaussianBlur
import clip

from multimodal.sam_concept_registry import build_sam_concept_registry


# directories and filenames
# must be consistent with multimodal_saycam_data_module
EVAL_DATA_DIR = Path(os.environ.get("BABYMIND_DATA_DIR", "./expt_saycam"))
EVAL_METADATA_FILENAME = "eval_dev.json"

# default arguments
# dataloader arguments
BATCH_SIZE = 4
VAL_BATCH_SIZE = 16
NUM_WORKERS = 4
EVAL_INCLUDE_SOS_EOS = False

# evaluation arguments
N_VAL_DATALOADERS_PER_SPLIT = 2
TEST_WHILE_VAL = False
EVAL_TYPE = "image"

# sampling arguments
MAX_LEN_UTTERANCE = 25

# training arguments
AUGMENT_FRAMES = False

# special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
SOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3

# image arguments
IMAGE_H = 224
IMAGE_W = 224

# image transforms
normalizer = transforms.Normalize(
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225],
)

CLIP_EVAL = False


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    if dist.is_available() and dist.is_initialized():
        worker_seed = (worker_seed + dist.get_rank()) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------
# SAM prepacked index
# ---------------------------------------------------------------------
@dataclass
class SamPrepackedIndex:
    """
    Index for prepacked SAM masks.

    Adds:
      - Optional min_masks_per_concept (uses concept_frequency.json)
      - Optional verbose summary on load
      - Lightweight runtime stats counters (per process/worker)
    """
    root: Path
    frame_to_file: Dict[str, str]
    concepts: Tuple[str, ...]
    cid_remap: Optional[torch.Tensor] = None

    cache_size: int = 0
    _cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False
    )

    # New: optional stats
    _stats: Dict[str, int] = field(default_factory=lambda: defaultdict(int), init=False, repr=False)
    _dropped_concepts: Dict[str, str] = field(default_factory=dict, init=False, repr=False)  # name -> reason

    @staticmethod
    def _load_frequency_counts(freq_path: Path) -> Dict[str, int]:
        """
        Reads concept_frequency.json and returns a dict: {concept_lower: count_int}.
        Tries keys in this order:
          - counts_packed_successfully
          - counts_parsed_from_filenames
          - counts
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
                kk = str(k).strip().lower()
                out[kk] = int(v)
            except Exception:
                continue
        return out

    @classmethod
    def load(
        cls,
        root: Union[str, Path],
        concept2idx: Optional[Dict[str, int]] = None,
        cache_size: int = 0,
        *,
        min_masks_per_concept: int = 0,
        concept_frequency_json: Optional[Union[str, Path]] = None,
        cid_remap: Optional[torch.Tensor] = None,   # NEW
        dropped_concepts: Optional[Dict[str, str]] = None,  # NEW
        verbose: bool = False,
    ) -> "SamPrepackedIndex":
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

        frame_to_file_raw = frame_to_file
        frame_to_file = {}

        for k, v in frame_to_file_raw.items():
            ks = str(k)
            vs = str(v)

            # original key
            frame_to_file[ks] = vs

            p = Path(ks)
            if p.suffix == "":
                # index stored a stem like "S_...._26"
                frame_to_file[ks + ".jpg"] = vs
            else:
                # index stored "S_...._26.jpg" (not your case, but make it robust)
                frame_to_file[p.stem] = vs

        concepts_list = vocab.get("concepts", [])
        concepts = tuple(concepts_list)

        # NEW: if caller provides cid_remap, use it and skip internal filtering
        if cid_remap is not None:
            cid_remap_t = torch.as_tensor(cid_remap, dtype=torch.long)
            if int(cid_remap_t.numel()) != int(len(concepts)):
                raise ValueError(
                    f"cid_remap length mismatch: got {cid_remap_t.numel()} expected {len(concepts)}"
                )
            inst = cls(
                root=root,
                frame_to_file=frame_to_file,
                concepts=concepts,
                cid_remap=cid_remap_t,
                cache_size=int(cache_size),
            )
            inst._dropped_concepts = dropped_concepts or {}

            if verbose:
                total_frames = len(frame_to_file)
                total_concepts = len(concepts)
                kept_concepts = int((cid_remap_t >= 0).sum().item())
                print("[SAM] SamPrepackedIndex loaded (precomputed cid_remap)")
                print(f"  root: {root}")
                print(f"  frames in index: {total_frames}")
                print(f"  concepts in vocab: {total_concepts}")
                print(f"  kept concepts: {kept_concepts}")
                print(f"  dropped concepts: {len(inst._dropped_concepts)}")
            return inst

        # Load counts if we want frequency filtering
        freq_counts: Dict[str, int] = {}
        if int(min_masks_per_concept) > 0:
            freq_path = Path(concept_frequency_json) if concept_frequency_json is not None else (root / "concept_frequency.json")
            freq_counts = cls._load_frequency_counts(freq_path)

        # Build remap:
        # - if concept2idx is provided: local_id -> global_id (or -1 if unknown)
        # - if not provided: local_id -> local_id
        remap: List[int] = []
        dropped: Dict[str, str] = {}

        for local_id, name in enumerate(concepts):
            name_l = str(name).strip().lower()

            if concept2idx is not None:
                gid = int(concept2idx.get(name_l, -1))
                if gid < 0:
                    dropped[name_l] = "not_in_concept2idx"
            else:
                gid = int(local_id)

            if int(min_masks_per_concept) > 0 and freq_counts:
                c = int(freq_counts.get(name_l, 0))
                if c < int(min_masks_per_concept):
                    gid = -1
                    dropped[name_l] = f"freq<{int(min_masks_per_concept)} (count={c})"

            remap.append(int(gid))

        cid_remap = torch.tensor(remap, dtype=torch.long)

        inst = cls(
            root=root,
            frame_to_file=frame_to_file,
            concepts=concepts,
            cid_remap=cid_remap,
            cache_size=int(cache_size),
        )
        inst._dropped_concepts = dropped

        if verbose:
            total_frames = len(frame_to_file)
            total_concepts = len(concepts)
            kept_concepts = sum(1 for x in remap if int(x) >= 0)
            print("[SAM] SamPrepackedIndex loaded")
            print(f"  root: {root}")
            print(f"  frames in index: {total_frames}")
            print(f"  concepts in vocab: {total_concepts}")
            print(f"  kept concepts: {kept_concepts}")
            print(f"  dropped concepts: {len(dropped)}")
            if int(min_masks_per_concept) > 0:
                print(f"  min_masks_per_concept: {int(min_masks_per_concept)}")
            if dropped:
                # Print a small preview
                preview = list(dropped.items())[:30]
                print("  dropped preview (up to 30):")
                for n, r in preview:
                    print(f"    - {n}: {r}")

        return inst

    def report_stats(self, *, reset: bool = False) -> Dict[str, int]:
        out = dict(self._stats)
        if reset:
            self._stats.clear()
        return out

    def get_masks_for_relpath(
        self,
        frame_relpath: str,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
          masks: (M,H,W) float32 in {0.,1.}
          concept_ids: (M,) long (after remap, if enabled)
        """
        self._stats["frames_requested"] += 1

        if self.cache_size > 0 and frame_relpath in self._cache:
            self._stats["frames_cache_hit"] += 1
            masks, concept_ids = self._cache[frame_relpath]
            return masks.clone(), concept_ids.clone()

        fname = self.frame_to_file.get(frame_relpath)
        if fname is None:
            self._stats["frames_not_in_index"] += 1
            return None

        pt_path = self.root / fname
        if not pt_path.is_file():
            self._stats["frames_missing_pt"] += 1
            return None

        self._stats["frames_loaded_pt"] += 1

        data = torch.load(pt_path, map_location="cpu")
        masks = data["masks"].float()
        masks = (masks > 0.5).float()
        concept_ids = data["concept_ids"].long()

        self._stats["masks_total_before_filter"] += int(concept_ids.numel())

        if self.cid_remap is not None:
            mapped = self.cid_remap[concept_ids]

            # Stats only (do NOT filter masks)
            dropped_n = int((mapped < 0).sum().item())
            kept_n = int(mapped.numel()) - dropped_n
            self._stats["masks_kept"] += kept_n
            self._stats["masks_dropped"] += dropped_n
            if kept_n == 0:
                self._stats["frames_all_masks_dropped"] += 1  # informational only

            # KEEP ALL masks; concept id may be -1 (unknown)
            concept_ids = mapped
        else:
            self._stats["masks_kept"] += int(concept_ids.numel())

        if masks.numel() == 0:
            self._stats["frames_empty_masks"] += 1
            return None

        if self.cache_size > 0:
            self._cache[frame_relpath] = (masks.clone(), concept_ids.clone())
            if len(self._cache) > self.cache_size:
                oldest_key = next(iter(self._cache.keys()))
                del self._cache[oldest_key]

        return masks, concept_ids


def read_vocab(vocab_filename):
    with open(vocab_filename) as f:
        return json.load(f)


def load_data(filename):
    with open(filename) as f:
        data = json.load(f)
    return data["data"]


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class MultiModalDataset(Dataset):
    """Abstract Dataset that returns paired image-utterances."""

    def __init__(
        self,
        sam_prepacked_dir: Optional[str] = None,
        sam_frames_root: Optional[str] = None,
        sam_concept2idx: Optional[Dict[str, int]] = None,
        sam_cache_size: int = 0,
        sam_min_masks_per_concept: int = 0,
        sam_concept_frequency_json: Optional[str] = None,
        sam_verbose_stats: bool = False,
        sam_cid_remap: Optional[torch.Tensor] = None,
        sam_dropped_concepts: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.sam_index: Optional[SamPrepackedIndex] = None
        self.sam_frames_root: Optional[Path] = None

        if sam_prepacked_dir is not None:
            self.sam_index = SamPrepackedIndex.load(
                sam_prepacked_dir,
                concept2idx=sam_concept2idx,
                cache_size=sam_cache_size,
                min_masks_per_concept=int(sam_min_masks_per_concept),
                concept_frequency_json=sam_concept_frequency_json,
                cid_remap=sam_cid_remap,
                dropped_concepts=sam_dropped_concepts,
                verbose=bool(sam_verbose_stats),
            )

        if sam_frames_root is not None:
            self.sam_frames_root = Path(sam_frames_root)

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any]:
        raise NotImplementedError

    def get_sam_meta_for_image_path(
        self,
        image_path: Union[str, Path],
    ) -> Optional[Dict[str, Any]]:
        if self.sam_index is None or self.sam_frames_root is None:
            return None

        image_path = Path(image_path)
        try:
            frame_relpath = str(image_path.relative_to(self.sam_frames_root))
        except ValueError:
            frame_relpath = os.path.relpath(str(image_path), str(self.sam_frames_root))

        out = self.sam_index.get_masks_for_relpath(frame_relpath)
        if out is None:
            return None

        masks, concept_ids = out
        if masks.numel() == 0:
            return None

        if masks.dim() == 3:
            masks = masks.unsqueeze(1)  # (K,1,H,W)
        elif masks.dim() == 4 and masks.shape[1] == 1:
            pass
        else:
            raise ValueError(f"Expected SAM masks (K,H,W) or (K,1,H,W), got {masks.shape}")

        meta = {
            "sam_mask": masks,
            "sam_mask_concept_id": concept_ids,
            "sam_mask_count": torch.tensor(masks.shape[0], dtype=torch.long),
        }
        return meta


def multiModalDataset_collate_fn(batch):
    """
    Collate function that supports:
      - (img, idxs, len, raw) legacy
      - (img, idxs, len, raw, meta_dict)
      - ((img, idxs, len, raw), meta_dict)

    Supports:
      - img: (3,H,W) or (M,3,H,W) -> batched to (B,3,H,W) or (B,M,3,H,W)
      - meta["sam_mask"]: (K,1,H,W) or (M,K,1,H,W) -> batched to (B,M,K,1,H,W) (always when present)
      - meta["sam_mask_concept_id"]: (K,) or (M,K) -> batched to (B,M,K)
      - meta["sam_mask_count"]: scalar or (M,) -> batched to (B,M)
      - meta["frame_idx"]: scalar or (M,) -> batched to (B,M)
    """

    first = batch[0]

    def _pack_core(img, utterance_idxs, utterance_length, raw_utterance):
        img = torch.stack(img, 0)

        utterance_idxs = pad_sequence(
            utterance_idxs,
            batch_first=True,
            padding_value=PAD_TOKEN_ID,
        )
        utterance_length = torch.tensor(utterance_length, dtype=torch.long)

        if utterance_idxs.size(1) > MAX_LEN_UTTERANCE:
            utterance_idxs = utterance_idxs[:, :MAX_LEN_UTTERANCE]
            utterance_length = torch.minimum(
                utterance_length,
                torch.tensor(MAX_LEN_UTTERANCE, dtype=torch.long),
            )

        raw_utterance = list(raw_utterance)
        return img, utterance_idxs, utterance_length, raw_utterance

    def _to_1d_long(v) -> torch.Tensor:
        if torch.is_tensor(v):
            t = v.detach().to(dtype=torch.long).view(-1)
            return t
        if isinstance(v, (list, tuple, np.ndarray)):
            return torch.tensor([int(x) for x in v], dtype=torch.long).view(-1)
        return torch.tensor([int(v)], dtype=torch.long)

    def _to_1d_float(v) -> torch.Tensor:
        if torch.is_tensor(v):
            return v.detach().to(dtype=torch.float32).view(-1)
        if isinstance(v, (list, tuple, np.ndarray)):
            return torch.tensor([float(x) for x in v], dtype=torch.float32).view(-1)
        return torch.tensor([float(v)], dtype=torch.float32)

    def _batch_meta(meta_seq: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(meta_seq) == 0:
            return {}

        B = len(meta_seq)
        all_keys = set()
        for m in meta_seq:
            all_keys.update(m.keys())

        meta_out: Dict[str, Any] = {}

        # Keep clip_id as a python object list (string or list[str])
        if "clip_id" in all_keys:
            meta_out["clip_id"] = [m.get("clip_id") for m in meta_seq]

        # frame_idx: scalar or (M,)
        if "frame_idx" in all_keys:
            fi_list = [_to_1d_long(m.get("frame_idx", -1)) for m in meta_seq]
            Mmax = max(int(t.numel()) for t in fi_list) if fi_list else 1
            fi_pad = torch.full((B, Mmax), -1, dtype=torch.long)
            for i, t in enumerate(fi_list):
                mi = int(t.numel())
                fi_pad[i, :mi] = t
            meta_out["frame_idx"] = fi_pad

        # frame_filename: keep list[str] or str as python objects
        if "frame_filename" in all_keys:
            meta_out["frame_filename"] = [m.get("frame_filename", "") for m in meta_seq]

        # SAM masks: normalize to (Mi,Ki,1,H,W) and pad to (B,Mmax,Kmax,1,H,W)
        if "sam_mask" in all_keys:
            raw_masks = [m.get("sam_mask") for m in meta_seq]
            proto = None
            for sm in raw_masks:
                if sm is not None:
                    proto = sm
                    break
            if proto is None:
                meta_out["sam_mask"] = torch.empty(B, 0, 0, 1, IMAGE_H, IMAGE_W, dtype=torch.float32)
                meta_out["sam_mask_concept_id"] = torch.empty(B, 0, 0, dtype=torch.long)
                meta_out["sam_mask_count"] = torch.empty(B, 0, dtype=torch.long)
            else:
                norm_masks: List[torch.Tensor] = []
                Mi_list: List[int] = []
                Ki_list: List[int] = []

                # Determine H,W,C
                if proto.dim() == 4:  # (K,1,H,W)
                    _, C, H, W = proto.shape
                elif proto.dim() == 5:  # (M,K,1,H,W)
                    _, _, C, H, W = proto.shape
                else:
                    raise ValueError(f"sam_mask must be 4D or 5D, got {proto.shape}")

                for sm in raw_masks:
                    if sm is None:
                        t = torch.empty(0, 0, C, H, W, dtype=torch.float32)
                    else:
                        t = sm
                        if t.dim() == 4:
                            t = t.unsqueeze(0)  # (1,K,1,H,W)
                        if t.dim() != 5:
                            raise ValueError(f"sam_mask must be 4D or 5D per example, got {t.shape}")
                        t = t.to(dtype=torch.float32)
                    norm_masks.append(t)
                    Mi_list.append(int(t.size(0)))
                    Ki_list.append(int(t.size(1)) if t.numel() > 0 else 0)

                Mmax = max(Mi_list) if Mi_list else 1
                Kmax = max(Ki_list) if Ki_list else 0

                padded = torch.zeros(B, Mmax, Kmax, C, H, W, dtype=torch.float32)
                for i, t in enumerate(norm_masks):
                    Mi, Ki = int(t.size(0)), int(t.size(1)) if t.numel() > 0 else 0
                    if Mi > 0 and Ki > 0:
                        padded[i, :Mi, :Ki] = t
                meta_out["sam_mask"] = padded

                # concept ids
                if "sam_mask_concept_id" in all_keys:
                    raw_cids = [m.get("sam_mask_concept_id") for m in meta_seq]
                    norm_cids: List[torch.Tensor] = []
                    for cid in raw_cids:
                        if cid is None:
                            norm_cids.append(torch.empty(0, 0, dtype=torch.long))
                        else:
                            t = cid
                            if torch.is_tensor(t):
                                t = t.detach().to(dtype=torch.long)
                            else:
                                t = torch.as_tensor(t, dtype=torch.long)
                            if t.dim() == 1:
                                t = t.unsqueeze(0)  # (1,K)
                            if t.dim() != 2:
                                raise ValueError(f"sam_mask_concept_id must be 1D or 2D per example, got {t.shape}")
                            norm_cids.append(t)
                    cids_pad = torch.full((B, Mmax, Kmax), -1, dtype=torch.long)
                    for i, t in enumerate(norm_cids):
                        Mi, Ki = int(t.size(0)), int(t.size(1)) if t.numel() > 0 else 0
                        if Mi > 0 and Ki > 0:
                            cids_pad[i, :Mi, :Ki] = t
                    meta_out["sam_mask_concept_id"] = cids_pad

                # mask counts
                if "sam_mask_count" in all_keys:
                    raw_cnt = [m.get("sam_mask_count") for m in meta_seq]
                    cnt_list = [_to_1d_long(c if c is not None else 0) for c in raw_cnt]
                    cnt_pad = torch.zeros(B, Mmax, dtype=torch.long)
                    for i, t in enumerate(cnt_list):
                        Mi = int(min(t.numel(), Mmax))
                        if Mi > 0:
                            cnt_pad[i, :Mi] = t[:Mi]
                    meta_out["sam_mask_count"] = cnt_pad
                else:
                    # fall back to Ki_list as counts (per frame unknown, so set all frames to Ki)
                    cnt_pad = torch.zeros(B, Mmax, dtype=torch.long)
                    for i in range(B):
                        if Ki_list[i] > 0:
                            cnt_pad[i, :Mi_list[i]] = Ki_list[i]
                    meta_out["sam_mask_count"] = cnt_pad

                # alias for legacy key
                meta_out["vm_concept_id"] = meta_out.get("sam_mask_concept_id")

        return meta_out

    # Case A: flat 5-tuples
    if isinstance(first, tuple) and len(first) == 5 and isinstance(first[4], dict):
        img, utterance_idxs, utterance_length, raw_utterance, meta = zip(*batch)
        img, utterance_idxs, utterance_length, raw_utterance = _pack_core(
            img, utterance_idxs, utterance_length, raw_utterance
        )
        meta_out = _batch_meta(list(meta))
        return img, utterance_idxs, utterance_length, raw_utterance, meta_out

    # Case B: nested ((core), meta)
    if (
        isinstance(first, tuple)
        and len(first) == 2
        and isinstance(first[0], (tuple, list))
        and isinstance(first[1], dict)
    ):
        core, meta = zip(*batch)
        img, utterance_idxs, utterance_length, raw_utterance = zip(*core)
        img, utterance_idxs, utterance_length, raw_utterance = _pack_core(
            img, utterance_idxs, utterance_length, raw_utterance
        )
        meta_out = _batch_meta(list(meta))
        return img, utterance_idxs, utterance_length, raw_utterance, meta_out

    # Case C: legacy 4-tuples
    img, utterance_idxs, utterance_length, raw_utterance = zip(*batch)
    return _pack_core(img, utterance_idxs, utterance_length, raw_utterance)


class LabeledSEvalDataset(Dataset):
    """Dataset that returns a set of referents and a target word for evaluation."""

    def __init__(
        self,
        data,
        vocab,
        transform,
        eval_include_sos_eos: bool = False,
        clip_eval: bool = False,
    ):
        self.data = data
        self.vocab = vocab
        self.transform = transform
        self.eval_include_sos_eos = eval_include_sos_eos
        self.clip_eval = clip_eval

    def __getitem__(self, idx):
        # read trial information
        trial = self.data[idx]

        # read in images (target and foils)
        # target image is always the first index
        n_imgs = len(trial["foil_img_filenames"]) + 1
        imgs = torch.zeros((n_imgs, 3, IMAGE_H, IMAGE_W))
        target_img_filename = trial["target_img_filename"]
        imgs[0] = self.transform(Image.open(target_img_filename).convert("RGB"))
        for i, foil_img_filename in enumerate(trial["foil_img_filenames"]):
            imgs[i + 1] = self.transform(Image.open(foil_img_filename).convert("RGB"))

        # get target category index from vocab as a single utterance
        raw_label = trial["target_category"]

        if not self.clip_eval:
            # use SAYCam vocab/tokenizer
            label = [self.vocab[raw_label]]
            if self.eval_include_sos_eos:
                # label is [<sos>, label, <eos>] to match LM training
                label = [SOS_TOKEN_ID] + label + [EOS_TOKEN_ID]

            label = torch.LongTensor(label)
            label_len = len(label)
        else:
            # use CLIP tokenizer
            label = clip.tokenize(raw_label)
            label_len = len(label)

        return imgs, label, label_len, [raw_label]

    def __len__(self):
        return len(self.data)


class LabeledSTextEvalDataset(Dataset):
    """Dataset that returns a single referent and multiple target words for evaluation."""

    def __init__(
        self,
        data,
        vocab,
        transform,
        eval_include_sos_eos: bool = False,
        clip_eval: bool = False,
    ):
        self.data = data
        self.vocab = vocab
        self.transform = transform
        self.eval_include_sos_eos = eval_include_sos_eos
        self.clip_eval = clip_eval

    def __getitem__(self, idx):
        # read trial information
        trial = self.data[idx]

        # read in target image
        img = torch.zeros((1, 3, IMAGE_H, IMAGE_W))
        target_img_filename = trial["target_img_filename"]
        img[0] = self.transform(Image.open(target_img_filename).convert("RGB"))

        # get target category and foil categories
        raw_target_label = trial["target_category"]
        raw_foil_labels = trial["foil_categories"]
        raw_labels = [raw_target_label] + raw_foil_labels

        labels = []
        labels_len = []
        for raw_label in raw_labels:
            if not self.clip_eval:
                # use SAYCam vocab/tokenizer
                label = [self.vocab[raw_label]]
                if self.eval_include_sos_eos:
                    label = [SOS_TOKEN_ID] + label + [EOS_TOKEN_ID]
                labels.append(label)
                labels_len.append(len(label))
            else:
                # use CLIP tokenizer
                label = clip.tokenize(raw_label)
                labels.append(label)
                labels_len.append(len(label))

        if not self.clip_eval:
            # convert list of labels to tensor
            labels = torch.LongTensor(labels)
        else:
            # labels are already tensors, so need to concatenate
            labels = torch.cat(labels, dim=0)

        return img, labels, labels_len, [raw_target_label]

    def __len__(self):
        return len(self.data)


class MultiModalDataModule(pl.LightningDataModule):
    """The abstract data module consisting of images and the associated utterances."""

    def __init__(self, args=None) -> None:
        super().__init__()

        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.drop_last = self.args.get("drop_last", False)
        self.val_batch_size = self.args.get("val_batch_size", VAL_BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)
        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))
        self.augment_frames = self.args.get("augment_frames", AUGMENT_FRAMES)
        self.eval_include_sos_eos = self.args.get("eval_include_sos_eos", EVAL_INCLUDE_SOS_EOS)
        self.test_while_val = self.args.get("test_while_val", TEST_WHILE_VAL)
        self.eval_type = self.args.get("eval_type", EVAL_TYPE)
        self.eval_metadata_filename = self.args.get("eval_metadata_filename", EVAL_METADATA_FILENAME)
        self.clip_eval = self.args.get("clip_eval", CLIP_EVAL)

        # SAM prepacked config
        sam_masks_dir = self.args.get("sam_masks_dir", None)
        sam_prepacked_dir = self.args.get("sam_prepacked_dir", None)
        sam_frames_root = self.args.get("sam_frames_root", None)
        if sam_masks_dir is not None:
            if sam_frames_root is None:
                sam_frames_root = sam_masks_dir
            if sam_prepacked_dir is None:
                sam_prepacked_dir = os.path.join(sam_masks_dir, "sam_prepacked")
        self.sam_prepacked_dir: Optional[str] = sam_prepacked_dir
        self.sam_frames_root: Optional[str] = sam_frames_root

        print(f"Using metadata file: {self.eval_metadata_filename}")
        if self.sam_frames_root is not None:
            print(f"SAM frames root : {self.sam_frames_root}")
        if self.sam_prepacked_dir is not None:
            print(f"SAM prepacked dir : {self.sam_prepacked_dir}")

        if self.augment_frames:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop((IMAGE_H, IMAGE_W), scale=(0.2, 1.0)),
                    transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalizer,
                ]
            )
        elif self.clip_eval:
            print("Using CLIP transforms for evaluation")
            self.transform = transforms.Compose(
                [
                    transforms.Resize(IMAGE_H, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(IMAGE_H),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )
        else:
            print("Using base transforms")
            self.transform = transforms.Compose([transforms.ToTensor(), normalizer])

        self.base_transform = transforms.Compose([transforms.ToTensor(), normalizer])

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
        parser.add_argument("--drop_last", action="store_true")
        parser.add_argument("--val_batch_size", type=int, default=VAL_BATCH_SIZE)
        parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
        parser.add_argument("--augment_frames", action="store_true")
        parser.add_argument("--eval_include_sos_eos", action="store_true")
        parser.add_argument("--test_while_val", action="store_true")
        parser.add_argument("--eval_type", type=str, default="image", choices=["image", "text"])
        parser.add_argument("--eval_metadata_filename", type=str, default="eval_filtered_dev.json")
        parser.add_argument("--clip_eval", action="store_true")

        parser.add_argument("--sam_prepacked_dir", type=str, default=None)
        parser.add_argument("--sam_frames_root", type=str, default=None)
        return parser

    def prepare_data(self, *args, **kwargs) -> None:
        print("Calling prepare_data!")

    def setup(self, *args, **kwargs) -> None:
        print("Calling setup!")

        # read vocab
        vocab = self.read_vocab()

        # Build SAM registry once (single source-of-truth)
        use_sam = bool(getattr(self, "use_sam_masks", False))
        mask_source = str(self.args.get("mil_mask_source", "sam")).lower()
        self.sam_registry = None
        if use_sam and mask_source == "sam" and self.sam_prepacked_dir is not None:
            freq_json = self.args.get("sam_concept_frequency_json", None)
            min_cnt = int(self.args.get("sam_min_masks_per_concept", 0))
            concept_list_file = self.args.get("concept_list_file", None)
            if concept_list_file is not None and not Path(concept_list_file).is_file():
                concept_list_file = None

            self.sam_registry = build_sam_concept_registry(
                sam_prepacked_dir=self.sam_prepacked_dir,
                concept_frequency_json=freq_json,
                min_masks_per_concept=min_cnt,
                concept_list_file=concept_list_file,
                alpha=float(self.args.get("mask_freq_alpha", 0.25)),
                clip_min=float(self.args.get("mask_freq_clip_min", 0.5)),
                clip_max=float(self.args.get("mask_freq_clip_max", 2.0)),
                verbose=bool(self.args.get("sam_verbose_stats", False)),
            )

        # read and create image-text data splits (train/val/test)
        self.datasets = self.create_datasets(vocab)

        # read and create eval data splits (val/test)
        self.eval_datasets = self.create_eval_datasets(vocab)

    def read_vocab(self):
        raise NotImplementedError

    def create_datasets(self, vocab):
        raise NotImplementedError

    def create_eval_datasets(self, vocab):
        eval_datasets: Dict[str, Dataset] = {}
        eval_dev_metadata_filename = EVAL_DATA_DIR / self.eval_metadata_filename
        eval_test_metadata_filename = EVAL_DATA_DIR / self.eval_metadata_filename.replace("dev", "test")
        for split, filename in [("val", eval_dev_metadata_filename), ("test", eval_test_metadata_filename)]:
            data = load_data(filename)

            if self.eval_type == "image":
                dataset = LabeledSEvalDataset(
                    data, vocab, self.transform, self.eval_include_sos_eos, self.clip_eval
                )
            elif self.eval_type == "text":
                dataset = LabeledSTextEvalDataset(
                    data, vocab, self.transform, self.eval_include_sos_eos, self.clip_eval
                )
            else:
                raise ValueError(f"Unknown eval_type: {self.eval_type}")
            eval_datasets[split] = dataset

        return eval_datasets

    def _dist_info(self) -> Tuple[int, int]:
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            world_size = getattr(trainer, "world_size", None)
            rank = getattr(trainer, "global_rank", None)
            if world_size is not None and rank is not None:
                return int(world_size), int(rank)
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
        return world_size, rank

    def train_dataloader(self, batch_size=None, shuffle=True, drop_last=None):
        if batch_size is None:
            batch_size = self.batch_size
        if drop_last is None:
            drop_last = self.drop_last

        dataset = self.datasets["train"]
        sampler = None
        world_size, rank = self._dist_info()
        if world_size > 1 and not isinstance(dataset, IterableDataset):
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            shuffle = False

        return DataLoader(
            dataset,
            collate_fn=multiModalDataset_collate_fn,
            shuffle=shuffle,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.num_workers > 0),
            worker_init_fn=_seed_worker,
        )

    def val_test_dataloader(self, dataset, eval_dataset, batch_size=None, shuffle=False, drop_last=False):
        if batch_size is None:
            batch_size = self.val_batch_size

        dataloader = DataLoader(
            dataset,
            collate_fn=multiModalDataset_collate_fn,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.num_workers > 0),
            worker_init_fn=_seed_worker,
        )

        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=multiModalDataset_collate_fn,
            shuffle=False,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.num_workers > 0),
            worker_init_fn=_seed_worker,
        )

        return [dataloader, eval_dataloader]

    def val_dataloader(self, batch_size=None, shuffle=False, drop_last=False):
        dataloaders = self.val_test_dataloader(
            self.datasets['val'],
            self.eval_datasets['val'],
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        if self.test_while_val:
            dataloaders += self.test_dataloader(
                batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

        return dataloaders

    def test_dataloader(self, batch_size=None, shuffle=False, drop_last=False):
        return self.val_test_dataloader(
            self.datasets['test'],
            self.eval_datasets['test'],
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )


def load_and_print_info(data_module_class):
    # parse args
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()

    # set up data module
    data = data_module_class(args)
    data.prepare_data()

    print(data)
