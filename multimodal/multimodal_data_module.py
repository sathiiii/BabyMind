from pathlib import Path
from typing import Any, Tuple, Optional, Dict, List, Union
import json
import argparse
import os
from dataclasses import dataclass, field

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import pytorch_lightning as pl

from multimodal.utils import GaussianBlur

import clip

# directories and filenames
# must be consistent with multimodal_saycam_data_module
EVAL_DATA_DIR = Path("./expt_saycam")
EVAL_METADATA_FILENAME = "eval_dev.json"
# EVAL_DEV_METADATA_FILENAME = EVAL_DATA_DIR / "eval_dev.json"
# EVAL_TEST_METADATA_FILENAME = EVAL_DATA_DIR / "eval_test.json"

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


# ---------------------------------------------------------------------
# SAM prepacked index
# ---------------------------------------------------------------------
@dataclass
class SamPrepackedIndex:
    """
    Index for prepacked SAM masks. Directory layout (produced by prepack_sam_masks.py):

        sam_prepacked/
        concept_vocab.json        # {"concepts": [name0, name1, ...]}
        sam_prepacked_index.json  # {"sub1/vid1/frame_0001.jpg": "sub1_vid1_frame_0001.pt", ...}
        *.pt                      # each: {"masks": (M,H,W) uint8, "concept_ids": (M,) int16}

    If concept2idx is provided, concept_ids from the .pt file are mapped to global
    ids and any concept not in concept2idx is dropped.
    """

    root: Path
    frame_to_file: Dict[str, str]
    concepts: Tuple[str, ...]  # local id -> name
    cid_remap: Optional[torch.Tensor] = None  # local id -> global id or -1
    cache_size: int = 0

    _cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @classmethod
    def load(
        cls,
        root: Union[str, Path],
        concept2idx: Optional[Dict[str, int]] = None,
        cache_size: int = 0,
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
        concepts_list = vocab.get("concepts", [])
        concepts = tuple(concepts_list)

        cid_remap = None
        if concept2idx is not None:
            remap: List[int] = []
            for name in concepts:
                gid = concept2idx.get(str(name).lower(), -1)
                remap.append(int(gid))
            cid_remap = torch.tensor(remap, dtype=torch.long)

        return cls(
            root=root,
            frame_to_file=frame_to_file,
            concepts=concepts,
            cid_remap=cid_remap,
            cache_size=int(cache_size),
        )

    def get_masks_for_relpath(
        self,
        frame_relpath: str,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        frame_relpath must match the key stored in sam_prepacked_index.json,
        for example "sub1/vid1/frame_000123.jpg".

        Returns:
            masks: (M, H, W) float32 in {0., 1.}, filtered to concepts in concept2idx
            concept_ids: (M,) long, global concept ids (from concept_list_file)
            or None if no usable masks.
        """
        # small in memory cache for frequently reused frames
        if self.cache_size > 0 and frame_relpath in self._cache:
            masks, concept_ids = self._cache[frame_relpath]
            return masks.clone(), concept_ids.clone()

        fname = self.frame_to_file.get(frame_relpath)
        if fname is None:
            return None

        pt_path = self.root / fname
        if not pt_path.is_file():
            return None

        data = torch.load(pt_path, map_location="cpu")
        masks = data["masks"].float()  # uint8 -> float
        masks = (masks > 0.5).float()
        concept_ids = data["concept_ids"].long()  # local ids in concept_vocab.json

        # map local concept ids to global ids and drop ones not in concept2idx
        if self.cid_remap is not None:
            mapped = self.cid_remap[concept_ids]  # (M,)
            keep = mapped >= 0
            if not bool(keep.any()):
                return None
            masks = masks[keep]
            concept_ids = mapped[keep]

        if masks.numel() == 0:
            return None

        # cache result
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
        return data['data']


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class MultiModalDataset(Dataset):
    """
    Abstract Dataset that returns paired image-utterances.
    """

    def __init__(
        self,
        sam_prepacked_dir: Optional[str] = None,
        sam_frames_root: Optional[str] = None,
        sam_concept2idx: Optional[Dict[str, int]] = None,
        sam_cache_size: int = 0,
    ):
        super().__init__()
        self.sam_index: Optional[SamPrepackedIndex] = None
        self.sam_frames_root: Optional[Path] = None

        if sam_prepacked_dir is not None:
            self.sam_index = SamPrepackedIndex.load(
                sam_prepacked_dir,
                concept2idx=sam_concept2idx,
                cache_size=sam_cache_size,
            )

        if sam_frames_root is not None:
            self.sam_frames_root = Path(sam_frames_root)

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any]:
        """
        Returns an image utterance pair in tuple
        (img, utterance_idxs, utterance_length, raw_utterances).

        raw_utterances: a list of str, each of which is a sentence with
        space separated tokens.
        """
        raise NotImplementedError

    def get_sam_meta_for_image_path(
        self,
        image_path: Union[str, Path],
    ) -> Optional[Dict[str, Any]]:
        """
        Convenience helper for subclasses that want to attach SAM masks.

        Given an absolute image_path, it computes the frame_relpath with
        respect to self.sam_frames_root and looks up the corresponding
        prepacked masks.

        Returns a meta dict with keys:

            sam_mask: (K, 1, H, W) float32 in {0., 1.}
            sam_mask_concept_id: (K,) long
            sam_mask_count: scalar LongTensor

        or None if no SAM index or no masks for this frame.
        """
        if self.sam_index is None or self.sam_frames_root is None:
            return None

        image_path = Path(image_path)
        try:
            frame_relpath = str(image_path.relative_to(self.sam_frames_root))
        except ValueError:
            frame_relpath = os.path.relpath(
                str(image_path),
                str(self.sam_frames_root),
            )

        out = self.sam_index.get_masks_for_relpath(frame_relpath)
        if out is None:
            return None
        masks, concept_ids = out  # (K,H,W), (K,)

        if masks.numel() == 0:
            return None

        # convert to (K,1,H,W) if needed
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        elif masks.dim() == 4 and masks.shape[1] == 1:
            pass
        else:
            raise ValueError(
                f"Expected SAM masks of shape (K,H,W) or (K,1,H,W), got {masks.shape}"
            )

        meta = {
            "sam_mask": masks,
            "sam_mask_concept_id": concept_ids,
            "sam_mask_count": torch.tensor(masks.shape[0], dtype=torch.long),
        }
        return meta


def multiModalDataset_collate_fn(batch):
    """
    Collate function that supports:

      * (img, idxs, len, raw) legacy, no meta
      * (img, idxs, len, raw, meta_dict)
      * ((img, idxs, len, raw), meta_dict)

    For meta_dict, if present, it preserves and batches:

      * clip_id: list[str]
      * frame_idx: LongTensor[B]
      * frame_filename: list[str] (if present)
      * sam_mask: FloatTensor[B, K_max, 1, H, W] (padded)
      * sam_mask_concept_id: LongTensor[B, K_max] (padded with -1)
      * sam_mask_count: LongTensor[B] (real count per sample)
      * vm_concept_id: LongTensor[B, K_max] (padded with -1)

    The batching of meta is robust to some examples missing a field.
    For mask related fields, missing entries are treated as zero masks.
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

    def _batch_meta(meta_seq: List[Dict[str, Any]]):
        """
        Batch a sequence of per example meta dicts into a single meta dict.

        Handles variable number of SAM masks by padding to K_max.

        If some examples do not have a field, they are treated as having
        empty values for that field.
        """
        if len(meta_seq) == 0:
            return {}

        B = len(meta_seq)
        all_keys = set()
        for m in meta_seq:
            all_keys.update(m.keys())

        meta_out: Dict[str, Any] = {}
        K_sizes: Optional[List[int]] = None
        K_max: int = 0

        # clip id and frame index
        if "clip_id" in all_keys:
            meta_out["clip_id"] = [m.get("clip_id") for m in meta_seq]

        if "frame_idx" in all_keys:
            meta_out["frame_idx"] = torch.tensor(
                [int(m.get("frame_idx", -1)) for m in meta_seq],
                dtype=torch.long,
            )

        # optionally keep frame filename for debugging
        if "frame_filename" in all_keys:
            meta_out["frame_filename"] = [
                m.get("frame_filename", "") for m in meta_seq
            ]

        # SAM masks
        if "sam_mask" in all_keys:
            sam_masks_raw = [m.get("sam_mask") for m in meta_seq]

            # find a prototype tensor to get dtype and spatial shape
            proto = None
            for sm in sam_masks_raw:
                if sm is not None:
                    proto = sm
                    break

            if proto is None:
                # no masks at all in this batch
                meta_out["sam_mask"] = torch.empty(
                    B,
                    0,
                    1,
                    IMAGE_H,
                    IMAGE_W,
                    dtype=torch.float32,
                )
                K_sizes = [0 for _ in range(B)]
                K_max = 0
            else:
                # normalise all sam masks to (K,1,H,W)
                norm_sam_masks: List[torch.Tensor] = []
                if proto.dim() == 3:
                    _, H, W = proto.shape
                    C = 1
                elif proto.dim() == 4:
                    _, C, H, W = proto.shape
                else:
                    raise ValueError(
                        "Expected SAM mask prototype of shape (K,H,W) or "
                        f"(K,1,H,W), got {proto.shape}"
                    )

                for sm in sam_masks_raw:
                    if sm is None:
                        norm_sam_masks.append(
                            torch.empty(
                                0,
                                C,
                                H,
                                W,
                                dtype=proto.dtype,
                            )
                        )
                    else:
                        t = sm
                        if t.dim() == 3:
                            t = t.unsqueeze(1)
                        norm_sam_masks.append(t)

                K_sizes = [sm.shape[0] for sm in norm_sam_masks]
                K_max = max(K_sizes) if K_sizes else 0

                if K_max == 0:
                    meta_out["sam_mask"] = torch.empty(
                        B,
                        0,
                        C,
                        H,
                        W,
                        dtype=proto.dtype,
                    )
                else:
                    padded = torch.zeros(
                        B,
                        K_max,
                        C,
                        H,
                        W,
                        dtype=proto.dtype,
                    )
                    for i, sm in enumerate(norm_sam_masks):
                        Ki = sm.shape[0]
                        if Ki > 0:
                            padded[i, :Ki] = sm
                    meta_out["sam_mask"] = padded

        # sam_mask_concept_id
        if "sam_mask_concept_id" in all_keys:
            sam_cids_raw = [m.get("sam_mask_concept_id") for m in meta_seq]
            norm_cids: List[torch.Tensor] = []
            for cid in sam_cids_raw:
                if cid is None:
                    norm_cids.append(torch.empty(0, dtype=torch.long))
                else:
                    cid_t = torch.as_tensor(cid, dtype=torch.long).view(-1)
                    norm_cids.append(cid_t)

            if K_sizes is None:
                K_sizes = [cid.shape[0] for cid in norm_cids]
                K_max = max(K_sizes) if K_sizes else 0

            if K_max == 0:
                meta_out["sam_mask_concept_id"] = torch.full(
                    (B, 0),
                    fill_value=-1,
                    dtype=torch.long,
                )
            else:
                padded_cids = torch.full(
                    (B, K_max),
                    fill_value=-1,
                    dtype=torch.long,
                )
                for i, cid in enumerate(norm_cids):
                    Ki = cid.shape[0]
                    if Ki > 0:
                        padded_cids[i, :Ki] = cid
                meta_out["sam_mask_concept_id"] = padded_cids

        # sam_mask_count
        if "sam_mask_count" in all_keys:
            meta_out["sam_mask_count"] = torch.tensor(
                [
                    int(
                        m.get(
                            "sam_mask_count",
                            (K_sizes[i] if K_sizes is not None else 0),
                        )
                    )
                    for i, m in enumerate(meta_seq)
                ],
                dtype=torch.long,
            )
        elif K_sizes is not None:
            meta_out["sam_mask_count"] = torch.tensor(
                K_sizes,
                dtype=torch.long,
            )

        # vm_concept_id
        if "vm_concept_id" in all_keys:
            vm_cids_raw = [m.get("vm_concept_id") for m in meta_seq]
            norm_vm: List[torch.Tensor] = []
            for cid in vm_cids_raw:
                if cid is None:
                    norm_vm.append(torch.empty(0, dtype=torch.long))
                else:
                    cid_t = torch.as_tensor(cid, dtype=torch.long).view(-1)
                    norm_vm.append(cid_t)

            if K_sizes is None:
                K_sizes = [cid.shape[0] for cid in norm_vm]
                K_max = max(K_sizes) if K_sizes else 0

            if K_max == 0:
                meta_out["vm_concept_id"] = torch.full(
                    (B, 0),
                    fill_value=-1,
                    dtype=torch.long,
                )
            else:
                padded_vm = torch.full(
                    (B, K_max),
                    fill_value=-1,
                    dtype=torch.long,
                )
                for i, cid in enumerate(norm_vm):
                    Ki = cid.shape[0]
                    if Ki > 0:
                        padded_vm[i, :Ki] = cid
                meta_out["vm_concept_id"] = padded_vm

        return meta_out

    # Case A: flat 5 tuples with meta dict
    if isinstance(first, tuple) and len(first) == 5 and isinstance(first[4], dict):
        img, utterance_idxs, utterance_length, raw_utterance, meta = zip(*batch)
        img, utterance_idxs, utterance_length, raw_utterance = _pack_core(
            img,
            utterance_idxs,
            utterance_length,
            raw_utterance,
        )
        meta_out = _batch_meta(list(meta))
        return img, utterance_idxs, utterance_length, raw_utterance, meta_out

    # Case B: nested ((core), meta) with meta dict
    if (
        isinstance(first, tuple)
        and len(first) == 2
        and isinstance(first[0], (tuple, list))
        and isinstance(first[1], dict)
    ):
        core, meta = zip(*batch)
        img, utterance_idxs, utterance_length, raw_utterance = zip(*core)
        img, utterance_idxs, utterance_length, raw_utterance = _pack_core(
            img,
            utterance_idxs,
            utterance_length,
            raw_utterance,
        )
        meta_out = _batch_meta(list(meta))
        return img, utterance_idxs, utterance_length, raw_utterance, meta_out

    # Case C: legacy 4 tuples (no meta)
    img, utterance_idxs, utterance_length, raw_utterance = zip(*batch)
    return _pack_core(img, utterance_idxs, utterance_length, raw_utterance)


class LabeledSEvalDataset(Dataset):
    """
    Dataset that returns a set of referents and a target word for evaluation.
    """

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
        imgs[0] = self.transform(
            Image.open(target_img_filename).convert("RGB")
        )

        for i, foil_img_filename in enumerate(trial["foil_img_filenames"]):
            imgs[i + 1] = self.transform(
                Image.open(foil_img_filename).convert("RGB")
            )

        # get target category index from vocab as a single utterance
        raw_label = trial["target_category"]

        if not self.clip_eval:
            # use SAYCam vocab tokenizer
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
    """
    Dataset that returns a single referent and multiple target words for evaluation.
    """

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
        img[0] = self.transform(
            Image.open(target_img_filename).convert("RGB")
        )

        # get target category and foil categories
        raw_target_label = trial["target_category"]
        raw_foil_labels = trial["foil_categories"]
        raw_labels = [raw_target_label] + raw_foil_labels
        labels = []
        labels_len = []
        for raw_label in raw_labels:
            if not self.clip_eval:
                # use SAYCam vocab tokenizer
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
    """
    The abstract data module consisting of images and the associated utterances.
    """

    def __init__(self, args=None) -> None:
        super().__init__()

        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.drop_last = self.args.get("drop_last", False)
        self.val_batch_size = self.args.get("val_batch_size", VAL_BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)
        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))
        self.augment_frames = self.args.get(
            "augment_frames", AUGMENT_FRAMES
        )
        self.eval_include_sos_eos = self.args.get(
            "eval_include_sos_eos",
            EVAL_INCLUDE_SOS_EOS,
        )
        self.test_while_val = self.args.get(
            "test_while_val",
            TEST_WHILE_VAL,
        )
        self.eval_type = self.args.get("eval_type", EVAL_TYPE)
        self.eval_metadata_filename = self.args.get(
            "eval_metadata_filename",
            EVAL_METADATA_FILENAME,
        )
        self.clip_eval = self.args.get("clip_eval", CLIP_EVAL)

        # SAM prepacked config
        # sam_masks_dir comes from multimodal_lit.py (for example expt_saycam/train_sam_masks)
        sam_masks_dir = self.args.get("sam_masks_dir", None)
        sam_prepacked_dir = self.args.get("sam_prepacked_dir", None)
        sam_frames_root = self.args.get("sam_frames_root", None)

        # If sam_masks_dir is provided, derive defaults:
        # sam_frames_root = sam_masks_dir
        # sam_prepacked_dir = sam_masks_dir / "sam_prepacked"
        if sam_masks_dir is not None:
            if sam_frames_root is None:
                sam_frames_root = sam_masks_dir
            if sam_prepacked_dir is None:
                sam_prepacked_dir = os.path.join(
                    sam_masks_dir,
                    "sam_prepacked",
                )

        self.sam_prepacked_dir: Optional[str] = sam_prepacked_dir
        self.sam_frames_root: Optional[str] = sam_frames_root

        # check which metadata file is being used
        print(f"Using metadata file: {self.eval_metadata_filename}")
        if self.sam_frames_root is not None:
            print(f"SAM frames root    : {self.sam_frames_root}")
        if self.sam_prepacked_dir is not None:
            print(f"SAM prepacked dir  : {self.sam_prepacked_dir}")

        if self.augment_frames:
            # add same augmentations as Emin used
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        (IMAGE_H, IMAGE_W),
                        scale=(0.2, 1.0),
                    ),
                    # transforms.RandomApply([transforms.ColorJitter(0.9, 0.9, 0.9, 0.5)], p=0.9),
                    # transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply(
                        [GaussianBlur([0.1, 2.0])],
                        p=0.5,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalizer,
                ]
            )
        elif self.clip_eval:
            print("Using CLIP transforms for evaluation")
            # use CLIP transforms (for CLIP evaluation only)
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        IMAGE_H,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.CenterCrop(IMAGE_H),
                    # _convert_image_to_rgb,  # commented out since we convert to RGB
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )
        else:
            print("Using base transforms")
            # just convert to tensor and normalize
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalizer,
                ]
            )

        # keep base transform for val and test
        self.base_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalizer,
            ]
        )

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size",
            type=int,
            default=BATCH_SIZE,
            help="Number of examples to operate on per train step.",
        )
        parser.add_argument(
            "--drop_last",
            action="store_true",
            help="Drop the last not full batch.",
        )
        parser.add_argument(
            "--val_batch_size",
            type=int,
            default=VAL_BATCH_SIZE,
            help=(
                "Number of examples to operate on per forward step "
                "during validation."
            ),
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=NUM_WORKERS,
            help="Number of additional processes to load data.",
        )
        parser.add_argument(
            "--augment_frames",
            action="store_true",
            help="Apply data augmentation to images.",
        )
        parser.add_argument(
            "--eval_include_sos_eos",
            action="store_true",
            help="Add <sos> and <eos> tokens during evaluation",
        )
        parser.add_argument(
            "--test_while_val",
            action="store_true",
            help="Evaluate test set during validation (for COCO only).",
        )
        parser.add_argument(
            "--eval_type",
            type=str,
            default="image",
            choices=["image", "text"],
            help="Run evaluation using multiple images or multiple labels",
        )
        parser.add_argument(
            "--eval_metadata_filename",
            type=str,
            default="eval_filtered_dev.json",
            help=(
                "JSON file with metadata for (dev) evaluation split to use"
            ),
        )
        parser.add_argument(
            "--clip_eval",
            action="store_true",
            help="Perform evaluation using CLIP",
        )
        parser.add_argument(
            "--sam_prepacked_dir",
            type=str,
            default=None,
            help=(
                "Directory with prepacked SAM mask .pt files and index."
            ),
        )
        parser.add_argument(
            "--sam_frames_root",
            type=str,
            default=None,
            help=(
                "Root directory whose relative paths match "
                "sam_prepacked_index keys."
            ),
        )
        return parser

    def prepare_data(self, *args, **kwargs) -> None:
        print("Calling prepare_data!")

    def setup(self, *args, **kwargs) -> None:
        print("Calling setup!")

        # read vocab
        vocab = self.read_vocab()

        # read and create image text data splits (train val test)
        self.datasets = self.create_datasets(vocab)

        # read and create eval data splits (val test)
        self.eval_datasets = self.create_eval_datasets(vocab)

    def read_vocab(self):
        raise NotImplementedError

    def create_datasets(self, vocab):
        raise NotImplementedError

    def create_eval_datasets(self, vocab):
        eval_datasets: Dict[str, Dataset] = {}

        eval_dev_metadata_filename = EVAL_DATA_DIR / self.eval_metadata_filename
        eval_test_metadata_filename = EVAL_DATA_DIR / self.eval_metadata_filename.replace(
            "dev",
            "test",
        )

        for split, filename in [
            ("val", eval_dev_metadata_filename),
            ("test", eval_test_metadata_filename),
        ]:
            data = load_data(filename)

            if self.eval_type == "image":
                dataset = LabeledSEvalDataset(
                    data,
                    vocab,
                    self.transform,
                    self.eval_include_sos_eos,
                    self.clip_eval,
                )
            elif self.eval_type == "text":
                dataset = LabeledSTextEvalDataset(
                    data,
                    vocab,
                    self.transform,
                    self.eval_include_sos_eos,
                    self.clip_eval,
                )
            else:
                raise ValueError(f"Unknown eval_type: {self.eval_type}")

            eval_datasets[split] = dataset

        return eval_datasets

    def train_dataloader(self, batch_size=None, shuffle=True, drop_last=None):
        if batch_size is None:
            batch_size = self.batch_size
        if drop_last is None:
            drop_last = self.drop_last

        return DataLoader(
            self.datasets["train"],
            collate_fn=multiModalDataset_collate_fn,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_test_dataloader(
        self,
        dataset,
        eval_dataset,
        batch_size=None,
        shuffle=False,
        drop_last=False,
    ):
        if batch_size is None:
            batch_size = self.val_batch_size

        dataloader = DataLoader(
            dataset,
            collate_fn=multiModalDataset_collate_fn,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=multiModalDataset_collate_fn,
            shuffle=shuffle,
            # batch_size=self.batch_size // 4,  # divide by 4 here since eval trials have 4 images
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        return [dataloader, eval_dataloader]

    def val_dataloader(self, batch_size=None, shuffle=False, drop_last=False):
        dataloaders = self.val_test_dataloader(
            self.datasets["val"],
            self.eval_datasets["val"],
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        if self.test_while_val:
            dataloaders += self.test_dataloader(
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
            )

        return dataloaders

    def test_dataloader(self, batch_size=None, shuffle=False, drop_last=False):
        return self.val_test_dataloader(
            self.datasets["test"],
            self.eval_datasets["test"],
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
