# multimodal/imagenet_val_data_module.py
from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import clip

from multimodal.multimodal_data_module import (
    EVAL_DATA_DIR,
    IMAGE_H,
    IMAGE_W,
    normalizer,
    multiModalDataset_collate_fn,
    read_vocab,
    SOS_TOKEN_ID,
    EOS_TOKEN_ID,
    UNK_TOKEN_ID,
)

# ---------------------------
# Small utilities
# ---------------------------

_WNID_RE = re.compile(r"^[nvar]\d{8}$")  # ImageNet uses 'n' + 8 digits

# Lightweight aliases to increase mapping into a small vocab (used mainly in vocab mode).
_ALIAS: Dict[str, List[str]] = {
    "cat": ["kitty", "kitten"],
    "kitty": ["cat", "kitten"],
    "kitten": ["cat", "kitty"],
    "dog": ["puppy", "doggy"],
    "puppy": ["dog", "doggy"],
    "rabbit": ["bunny"],
    "bunny": ["rabbit"],
    "horse": ["pony"],
    "pony": ["horse"],
    "bicycle": ["bike"],
    "bike": ["bicycle"],
    "automobile": ["car"],
    "car": ["automobile"],
    "airplane": ["plane", "aeroplane", "aircraft"],
    "plane": ["airplane", "aeroplane", "aircraft"],
    "couch": ["sofa"],
    "sofa": ["couch"],
    "television": ["tv"],
    "tv": ["television"],
    "telephone": ["phone", "cellphone", "cell phone"],
    "phone": ["telephone", "cellphone", "cell phone"],
}

# Tokens/phrases we never want as final labels
_ALLOW_SHORT = {"tv", "ox"}
_BAD_LABELS = {
    # function words
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "from",
    "by",
    "at",
    "is",
    "it",
    "this",
    "that",
    "these",
    "those",
    # number-ish words (ImageNet glosses like “one-humped camel”)
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    # generic junk
    "thing",
    "stuff",
    "piece",
    "object",
    # generic adjectives that appear in glosses and create nonsense mappings
    "old",
    "new",
    "young",
}

def _atomic_write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)

def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def _as_path_or_none(x) -> Optional[Path]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    return Path(s).expanduser()

def _require_file_path(p: Path, name: str) -> None:
    # Only error if it exists and is a directory (common bug: Path("") -> ".")
    if p.exists() and p.is_dir():
        raise IsADirectoryError(f"[imagenet_eval] {name} points to a directory (expected a file): {p}")

def _require_dir_path(p: Path, name: str) -> None:
    if p.exists() and not p.is_dir():
        raise NotADirectoryError(f"[imagenet_eval] {name} is not a directory: {p}")

def _normalize_name(s: str) -> str:
    s = str(s).strip().lower()
    s = " ".join(s.split())
    return s

def _clean_phrase(s: str) -> str:
    s = _normalize_name(str(s)).replace("_", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = " ".join(s.split())
    return s

def _singularize(w: str) -> str:
    w = _normalize_name(w)
    if w.endswith("ies") and len(w) > 3:
        return w[:-3] + "y"
    if w.endswith("ses") and len(w) > 3:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w

def _pluralize(w: str) -> str:
    w = _normalize_name(w)
    if len(w) <= 2:
        return w
    if w.endswith("y") and len(w) > 2 and w[-2] not in "aeiou":
        return w[:-1] + "ies"
    if w.endswith(("s", "x", "z", "ch", "sh")):
        return w + "es"
    return w + "s"

def _is_good_label(label: str) -> bool:
    lab = _clean_phrase(label)
    if not lab:
        return False
    if re.fullmatch(r"[a-z]", lab):  # single letter
        return False
    if re.fullmatch(r"\d+", lab):  # digits only
        return False
    if len(lab) < 3 and lab not in _ALLOW_SHORT:
        return False
    if lab in _BAD_LABELS:
        return False
    return True

def _expand_aliases(phrase_or_token: str) -> List[str]:
    key = _clean_phrase(phrase_or_token)
    out = [key] if key else []
    for alt in _ALIAS.get(key, []):
        a = _clean_phrase(alt)
        if a and a not in out:
            out.append(a)
    return out

def _candidate_strings_phrase(name: str, include_head: bool = True) -> List[str]:
    """
    Candidate strings for matching / canonicalization:
      - cleaned phrase
      - underscore/no-space variants
      - optionally head noun singular/plural
    """
    n = _clean_phrase(name)
    if not n:
        return []

    cands: List[str] = [
        n,
        n.replace(" ", "_"),
        n.replace(" ", ""),
    ]

    if include_head:
        toks = n.split(" ")
        if toks:
            head = _singularize(toks[-1])
            cands.extend([head, _pluralize(head), head.replace(" ", "_"), head.replace(" ", "")])

    out: List[str] = []
    seen = set()
    for c in cands:
        cc = _normalize_name(c)
        if cc and cc not in seen:
            seen.add(cc)
            out.append(cc)
    return out

def _head_tokens(name: str) -> List[str]:
    n = _clean_phrase(name)
    toks = [t for t in n.split(" ") if t]
    if not toks:
        return []
    head = _singularize(toks[-1])
    return list(dict.fromkeys([head, _pluralize(head)]))

def _map_phrase_to_vocab_label(phrase: str, vocab_set: set[str]) -> Optional[str]:
    """
    Map a single English phrase into a vocab key.
    Strategy:
      1) phrase-level candidates (with alias expansion)
      2) token/head-noun fallback (with alias expansion)
    """
    n = _clean_phrase(phrase)
    if not n:
        return None

    # 1) phrase candidates
    for cand0 in _candidate_strings_phrase(n, include_head=False):
        for cand in _expand_aliases(cand0):
            if cand in vocab_set and _is_good_label(cand):
                return cand

    # also try head noun candidates of the phrase
    for cand0 in _candidate_strings_phrase(n, include_head=True):
        for cand in _expand_aliases(cand0):
            if cand in vocab_set and _is_good_label(cand):
                return cand

    # 2) token fallback: head token first, then others
    toks = [t for t in n.split(" ") if t]
    if toks:
        ordered = [toks[-1]] + toks[:-1] if len(toks) > 1 else toks
        for t in ordered:
            for ali in _expand_aliases(t):
                for cand0 in _candidate_strings_phrase(ali, include_head=False):
                    if cand0 in vocab_set and _is_good_label(cand0):
                        return cand0

    return None

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _crop_square_from_bbox(im: Image.Image, bbox_xywh: List[float]) -> Image.Image:
    """
    Crop bbox and pad to square with white background (keeps object centered).
    bbox_xywh: [x, y, w, h] in pixel coords.
    """
    x, y, w, h = bbox_xywh
    W, H = im.size

    x1 = int(_clamp(x, 0, W - 1))
    y1 = int(_clamp(y, 0, H - 1))
    x2 = int(_clamp(x + w, x1 + 1, W))
    y2 = int(_clamp(y + h, y1 + 1, H))

    crop = im.crop((x1, y1, x2, y2)).convert("RGB")
    cw, ch = crop.size
    s = max(cw, ch)
    out = Image.new("RGB", (s, s), (255, 255, 255))
    out.paste(crop, ((s - cw) // 2, (s - ch) // 2))
    return out

def _infer_imagenet_val_image_dir(images_root: Path) -> Path:
    """
    Accept common layouts for ImageNet val images and return the directory
    that directly contains ILSVRC2012_val_*.JPEG.
    """
    images_root = images_root.expanduser()
    if not images_root.exists():
        raise FileNotFoundError(f"[imagenet_eval] images root not found: {images_root}")

    candidates = [
        images_root,
        images_root / "val",
        images_root / "imgs",
        images_root / "images",
        images_root / "ILSVRC2012_img_val",
        images_root / "ILSVRC2012_img_val" / "val",
        images_root / "ILSVRC2012_img_val" / "imgs",
    ]

    for c in candidates:
        if c.exists() and c.is_dir():
            if any(c.glob("ILSVRC2012_val_*.JPEG")):
                return c

    # Lightweight fallback: search depth 2 for a matching directory
    for c in candidates:
        if c.exists() and c.is_dir():
            for sub in c.glob("*"):
                if sub.is_dir() and any(sub.glob("ILSVRC2012_val_*.JPEG")):
                    return sub

    raise FileNotFoundError(
        f"[imagenet_eval] Could not find ILSVRC2012_val_*.JPEG under: {images_root}"
    )

def _infer_imagenet_bbox_xml_dir(bbox_root: Path) -> Path:
    """
    Accept common layouts for ILSVRC val bbox xmls and return the directory
    that directly contains *.xml files matching the image stems.
    """
    bbox_root = bbox_root.expanduser()
    if not bbox_root.exists():
        raise FileNotFoundError(f"[imagenet_eval] bbox root not found: {bbox_root}")

    candidates = [
        bbox_root,
        bbox_root / "val",
        bbox_root / "ILSVRC2012_bbox_val_v3",
        bbox_root / "ILSVRC2012_bbox_val_v3" / "val",
        bbox_root / "Annotations",
        bbox_root / "Annotations" / "val",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            if any(c.glob("*.xml")):
                return c

    raise FileNotFoundError(
        f"[imagenet_eval] Could not find any .xml files under: {bbox_root} (tried common subdirs)"
    )

def _parse_imagenet_bbox_xml(xml_path: Path) -> Tuple[Optional[str], Optional[List[float]]]:
    """
    Returns (wnid, bbox_xywh_union).
    ILSVRC XML coords are xmin,ymin,xmax,ymax.
    We convert to xywh and take union if multiple objects exist.
    """
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return None, None

    wnids: List[str] = []
    boxes_xyxy: List[Tuple[float, float, float, float]] = []

    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name:
            wnids.append(name.strip())

        bb = obj.find("bndbox")
        if bb is None:
            continue

        try:
            xmin = float(bb.findtext("xmin"))
            ymin = float(bb.findtext("ymin"))
            xmax = float(bb.findtext("xmax"))
            ymax = float(bb.findtext("ymax"))
            boxes_xyxy.append((xmin, ymin, xmax, ymax))
        except Exception:
            continue

    wnid = wnids[0] if wnids else None
    if wnid is None:
        folder = root.findtext("folder")
        if folder:
            wnid = folder.strip()

    bbox_xywh = None
    if boxes_xyxy:
        x1 = min(b[0] for b in boxes_xyxy)
        y1 = min(b[1] for b in boxes_xyxy)
        x2 = max(b[2] for b in boxes_xyxy)
        y2 = max(b[3] for b in boxes_xyxy)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        bbox_xywh = [float(x1), float(y1), float(w), float(h)]

    return wnid, bbox_xywh

def _load_words_txt(words_txt_path: Path) -> Dict[str, List[str]]:
    """
    words.txt lines can be separated by spaces OR tabs, e.g.
      n01440764 tench, Tinca tinca
      n01440764\ttench, Tinca tinca
    """
    wnid2names: Dict[str, List[str]] = {}
    with open(words_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(None, 1)
            if len(parts) != 2:
                continue

            wnid, names = parts[0].strip(), parts[1].strip()
            if not wnid:
                continue

            raw = [x.strip() for x in names.split(",") if x.strip()]
            out: List[str] = []
            seen = set()
            for r in raw:
                r2 = _clean_phrase(r)
                if r2 and r2 not in seen:
                    seen.add(r2)
                    out.append(r2)

            if out:
                wnid2names[wnid] = out

    return wnid2names

def _wnid_to_wordnet_names(wnid: str) -> List[str]:
    """
    Fallback: derive names from WordNet using wnid offset.
    Requires nltk + wordnet corpus.
    """
    try:
        from nltk.corpus import wordnet as wn  # type: ignore
    except Exception:
        return []

    if not _WNID_RE.match(wnid):
        return []

    try:
        pos = wnid[0]
        offset = int(wnid[1:])
        syn = wn.synset_from_pos_and_offset(pos, offset)
        raw = [x.replace("_", " ") for x in syn.lemma_names()]
        out: List[str] = []
        seen = set()
        for r in raw:
            r2 = _clean_phrase(r)
            if r2 and r2 not in seen:
                seen.add(r2)
                out.append(r2)
        return out
    except Exception:
        return []

def _load_meta_mat(meta_mat_path: Path) -> Dict[str, List[str]]:
    """
    Load ImageNet devkit meta.mat and return wnid -> list of synonym phrases (from 'words').
    Uses scipy.io.loadmat if available.
    """
    try:
        from scipy.io import loadmat  # type: ignore
    except Exception:
        return {}

    try:
        mat = loadmat(str(meta_mat_path), squeeze_me=True, struct_as_record=False)
    except Exception:
        return {}

    synsets = mat.get("synsets", None)
    if synsets is None:
        return {}

    wnid2names: Dict[str, List[str]] = {}

    def _to_py_str(x) -> Optional[str]:
        if x is None:
            return None
        if isinstance(x, str):
            return x
        if isinstance(x, (bytes, bytearray)):
            try:
                return x.decode("utf-8", errors="ignore")
            except Exception:
                return None
        try:
            if hasattr(x, "item"):
                v = x.item()
                if isinstance(v, str):
                    return v
        except Exception:
            pass
        return str(x)

    def _to_py_int(x) -> Optional[int]:
        if x is None:
            return None
        try:
            if isinstance(x, (int, np.integer)):
                return int(x)
            if hasattr(x, "item"):
                v = x.item()
                if isinstance(v, (int, np.integer)):
                    return int(v)
                if isinstance(v, float):
                    return int(v)
        except Exception:
            pass
        try:
            return int(x)
        except Exception:
            return None

    for s in np.ravel(synsets):
        wnid = None
        words = None
        ilsvrc_id = None

        if hasattr(s, "__dict__"):
            wnid = _to_py_str(getattr(s, "WNID", None))
            words = _to_py_str(getattr(s, "words", None))
            ilsvrc_id = _to_py_int(getattr(s, "ILSVRC2012_ID", None))
        else:
            try:
                if hasattr(s, "dtype") and s.dtype.names:
                    wnid = _to_py_str(s["WNID"])
                    words = _to_py_str(s["words"])
                    ilsvrc_id = _to_py_int(s["ILSVRC2012_ID"])
            except Exception:
                wnid = None

        if wnid is None or words is None:
            continue
        if not _WNID_RE.match(wnid):
            continue
        if ilsvrc_id is None or ilsvrc_id <= 0:
            continue

        raw = [x.strip() for x in str(words).split(",") if x.strip()]
        out: List[str] = []
        seen = set()
        for r in raw:
            r2 = _clean_phrase(r)
            if r2 and r2 not in seen:
                seen.add(r2)
                out.append(r2)

        if out:
            wnid2names[wnid] = out

    return wnid2names


# ---------------------------
# Dataset (forced-choice)
# ---------------------------

class ImageNetForcedChoiceEvalDataset(Dataset):
    def __init__(
        self,
        trials: List[dict],
        vocab: Dict[str, int],
        imagenet_images_dir: Path,
        eval_type: str = "image",
        eval_include_sos_eos: bool = False,
        clip_eval: bool = False,
        use_bboxes: bool = True,
    ):
        self.trials = trials
        self.vocab = vocab
        self.imagenet_images_dir = imagenet_images_dir
        self.eval_type = eval_type
        self.eval_include_sos_eos = eval_include_sos_eos
        self.clip_eval = clip_eval
        self.use_bboxes = use_bboxes

        if self.clip_eval:
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
            self.transform = transforms.Compose(
                [
                    transforms.Resize((IMAGE_H, IMAGE_W), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    normalizer,
                ]
            )

    def __len__(self) -> int:
        return len(self.trials)

    def _encode_label(self, raw_label: str) -> Tuple[torch.Tensor, int]:
        if self.clip_eval:
            t = clip.tokenize(raw_label)  # (1,77)
            return t, int(t.numel())

        token_id = self.vocab.get(raw_label, UNK_TOKEN_ID)
        ids = [token_id]
        if self.eval_include_sos_eos:
            ids = [SOS_TOKEN_ID] + ids + [EOS_TOKEN_ID]
        t = torch.LongTensor(ids)
        return t, len(ids)

    def _load_image(self, inst: dict) -> torch.Tensor:
        img_path = self.imagenet_images_dir / inst["file_name"]
        im = Image.open(img_path).convert("RGB")

        bbox = inst.get("bbox", None)
        if self.use_bboxes and bbox is not None:
            im = _crop_square_from_bbox(im, bbox)

        return self.transform(im)

    def __getitem__(self, idx: int):
        trial = self.trials[idx]
        target_label = trial["target_category"]
        foil_labels = trial["foil_categories"]

        if self.eval_type == "image":
            n_imgs = 1 + len(trial["foil_instances"])
            imgs = torch.zeros((n_imgs, 3, IMAGE_H, IMAGE_W))

            imgs[0] = self._load_image(trial["target_instance"])
            for j, f_inst in enumerate(trial["foil_instances"]):
                imgs[j + 1] = self._load_image(f_inst)

            label, label_len = self._encode_label(target_label)
            return imgs, label, label_len, [target_label]

        if self.eval_type == "text":
            img = torch.zeros((1, 3, IMAGE_H, IMAGE_W))
            img[0] = self._load_image(trial["target_instance"])

            raw_labels = [target_label] + foil_labels
            labels = []
            lens = []
            for r in raw_labels:
                t, l = self._encode_label(r)
                labels.append(t)
                lens.append(l)

            if self.clip_eval:
                labels = torch.cat(labels, dim=0)  # (K,77)
            else:
                labels = torch.stack(labels, dim=0)  # (K,L)

            return img, labels, lens, [target_label]

        raise ValueError(f"Unknown eval_type: {self.eval_type}")


# ---------------------------
# DataModule
# ---------------------------

@dataclass
class ImageNetEvalConfig:
    imagenet_data_dir: Path
    imagenet_images_dir: Path
    imagenet_bbox_xml_dir: Path

    imagenet_words_txt: Optional[Path]
    imagenet_meta_mat: Optional[Path]

    eval_metadata_path: Path
    vocab_path: Path

    label_map_path: Path
    label_overrides_path: Optional[Path] = None
    regenerate_label_map: bool = False

    label_mode: str = "vocab"  # "vocab" | "canonical" | "raw"

    n_foils: int = 3  # 4-way forced-choice => 3 foils
    n_repeats: int = 1
    max_images_per_label: int = 50

    min_box_area: float = 32 * 32
    min_box_side: float = 16

    seed: int = 0
    regenerate_trials: bool = False

    use_bboxes: bool = True


class ImageNetValDataModule(pl.LightningDataModule):
    """
    Forced-choice evaluation on ImageNet val.

    label_mode:
      - "vocab": map wnids into CVCL vocab (via overrides/synonyms/variants). Drop wnids that cannot be mapped.
      - "canonical": choose a clean English label per wnid (via words.txt/meta.mat/wordnet + overrides),
                     but DO NOT require it to exist in the CVCL vocab (enables OOV trials for neuron branch).
      - "raw": choose a minimally-cleaned English label per wnid (basically the first usable gloss),
               but DO NOT require it to exist in the CVCL vocab.

    Writes:
      Trials:    EVAL_DATA_DIR / <eval_metadata_filename>
      Label map: EVAL_DATA_DIR / <eval_metadata_stem>_label_map.json

    Notes:
      - In "canonical"/"raw", OOV labels are kept in the trials JSON so eval.py can route them
        to the neuron-classifier branch.
      - The label_map and trials cache are validated against the current label_mode to prevent
        accidental reuse of a vocab-mode cache when switching to canonical/raw.
    """

    def __init__(self, args=None) -> None:
        super().__init__()
        self.args = args

        data_dir = Path(getattr(args, "imagenet_data_dir", "eval_datasets/imagenet")).expanduser()

        images_root_arg = _as_path_or_none(getattr(args, "imagenet_images_dir", None))
        images_root = images_root_arg if images_root_arg is not None else data_dir
        images_dir = _infer_imagenet_val_image_dir(images_root)

        bbox_root_arg = _as_path_or_none(getattr(args, "imagenet_bbox_dir", None))
        bbox_root = bbox_root_arg if bbox_root_arg is not None else data_dir
        bbox_xml_dir = _infer_imagenet_bbox_xml_dir(bbox_root)

        # Optional: words.txt
        words_txt_arg = _as_path_or_none(getattr(args, "imagenet_words_txt", None))
        if words_txt_arg is None:
            candidates = [
                data_dir / "words.txt",
                data_dir / "ILSVRC2012_devkit_t12" / "data" / "words.txt",
            ]
            words_txt_arg = next((p for p in candidates if p.exists() and p.is_file()), None)

        # Optional: meta.mat from devkit
        meta_mat_arg = _as_path_or_none(getattr(args, "imagenet_meta_mat", None))
        if meta_mat_arg is None:
            candidates = [
                data_dir / "meta.mat",
                data_dir / "ILSVRC2012_devkit_t12" / "data" / "meta.mat",
            ]
            meta_mat_arg = next((p for p in candidates if p.exists() and p.is_file()), None)

        eval_meta = Path(getattr(args, "eval_metadata_filename", "eval_imagenet_val.json"))
        eval_meta_path = (EVAL_DATA_DIR / eval_meta).resolve()

        label_map_arg = _as_path_or_none(getattr(args, "imagenet_label_map_path", None))
        label_map_path = (
            label_map_arg
            if label_map_arg is not None
            else eval_meta_path.with_name(eval_meta_path.stem + "_label_map.json")
        )

        overrides_arg = _as_path_or_none(getattr(args, "imagenet_label_overrides_json", None))

        vocab_arg = getattr(args, "vocab_filename", None) or getattr(args, "saycam_vocab_filename", None)
        vocab_path = _as_path_or_none(vocab_arg)
        if vocab_path is None:
            candidates = [
                Path("vocab.json"),
                Path("expt_saycam") / "vocab.json",
                Path(os.environ.get("SAYCAM_VOCAB", str(EVAL_DATA_DIR / "vocab.json"))),
            ]
            vocab_path = next((p for p in candidates if p.exists()), candidates[-1])

        _require_file_path(label_map_path, "imagenet_label_map_path")
        _require_file_path(vocab_path, "vocab_filename")
        _require_dir_path(images_dir, "imagenet_images_dir")
        _require_dir_path(bbox_xml_dir, "imagenet_bbox_dir")

        use_bboxes = not bool(getattr(args, "imagenet_no_bboxes", False))

        label_mode = str(getattr(args, "imagenet_label_mode", "vocab")).strip().lower()
        if label_mode not in {"vocab", "canonical", "raw"}:
            raise ValueError(f"[imagenet_eval] Unknown imagenet_label_mode={label_mode} (expected vocab|canonical|raw)")

        self.cfg = ImageNetEvalConfig(
            imagenet_data_dir=data_dir,
            imagenet_images_dir=images_dir,
            imagenet_bbox_xml_dir=bbox_xml_dir,
            imagenet_words_txt=words_txt_arg,
            imagenet_meta_mat=meta_mat_arg,
            eval_metadata_path=eval_meta_path,
            vocab_path=vocab_path,
            label_map_path=label_map_path,
            label_overrides_path=overrides_arg,
            regenerate_label_map=bool(getattr(args, "imagenet_regenerate_label_map", False)),
            label_mode=label_mode,
            n_foils=int(getattr(args, "imagenet_n_foils", 3)),
            n_repeats=int(getattr(args, "imagenet_n_repeats", 1)),
            max_images_per_label=int(getattr(args, "imagenet_max_images_per_label", 50)),
            min_box_area=float(getattr(args, "imagenet_min_box_area", 32 * 32)),
            min_box_side=float(getattr(args, "imagenet_min_box_side", 16)),
            seed=int(getattr(args, "imagenet_seed", 0)),
            regenerate_trials=bool(getattr(args, "imagenet_regenerate_trials", False)),
            use_bboxes=use_bboxes,
        )

        self.eval_type = getattr(args, "eval_type", "image")
        self.eval_include_sos_eos = bool(getattr(args, "eval_include_sos_eos", False))
        self.clip_eval = bool(getattr(args, "clip_eval", False))

        # Populated in setup()
        self.trials: List[dict] = []
        self.labels: List[str] = []
        self.instances: List[dict] = []
        self.vocab: Dict[str, int] = {}

    # ---------------------------
    # Vocab
    # ---------------------------

    def read_vocab(self) -> Dict[str, int]:
        _require_file_path(self.cfg.vocab_path, "vocab_filename")
        return read_vocab(self.cfg.vocab_path)

    # ---------------------------
    # Overrides + label map cache
    # ---------------------------

    def _load_overrides(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Supports overrides keyed by wnid or by raw class name phrase.
        Example JSON:
          {
            "n02123045": "cat",
            "tabby cat": "cat"
          }
        """
        p = self.cfg.label_overrides_path
        if p is None:
            return {}, {}
        if not p.exists():
            raise FileNotFoundError(f"[imagenet_eval] overrides json not found: {p}")
        if p.is_dir():
            raise IsADirectoryError(f"[imagenet_eval] overrides path is a directory: {p}")

        raw = _load_json(p)
        by_wnid: Dict[str, str] = {}
        by_name: Dict[str, str] = {}
        for k, v in raw.items():
            kk = str(k).strip()
            vv = _clean_phrase(str(v))
            if not vv or not _is_good_label(vv):
                continue
            if _WNID_RE.match(kk):
                by_wnid[kk] = vv
            else:
                by_name[_clean_phrase(kk)] = vv
        return by_wnid, by_name

    def _load_label_map_file(self) -> Optional[Dict[str, str]]:
        """
        Returns wnid -> mapped_label (for the CURRENT label_mode) if present and valid.
        If label_mode mismatches, or the map has invalid labels, returns None (forces rebuild).
        """
        p = self.cfg.label_map_path
        if p.exists() and p.is_dir():
            raise IsADirectoryError(f"[imagenet_eval] label_map_path is a directory (expected a file): {p}")
        if not p.exists():
            return None

        obj = _load_json(p)
        stats = obj.get("stats", {}) if isinstance(obj, dict) else {}
        mode_in_file = str(stats.get("label_mode", "")).strip().lower()
        if mode_in_file and mode_in_file != self.cfg.label_mode:
            print(
                f"[imagenet_eval] label-map cache label_mode mismatch: file={mode_in_file} "
                f"current={self.cfg.label_mode}. Rebuilding label map."
            )
            return None

        mp = None
        if isinstance(obj, dict) and "map" in obj and isinstance(obj["map"], dict):
            mp = obj["map"]
        elif isinstance(obj, dict):
            mp = obj
        else:
            return None

        out: Dict[str, str] = {}
        for k, v in mp.items():
            kk = str(k).strip()
            vv = _clean_phrase(str(v))
            if not kk or not vv:
                continue
            if not _is_good_label(vv):
                continue
            out[kk] = vv

        if not out:
            return None
        return out

    # ---------------------------
    # WNID -> names
    # ---------------------------

    def _load_wnid_names_source(self) -> Tuple[Dict[str, List[str]], str]:
        """
        Load wnid -> list of synonym phrases.
        Priority:
          1) words.txt
          2) meta.mat (requires scipy)
          3) empty dict (then per-wnid fallback to nltk WordNet)
        """
        if self.cfg.imagenet_words_txt is not None and self.cfg.imagenet_words_txt.exists():
            return _load_words_txt(self.cfg.imagenet_words_txt), f"words_txt:{self.cfg.imagenet_words_txt}"

        if self.cfg.imagenet_meta_mat is not None and self.cfg.imagenet_meta_mat.exists():
            mp = _load_meta_mat(self.cfg.imagenet_meta_mat)
            if mp:
                return mp, f"meta_mat:{self.cfg.imagenet_meta_mat}"
            return {}, f"meta_mat_unusable:{self.cfg.imagenet_meta_mat}"

        return {}, "wordnet_nltk"

    def _names_for_wnid(self, wnid: str, wnid2names: Dict[str, List[str]]) -> List[str]:
        names = wnid2names.get(wnid, []) or []
        if names:
            return names
        # fallback per wnid
        return _wnid_to_wordnet_names(wnid)

    # ---------------------------
    # Label picking / mapping
    # ---------------------------

    def _pick_label_for_wnid(
        self,
        wnid: str,
        names: List[str],
        vocab_set: set[str],
        overrides_by_wnid: Dict[str, str],
        overrides_by_name: Dict[str, str],
    ) -> Optional[str]:
        """
        Return mapped label for this wnid under the current cfg.label_mode.
        """
        mode = self.cfg.label_mode

        # 1) explicit override by wnid
        if wnid in overrides_by_wnid:
            cand = _clean_phrase(overrides_by_wnid[wnid])
            if not _is_good_label(cand):
                return None
            if mode == "vocab":
                return cand if cand in vocab_set else None
            return cand

        # 2) derive from names list
        for nm in names:
            nn = _clean_phrase(nm)
            if not nn:
                continue

            # override by phrase
            if nn in overrides_by_name:
                cand = _clean_phrase(overrides_by_name[nn])
                if not _is_good_label(cand):
                    continue
                if mode == "vocab":
                    if cand in vocab_set:
                        return cand
                    continue
                return cand

            if mode == "raw":
                # minimal: first usable cleaned phrase
                if _is_good_label(nn):
                    return nn
                continue

            if mode == "canonical":
                # prefer full phrase, then fall back to head noun variants if needed
                for cand0 in _candidate_strings_phrase(nn, include_head=True):
                    if _is_good_label(cand0):
                        return cand0
                continue

            # mode == "vocab"
            mapped = _map_phrase_to_vocab_label(nn, vocab_set)
            if mapped is not None:
                return mapped

        return None

    def _build_label_map(self, wnids: List[str], vocab: Dict[str, int]) -> Dict[str, str]:
        """
        Build wnid -> mapped_label according to cfg.label_mode.
        """
        vocab_set = set(vocab.keys())
        overrides_by_wnid, overrides_by_name = self._load_overrides()
        wnid2names, source = self._load_wnid_names_source()

        label_map: Dict[str, str] = {}
        dropped: List[str] = []
        mapped_examples: List[Tuple[str, str]] = []

        for wnid in wnids:
            names = self._names_for_wnid(wnid, wnid2names)
            lab = self._pick_label_for_wnid(
                wnid=wnid,
                names=names,
                vocab_set=vocab_set,
                overrides_by_wnid=overrides_by_wnid,
                overrides_by_name=overrides_by_name,
            )
            if lab is None:
                dropped.append(wnid)
                continue

            # final mode checks
            if self.cfg.label_mode == "vocab" and lab not in vocab_set:
                dropped.append(wnid)
                continue
            if not _is_good_label(lab):
                dropped.append(wnid)
                continue

            label_map[wnid] = lab
            if len(mapped_examples) < 25:
                mapped_examples.append((wnid, lab))

        stats = {
            "label_mode": self.cfg.label_mode,
            "n_wnids_seen": len(wnids),
            "n_mapped": len(label_map),
            "n_dropped": len(dropped),
            "examples_mapped": mapped_examples,
            "examples_dropped": dropped[:25],
            "names_source": source,
        }

        _atomic_write_json({"map": label_map, "stats": stats}, self.cfg.label_map_path)
        print(f"[imagenet_eval] wrote label map: {self.cfg.label_map_path}")
        print(f"[imagenet_eval] label-map stats: {stats}")
        return label_map

    def _get_or_create_label_map(self, wnids: List[str], vocab: Dict[str, int]) -> Dict[str, str]:
        if not self.cfg.regenerate_label_map:
            loaded = self._load_label_map_file()
            if loaded is not None:
                # loaded contains wnid->label entries. missing wnids are treated as dropped.
                return loaded
        return self._build_label_map(wnids, vocab)

    # ---------------------------
    # Instances
    # ---------------------------

    def _collect_instances(self) -> Tuple[List[dict], List[str]]:
        """
        Collect instances from val images + their xml.

        Returns:
          (instances, wnids_unique)

        Each instance dict:
          {
            "file_name": "<ILSVRC2012_val_...JPEG>",
            "wnid": "<n########>",
            "bbox": [x, y, w, h] or None
          }
        """
        img_dir = self.cfg.imagenet_images_dir
        xml_dir = self.cfg.imagenet_bbox_xml_dir

        img_paths = sorted(img_dir.glob("ILSVRC2012_val_*.JPEG"))
        if not img_paths:
            raise FileNotFoundError(f"[imagenet_eval] No ILSVRC2012_val_*.JPEG found in: {img_dir}")

        instances: List[dict] = []
        wnids: List[str] = []

        for img_path in img_paths:
            xml_path = xml_dir / (img_path.stem + ".xml")
            if not xml_path.exists():
                continue

            wnid, bbox = _parse_imagenet_bbox_xml(xml_path)
            if wnid is None:
                continue

            if bbox is not None:
                _, _, w, h = bbox
                if (w * h) < self.cfg.min_box_area:
                    continue
                if w < self.cfg.min_box_side or h < self.cfg.min_box_side:
                    continue

            instances.append(
                {
                    "file_name": img_path.name,
                    "wnid": wnid,
                    "bbox": bbox,
                }
            )
            wnids.append(wnid)

        wnids_unique = sorted(list({w for w in wnids}))
        if not instances:
            raise RuntimeError(
                "[imagenet_eval] Found 0 usable instances. Check:\n"
                f"  imagenet_images_dir: {img_dir}\n"
                f"  imagenet_bbox_xml_dir: {xml_dir}\n"
                "and that XML filenames match image stems."
            )

        return instances, wnids_unique

    # ---------------------------
    # Trials cache validation
    # ---------------------------

    def _existing_trials_are_valid(self, vocab: Dict[str, int]) -> bool:
        """
        Validate cached trials against current label_mode to prevent accidental
        reuse of vocab-mode trials in canonical/raw (or vice versa).
        """
        p = self.cfg.eval_metadata_path
        if not p.exists():
            return False
        try:
            meta = _load_json(p)
        except Exception:
            return False

        cfg = meta.get("config", {}) if isinstance(meta, dict) else {}
        mode_in_file = str(cfg.get("label_mode", "")).strip().lower()
        if mode_in_file and mode_in_file != self.cfg.label_mode:
            return False

        labels = meta.get("labels", None)
        if labels is None:
            data = meta.get("data", [])
            labels = sorted({t.get("target_category", "") for t in data if isinstance(t, dict)})

        if not isinstance(labels, list) or not labels:
            return False

        bad: List[str] = []
        if self.cfg.label_mode == "vocab":
            for l in labels:
                ll = str(l)
                if (not _is_good_label(ll)) or (ll not in vocab):
                    bad.append(ll)
        else:
            for l in labels:
                ll = str(l)
                if not _is_good_label(ll):
                    bad.append(ll)

        if bad:
            return False

        return True

    # ---------------------------
    # Trial generation
    # ---------------------------

    def _generate_trials(self, vocab: Dict[str, int]) -> None:
        cfg = self.cfg

        _require_dir_path(cfg.imagenet_images_dir, "imagenet_images_dir")
        _require_dir_path(cfg.imagenet_bbox_xml_dir, "imagenet_bbox_dir")
        _require_file_path(cfg.vocab_path, "vocab_filename")
        _require_file_path(cfg.label_map_path, "imagenet_label_map_path")

        if cfg.eval_metadata_path.exists() and not cfg.regenerate_trials:
            if self._existing_trials_are_valid(vocab):
                return

        # Collect instances from filesystem
        instances, wnids = self._collect_instances()

        # Build or load label map for this label_mode
        label_map = self._get_or_create_label_map(wnids, vocab)

        rng = np.random.default_rng(cfg.seed)
        vocab_set = set(vocab.keys())

        pools: Dict[str, List[dict]] = {}        # mapped_label -> list[instance]
        wnid_sources: Dict[str, List[str]] = {}  # mapped_label -> list[wnid] that contribute

        for inst in instances:
            wnid = inst["wnid"]
            mapped = label_map.get(wnid, None)
            if mapped is None:
                continue
            mapped = _clean_phrase(mapped)

            if not _is_good_label(mapped):
                continue
            if cfg.label_mode == "vocab" and mapped not in vocab_set:
                continue

            pools.setdefault(mapped, []).append(inst)
            wnid_sources.setdefault(mapped, [])
            if wnid not in wnid_sources[mapped]:
                wnid_sources[mapped].append(wnid)

        labels = sorted([k for k, v in pools.items() if len(v) > 0])

        if len(labels) < cfg.n_foils + 1:
            raise RuntimeError(
                f"[imagenet_eval] Not enough labels after mapping/filtering. "
                f"Have {len(labels)} labels; need at least {cfg.n_foils + 1}.\n"
                f"label_mode={cfg.label_mode} | label_map_path={cfg.label_map_path}"
            )

        # Cap per label
        for lab in labels:
            lst = pools[lab]
            rng.shuffle(lst)
            pools[lab] = lst[: cfg.max_images_per_label]

        # Create trials
        trials: List[dict] = []
        trial_num = 0
        for target_lab in labels:
            foil_candidates = [l for l in labels if l != target_lab]
            for t_inst in pools[target_lab]:
                for _ in range(cfg.n_repeats):
                    foil_labs = rng.choice(foil_candidates, size=cfg.n_foils, replace=False).tolist()
                    foil_instances: List[dict] = []
                    for fl in foil_labs:
                        foil_pool = pools[fl]
                        foil_instances.append(foil_pool[int(rng.integers(0, len(foil_pool)))])
                    trials.append(
                        {
                            "trial_num": trial_num,
                            "target_category": target_lab,
                            "foil_categories": foil_labs,
                            "target_instance": t_inst,
                            "foil_instances": foil_instances,
                        }
                    )
                    trial_num += 1

        # Store instances used (union of pools) for neuron probe collection
        used_instances: List[dict] = []
        for lab in labels:
            used_instances.extend(pools[lab])

        meta = {
            "data": trials,
            "labels": labels,
            "instances": used_instances,
            "label_map_path": str(cfg.label_map_path),
            "wnid_sources": wnid_sources,
            "config": {
                "label_mode": cfg.label_mode,
                "n_foils": cfg.n_foils,
                "n_repeats": cfg.n_repeats,
                "max_images_per_label": cfg.max_images_per_label,
                "min_box_area": cfg.min_box_area,
                "min_box_side": cfg.min_box_side,
                "seed": cfg.seed,
                "use_bboxes": cfg.use_bboxes,
                "words_txt": str(cfg.imagenet_words_txt) if cfg.imagenet_words_txt else None,
                "meta_mat": str(cfg.imagenet_meta_mat) if cfg.imagenet_meta_mat else None,
                "images_dir": str(cfg.imagenet_images_dir),
                "bbox_xml_dir": str(cfg.imagenet_bbox_xml_dir),
            },
        }
        _atomic_write_json(meta, cfg.eval_metadata_path)
        print(f"[imagenet_eval] wrote trials: {cfg.eval_metadata_path} (n={len(trials)})")
        print(f"[imagenet_eval] n_labels: {len(labels)} | label_mode={cfg.label_mode}")

    # ---------------------------
    # Lightning hooks
    # ---------------------------

    def setup(self, *args, **kwargs) -> None:
        vocab = self.read_vocab()
        self._generate_trials(vocab)

        meta = _load_json(self.cfg.eval_metadata_path)
        self.trials = meta["data"]
        self.labels = meta.get("labels", sorted({t["target_category"] for t in self.trials}))
        self.instances = meta.get("instances", [])
        self.vocab = vocab

        # Sanity checks
        bad = [l for l in self.labels if not _is_good_label(str(l))]
        if bad:
            raise RuntimeError(
                f"[imagenet_eval] Found invalid labels in loaded trials: {bad[:20]}.\n"
                "Delete the cached eval file / label map or regenerate:\n"
                "  --imagenet_regenerate_label_map --imagenet_regenerate_trials"
            )

        if self.cfg.label_mode == "vocab":
            missing = [l for l in self.labels if str(l) not in self.vocab]
            if missing:
                raise RuntimeError(
                    f"[imagenet_eval] label_mode=vocab but some labels are not in vocab: {missing[:20]}.\n"
                    "Regenerate caches with:\n"
                    "  --imagenet_regenerate_label_map --imagenet_regenerate_trials"
                )

        self.eval_dataset = ImageNetForcedChoiceEvalDataset(
            trials=self.trials,
            vocab=self.vocab,
            imagenet_images_dir=self.cfg.imagenet_images_dir,
            eval_type=self.eval_type,
            eval_include_sos_eos=self.eval_include_sos_eos,
            clip_eval=self.clip_eval,
            use_bboxes=self.cfg.use_bboxes,
        )

    def test_dataloader(self, shuffle: bool = False):
        return DataLoader(
            self.eval_dataset,
            collate_fn=multiModalDataset_collate_fn,
            shuffle=shuffle,
            batch_size=1,
            num_workers=4,
            pin_memory=False,
        )
