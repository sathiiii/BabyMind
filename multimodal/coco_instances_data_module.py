# multimodal/coco_instances_data_module.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import clip

from multimodal.multimodal_data_module import (
    EVAL_DATA_DIR,
    IMAGE_H, IMAGE_W,
    normalizer,
    multiModalDataset_collate_fn,
    read_vocab,
    SOS_TOKEN_ID, EOS_TOKEN_ID,
    UNK_TOKEN_ID,
)

# ---------------------------
# Small utilities
# ---------------------------

def _atomic_write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _normalize_name(s: str) -> str:
    s = str(s).strip().lower()
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


# Reject garbage labels that can exist in conversational vocabs
_BAD_LABELS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "from", "by", "at",
    "is", "it", "this", "that", "these", "those",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "old", "new", "young",
    "thing", "stuff", "piece",
    # COCO-specific "bad label" pitfalls if we fall back to non-head tokens
    "parking",
}

_ALLOW_SHORT = {"tv"}


def _is_good_label(label: str) -> bool:
    lab = _normalize_name(label)
    if not lab:
        return False
    if re.fullmatch(r"[a-z]", lab):
        return False
    if re.fullmatch(r"\d+", lab):
        return False
    if len(lab) < 3 and lab not in _ALLOW_SHORT:
        return False
    if lab in _BAD_LABELS:
        return False
    return True


def _clean_phrase(name: str) -> str:
    n = _normalize_name(name).replace("_", " ")
    n = re.sub(r"[^a-z0-9 ]+", " ", n)
    n = " ".join(n.split())
    return n


def _candidate_strings_phrase(name: str, include_head: bool = True) -> List[str]:
    """
    Candidate strings from a phrase:
      - normalized phrase
      - underscore/no-space variants
      - optional head noun variants (singular + plural)
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
        head = _singularize(toks[-1]) if toks else n
        head_pl = _pluralize(head)
        cands.extend([head, head_pl])

    out: List[str] = []
    seen = set()
    for c in cands:
        cc = _normalize_name(c)
        if cc and cc not in seen:
            seen.add(cc)
            out.append(cc)
    return out


def _head_tokens(name: str) -> List[str]:
    """
    COCO category names are curated, and the head noun is usually the best fallback.
    We intentionally avoid using other tokens (like 'parking' in 'parking meter').
    """
    n = _clean_phrase(name)
    toks = [t for t in n.split(" ") if t]
    if not toks:
        return []
    head = _singularize(toks[-1])
    return list(dict.fromkeys([head, _pluralize(head)]))


def _crop_square_from_bbox(im: Image.Image, bbox_xywh: List[float]) -> Image.Image:
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


def _as_path_or_none(x) -> Optional[Path]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    return Path(s).expanduser()


def _require_file_path(p: Path, name: str) -> None:
    if p.exists() and p.is_dir():
        raise IsADirectoryError(f"[coco_eval] {name} points to a directory (expected a file): {p}")


def _require_dir_path(p: Path, name: str) -> None:
    if p.exists() and not p.is_dir():
        raise NotADirectoryError(f"[coco_eval] {name} is not a directory: {p}")


# ---------------------------
# COCO-specific mapping table
# ---------------------------
# Keys are COCO category names (normalized).
# Values are ordered candidate vocab labels (normalized).
# - If the list is empty, we DROP the category by default (unless user override JSON maps it).
#
# This table is designed for your provided vocab, for example:
#   laptop -> computer (since "laptop" is not in vocab, but "computer" is).
#
# You can always override with --coco_label_overrides_json.
_COCO_CLASS_CANDIDATES: Dict[str, List[str]] = {
    # vehicles and electronics
    "motorcycle": ["bike", "bicycle"],
    "laptop": ["computer", "ipad", "monitor"],
    "cell phone": ["phone"],
    "refrigerator": ["fridge"],

    # household / furniture / containers
    "handbag": ["purse", "bag"],
    "suitcase": ["bag", "box", "case"],
    "potted plant": ["plant", "pot"],
    "dining table": ["table"],
    "vase": ["pot"],

    # signs / street stuff
    "traffic light": ["light"],
    "stop sign": ["sign"],

    # sports / toys / tools
    "sports ball": ["ball"],
    "baseball glove": ["gloves"],
    "baseball bat": ["stick"],
    "tennis racket": ["stick"],
    "frisbee": ["toy"],

    # food
    "wine glass": ["glass"],
    "hot dog": ["sausage"],
    "donut": ["cookie", "cake"],

    # animals / toys
    "teddy bear": ["teddy", "bear"],

    # appliances
    "hair drier": ["dryer"],
    "toaster": ["toast", "oven"],

    # hard-to-map categories: drop by default
    "parking meter": [],
    "fire hydrant": [],
    "tie": [],
    "skis": [],
    "snowboard": [],
    "surfboard": [],
    "broccoli": [],
    "remote": [],
    "keyboard": [],
}


# ---------------------------
# Dataset
# ---------------------------

class COCOForcedChoiceEvalDataset(Dataset):
    def __init__(
        self,
        trials: List[dict],
        vocab: Dict[str, int],
        coco_images_dir: Path,
        eval_type: str = "image",
        eval_include_sos_eos: bool = False,
        clip_eval: bool = False,
    ):
        self.trials = trials
        self.vocab = vocab
        self.coco_images_dir = coco_images_dir
        self.eval_type = eval_type
        self.eval_include_sos_eos = eval_include_sos_eos
        self.clip_eval = clip_eval

        if self.clip_eval:
            self.transform = transforms.Compose([
                transforms.Resize(IMAGE_H, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(IMAGE_H),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_H, IMAGE_W), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalizer,
            ])

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

    def __getitem__(self, idx: int):
        trial = self.trials[idx]
        target_label = trial["target_category"]
        foil_labels = trial["foil_categories"]

        if self.eval_type == "image":
            n_imgs = 1 + len(trial["foil_instances"])
            imgs = torch.zeros((n_imgs, 3, IMAGE_H, IMAGE_W))

            t_inst = trial["target_instance"]
            t_path = self.coco_images_dir / t_inst["file_name"]
            t_im = Image.open(t_path).convert("RGB")
            t_crop = _crop_square_from_bbox(t_im, t_inst["bbox"])
            imgs[0] = self.transform(t_crop)

            for j, f_inst in enumerate(trial["foil_instances"]):
                f_path = self.coco_images_dir / f_inst["file_name"]
                f_im = Image.open(f_path).convert("RGB")
                f_crop = _crop_square_from_bbox(f_im, f_inst["bbox"])
                imgs[j + 1] = self.transform(f_crop)

            label, label_len = self._encode_label(target_label)
            return imgs, label, label_len, [target_label]

        if self.eval_type == "text":
            t_inst = trial["target_instance"]
            t_path = self.coco_images_dir / t_inst["file_name"]
            t_im = Image.open(t_path).convert("RGB")
            t_crop = _crop_square_from_bbox(t_im, t_inst["bbox"])

            img = torch.zeros((1, 3, IMAGE_H, IMAGE_W))
            img[0] = self.transform(t_crop)

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
class COCOEvalConfig:
    coco_data_dir: Path
    coco_images_dir: Path
    instances_json: Path

    eval_metadata_path: Path
    vocab_path: Path

    label_map_path: Path
    label_overrides_path: Optional[Path] = None
    regenerate_label_map: bool = False

    n_foils: int = 3
    n_repeats: int = 5
    max_instances_per_label: int = 200
    min_box_area: float = 32 * 32
    min_box_side: float = 16
    seed: int = 0
    regenerate_trials: bool = False

    label_mode: str = "vocab"  # "vocab" | "canonical" | "raw"


class COCOInstancesDataModule(pl.LightningDataModule):
    """
    Forced-choice evaluation on COCO using instance crops from instances_val2017.json.

    Key behavior:
      - Build a COCO category -> vocab label map:
          * exact match if possible
          * else COCO-specific candidates table (covers laptop->computer, hot dog->sausage, etc)
          * else conservative fallback using phrase/head variants
          * never maps to "bad labels"
      - Filter to in-vocab mapped labels
      - Build Konkle-style trials: target crop + N foil crops
    """

    def __init__(self, args=None) -> None:
        super().__init__()
        self.args = args

        coco_dir = Path(getattr(args, "coco_data_dir", "eval_datasets/coco")).expanduser()

        coco_images_dir_arg = getattr(args, "coco_images_dir", None)
        coco_images_dir_p = _as_path_or_none(coco_images_dir_arg)
        images_dir = coco_images_dir_p if coco_images_dir_p is not None else (coco_dir / "val2017")

        coco_instances_json_arg = getattr(args, "coco_instances_json", None)
        coco_instances_json_p = _as_path_or_none(coco_instances_json_arg)
        inst_json = coco_instances_json_p if coco_instances_json_p is not None else (coco_dir / "annotations" / "instances_val2017.json")

        eval_meta = Path(getattr(args, "eval_metadata_filename", "eval_coco_instances.json"))
        eval_meta_path = (EVAL_DATA_DIR / eval_meta).resolve()

        label_map_arg = getattr(args, "coco_label_map_path", None)
        label_map_p = _as_path_or_none(label_map_arg)
        label_map_path = label_map_p if label_map_p is not None else eval_meta_path.with_name(eval_meta_path.stem + "_label_map.json")

        overrides_arg = getattr(args, "coco_label_overrides_json", None)
        overrides_path = _as_path_or_none(overrides_arg)

        vocab_arg = getattr(args, "vocab_filename", None) or getattr(args, "saycam_vocab_filename", None)
        vocab_path = _as_path_or_none(vocab_arg)
        if vocab_path is None:
            candidates = [
                Path("vocab.json"),
                Path("expt_saycam") / "vocab.json",
                Path(os.environ.get("SAYCAM_VOCAB", str(EVAL_DATA_DIR / "vocab.json"))),
            ]
            vocab_path = next((p for p in candidates if p.exists()), candidates[-1])

        _require_file_path(label_map_path, "coco_label_map_path")
        _require_file_path(vocab_path, "vocab_filename")

        label_mode = str(getattr(args, "coco_label_mode", "vocab")).strip().lower()
        if label_mode not in {"vocab", "canonical", "raw"}:
            raise ValueError(f"[coco_eval] Unknown coco_label_mode={label_mode} (expected vocab|canonical|raw)")

        self.cfg = COCOEvalConfig(
            coco_data_dir=coco_dir,
            coco_images_dir=images_dir,
            instances_json=inst_json,
            eval_metadata_path=eval_meta_path,
            vocab_path=vocab_path,
            label_map_path=label_map_path,
            label_overrides_path=overrides_path,
            regenerate_label_map=bool(getattr(args, "coco_regenerate_label_map", False)),
            n_foils=int(getattr(args, "coco_n_foils", 3)),
            n_repeats=int(getattr(args, "coco_n_repeats", 5)),
            max_instances_per_label=int(getattr(args, "coco_max_instances_per_label", 200)),
            min_box_area=float(getattr(args, "coco_min_box_area", 32 * 32)),
            min_box_side=float(getattr(args, "coco_min_box_side", 16)),
            seed=int(getattr(args, "coco_seed", 0)),
            regenerate_trials=bool(getattr(args, "coco_regenerate_trials", False)),
            label_mode=label_mode,
        )

        self.eval_type = getattr(args, "eval_type", "image")
        self.eval_include_sos_eos = bool(getattr(args, "eval_include_sos_eos", False))
        self.clip_eval = bool(getattr(args, "clip_eval", False))

    def read_vocab(self) -> Dict[str, int]:
        _require_file_path(self.cfg.vocab_path, "vocab_filename")
        return read_vocab(self.cfg.vocab_path)

    def _load_overrides(self) -> Dict[str, str]:
        """
        Overrides json maps COCO category names -> vocab labels.
        Example:
          { "parking meter": "money" }
        """
        if self.cfg.label_overrides_path is None:
            return {}
        if not self.cfg.label_overrides_path.exists():
            raise FileNotFoundError(f"Overrides json not found: {self.cfg.label_overrides_path}")
        if self.cfg.label_overrides_path.is_dir():
            raise IsADirectoryError(f"Overrides path is a directory: {self.cfg.label_overrides_path}")

        raw = _load_json(self.cfg.label_overrides_path)
        out: Dict[str, str] = {}
        for k, v in raw.items():
            kk = _normalize_name(k)
            vv = _normalize_name(v)
            if kk and vv:
                out[kk] = vv
        return out

    def _load_label_map_file(self) -> Optional[Dict[str, str]]:
        p = self.cfg.label_map_path
        if p.exists() and p.is_dir():
            raise IsADirectoryError(f"[coco_eval] label_map_path is a directory (expected a file): {p}")
        if not p.exists():
            return None

        obj = _load_json(p)
        if isinstance(obj, dict) and "map" in obj and isinstance(obj["map"], dict):
            return {_normalize_name(k): _normalize_name(v) for k, v in obj["map"].items()}
        if isinstance(obj, dict):
            return {_normalize_name(k): _normalize_name(v) for k, v in obj.items()}
        return None

    def _map_one_category(
        self,
        coco_name_raw: str,
        vocab_set: set,
        overrides: Dict[str, str],
    ) -> Tuple[Optional[str], str]:
        """
        Returns (mapped_label or None, reason).
        """
        cn = _normalize_name(coco_name_raw)

        # 0) user override wins
        if cn in overrides:
            vv = _normalize_name(overrides[cn])
            if vv in vocab_set and _is_good_label(vv):
                return vv, "override"
            return None, "override_oov_or_bad"

        # 1) exact match
        if cn in vocab_set and _is_good_label(cn):
            return cn, "exact"

        # 2) COCO-specific candidates (including "drop by default")
        if cn in _COCO_CLASS_CANDIDATES:
            cands = _COCO_CLASS_CANDIDATES[cn]
            if len(cands) == 0:
                return None, "blocked_default"

            for cand in cands:
                cand_n = _normalize_name(cand)
                # try a few variants just in case (usually cand is a single token)
                for v in _candidate_strings_phrase(cand_n, include_head=True):
                    if v in vocab_set and _is_good_label(v):
                        return v, f"coco_candidate:{cn}->{v}"

        # 3) phrase-level variants (no head yet)
        for cand in _candidate_strings_phrase(cn, include_head=False):
            if cand in vocab_set and _is_good_label(cand):
                return cand, "phrase_variant"

        # 4) head noun fallback
        for cand in _candidate_strings_phrase(cn, include_head=True):
            if cand in vocab_set and _is_good_label(cand):
                return cand, "head_variant"

        # 5) last resort: head token (singular/plural) only
        for tok in _head_tokens(cn):
            if tok in vocab_set and _is_good_label(tok):
                return tok, "head_token"

        return None, "no_match"

    def _build_label_map(self, coco_cat_names: List[str], vocab: Dict[str, int]) -> Dict[str, str]:
        vocab_set = set(vocab.keys())
        overrides = self._load_overrides()

        label_map: Dict[str, str] = {}
        per_category: List[dict] = []

        n_exact = 0
        n_mapped = 0
        n_dropped = 0

        for raw in coco_cat_names:
            cn = _normalize_name(raw)
            mapped, reason = self._map_one_category(cn, vocab_set=vocab_set, overrides=overrides)

            if mapped is None:
                n_dropped += 1
                per_category.append({"coco": cn, "mapped": None, "reason": reason})
                continue

            if mapped == cn:
                n_exact += 1
                per_category.append({"coco": cn, "mapped": cn, "reason": reason})
            else:
                n_mapped += 1
                label_map[cn] = mapped
                per_category.append({"coco": cn, "mapped": mapped, "reason": reason})

        stats = {
            "n_coco_categories": len(coco_cat_names),
            "n_exact_in_vocab": n_exact,
            "n_mapped_into_vocab": n_mapped,
            "n_dropped": n_dropped,
            "per_category": per_category,  # full list (only 80 rows)
            "note": (
                "Mapping uses: user overrides, exact match, COCO candidates table "
                "(covers laptop->computer), then conservative fallbacks."
            ),
        }

        _atomic_write_json({"map": label_map, "stats": stats}, self.cfg.label_map_path)
        print(f"[coco_eval] wrote label map: {self.cfg.label_map_path}")
        print(f"[coco_eval] label-map stats: { {k: stats[k] for k in ['n_coco_categories','n_exact_in_vocab','n_mapped_into_vocab','n_dropped']} }")
        return label_map

    def _get_or_create_label_map(self, coco_cat_names: List[str], vocab: Dict[str, int]) -> Dict[str, str]:
        if not self.cfg.regenerate_label_map:
            loaded = self._load_label_map_file()
            if loaded is not None:
                return loaded
        return self._build_label_map(coco_cat_names, vocab)

    def _generate_trials(self, vocab: Dict[str, int]) -> None:
        cfg = self.cfg
        if cfg.eval_metadata_path.exists() and not cfg.regenerate_trials:
            return

        if not cfg.instances_json.exists():
            raise FileNotFoundError(f"Missing COCO instances json: {cfg.instances_json}")
        if not cfg.coco_images_dir.exists():
            raise FileNotFoundError(f"Missing COCO images dir: {cfg.coco_images_dir}")
        if not cfg.vocab_path.exists():
            raise FileNotFoundError(f"Missing vocab file: {cfg.vocab_path}")

        _require_dir_path(cfg.coco_images_dir, "coco_images_dir")
        _require_file_path(cfg.instances_json, "coco_instances_json")
        _require_file_path(cfg.vocab_path, "vocab_filename")
        _require_file_path(cfg.label_map_path, "coco_label_map_path")

        coco = _load_json(cfg.instances_json)
        id2file = {im["id"]: im["file_name"] for im in coco["images"]}
        id2catname = {c["id"]: c["name"] for c in coco["categories"]}

        coco_cat_names = [c["name"] for c in coco["categories"]]
        label_map = self._get_or_create_label_map(coco_cat_names, vocab)

        rng = np.random.default_rng(cfg.seed)

        pools: Dict[str, List[dict]] = {}
        label_sources: Dict[str, List[str]] = {}

        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0) == 1:
                continue

            area = float(ann.get("area", 0.0))
            if area < cfg.min_box_area:
                continue

            x, y, w, h = ann["bbox"]
            if w < cfg.min_box_side or h < cfg.min_box_side:
                continue

            coco_name = _normalize_name(id2catname[ann["category_id"]])
            mapped = label_map.get(coco_name, coco_name)

            if cfg.label_mode == "vocab" and mapped not in vocab:
                continue
            if not _is_good_label(mapped):
                continue

            inst = {
                "image_id": ann["image_id"],
                "file_name": id2file[ann["image_id"]],
                "bbox": [float(x), float(y), float(w), float(h)],
                "category": mapped,
            }
            pools.setdefault(mapped, []).append(inst)
            label_sources.setdefault(mapped, [])
            if coco_name not in label_sources[mapped]:
                label_sources[mapped].append(coco_name)

        labels = sorted([k for k, v in pools.items() if len(v) > 0])
        if len(labels) < cfg.n_foils + 1:
            raise RuntimeError(f"Not enough in-vocab labels after mapping/filtering. Have {len(labels)}.")

        for lab in labels:
            lst = pools[lab]
            rng.shuffle(lst)
            pools[lab] = lst[: cfg.max_instances_per_label]

        trials: List[dict] = []
        trial_num = 0
        for target_lab in labels:
            foil_candidates = [l for l in labels if l != target_lab]
            target_instances = pools[target_lab]

            for t_inst in target_instances:
                for _ in range(cfg.n_repeats):
                    foil_labs = rng.choice(foil_candidates, size=cfg.n_foils, replace=False).tolist()
                    foil_instances = []
                    for fl in foil_labs:
                        foil_instances.append(pools[fl][int(rng.integers(0, len(pools[fl])))])
                    trials.append({
                        "trial_num": trial_num,
                        "target_category": target_lab,
                        "foil_categories": foil_labs,
                        "target_instance": t_inst,
                        "foil_instances": foil_instances,
                    })
                    trial_num += 1

        meta = {
            "data": trials,
            "labels": labels,
            "label_map_path": str(cfg.label_map_path),
            "label_sources": label_sources,
            "config": {
                "n_foils": cfg.n_foils,
                "n_repeats": cfg.n_repeats,
                "max_instances_per_label": cfg.max_instances_per_label,
                "min_box_area": cfg.min_box_area,
                "min_box_side": cfg.min_box_side,
                "seed": cfg.seed,
            }
        }
        _atomic_write_json(meta, cfg.eval_metadata_path)
        print(f"[coco_eval] wrote trials: {cfg.eval_metadata_path} (n={len(trials)})")
        print(f"[coco_eval] n_labels: {len(labels)}")

    def setup(self, *args, **kwargs) -> None:
        vocab = self.read_vocab()
        self._generate_trials(vocab)

        meta = _load_json(self.cfg.eval_metadata_path)
        self.trials = meta["data"]
        self.labels = meta.get("labels", sorted({t["target_category"] for t in self.trials}))
        self.vocab = vocab

        self.eval_dataset = COCOForcedChoiceEvalDataset(
            trials=self.trials,
            vocab=self.vocab,
            coco_images_dir=self.cfg.coco_images_dir,
            eval_type=self.eval_type,
            eval_include_sos_eos=self.eval_include_sos_eos,
            clip_eval=self.clip_eval,
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
