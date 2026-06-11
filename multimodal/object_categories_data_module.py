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


def _as_path_or_none(x) -> Optional[Path]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    return Path(s).expanduser()


def _require_file_path(p: Path, name: str) -> None:
    if p.exists() and p.is_dir():
        raise IsADirectoryError(f"[konkle_eval] {name} points to a directory (expected a file): {p}")


def _require_dir_path(p: Path, name: str) -> None:
    if p.exists() and not p.is_dir():
        raise NotADirectoryError(f"[konkle_eval] {name} is not a directory: {p}")


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


def _clean_phrase(name: str) -> str:
    n = _normalize_name(name).replace("_", " ")
    n = re.sub(r"[^a-z0-9 ]+", " ", n)
    n = " ".join(n.split())
    return n


_ALLOW_SHORT = {"tv"}
_BAD_LABELS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "from", "by", "at",
    "is", "it", "this", "that", "these", "those",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "thing", "stuff", "piece",
}


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


def _candidate_strings_phrase(name: str, include_head: bool = True) -> List[str]:
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
        cands.extend([head, _pluralize(head)])

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


# ---------------------------
# Synonyms (token + phrase)
# ---------------------------

_TOKEN_SYNONYMS: Dict[str, List[str]] = {
    "cat": ["kitty", "kitten", "kitties"],
    "dog": ["puppy", "doggy", "doggies"],
    "rabbit": ["bunny", "bunnies"],
    "horse": ["pony", "horses"],
    "bicycle": ["bike", "bikes"],
    "airplane": ["plane", "aeroplane", "aircraft"],
    "couch": ["sofa"],
    "television": ["tv"],
    "refrigerator": ["fridge"],
    "truck": ["lorry", "trucks"],
    "car": ["automobile", "cars"],
    "cellphone": ["phone"],
    "laptop": ["computer", "notebook"],
    "notebook": ["laptop", "computer"],
}

_PHRASE_SYNONYMS: Dict[str, List[str]] = {
    "cell phone": ["phone"],
    "cellphone": ["phone"],
    "mobile phone": ["phone"],
    "laptop computer": ["laptop", "computer"],
    "teddy bear": ["teddy", "bear"],
    "wine glass": ["glass"],
    "potted plant": ["plant", "pot"],
    "dining table": ["table"],
    "hot dog": ["sausage"],
}

_REV_TOKEN_SYNONYMS: Dict[str, List[str]] = {}
for canon, alts in _TOKEN_SYNONYMS.items():
    canon_n = _normalize_name(canon)
    group = [canon_n] + [_normalize_name(x) for x in alts]
    for a in group:
        _REV_TOKEN_SYNONYMS.setdefault(a, [])
        for b in group:
            if b not in _REV_TOKEN_SYNONYMS[a]:
                _REV_TOKEN_SYNONYMS[a].append(b)


def _synonym_candidates(name: str) -> List[str]:
    n = _clean_phrase(name)
    out: List[str] = []
    seen = set()

    def add(x: str) -> None:
        xx = _normalize_name(x)
        if xx and xx not in seen:
            seen.add(xx)
            out.append(xx)

    for c in _candidate_strings_phrase(n, include_head=False):
        add(c)

    if n in _PHRASE_SYNONYMS:
        for s in _PHRASE_SYNONYMS[n]:
            for c in _candidate_strings_phrase(s, include_head=True):
                add(c)

    toks = n.split(" ")
    if len(toks) == 1 and toks[0] in _REV_TOKEN_SYNONYMS:
        for s in _REV_TOKEN_SYNONYMS[toks[0]]:
            add(s)

    for h in _head_tokens(n):
        add(h)

    return out


# ---------------------------
# Dataset (forced-choice)
# ---------------------------

class KonkleForcedChoiceEvalDataset(Dataset):
    def __init__(
        self,
        trials: List[dict],
        vocab: Dict[str, int],
        images_root: Path,
        eval_type: str = "image",
        eval_include_sos_eos: bool = False,
        clip_eval: bool = False,
    ):
        self.trials = trials
        self.vocab = vocab
        self.images_root = images_root
        self.eval_type = eval_type
        self.eval_include_sos_eos = eval_include_sos_eos
        self.clip_eval = clip_eval

        if self.clip_eval:
            print("Using CLIP transforms for evaluation")
            # use CLIP transforms
            self.transform = transforms.Compose([
                transforms.Resize(IMAGE_H, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(IMAGE_H),
                # _convert_image_to_rgb,  # commeting out since we convert to RGB
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ])
        else:
            print("Using base transforms for evaluation")
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_H, IMAGE_W), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalizer,
            ])

    def __len__(self) -> int:
        return len(self.trials)

    def _encode_label(self, raw_label: str) -> Tuple[torch.Tensor, int]:
        if self.clip_eval:
            t = clip.tokenize(raw_label)
            return t, int(t.numel())

        token_id = self.vocab.get(raw_label, UNK_TOKEN_ID)
        ids = [token_id]
        if self.eval_include_sos_eos:
            ids = [SOS_TOKEN_ID] + ids + [EOS_TOKEN_ID]
        t = torch.LongTensor(ids)
        return t, len(ids)

    def _load_image(self, inst: dict) -> torch.Tensor:
        rel = inst["relpath"]
        p = self.images_root / rel
        im = Image.open(p).convert("RGB")
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
                labels = torch.cat(labels, dim=0)
            else:
                labels = torch.stack(labels, dim=0)

            return img, labels, lens, [target_label]

        raise ValueError(f"Unknown eval_type: {self.eval_type}")


# ---------------------------
# DataModule
# ---------------------------

@dataclass
class KonkleEvalConfig:
    konkle_data_dir: Path
    categories_dir: Path

    eval_metadata_path: Path
    vocab_path: Path

    label_map_path: Path
    label_overrides_path: Optional[Path] = None
    regenerate_label_map: bool = False

    label_mode: str = "vocab"  # "vocab" | "canonical" | "raw"

    n_foils: int = 3
    n_repeats: int = 5
    max_images_per_label: int = 200
    seed: int = 0
    regenerate_trials: bool = False


class KonkleObjectCategoriesDataModule(pl.LightningDataModule):
    """
    Konkle/Brady-style object category evaluation with forced-choice trials.

    label_mode:
      - "vocab": map categories into CVCL vocab (via overrides/exact/synonyms). Drop categories that cannot be mapped.
      - "canonical": canonicalize category strings (via overrides/synonyms/variants) but DO NOT require they be in vocab.
      - "raw": use cleaned raw folder names (underscores->spaces) with minimal normalization.
    """

    def __init__(self, args=None) -> None:
        super().__init__()
        self.args = args

        data_dir = Path(getattr(args, "konkle_data_dir", "eval_datasets/17-objects")).expanduser()

        categories_dir_arg = _as_path_or_none(getattr(args, "konkle_categories_dir", None))
        categories_dir = categories_dir_arg if categories_dir_arg is not None else (data_dir / "object_categories")

        eval_meta = Path(getattr(args, "eval_metadata_filename", "eval_konkle_object_categories.json"))
        eval_meta_path = (EVAL_DATA_DIR / eval_meta).resolve()

        label_map_arg = _as_path_or_none(getattr(args, "konkle_label_map_path", None))
        label_map_path = label_map_arg if label_map_arg is not None else eval_meta_path.with_name(
            eval_meta_path.stem + "_label_map.json"
        )

        overrides_arg = _as_path_or_none(getattr(args, "konkle_label_overrides_json", None))

        vocab_arg = getattr(args, "vocab_filename", None) or getattr(args, "saycam_vocab_filename", None)
        vocab_path = _as_path_or_none(vocab_arg)
        if vocab_path is None:
            candidates = [
                Path("vocab.json"),
                Path("expt_saycam") / "vocab.json",
                data_dir / "vocab.json",
                Path(os.environ.get("SAYCAM_VOCAB", str(EVAL_DATA_DIR / "vocab.json"))),
            ]
            vocab_path = next((p for p in candidates if p.exists()), candidates[-1])

        _require_file_path(vocab_path, "vocab_filename")
        _require_file_path(label_map_path, "konkle_label_map_path")
        _require_dir_path(categories_dir, "konkle_categories_dir")

        label_mode = str(getattr(args, "konkle_label_mode", "vocab")).strip().lower()
        if label_mode not in {"vocab", "canonical", "raw"}:
            raise ValueError(f"[konkle_eval] Unknown konkle_label_mode={label_mode} (expected vocab|canonical|raw)")

        self.cfg = KonkleEvalConfig(
            konkle_data_dir=data_dir,
            categories_dir=categories_dir,
            eval_metadata_path=eval_meta_path,
            vocab_path=vocab_path,
            label_map_path=label_map_path,
            label_overrides_path=overrides_arg,
            regenerate_label_map=bool(getattr(args, "konkle_regenerate_label_map", False)),
            label_mode=label_mode,
            n_foils=int(getattr(args, "konkle_n_foils", 3)),
            n_repeats=int(getattr(args, "konkle_n_repeats", 5)),
            max_images_per_label=int(getattr(args, "konkle_max_images_per_label", 200)),
            seed=int(getattr(args, "konkle_seed", 0)),
            regenerate_trials=bool(getattr(args, "konkle_regenerate_trials", False)),
        )

        self.eval_type = getattr(args, "eval_type", "image")
        self.eval_include_sos_eos = bool(getattr(args, "eval_include_sos_eos", False))
        self.clip_eval = bool(getattr(args, "clip_eval", False))

    def read_vocab(self) -> Dict[str, int]:
        _require_file_path(self.cfg.vocab_path, "vocab_filename")
        return read_vocab(self.cfg.vocab_path)

    def _load_overrides(self) -> Dict[str, str]:
        if self.cfg.label_overrides_path is None:
            return {}
        p = self.cfg.label_overrides_path
        if not p.exists():
            raise FileNotFoundError(f"[konkle_eval] overrides json not found: {p}")
        if p.is_dir():
            raise IsADirectoryError(f"[konkle_eval] overrides path is a directory: {p}")

        raw = _load_json(p)
        out: Dict[str, str] = {}
        for k, v in raw.items():
            kk = _normalize_name(k)
            vv = _clean_phrase(v)
            if kk and vv:
                out[kk] = vv
        return out

    def _load_label_map_file(self) -> Optional[Dict[str, str]]:
        p = self.cfg.label_map_path
        if p.exists() and p.is_dir():
            raise IsADirectoryError(f"[konkle_eval] label_map_path is a directory (expected a file): {p}")
        if not p.exists():
            return None

        obj = _load_json(p)
        if isinstance(obj, dict) and "map" in obj and isinstance(obj["map"], dict):
            return {_normalize_name(k): _normalize_name(v) for k, v in obj["map"].items()}
        if isinstance(obj, dict):
            return {_normalize_name(k): _normalize_name(v) for k, v in obj.items()}
        return None

    def _pick_canonical(self, raw_cat: str, overrides: Dict[str, str]) -> Optional[str]:
        rc = _normalize_name(raw_cat)

        if rc in overrides:
            vv = _clean_phrase(overrides[rc])
            if _is_good_label(vv):
                return vv
            return None

        cleaned = _clean_phrase(rc)
        if _is_good_label(cleaned):
            return cleaned

        for cand in _synonym_candidates(rc):
            if _is_good_label(cand):
                return cand

        return None

    def _build_label_map(self, raw_categories: List[str], vocab: Dict[str, int]) -> Dict[str, Optional[str]]:
        mode = self.cfg.label_mode
        vocab_set = set(vocab.keys())
        overrides = self._load_overrides()

        label_map: Dict[str, Optional[str]] = {}
        per_cat: List[dict] = []

        for raw in raw_categories:
            rc = _normalize_name(raw)

            if mode == "raw":
                mapped = _clean_phrase(rc)
                if _is_good_label(mapped):
                    label_map[rc] = mapped
                    per_cat.append({"raw": rc, "mapped": mapped, "reason": "raw_clean"})
                else:
                    label_map[rc] = None
                    per_cat.append({"raw": rc, "mapped": None, "reason": "raw_bad"})
                continue

            if mode == "canonical":
                mapped = self._pick_canonical(rc, overrides)
                if mapped is None:
                    per_cat.append({"raw": rc, "mapped": None, "reason": "canonical_no_match"})
                    label_map[rc] = None
                else:
                    per_cat.append({"raw": rc, "mapped": mapped, "reason": "canonical"})
                    label_map[rc] = mapped
                continue

            # mode == "vocab"
            if rc in overrides:
                vv = _normalize_name(overrides[rc])
                if vv in vocab_set and _is_good_label(vv):
                    label_map[rc] = vv
                    per_cat.append({"raw": rc, "mapped": vv, "reason": "override"})
                else:
                    label_map[rc] = None
                    per_cat.append({"raw": rc, "mapped": None, "reason": "override_oov_or_bad"})
                continue

            if rc in vocab_set and _is_good_label(rc):
                label_map[rc] = rc
                per_cat.append({"raw": rc, "mapped": rc, "reason": "exact"})
                continue

            mapped = None
            for cand in _synonym_candidates(rc):
                if cand in vocab_set and _is_good_label(cand):
                    mapped = cand
                    break

            if mapped is None:
                label_map[rc] = None
                per_cat.append({"raw": rc, "mapped": None, "reason": "vocab_no_match"})
                continue

            label_map[rc] = mapped
            per_cat.append({"raw": rc, "mapped": mapped, "reason": "synonym_or_variant"})

        stats = {
            "label_mode": mode,
            "n_raw_categories": len(raw_categories),
            "n_kept": int(sum(1 for v in label_map.values() if v is not None)),
            "n_dropped": int(sum(1 for v in label_map.values() if v is None)),
            "per_category": per_cat,
        }

        _atomic_write_json(
            {"map": {k: v for k, v in label_map.items() if v is not None}, "stats": stats},
            self.cfg.label_map_path,
        )
        print(f"[konkle_eval] wrote label map: {self.cfg.label_map_path}")
        print(
            f"[konkle_eval] label-map stats: "
            f"{{'label_mode': '{mode}', 'n_raw_categories': {stats['n_raw_categories']}, "
            f"'n_kept': {stats['n_kept']}, 'n_dropped': {stats['n_dropped']}}}"
        )

        return label_map

    def _get_or_create_label_map(self, raw_categories: List[str], vocab: Dict[str, int]) -> Dict[str, Optional[str]]:
        if not self.cfg.regenerate_label_map:
            loaded = self._load_label_map_file()
            if loaded is not None:
                out: Dict[str, Optional[str]] = {rc: loaded.get(_normalize_name(rc), None) for rc in raw_categories}
                return out
        return self._build_label_map(raw_categories, vocab)

    def _collect_instances(self) -> Tuple[List[dict], List[str]]:
        cat_dir = self.cfg.categories_dir
        if not cat_dir.exists():
            raise FileNotFoundError(f"[konkle_eval] categories_dir not found: {cat_dir}")

        category_dirs = sorted([p for p in cat_dir.iterdir() if p.is_dir()])
        if len(category_dirs) == 0:
            raise RuntimeError(f"[konkle_eval] No category subdirectories found under: {cat_dir}")

        instances: List[dict] = []
        raw_cats: List[str] = []

        exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
        for d in category_dirs:
            raw = _normalize_name(d.name)
            raw_cats.append(raw)

            img_paths = [p for p in d.iterdir() if p.is_file() and p.suffix in exts]
            img_paths = sorted(img_paths)

            for p in img_paths:
                rel = Path(d.name) / p.name
                instances.append({
                    "relpath": rel.as_posix(),
                    "raw_category": raw,
                })

        raw_cats_unique = sorted(list({c for c in raw_cats}))
        if len(instances) == 0:
            raise RuntimeError(f"[konkle_eval] Found 0 images under: {cat_dir}")

        return instances, raw_cats_unique

    def _generate_trials(self, vocab: Dict[str, int]) -> None:
        cfg = self.cfg
        if cfg.eval_metadata_path.exists() and not cfg.regenerate_trials:
            return

        _require_dir_path(cfg.categories_dir, "konkle_categories_dir")
        _require_file_path(cfg.vocab_path, "vocab_filename")
        _require_file_path(cfg.label_map_path, "konkle_label_map_path")

        instances, raw_categories = self._collect_instances()
        label_map = self._get_or_create_label_map(raw_categories, vocab)

        rng = np.random.default_rng(cfg.seed)

        pools: Dict[str, List[dict]] = {}
        label_sources: Dict[str, List[str]] = {}

        for inst in instances:
            raw = inst["raw_category"]
            mapped = label_map.get(raw, None)
            if mapped is None:
                continue

            if cfg.label_mode == "vocab" and mapped not in vocab:
                continue

            if not _is_good_label(mapped):
                continue

            pools.setdefault(mapped, []).append(inst)
            label_sources.setdefault(mapped, [])
            if raw not in label_sources[mapped]:
                label_sources[mapped].append(raw)

        labels = sorted([k for k, v in pools.items() if len(v) > 0])
        if len(labels) < cfg.n_foils + 1:
            raise RuntimeError(
                f"[konkle_eval] Not enough labels after mapping/filtering. "
                f"Have {len(labels)} labels, need at least {cfg.n_foils + 1}."
            )

        for lab in labels:
            lst = pools[lab]
            rng.shuffle(lst)
            pools[lab] = lst[: cfg.max_images_per_label]

        trials: List[dict] = []
        trial_num = 0

        for target_lab in labels:
            foil_candidates = [l for l in labels if l != target_lab]
            for t_inst in pools[target_lab]:
                for _ in range(cfg.n_repeats):
                    foil_labs = rng.choice(foil_candidates, size=cfg.n_foils, replace=False).tolist()
                    foil_instances = []
                    for fl in foil_labs:
                        foil_pool = pools[fl]
                        foil_instances.append(foil_pool[int(rng.integers(0, len(foil_pool)))])
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
            "instances": instances,
            "label_map_path": str(cfg.label_map_path),
            "label_sources": label_sources,
            "config": {
                "konkle_data_dir": str(cfg.konkle_data_dir),
                "categories_dir": str(cfg.categories_dir),
                "label_mode": cfg.label_mode,
                "n_foils": cfg.n_foils,
                "n_repeats": cfg.n_repeats,
                "max_images_per_label": cfg.max_images_per_label,
                "seed": cfg.seed,
            }
        }
        _atomic_write_json(meta, cfg.eval_metadata_path)
        print(f"[konkle_eval] wrote trials: {cfg.eval_metadata_path} (n={len(trials)})")
        print(f"[konkle_eval] n_labels: {len(labels)}")

    def setup(self, *args, **kwargs) -> None:
        vocab = self.read_vocab()
        self._generate_trials(vocab)

        meta = _load_json(self.cfg.eval_metadata_path)
        self.trials = meta["data"]
        self.labels = meta.get("labels", sorted({t["target_category"] for t in self.trials}))
        self.instances = meta.get("instances", [])
        self.vocab = vocab

        self.eval_dataset = KonkleForcedChoiceEvalDataset(
            trials=self.trials,
            vocab=self.vocab,
            images_root=self.cfg.categories_dir,
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
