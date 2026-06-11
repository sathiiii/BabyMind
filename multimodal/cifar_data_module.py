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
from torchvision.datasets import CIFAR10, CIFAR100
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
# Utilities
# ---------------------------

_ALLOW_SHORT = {"tv"}  # allow short-but-legit labels
_BAD_LABELS = {
    # common function words (avoid garbage mapping if they exist in vocab)
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "from", "by", "at",
    # number-ish words (avoid mapping to these if they exist in vocab)
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
}

# Helpful aliases/synonyms for CIFAR names -> typical vocab words
# (Used only as fallbacks if exact matching fails.)
_ALIAS = {
    # CIFAR-10
    "automobile": ["car"],
    "airplane": ["plane"],
    # CIFAR-100
    "telephone": ["phone", "cellphone", "cell phone"],
    "television": ["tv"],
    "couch": ["sofa"],
    "pickup truck": ["truck"],
    "lawn mower": ["mower"],
    "streetcar": ["tram", "trolley", "car"],
    "sweet pepper": ["pepper"],
    "maple tree": ["tree"],
    "oak tree": ["tree"],
    "palm tree": ["tree"],
    "pine tree": ["tree"],
    "willow tree": ["tree"],
    "aquarium fish": ["fish"],
    "flatfish": ["fish"],
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
    # only error if it exists and is a directory (common bug: "" -> ".")
    if p.exists() and p.is_dir():
        raise IsADirectoryError(f"[cifar_eval] {name} points to a directory (expected a file): {p}")

def _require_dir_path(p: Path, name: str) -> None:
    if p.exists() and not p.is_dir():
        raise NotADirectoryError(f"[cifar_eval] {name} is not a directory: {p}")

def _normalize_name(s: str) -> str:
    s = str(s).strip().lower()
    s = " ".join(s.split())
    return s

def _is_good_label(label: str) -> bool:
    lab = _normalize_name(label)
    if not lab:
        return False
    if re.fullmatch(r"[a-z]", lab):  # single letter
        return False
    if re.fullmatch(r"\d+", lab):  # digits
        return False
    if len(lab) < 3 and lab not in _ALLOW_SHORT:
        return False
    if lab in _BAD_LABELS:
        return False
    return True

def _candidate_strings(name: str) -> List[str]:
    """
    Candidate vocab strings from a raw class name:
    - normalize underscores to spaces
    - remove punctuation
    - phrase + underscore/no-space variants
    - head noun
    - token candidates
    """
    n = _normalize_name(name.replace("_", " "))
    n = re.sub(r"[^a-z0-9 ]+", " ", n)
    n = " ".join(n.split())

    toks = [t for t in n.split(" ") if t]
    head = toks[-1] if toks else n

    cands = []
    if n:
        cands += [n, n.replace(" ", "_"), n.replace(" ", "")]
    if head:
        cands += [head, head.replace(" ", "_"), head.replace(" ", "")]

    # also try individual tokens (head first)
    if len(toks) > 1:
        for t in [toks[-1]] + toks[:-1]:
            if t:
                cands += [t, t.replace(" ", "_"), t.replace(" ", "")]

    # unique, preserve order
    out, seen = [], set()
    for c in cands:
        c2 = _normalize_name(c)
        if c2 and c2 not in seen:
            seen.add(c2)
            out.append(c2)
    return out

def _expand_aliases(phrase_or_token: str) -> List[str]:
    key = _normalize_name(phrase_or_token.replace("_", " "))
    out = [key]
    for alt in _ALIAS.get(key, []):
        a = _normalize_name(str(alt).replace("_", " "))
        if a and a not in out:
            out.append(a)
    return out

def _map_cifar_class_to_vocab_label(
    cifar_class_name: str,
    vocab_set: set[str],
) -> Optional[str]:
    """
    Returns a vocab key (string) to use as the evaluation label, or None if no match.
    Uses:
      1) exact / candidate phrase variants
      2) alias expansion on phrase
      3) token-level matching + aliases
    """
    raw = _normalize_name(cifar_class_name.replace("_", " "))
    if not raw:
        return None

    # 1) phrase candidates (exact + variants)
    for cand in _candidate_strings(raw):
        if cand in vocab_set and _is_good_label(cand):
            return cand

    # 2) phrase-level aliases
    for ali in _expand_aliases(raw):
        for cand in _candidate_strings(ali):
            if cand in vocab_set and _is_good_label(cand):
                return cand

    # 3) token-level fallback (head noun then others), with aliases
    toks = [t for t in re.sub(r"[^a-z0-9 ]+", " ", raw).split() if t]
    if toks:
        ordered = [toks[-1]] + toks[:-1] if len(toks) > 1 else toks
        for t in ordered:
            for ali in _expand_aliases(t):
                for cand in _candidate_strings(ali):
                    if cand in vocab_set and _is_good_label(cand):
                        return cand

    return None

# ---------------------------
# Dataset
# ---------------------------

class CIFARForcedChoiceEvalDataset(Dataset):
    def __init__(
        self,
        base_dataset,  # CIFAR10 or CIFAR100
        trials: List[dict],
        vocab: Dict[str, int],
        eval_type: str = "image",
        eval_include_sos_eos: bool = False,
        clip_eval: bool = False,
    ):
        self.base = base_dataset
        self.trials = trials
        self.vocab = vocab
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

    def _load_image_by_index(self, idx: int) -> torch.Tensor:
        arr = self.base.data[idx]  # uint8 HxWxC
        im = Image.fromarray(arr).convert("RGB")
        return self.transform(im)

    def __getitem__(self, idx: int):
        trial = self.trials[idx]
        target_label = trial["target_category"]
        foil_labels = trial["foil_categories"]

        if self.eval_type == "image":
            foil_instances = trial["foil_instances"]
            n_imgs = 1 + len(foil_instances)

            imgs = torch.zeros((n_imgs, 3, IMAGE_H, IMAGE_W))
            imgs[0] = self._load_image_by_index(trial["target_instance"]["index"])
            for j, f_inst in enumerate(foil_instances):
                imgs[j + 1] = self._load_image_by_index(f_inst["index"])

            label, label_len = self._encode_label(target_label)
            return imgs, label, label_len, [target_label]

        if self.eval_type == "text":
            img = torch.zeros((1, 3, IMAGE_H, IMAGE_W))
            img[0] = self._load_image_by_index(trial["target_instance"]["index"])

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
class CIFAREvalConfig:
    cifar_data_dir: Path
    cifar_dataset: str  # "cifar10" or "cifar100"

    eval_metadata_path: Path
    vocab_path: Path

    label_map_path: Path
    label_overrides_path: Optional[Path] = None
    regenerate_label_map: bool = False

    n_foils: int = 3
    n_repeats: int = 1
    max_images_per_label: int = 100  # CIFAR100 test has 100 imgs/class; CIFAR10 test has 1000

    seed: int = 0
    regenerate_trials: bool = False

    label_mode: str = "vocab"  # "vocab" | "canonical" | "raw"


class CIFARForcedChoiceDataModule(pl.LightningDataModule):
    """
    Forced-choice evaluation on CIFAR-10 or CIFAR-100 (test split, local files).

    - Loads CIFAR from extracted "python version" directories using torchvision (download=False)
    - Builds a CIFAR-class-name -> vocab-label map automatically
    - Filters to labels present in your CVCL vocab
    - Generates trials: target + N foils (default N=3 => 4-way forced-choice)
    - Writes:
        Trials:    EVAL_DATA_DIR / <eval_metadata_filename>
        Label map: EVAL_DATA_DIR / <eval_metadata_stem>_label_map.json
    """

    def __init__(self, args=None) -> None:
        super().__init__()
        self.args = args

        data_dir = Path(getattr(args, "cifar_data_dir", "eval_datasets/cifar")).expanduser()

        # Determine dataset: prefer explicit --cifar_dataset, else infer from eval_dataset
        ds = getattr(args, "cifar_dataset", None)
        if ds is None:
            ds = getattr(args, "eval_dataset", "cifar10")
        ds = str(ds).strip().lower()
        if ds in ("10", "cifar-10", "cifar_10"):
            ds = "cifar10"
        if ds in ("100", "cifar-100", "cifar_100"):
            ds = "cifar100"
        if ds not in ("cifar10", "cifar100"):
            raise ValueError(f"[cifar_eval] Unknown cifar_dataset: {ds}")

        eval_meta = Path(getattr(args, "eval_metadata_filename", f"eval_{ds}.json"))
        eval_meta_path = (EVAL_DATA_DIR / eval_meta).resolve()

        label_map_arg = _as_path_or_none(getattr(args, "cifar_label_map_path", None))
        label_map_path = label_map_arg if label_map_arg is not None else eval_meta_path.with_name(eval_meta_path.stem + "_label_map.json")

        overrides_arg = _as_path_or_none(getattr(args, "cifar_label_overrides_json", None))

        vocab_arg = getattr(args, "vocab_filename", None) or getattr(args, "saycam_vocab_filename", None)
        vocab_path = _as_path_or_none(vocab_arg)
        if vocab_path is None:
            candidates = [
                Path("vocab.json"),
                Path("expt_saycam") / "vocab.json",
                Path(os.environ.get("SAYCAM_VOCAB", str(EVAL_DATA_DIR / "vocab.json"))),
            ]
            vocab_path = next((p for p in candidates if p.exists()), candidates[-1])

        _require_file_path(vocab_path, "vocab_filename")
        _require_file_path(label_map_path, "cifar_label_map_path")
        _require_dir_path(data_dir, "cifar_data_dir")

        label_mode = str(getattr(args, "cifar_label_mode", "vocab")).strip().lower()
        if label_mode not in {"vocab", "canonical", "raw"}:
            raise ValueError(f"[cifar_eval] Unknown cifar_label_mode={label_mode} (expected vocab|canonical|raw)")
        
        self.cfg = CIFAREvalConfig(
            cifar_data_dir=data_dir,
            cifar_dataset=ds,
            eval_metadata_path=eval_meta_path,
            vocab_path=vocab_path,
            label_map_path=label_map_path,
            label_overrides_path=overrides_arg,
            regenerate_label_map=bool(getattr(args, "cifar_regenerate_label_map", False)),
            n_foils=int(getattr(args, "cifar_n_foils", 3)),
            n_repeats=int(getattr(args, "cifar_n_repeats", 1)),
            max_images_per_label=int(getattr(args, "cifar_max_images_per_label", 100)),
            seed=int(getattr(args, "cifar_seed", 0)),
            regenerate_trials=bool(getattr(args, "cifar_regenerate_trials", False)),
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
        Optional JSON mapping CIFAR class name -> vocab label.
        Example:
          {
            "automobile": "car",
            "telephone": "phone"
          }
        Keys are compared after normalization (underscores/spaces/punct collapsed).
        """
        p = self.cfg.label_overrides_path
        if p is None:
            return {}
        if not p.exists():
            raise FileNotFoundError(f"[cifar_eval] overrides json not found: {p}")
        if p.is_dir():
            raise IsADirectoryError(f"[cifar_eval] overrides path is a directory: {p}")

        raw = _load_json(p)
        out: Dict[str, str] = {}
        for k, v in raw.items():
            kk = _normalize_name(str(k).replace("_", " "))
            vv = _normalize_name(str(v).replace("_", " "))
            if kk and vv:
                out[kk] = vv
        return out

    def _load_label_map_file(self) -> Optional[Dict[str, str]]:
        p = self.cfg.label_map_path
        if p.exists() and p.is_dir():
            raise IsADirectoryError(f"[cifar_eval] label_map_path is a directory (expected a file): {p}")
        if not p.exists():
            return None

        obj = _load_json(p)
        if isinstance(obj, dict) and "map" in obj and isinstance(obj["map"], dict):
            return {str(k): _normalize_name(str(v)) for k, v in obj["map"].items()}
        if isinstance(obj, dict):
            return {str(k): _normalize_name(str(v)) for k, v in obj.items()}
        return None

    def _build_label_map(self, cifar_class_names: List[str], vocab: Dict[str, int]) -> Dict[str, str]:
        mode = self.cfg.label_mode
        vocab_set = set(vocab.keys())
        overrides = self._load_overrides()

        label_map: Dict[str, str] = {}
        mapped: List[Tuple[str, str]] = []
        dropped: List[str] = []

        def _clean_raw(name: str) -> str:
            s = _normalize_name(str(name).replace("_", " "))
            s = re.sub(r"[^a-z0-9 ]+", " ", s)
            return " ".join(s.split())

        for cname in cifar_class_names:
            raw_clean = _clean_raw(cname)
            key = _normalize_name(cname.replace("_", " "))

            # ----- overrides -----
            if key in overrides:
                tgt = _clean_raw(overrides[key])
                if mode == "vocab":
                    if tgt in vocab_set and _is_good_label(tgt):
                        label_map[cname] = tgt
                        mapped.append((cname, tgt))
                    else:
                        dropped.append(cname)
                else:
                    # canonical/raw: allow overrides even if not in vocab
                    if _is_good_label(tgt):
                        label_map[cname] = tgt
                        mapped.append((cname, tgt))
                    else:
                        dropped.append(cname)
                continue

            # ----- raw mode -----
            if mode == "raw":
                if _is_good_label(raw_clean):
                    label_map[cname] = raw_clean
                    mapped.append((cname, raw_clean))
                else:
                    dropped.append(cname)
                continue

            # ----- canonical/vocab: prefer vocab mapping if possible -----
            found = _map_cifar_class_to_vocab_label(cname, vocab_set)

            if found is not None:
                # found an in-vocab label (via exact/alias/token)
                label_map[cname] = found
                mapped.append((cname, found))
                continue

            if mode == "canonical":
                # keep the raw class label even if it is OOV
                if _is_good_label(raw_clean):
                    label_map[cname] = raw_clean
                    mapped.append((cname, raw_clean))
                else:
                    dropped.append(cname)
            else:
                # mode == "vocab": drop if not mappable into vocab
                dropped.append(cname)

        stats = {
            "dataset": self.cfg.cifar_dataset,
            "label_mode": mode,
            "n_cifar_classes": len(cifar_class_names),
            "n_mapped": len(label_map),
            "n_dropped": len(dropped),
            "examples_mapped": mapped[:25],
            "examples_dropped": dropped[:25],
        }

        _atomic_write_json({"map": label_map, "stats": stats}, self.cfg.label_map_path)
        print(f"[cifar_eval] wrote label map: {self.cfg.label_map_path}")
        print(f"[cifar_eval] label-map stats: {stats}")
        return label_map

    def _get_or_create_label_map(self, cifar_class_names: List[str], vocab: Dict[str, int]) -> Dict[str, str]:
        if not self.cfg.regenerate_label_map:
            loaded = self._load_label_map_file()
            if loaded is not None:
                return loaded
        return self._build_label_map(cifar_class_names, vocab)

    def _load_base_dataset(self):
        """
        Loads CIFAR test split from local extracted directories.
        """
        root = self.cfg.cifar_data_dir

        if self.cfg.cifar_dataset == "cifar10":
            # torchvision requires root/cifar-10-batches-py/
            need = root / "cifar-10-batches-py"
            if not need.exists():
                raise FileNotFoundError(
                    f"[cifar_eval] Missing {need}. Expected extracted CIFAR-10 python version under: {root}"
                )
            return CIFAR10(root=str(root), train=False, download=False)
        else:
            need = root / "cifar-100-python"
            if not need.exists():
                raise FileNotFoundError(
                    f"[cifar_eval] Missing {need}. Expected extracted CIFAR-100 python version under: {root}"
                )
            return CIFAR100(root=str(root), train=False, download=False)

    def _generate_trials(self, vocab: Dict[str, int]) -> None:
        cfg = self.cfg
        if cfg.eval_metadata_path.exists() and not cfg.regenerate_trials:
            return

        _require_dir_path(cfg.cifar_data_dir, "cifar_data_dir")
        _require_file_path(cfg.vocab_path, "vocab_filename")
        _require_file_path(cfg.label_map_path, "cifar_label_map_path")

        base = self._load_base_dataset()
        cifar_class_names = list(getattr(base, "classes", []))
        if not cifar_class_names:
            raise RuntimeError("[cifar_eval] torchvision CIFAR dataset has no .classes; cannot build mapping.")

        label_map = self._get_or_create_label_map(cifar_class_names, vocab)

        rng = np.random.default_rng(cfg.seed)

        # Build pools keyed by *final vocab label* (target_category)
        pools: Dict[str, List[int]] = {}  # label -> list of dataset indices
        sources: Dict[str, List[str]] = {}  # label -> original CIFAR class names that mapped into it

        # base.targets is list[int]
        targets = list(getattr(base, "targets", []))
        if len(targets) != len(base.data):
            raise RuntimeError("[cifar_eval] CIFAR base dataset targets/data length mismatch.")

        for idx, y in enumerate(targets):
            cname = cifar_class_names[int(y)]
            mapped = label_map.get(cname, None)
            if mapped is None:
                continue
            if cfg.label_mode == "vocab" and mapped not in vocab:
                continue
            if not _is_good_label(mapped):
                continue

            pools.setdefault(mapped, []).append(idx)
            sources.setdefault(mapped, [])
            if cname not in sources[mapped]:
                sources[mapped].append(cname)

        labels = sorted([k for k, v in pools.items() if len(v) > 0])
        if len(labels) < cfg.n_foils + 1:
            raise RuntimeError(
                f"[cifar_eval] Not enough in-vocab labels after mapping/filtering. "
                f"Have {len(labels)} labels; need at least {cfg.n_foils + 1}."
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
            for t_idx in pools[target_lab]:
                for _ in range(cfg.n_repeats):
                    foil_labs = rng.choice(foil_candidates, size=cfg.n_foils, replace=False).tolist()
                    foil_indices = []
                    for fl in foil_labs:
                        pool = pools[fl]
                        foil_indices.append(int(pool[int(rng.integers(0, len(pool)))]))

                    trials.append({
                        "trial_num": trial_num,
                        "target_category": target_lab,
                        "foil_categories": foil_labs,
                        "target_instance": {"index": int(t_idx)},
                        "foil_instances": [{"index": int(fi)} for fi in foil_indices],
                    })
                    trial_num += 1

        meta = {
            "data": trials,
            "labels": labels,
            "label_map_path": str(cfg.label_map_path),
            "sources": sources,
            "config": {
                "dataset": cfg.cifar_dataset,
                "n_foils": cfg.n_foils,
                "n_repeats": cfg.n_repeats,
                "max_images_per_label": cfg.max_images_per_label,
                "seed": cfg.seed,
            },
        }
        _atomic_write_json(meta, cfg.eval_metadata_path)
        print(f"[cifar_eval] wrote trials: {cfg.eval_metadata_path} (n={len(trials)})")
        print(f"[cifar_eval] n_labels: {len(labels)}")

    def setup(self, *args, **kwargs) -> None:
        vocab = self.read_vocab()
        self._generate_trials(vocab)

        meta = _load_json(self.cfg.eval_metadata_path)
        self.trials = meta["data"]
        self.labels = meta.get("labels", sorted({t["target_category"] for t in self.trials}))
        self.vocab = vocab

        self.base_dataset = self._load_base_dataset()

        self.eval_dataset = CIFARForcedChoiceEvalDataset(
            base_dataset=self.base_dataset,
            trials=self.trials,
            vocab=self.vocab,
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
