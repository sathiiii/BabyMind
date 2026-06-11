#!/usr/bin/env python3

"""Evaluation script.

Supported modes:

1) eval_mode=cvcl
   Standard forced-choice evaluation using the CVCL checkpoint (or CLIP with --clip_eval).

2) eval_mode=neuron_classifier
   Hybrid OOD+OOV evaluation (default):
     - OOD / in-vocab trials: CVCL-style evaluation.
     - OOV trials: neuron-classifier evaluation (CLIP-Dissect style mapping).

   If you want everything to run via the neuron-classifier, add --nc_all_labels.

Notes:
  - "OOV" here means labels that cannot be represented by the CVCL training vocab.
  - For eval_type=text, a trial is treated as OOV for CVCL scoring if any candidate label
    (target or foil) is OOV, since the CVCL text encoder cannot score it meaningfully.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from multimodal.multimodal_data_module import (
    EVAL_DATA_DIR,
    EOS_TOKEN_ID,
    IMAGE_H,
    IMAGE_W,
    SOS_TOKEN_ID,
    UNK_TOKEN_ID,
    load_data,
    normalizer,
)
from multimodal.multimodal_lit import MultiModalLitModel
from multimodal.multimodal_saycam_data_module import (
    DATA_DIR,
    EXTRACTED_FRAMES_DIRNAME,
    MultiModalSAYCamDataModule,
)
from multimodal.object_categories_data_module import KonkleObjectCategoriesDataModule
from multimodal.coco_instances_data_module import COCOInstancesDataModule
from multimodal.imagenet_val_data_module import ImageNetValDataModule
from multimodal.cifar_data_module import CIFARForcedChoiceDataModule
from train import _setup_parser

import clip


EVAL_FRAMES_DIRNAME = EVAL_DATA_DIR / "eval"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------


def _seed_everything(seed: int = 0) -> None:
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# Labeled-T (temporal) helpers
# -----------------------------------------------------------------------------

_SAYCAM_TAIL_RE = re.compile(r"^(?P<clip>.+)_(?P<utt>\d{3})_(?P<frm>\d{2})$")

# Cache per (clip_id, utt_str) -> sorted frame paths in train_5fps
_LABELED_T_UTT_CACHE: Dict[Tuple[str, str], List[Path]] = {}


def _encode_cvcl_image_feats(model: torch.nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    """
    Returns normalized image embeddings: (N,D).
    Works for MultiModalLitModel (expects .model.encode_image).
    """
    if hasattr(model, "model") and hasattr(model.model, "encode_image"):
        out = model.model.encode_image(imgs)
    elif hasattr(model, "encode_image"):
        out = model.encode_image(imgs)
    else:
        raise RuntimeError("[labeled_t] Could not find encode_image on model.")

    feats = out[0] if isinstance(out, (tuple, list)) else out
    return feats


def _encode_cvcl_text_feats(
    model: torch.nn.Module, label_ids: torch.Tensor, label_len: torch.Tensor
) -> torch.Tensor:
    """
    Returns normalized text embeddings: (L,D).
    Works for MultiModalLitModel (expects .model.encode_text).
    """
    if hasattr(model, "model") and hasattr(model.model, "encode_text"):
        out = model.model.encode_text(label_ids, label_len)
    elif hasattr(model, "encode_text"):
        out = model.encode_text(label_ids, label_len)
    else:
        raise RuntimeError("[labeled_t] Could not find encode_text on model.")

    feats = out[0] if isinstance(out, (tuple, list)) else out
    return feats


def _labeled_t_list_utterance_frames(
    frames_root: Path, clip_id: str, utt_str: str
) -> List[Path]:
    key = (clip_id, utt_str)
    if key in _LABELED_T_UTT_CACHE:
        return _LABELED_T_UTT_CACHE[key]

    # Train frames are typically .jpg, but be permissive.
    candidates = list(frames_root.glob(f"{clip_id}_{utt_str}_*.jpg"))
    candidates += list(frames_root.glob(f"{clip_id}_{utt_str}_*.jpeg"))
    candidates += list(frames_root.glob(f"{clip_id}_{utt_str}_*.png"))

    def _frm_num(p: Path) -> int:
        m = _SAYCAM_TAIL_RE.match(p.stem)
        if m is None:
            return 0
        return int(m.group("frm"))

    candidates = sorted(candidates, key=_frm_num)
    _LABELED_T_UTT_CACHE[key] = candidates
    return candidates


def _labeled_t_build_clip_paths(
    anchor_img: str | Path,
    *,
    frames_root: Path,
    num_frames: int,
    stride: int,
) -> List[Path]:
    """
    Build a temporal clip (list of frame paths) centered on anchor frame.
    If we cannot map anchor->train_5fps, we fallback to repeating the anchor.
    """
    anchor_path = Path(anchor_img)
    if not anchor_path.exists():
        raise FileNotFoundError(f"[labeled_t] anchor image missing: {anchor_path}")

    M = int(num_frames)
    if M <= 1:
        return [anchor_path]

    s = max(int(stride), 1)

    m = _SAYCAM_TAIL_RE.match(anchor_path.stem)
    if m is None:
        # Unknown naming -> cannot find neighbors
        return [anchor_path] * M

    clip_id = m.group("clip")
    utt_str = m.group("utt")
    frm_local = int(m.group("frm"))

    frames = _labeled_t_list_utterance_frames(frames_root, clip_id, utt_str)
    if len(frames) == 0:
        return [anchor_path] * M

    # Find the anchor index (best-effort).
    center = None
    for i, p in enumerate(frames):
        if p.stem == anchor_path.stem:
            center = i
            break
    if center is None:
        center = max(0, min(frm_local, len(frames) - 1))

    # Sample indices around center with stride; clamp at ends (duplicates OK).
    half = M // 2
    idxs: List[int] = []
    for j in range(M):
        off = j - half
        idx = center + off * s
        idx = max(0, min(idx, len(frames) - 1))
        idxs.append(idx)

    return [frames[i] for i in idxs]


def _labeled_t_load_clip_tensor(paths: List[Path], tfm) -> torch.Tensor:
    """
    Returns (M,3,H,W) float tensor
    """
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        imgs.append(tfm(im))
    return torch.stack(imgs, dim=0)


def _labeled_t_combine_scores(
    s_global: torch.Tensor,
    s_obj: Optional[torch.Tensor],
    *,
    mode: str,
    alpha: float,
) -> torch.Tensor:
    """
    s_global: (N,) or (L,)
    s_obj:    (N,) or (L,) or None
    """
    if s_obj is None:
        return s_global

    mode = str(mode).lower()
    if mode == "alpha":
        a = float(alpha)
        return a * s_global + (1.0 - a) * s_obj
    if mode == "sum":
        return s_global + s_obj
    if mode == "max":
        return torch.maximum(s_global, s_obj)

    raise ValueError(f"[labeled_t] Unknown --labeled_t_combine='{mode}' (use alpha|sum|max)")


def _labeled_t_object_scores(
    model: torch.nn.Module,
    x_bag: torch.Tensor,          # (B,M,3,H,W)
    text_feat: torch.Tensor,      # (L,D)
    *,
    exclude_null: bool = True,
) -> Optional[torch.Tensor]:
    """
    Returns object score matrix (B,L) or None if the model doesn't expose object candidates.
    """
    if not hasattr(model, "_compute_object_candidates"):
        return None

    # Call the internal candidate builder (be signature-tolerant).
    try:
        out = model._compute_object_candidates(x_bag=x_bag, sam_mask=None, sam_cid=None)
    except TypeError:
        try:
            out = model._compute_object_candidates(x_bag, None, None)
        except Exception:
            return None
    except Exception:
        return None

    cand_emb = None
    cand_mask = None

    if isinstance(out, dict):
        cand_emb = out.get("cand_emb", None)
        cand_mask = out.get("cand_mask", None)
    elif isinstance(out, (tuple, list)) and len(out) >= 2:
        cand_emb, cand_mask = out[0], out[1]

    if cand_emb is None or cand_mask is None:
        return None

    cand_emb = cand_emb.to(x_bag.device)          # (B,R,D)
    cand_mask = cand_mask.to(x_bag.device).bool() # (B,R)

    B, R, D = cand_emb.shape
    L = int(text_feat.shape[0])

    # Optionally drop the (packed) null candidate: "last valid" per sample.
    use_mask = cand_mask.clone()
    if exclude_null:
        counts = cand_mask.sum(dim=1).tolist()
        for b in range(B):
            n = int(counts[b])
            if n > 0:
                use_mask[b, n - 1] = False

    # sims: (B,R,L)
    sims = torch.einsum("brd,ld->brl", cand_emb, text_feat)

    # mask invalid candidates
    sims = sims.masked_fill(~use_mask.unsqueeze(-1), float("-inf"))

    # If everything is -inf (no candidates), return very low scores so global dominates.
    all_bad = torch.isinf(sims).all(dim=1)  # (B,L)
    sims = torch.where(all_bad.unsqueeze(1), torch.full_like(sims, -1e9), sims)

    # max over candidates
    s_obj = sims.max(dim=1).values  # (B,L)
    return s_obj


def _cvcl_labeled_t_eval_loop(
    *,
    model: torch.nn.Module,
    eval_data: List[Any],
    classes: List[str],
    vocab: Optional[Dict[str, int]],
    args: argparse.Namespace,
    checkpoint_name: str,
    config: Dict[str, Any],
) -> Tuple[List[dict], Dict[str, Any]]:
    """
    Labeled-T evaluation:
      - build a temporal clip (bag of frames) for each candidate image in the forced-choice trial
      - score using max-over-time global similarity + max-over-candidates object similarity
      - combine scores (alpha/sum/max)
    """
    if args.clip_eval:
        raise ValueError("[labeled_t] Not supported with --clip_eval (needs MIL/object candidates).")
    if vocab is None:
        raise RuntimeError("[labeled_t] Requires a CVCL vocab.")

    # Transform for frames
    tfm = _default_cvcl_transform()

    frames_root = Path(getattr(args, "labeled_t_frames_root", str(EXTRACTED_FRAMES_DIRNAME))).expanduser()
    num_frames = int(getattr(args, "labeled_t_num_frames", 5))
    stride = int(getattr(args, "labeled_t_stride", 1))
    combine_mode = str(getattr(args, "labeled_t_combine", "alpha"))
    alpha = float(getattr(args, "labeled_t_alpha", 0.5))
    exclude_null = bool(getattr(args, "labeled_t_exclude_null", True))

    # Optional kitty relabeling (keep consistent with existing eval)
    if args.use_kitty_label and "kitty" in vocab:
        if "cat" in classes:
            classes = [c for c in classes if c != "cat"]
        if "kitty" not in classes:
            classes = classes + ["kitty"]

    correct_combined = {c: 0 for c in classes}
    total = {c: 0 for c in classes}

    # Extra diagnostics
    correct_global = {c: 0 for c in classes}
    correct_obj = {c: 0 for c in classes}

    results: List[dict] = []

    # Override MIL mask source for evaluation (no SAM on test)
    # (works if your LightningModule stores args as dict or Namespace)
    if getattr(args, "labeled_t_mask_source", None) is not None:
        ms = str(args.labeled_t_mask_source).lower()
        if hasattr(model, "args"):
            try:
                if isinstance(model.args, dict):
                    model.args["mil_mask_source"] = ms
                else:
                    setattr(model.args, "mil_mask_source", ms)
            except Exception:
                pass

    model.eval()

    with torch.inference_mode():
        for i, trial in enumerate(eval_data):
            # target label (per trial)
            class_label = None
            if isinstance(trial, dict):
                class_label = trial.get("target_category", None)
            if class_label is None:
                continue

            # kitty patch
            if args.use_kitty_label and (class_label == "cat") and ("kitty" in vocab):
                class_label = "kitty"

            if class_label not in correct_combined:
                # unexpected label (skip safely)
                continue

            total[class_label] += 1

            # ---- build clips depending on eval_type ----
            if args.eval_type == "image":
                target_img = trial["target_img_filename"]
                foil_imgs = list(trial.get("foil_img_filenames", []) or [])
                anchors = [target_img] + foil_imgs  # candidate images

                clip_paths = [
                    _labeled_t_build_clip_paths(
                        a, frames_root=frames_root, num_frames=num_frames, stride=stride
                    )
                    for a in anchors
                ]
                x_bag = torch.stack([_labeled_t_load_clip_tensor(p, tfm) for p in clip_paths], dim=0)
                # (B,M,3,H,W)
                x_bag = x_bag.to(device)

                B = int(x_bag.shape[0])
                M = int(x_bag.shape[1])

                # encode text: single label
                label_ids, label_len = _encode_cvcl_label_tensor(
                    vocab=vocab,
                    labels=[class_label],
                    include_sos_eos=args.eval_include_sos_eos,
                )
                label_ids = label_ids.to(device)
                label_len = label_len.to(device)

                text_feat = _encode_cvcl_text_feats(model, label_ids, label_len)  # (1,D)

                # global: max over time
                x_flat = x_bag.reshape(B * M, *x_bag.shape[2:])
                img_feat = _encode_cvcl_image_feats(model, x_flat).reshape(B, M, -1)  # (B,M,D)
                sims = torch.einsum("bmd,ld->bml", img_feat, text_feat).max(dim=1).values  # (B,1)
                s_global = sims[:, 0]  # (B,)

                # object: max over candidates
                s_obj_mat = _labeled_t_object_scores(
                    model, x_bag, text_feat, exclude_null=exclude_null
                )
                s_obj = None if s_obj_mat is None else s_obj_mat[:, 0]  # (B,)

                s_comb = _labeled_t_combine_scores(s_global, s_obj, mode=combine_mode, alpha=alpha)

                pred_g = int(torch.argmax(s_global).item())
                pred_o = int(torch.argmax(s_obj).item()) if s_obj is not None else None
                pred_c = int(torch.argmax(s_comb).item())

                correct_g = (pred_g == 0)
                correct_o = (pred_o == 0) if pred_o is not None else None
                correct_c = (pred_c == 0)

            elif args.eval_type == "text":
                # one target clip, multiple labels
                target_img = trial["target_img_filename"]
                clip_paths = _labeled_t_build_clip_paths(
                    target_img, frames_root=frames_root, num_frames=num_frames, stride=stride
                )
                x_clip = _labeled_t_load_clip_tensor(clip_paths, tfm).unsqueeze(0).to(device)  # (1,M,3,H,W)

                categories = [trial["target_category"]] + list(trial.get("foil_categories", []) or [])
                if args.use_kitty_label and ("kitty" in vocab):
                    categories = ["kitty" if c == "cat" else c for c in categories]

                label_ids, label_len = _encode_cvcl_label_tensor(
                    vocab=vocab,
                    labels=categories,
                    include_sos_eos=args.eval_include_sos_eos,
                )
                label_ids = label_ids.to(device)
                label_len = label_len.to(device)

                text_feat = _encode_cvcl_text_feats(model, label_ids, label_len)  # (L,D)

                B = 1
                M = int(x_clip.shape[1])

                x_flat = x_clip.reshape(B * M, *x_clip.shape[2:])
                img_feat = _encode_cvcl_image_feats(model, x_flat).reshape(B, M, -1)  # (1,M,D)

                sims = torch.einsum("bmd,ld->bml", img_feat, text_feat).max(dim=1).values  # (1,L)
                s_global = sims[0]  # (L,)

                s_obj_mat = _labeled_t_object_scores(
                    model, x_clip, text_feat, exclude_null=exclude_null
                )
                s_obj = None if s_obj_mat is None else s_obj_mat[0]  # (L,)

                s_comb = _labeled_t_combine_scores(s_global, s_obj, mode=combine_mode, alpha=alpha)

                pred_g = int(torch.argmax(s_global).item())
                pred_o = int(torch.argmax(s_obj).item()) if s_obj is not None else None
                pred_c = int(torch.argmax(s_comb).item())

                correct_g = (pred_g == 0)
                correct_o = (pred_o == 0) if pred_o is not None else None
                correct_c = (pred_c == 0)

            else:
                raise ValueError(f"[labeled_t] Unknown eval_type: {args.eval_type}")

            correct_combined[class_label] += int(correct_c)
            correct_global[class_label] += int(correct_g)
            if correct_o is not None:
                correct_obj[class_label] += int(correct_o)

            # log trial
            curr_results = {
                "checkpoint": checkpoint_name,
                "model": config.get("model", None),
                "seed": config.get("seed", None),
                "cnn": config.get("cnn", None),
                "augment_frames": config.get("augment_frames", None),
                "multiple_frames": config.get("multiple_frames", None),
                "eval_mode": "cvcl",
                "eval_setting": "labeled_t",
                "eval_type": args.eval_type,
                "eval_dataset": args.eval_dataset,
                "stage": args.stage,
                "trial_idx": i,
                "target_label": trial.get("target_category", None) if isinstance(trial, dict) else None,
                "foil_labels": trial.get("foil_categories", None) if isinstance(trial, dict) else None,
                "pred_combined": pred_c,
                "pred_global": pred_g,
                "pred_obj": pred_o,
                "correct": bool(correct_c),
                "correct_global": bool(correct_g),
                "correct_obj": (bool(correct_o) if correct_o is not None else None),
                "labeled_t_num_frames": num_frames,
                "labeled_t_stride": stride,
                "labeled_t_combine": combine_mode,
                "labeled_t_alpha": alpha,
                # store scores (small lists)
                "scores_global": s_global.detach().cpu().tolist(),
                "scores_obj": (s_obj.detach().cpu().tolist() if s_obj is not None else None),
                "scores_combined": s_comb.detach().cpu().tolist(),
            }
            results.append(curr_results)

    # ---- metrics ----
    total_trials = sum(total.values())
    total_correct = sum(correct_combined.values())
    total_correct_g = sum(correct_global.values())
    total_correct_o = sum(correct_obj.values())

    overall_acc = float(total_correct) / float(total_trials) if total_trials > 0 else 0.0
    overall_acc_g = float(total_correct_g) / float(total_trials) if total_trials > 0 else 0.0
    overall_acc_o = (float(total_correct_o) / float(total_trials) if total_trials > 0 else None)

    per_class = {}
    per_class_g = {}
    per_class_o = {}

    for c in classes:
        t = int(total.get(c, 0))
        if t <= 0:
            per_class[c] = {"correct": 0, "total": 0, "accuracy": None}
            per_class_g[c] = {"correct": 0, "total": 0, "accuracy": None}
            per_class_o[c] = {"correct": 0, "total": 0, "accuracy": None}
            continue

        cc = int(correct_combined.get(c, 0))
        cg = int(correct_global.get(c, 0))
        co = int(correct_obj.get(c, 0))

        per_class[c] = {"correct": cc, "total": t, "accuracy": float(cc) / float(t)}
        per_class_g[c] = {"correct": cg, "total": t, "accuracy": float(cg) / float(t)}
        per_class_o[c] = {"correct": co, "total": t, "accuracy": float(co) / float(t)}

    valid_accs = [m["accuracy"] for m in per_class.values() if m["accuracy"] is not None]
    macro_acc = float(np.mean(valid_accs)) if valid_accs else None

    valid_accs_g = [m["accuracy"] for m in per_class_g.values() if m["accuracy"] is not None]
    macro_acc_g = float(np.mean(valid_accs_g)) if valid_accs_g else None

    valid_accs_o = [m["accuracy"] for m in per_class_o.values() if m["accuracy"] is not None]
    macro_acc_o = float(np.mean(valid_accs_o)) if valid_accs_o else None

    metrics = {
        "overall_accuracy": overall_acc,
        "macro_accuracy": macro_acc,
        "overall_accuracy_global": overall_acc_g,
        "macro_accuracy_global": macro_acc_g,
        "overall_accuracy_obj": overall_acc_o,
        "macro_accuracy_obj": macro_acc_o,
        "per_class": per_class,
        "per_class_global": per_class_g,
        "per_class_obj": per_class_o,
        "n_trials": int(total_trials),
        "labeled_t": {
            "frames_root": str(frames_root),
            "num_frames": int(num_frames),
            "stride": int(stride),
            "combine": combine_mode,
            "alpha": float(alpha),
            "exclude_null": bool(exclude_null),
        },
    }
    return results, metrics

# -----------------------------------------------------------------------------
# Checkpoint handling
# -----------------------------------------------------------------------------


def _resolve_ckpt_path(ckpt_arg: str) -> Path:
    ckpt_path = Path(ckpt_arg).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if ckpt_path.suffix != ".ckpt":
        raise ValueError(f"--checkpoint must point to a .ckpt file, got: {ckpt_path}")
    return ckpt_path


def _infer_run_tag(ckpt_path: Path) -> str:
    parts = list(ckpt_path.parts)
    if "checkpoints" in parts:
        i = parts.index("checkpoints")
        if i + 1 < len(parts):
            return parts[i + 1]
    return ckpt_path.parent.name


def _load_lit_disable_vm(ckpt_path: Path, map_location) -> MultiModalLitModel:
    """Load lightning module while forcing vm_enable=False, regardless of saved args."""

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    hp = ckpt.get("hyper_parameters", {})

    saved_args = hp.get("args", None)
    if saved_args is None:
        raise RuntimeError(
            "[eval] Checkpoint missing hyper_parameters['args']; cannot override vm_enable safely."
        )

    if isinstance(saved_args, dict):
        saved_args = argparse.Namespace(**saved_args)

    setattr(saved_args, "vm_enable", False)
    setattr(saved_args, "vm_lambda", 0.0)
    setattr(saved_args, "use_sam_masks", False)

    lit = MultiModalLitModel.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        map_location=map_location,
        strict=False,
        args=saved_args,
    )
    lit.eval()
    return lit


def _ckpt_key_sanity_report(ckpt_path: Path, lit: MultiModalLitModel) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if "state_dict" not in ckpt:
        print("[eval] WARNING: checkpoint missing 'state_dict' key; cannot sanity-check keys.")
        return

    ckpt_keys = set(ckpt["state_dict"].keys())
    model_keys = set(lit.state_dict().keys())

    unexpected = sorted(list(ckpt_keys - model_keys))
    missing = sorted(list(model_keys - ckpt_keys))

    print(f"[eval] checkpoint keys: {len(ckpt_keys)} | model keys: {len(model_keys)}")
    print(f"[eval] unexpected keys in checkpoint: {len(unexpected)}")
    print(f"[eval] missing keys from checkpoint: {len(missing)}")

    if unexpected:
        print("[eval] unexpected examples:", unexpected[:20])
    if missing:
        print("[eval] missing examples:", missing[:20])

    allowed_unexpected_prefixes = ("obj_encoder.", "vm.", "null_obj", "obj_proj", "vm_", "mask_")
    bad_unexpected = [k for k in unexpected if not k.startswith(allowed_unexpected_prefixes)]
    if bad_unexpected:
        raise RuntimeError(
            "[eval] Refusing to evaluate: checkpoint contains unexpected keys that are not whitelisted. "
            f"Examples: {bad_unexpected[:10]}"
        )


# -----------------------------------------------------------------------------
# Neuron-classifier utilities (CLIP-Dissect style)
# -----------------------------------------------------------------------------


_STOPWORDS = {
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
}


def _is_reasonable_token(tok: str) -> bool:
    t = str(tok).strip().lower()
    if not t:
        return False
    if t in _STOPWORDS:
        return False
    if t.startswith("<") and t.endswith(">"):
        return False
    if re.fullmatch(r"\d+", t):
        return False
    if len(t) < 3 and t not in {"tv"}:
        return False
    if re.search(r"[^a-z0-9_ ]", t):
        return False
    return True


def _clean_label_for_text(label: str) -> str:
    s = str(label).strip().lower()
    s = s.replace("_", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = " ".join(s.split())
    return s


def _sanitize_for_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    s = s.strip("_")
    return s[:120] if len(s) > 120 else s


def _get_by_path(root: Any, path: str) -> Any:
    cur = root
    for part in path.split("."):
        if part == "":
            continue
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def _auto_hook_path(model: torch.nn.Module) -> str:
    """Best-effort selection of a deep conv block to hook."""

    names = {n for n, _m in model.named_modules()}

    preferred = [
        "vision_encoder.model.layer4",
        "model.vision_encoder.model.layer4",
        "img_encoder.layer4",
        "model.img_encoder.layer4",
        "vision_encoder.model.layer3",
        "model.vision_encoder.model.layer3",
    ]
    for p in preferred:
        if p in names:
            return p

    convs: List[str] = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            convs.append(n)
    if convs:
        return convs[-1]

    layer4 = [n for n in names if "layer4" in n]
    if layer4:
        return sorted(layer4, key=len)[-1]

    raise RuntimeError(
        "[neuron_eval] Could not auto-select a hook module. "
        "Pass --nc_hook explicitly (e.g., vision_encoder.model.layer4)."
    )


def _resolve_hook_module(model: torch.nn.Module, hook_path: str) -> Tuple[str, torch.nn.Module]:
    if hook_path is None or str(hook_path).strip().lower() == "auto":
        hook_path = _auto_hook_path(model)
        print(f"[neuron_eval] Auto-selected hook module: {hook_path}")

    hook_path = str(hook_path)

    try:
        m = _get_by_path(model, hook_path)
        if not isinstance(m, torch.nn.Module):
            raise TypeError(f"Resolved object at {hook_path} is not a torch.nn.Module: {type(m)}")
        return hook_path, m
    except Exception:
        if not hook_path.startswith("model."):
            try:
                m = _get_by_path(model, "model." + hook_path)
                if not isinstance(m, torch.nn.Module):
                    raise TypeError(
                        f"Resolved object at model.{hook_path} is not a torch.nn.Module: {type(m)}"
                    )
                return "model." + hook_path, m
            except Exception:
                pass

        last = hook_path.split(".")[-1]
        try:
            matches = [n for n, _m in model.named_modules() if last in n]
            if matches:
                print(
                    f"[neuron_eval] Could not resolve hook_path='{hook_path}'. Candidates containing '{last}':"
                )
                for n in matches[:60]:
                    print(f"  - {n}")
        except Exception:
            pass
        raise


def _forward_images_for_hook(model: torch.nn.Module, imgs: torch.Tensor) -> None:
    """Forward only the vision branch to trigger the hook."""

    if hasattr(model, "encode_image") and callable(getattr(model, "encode_image")):
        _ = model.encode_image(imgs)
        return

    for attr in [
        "img_encoder",
        "image_encoder",
        "visual_encoder",
        "vision_encoder",
        "cnn",
        "visual",
    ]:
        if hasattr(model, attr):
            enc = getattr(model, attr)
            try:
                _ = enc(imgs)
                return
            except Exception:
                pass

    if hasattr(model, "model") and isinstance(getattr(model, "model"), torch.nn.Module):
        try:
            _forward_images_for_hook(getattr(model, "model"), imgs)
            return
        except Exception:
            pass

    b = int(imgs.shape[0])
    dummy_label = torch.full((b, 1), int(UNK_TOKEN_ID), dtype=torch.long, device=imgs.device)
    dummy_len = torch.ones((b,), dtype=torch.long, device=imgs.device)
    try:
        _ = model(imgs, dummy_label, dummy_len)
    except Exception:
        _ = model(imgs, dummy_label, dummy_len.unsqueeze(1))


class _ActivationCatcher:
    def __init__(self):
        self.out: Optional[torch.Tensor] = None

    def __call__(self, _module, _inp, out):
        self.out = out


def _reduce_activation_map(act: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
    if act is None:
        raise RuntimeError("Hook did not capture any activation output.")
    if act.dim() == 4:
        if reduce == "mean":
            return act.mean(dim=(2, 3))
        if reduce == "max":
            return act.amax(dim=(2, 3))
        raise ValueError(f"Unknown reduce={reduce} (expected mean|max)")
    if act.dim() == 2:
        return act
    raise ValueError(f"Unsupported activation dim={act.dim()} (expected 2 or 4)")


# -----------------------------------------------------------------------------
# Probe images for neuron labeling
# -----------------------------------------------------------------------------


try:
    from multimodal.coco_instances_data_module import _crop_square_from_bbox as _coco_crop_square
except Exception:

    def _coco_crop_square(im: Image.Image, bbox_xywh: List[float]) -> Image.Image:
        x, y, w, h = bbox_xywh
        cx = x + w * 0.5
        cy = y + h * 0.5
        side = max(w, h)
        left = int(round(cx - side * 0.5))
        top = int(round(cy - side * 0.5))
        right = int(round(cx + side * 0.5))
        bottom = int(round(cy + side * 0.5))
        left = max(0, left)
        top = max(0, top)
        right = min(im.width, right)
        bottom = min(im.height, bottom)
        if right <= left + 1 or bottom <= top + 1:
            return im
        return im.crop((left, top, right, bottom))


try:
    from multimodal.imagenet_val_data_module import _crop_square_from_bbox as _imagenet_crop_square
except Exception:

    def _imagenet_crop_square(im: Image.Image, bbox_xywh: List[float]) -> Image.Image:
        x, y, w, h = bbox_xywh
        cx = x + w * 0.5
        cy = y + h * 0.5
        side = max(w, h)
        left = int(round(cx - side * 0.5))
        top = int(round(cy - side * 0.5))
        right = int(round(cx + side * 0.5))
        bottom = int(round(cy + side * 0.5))
        left = max(0, left)
        top = max(0, top)
        right = min(im.width, right)
        bottom = min(im.height, bottom)
        if right <= left + 1 or bottom <= top + 1:
            return im
        return im.crop((left, top, right, bottom))


class _ProbeSpecDataset(Dataset):
    """Yields (cvcl_x, clip_x, id_str) from heterogeneous probe specs."""

    def __init__(self, specs: List[dict], cvcl_transform, clip_transform):
        self.specs = specs
        self.cvcl_transform = cvcl_transform
        self.clip_transform = clip_transform

    def __len__(self) -> int:
        return len(self.specs)

    def _load_pil(self, spec: dict) -> Image.Image:
        kind = spec["kind"]

        if kind == "path":
            p = Path(spec["path"])
            return Image.open(p).convert("RGB")

        if kind == "coco":
            p = Path(spec["images_dir"]) / spec["file_name"]
            im = Image.open(p).convert("RGB")
            bbox = spec.get("bbox", None)
            if bbox is not None:
                im = _coco_crop_square(im, bbox)
            return im

        if kind == "imagenet":
            p = Path(spec["images_dir"]) / spec["file_name"]
            im = Image.open(p).convert("RGB")
            if spec.get("use_bboxes", True):
                bbox = spec.get("bbox", None)
                if bbox is not None:
                    im = _imagenet_crop_square(im, bbox)
            return im

        if kind == "cifar":
            ds = spec["base_dataset"]
            idx = int(spec["index"])
            im, _y = ds[idx]
            if not isinstance(im, Image.Image):
                im = transforms.ToPILImage()(im)
            return im.convert("RGB")

        raise ValueError(f"Unknown probe spec kind: {kind}")

    def __getitem__(self, idx: int):
        spec = self.specs[idx]
        im = self._load_pil(spec)
        cvcl_x = self.cvcl_transform(im)
        clip_x = self.clip_transform(im)
        return cvcl_x, clip_x, spec.get("id", str(idx))


def _default_cvcl_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_H, IMAGE_W), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalizer,
        ]
    )


def _build_concept_set(
    dataset_labels: List[str],
    vocab: Optional[Dict[str, int]] = None,
    common_words_txt: Optional[Path] = None,
    max_vocab_words: int = 50000,
) -> List[str]:
    concepts: List[str] = []

    def add(x: str) -> None:
        s = _clean_label_for_text(x)
        if s and s not in concepts:
            concepts.append(s)

    for l in dataset_labels:
        add(l)

    if vocab is not None:
        words = list(vocab.keys())
        for w in words[:max_vocab_words]:
            if _is_reasonable_token(w):
                add(w)

    if common_words_txt is not None and common_words_txt.exists() and common_words_txt.is_file():
        try:
            with open(common_words_txt, "r") as f:
                for line in f:
                    w = line.strip()
                    if _is_reasonable_token(w):
                        add(w)
        except Exception as e:
            print(f"[neuron_eval] WARNING: could not read common_words_txt={common_words_txt}: {e}")

    return concepts


def _load_openai_clip(
    clip_model_name: str,
    device_: torch.device,
    download_root: Optional[str] = None,
) -> Tuple[torch.nn.Module, Any]:
    """Loads CLIP via the OpenAI `clip` package."""

    kw = {"device": device_}
    if download_root is not None:
        kw["download_root"] = str(Path(download_root).expanduser())

    try:
        model, preprocess = clip.load(clip_model_name, **kw)
        model.eval()
        return model, preprocess
    except Exception as e:
        msg = (
            f"[neuron_eval] Failed to load CLIP model '{clip_model_name}' via openai/clip. "
            "If you're offline, make sure the correct weight file exists in "
            f"{kw.get('download_root', '~/.cache/clip')}. Original error: {e}"
        )
        raise RuntimeError(msg) from e


def _encode_clip_text(
    clip_model,
    texts: List[str],
    prompt: str = "a photo of a {}",
    device_: torch.device = device,
    batch_size: int = 256,
) -> torch.Tensor:
    clip_model.eval()
    out_chunks: List[torch.Tensor] = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            prompts = [prompt.format(t) for t in chunk]
            tokens = clip.tokenize(prompts).to(device_)
            feats = clip_model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out_chunks.append(feats.detach().cpu())

    return torch.cat(out_chunks, dim=0)


def _collect_probe_specs(datamodule: Any) -> List[dict]:
    """Collect a heterogeneous list of image specs usable for neuron labeling."""

    specs: List[dict] = []

    if isinstance(datamodule, KonkleObjectCategoriesDataModule):
        root = Path(datamodule.cfg.categories_dir)
        for inst in getattr(datamodule, "instances", []) or []:
            rel = inst.get("relpath", None)
            if rel is None:
                continue
            p = root / rel
            if p.exists() and p.is_file():
                specs.append({"kind": "path", "path": str(p), "id": f"konkle::{rel}"})
        return specs

    if isinstance(datamodule, ImageNetValDataModule):
        root = Path(datamodule.cfg.imagenet_images_dir)
        use_bboxes = bool(getattr(datamodule.cfg, "use_bboxes", True))
        for inst in getattr(datamodule, "instances", []) or []:
            fn = inst.get("file_name", None)
            if fn is None:
                continue
            p = root / fn
            if not (p.exists() and p.is_file()):
                continue
            bbox = inst.get("bbox", None)
            specs.append(
                {
                    "kind": "imagenet",
                    "images_dir": str(root),
                    "file_name": fn,
                    "bbox": bbox,
                    "use_bboxes": use_bboxes,
                    "id": f"imagenet::{fn}::{bbox}",
                }
            )
        return specs

    if isinstance(datamodule, COCOInstancesDataModule):
        root = Path(datamodule.cfg.coco_images_dir)
        seen = set()
        trials = getattr(datamodule, "trials", []) or []
        for t in trials:
            for key in ["target_instance", "foil_instances"]:
                if key not in t:
                    continue
                items = t[key]
                if isinstance(items, dict):
                    items = [items]
                for inst in items:
                    fn = inst.get("file_name", None)
                    bbox = inst.get("bbox", None)
                    if fn is None or bbox is None:
                        continue
                    p = root / fn
                    if not (p.exists() and p.is_file()):
                        continue
                    sig = (fn, tuple(float(x) for x in bbox))
                    if sig in seen:
                        continue
                    seen.add(sig)
                    specs.append(
                        {
                            "kind": "coco",
                            "images_dir": str(root),
                            "file_name": fn,
                            "bbox": bbox,
                            "id": f"coco::{fn}::{bbox}",
                        }
                    )
        return specs

    if isinstance(datamodule, CIFARForcedChoiceDataModule):
        base_ds = getattr(datamodule, "base_dataset", None)
        if base_ds is None:
            return specs
        for i in range(len(base_ds)):
            specs.append(
                {"kind": "cifar", "base_dataset": base_ds, "index": int(i), "id": f"cifar::{i}"}
            )
        return specs

    raise RuntimeError(
        "[neuron_eval] Unsupported datamodule for probe collection. "
        f"Got: {datamodule.__class__.__name__}. "
        "Extend _collect_probe_specs() for this dataset."
    )


def _label_fingerprint(labels: List[str]) -> str:
    """Stable short hash used to prevent neuron-map cache collisions."""

    joined = "\n".join([str(x) for x in labels])
    h = hashlib.sha1(joined.encode("utf-8")).hexdigest()
    return h[:12]


def _build_neuron_classifier_maps(
    *,
    model: torch.nn.Module,
    datamodule: Any,
    dataset_labels: List[str],
    vocab: Optional[Dict[str, int]],
    common_words_txt: Optional[Path],
    hook_path: str,
    clip_model_name: str,
    clip_download_root: Optional[str],
    topk: int,
    probe_max_images: int,
    activation_reduce: str,
    prompt: str,
    cache_dir: Path,
    regenerate: bool = False,
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Resolve hook now so the cache key reflects the actual module.
    resolved_hook_path, _hook_module = _resolve_hook_module(model, hook_path)

    hook_tag = _sanitize_for_filename(resolved_hook_path)
    clip_tag = _sanitize_for_filename(clip_model_name)
    labels_hash = _label_fingerprint(sorted([_clean_label_for_text(x) for x in dataset_labels]))

    cache_path = (
        cache_dir
        / (
            f"neuron_map_{datamodule.__class__.__name__}_"
            f"{hook_tag}_{clip_tag}_topk{int(topk)}_probe{int(probe_max_images)}_"
            f"labels{len(dataset_labels)}_{labels_hash}.json"
        )
    )

    if cache_path.exists() and not regenerate:
        with open(cache_path, "r") as f:
            cached = json.load(f)
        label_to_neuron = {k: int(v) for k, v in cached["label_to_neuron"].items()}
        return label_to_neuron, cached

    # Probe specs
    probe_specs = _collect_probe_specs(datamodule)
    probe_specs = [s for s in probe_specs if s is not None]
    if len(probe_specs) == 0:
        raise RuntimeError("[neuron_eval] Could not collect any probe images.")

    if probe_max_images is not None and probe_max_images > 0 and len(probe_specs) > probe_max_images:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(probe_specs), size=int(probe_max_images), replace=False)
        probe_specs = [probe_specs[i] for i in idx]

    print(f"[neuron_eval] Probe images: {len(probe_specs)}")
    print(f"[neuron_eval] Loading CLIP backbone for neuron labeling: {clip_model_name}")
    clip_model, clip_preprocess = _load_openai_clip(
        clip_model_name=clip_model_name,
        device_=device,
        download_root=clip_download_root,
    )

    cvcl_transform = _default_cvcl_transform()
    probe_ds = _ProbeSpecDataset(probe_specs, cvcl_transform=cvcl_transform, clip_transform=clip_preprocess)

    # Keep num_workers=0 for robustness (CIFAR probe specs capture a dataset object).
    probe_dl = DataLoader(probe_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

    # Hook
    resolved_hook_path, hook_module = _resolve_hook_module(model, hook_path)
    catcher = _ActivationCatcher()
    hook_handle = hook_module.register_forward_hook(catcher)

    all_act: List[torch.Tensor] = []
    all_clip_img: List[torch.Tensor] = []

    model.eval()
    clip_model.eval()

    with torch.no_grad():
        for cvcl_x, clip_x, _ids in probe_dl:
            cvcl_x = cvcl_x.to(device)
            clip_x = clip_x.to(device)

            catcher.out = None
            _forward_images_for_hook(model, cvcl_x)
            act_bc = _reduce_activation_map(catcher.out, reduce=activation_reduce)
            all_act.append(act_bc.detach().cpu())

            img_feat = clip_model.encode_image(clip_x)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            all_clip_img.append(img_feat.detach().cpu())

    hook_handle.remove()

    act_mat = torch.cat(all_act, dim=0)  # (N,C)
    clip_img_mat = torch.cat(all_clip_img, dim=0)  # (N,D)

    n_probe, n_neurons = act_mat.shape
    _n_probe2, d_clip = clip_img_mat.shape

    k = int(min(int(topk), int(n_probe)))
    print(
        f"[neuron_eval] Hook: {resolved_hook_path} | neurons: {n_neurons} | CLIP dim: {d_clip} | top-k: {k}"
    )

    topk_vals, topk_idx = torch.topk(act_mat, k=k, dim=0, largest=True, sorted=False)
    _ = topk_vals

    topk_clip = clip_img_mat[topk_idx]  # (k,C,D)
    neuron_embed = topk_clip.mean(dim=0)
    neuron_embed = neuron_embed / (neuron_embed.norm(dim=-1, keepdim=True) + 1e-6)  # (C,D)

    concept_set = _build_concept_set(
        dataset_labels=dataset_labels,
        vocab=vocab,
        common_words_txt=common_words_txt,
        max_vocab_words=50000,
    )
    print(f"[neuron_eval] Concept set size |S| = {len(concept_set)}")

    text_emb = _encode_clip_text(
        clip_model=clip_model,
        texts=concept_set,
        prompt=prompt,
        device_=device,
        batch_size=256,
    )  # (|S|,D) on CPU

    text_emb_t = text_emb.t().contiguous()
    neuron_labels: List[str] = []
    neuron_label_scores: List[float] = []

    block = 256
    for start in range(0, n_neurons, block):
        end = min(start + block, n_neurons)
        sims = neuron_embed[start:end] @ text_emb_t
        best_scores, best_idx = torch.max(sims, dim=1)
        for s, j in zip(best_scores.tolist(), best_idx.tolist()):
            neuron_labels.append(concept_set[int(j)])
            neuron_label_scores.append(float(s))

    label_set_clean = [_clean_label_for_text(l) for l in dataset_labels]

    label_to_neuron_candidates: Dict[str, List[int]] = {}
    for ni, lab in enumerate(neuron_labels):
        if lab in label_set_clean:
            label_to_neuron_candidates.setdefault(lab, []).append(ni)

    label_to_neuron: Dict[str, int] = {}
    label_to_score: Dict[str, float] = {}

    for lab in label_set_clean:
        cands = label_to_neuron_candidates.get(lab, [])
        if not cands:
            continue
        best = max(cands, key=lambda idx: neuron_label_scores[idx])
        label_to_neuron[lab] = int(best)
        label_to_score[lab] = float(neuron_label_scores[best])

    missing = [l for l in label_set_clean if l not in label_to_neuron]
    if missing:
        print(
            f"[neuron_eval] WARNING: {len(missing)} labels had 0 labeled neurons; "
            "using direct similarity fallback (text embedding -> neuron embedding)."
        )
        miss_emb = _encode_clip_text(clip_model, missing, prompt=prompt, device_=device, batch_size=256)
        sims = miss_emb @ neuron_embed.t()
        best_scores, best_idx = torch.max(sims, dim=1)
        for lab, s, j in zip(missing, best_scores.tolist(), best_idx.tolist()):
            label_to_neuron[lab] = int(j)
            label_to_score[lab] = float(s)

    meta = {
        "label_to_neuron": {k: int(v) for k, v in label_to_neuron.items()},
        "label_to_score": {k: float(v) for k, v in label_to_score.items()},
        "hook_path": resolved_hook_path,
        "requested_hook_path": hook_path,
        "clip_model_name": clip_model_name,
        "clip_download_root": clip_download_root,
        "topk": int(topk),
        "probe_n": int(n_probe),
        "n_neurons": int(n_neurons),
        "activation_reduce": activation_reduce,
        "prompt": prompt,
        "concept_set_size": int(len(concept_set)),
        "labels_fingerprint": labels_hash,
        "note": "Neuron mapping built using CLIP image/text embeddings over top-k activating probe images (CLIP-Dissect-style).",
    }
    with open(cache_path, "w") as f:
        json.dump(meta, f)
    print(f"[neuron_eval] Saved neuron map cache: {cache_path}")

    return label_to_neuron, meta


# -----------------------------------------------------------------------------
# Hybrid eval helpers
# -----------------------------------------------------------------------------


def _maybe_map_cat_to_kitty(label: str, *, use_kitty_label: bool, vocab: Optional[Dict[str, int]]) -> str:
    if not use_kitty_label:
        return label
    if vocab is None:
        return label
    if str(label) == "cat" and "kitty" in vocab:
        return "kitty"
    return label


def _trial_categories(eval_data: List[Any], i: int) -> Optional[List[str]]:
    if i >= len(eval_data):
        return None
    if not isinstance(eval_data[i], dict):
        return None
    t = eval_data[i]
    tgt = t.get("target_category", None)
    foils = t.get("foil_categories", None)
    if tgt is None or foils is None:
        return None
    return [tgt] + list(foils)


def _cvcl_forward(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    labels: torch.Tensor,
    label_len: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Robust wrapper: some versions expect label_len shaped (N,) or (N,1)."""

    try:
        return model(imgs, labels, label_len)
    except Exception:
        if label_len.dim() == 1:
            return model(imgs, labels, label_len.unsqueeze(1))
        if label_len.dim() == 2 and label_len.shape[1] == 1:
            return model(imgs, labels, label_len.squeeze(1))
        raise


def _encode_cvcl_label_tensor(
    vocab: Dict[str, int],
    labels: List[str],
    include_sos_eos: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode label strings into CVCL label-id tensors.

    Returns:
      label_ids: (N, L)
      label_len: (N,)  (or (N,1) via _cvcl_forward retry)
    """

    ids_list: List[List[int]] = []
    for s in labels:
        if s not in vocab:
            raise KeyError(f"Label '{s}' not in vocab (cannot CVCL-eval this trial)")
        ids = [int(vocab[s])]
        if include_sos_eos:
            ids = [int(SOS_TOKEN_ID)] + ids + [int(EOS_TOKEN_ID)]
        ids_list.append(ids)

    L = len(ids_list[0])
    for ids in ids_list:
        if len(ids) != L:
            raise ValueError("All labels must have the same length (expected single-token labels)")

    label_ids = torch.LongTensor(ids_list)
    label_len = torch.LongTensor([L] * len(ids_list))
    return label_ids, label_len


def _should_use_cvcl_branch(
    *,
    args: argparse.Namespace,
    vocab: Optional[Dict[str, int]],
    target_label: str,
    curr_categories: Optional[List[str]],
) -> bool:
    """Decide whether to evaluate a trial with the CVCL text encoder.

    In hybrid mode:
      - image eval: requires target label in vocab.
      - text eval: requires all candidate labels in vocab.
    """

    if args.eval_mode == "cvcl":
        return True

    # neuron_classifier mode:
    if args.nc_all_labels:
        return False

    if vocab is None:
        return False

    vocab_set = set(vocab.keys())

    if args.eval_type == "image":
        return target_label in vocab_set

    # eval_type == "text"
    if curr_categories is None:
        return False
    return all(c in vocab_set for c in curr_categories)


def _ensure_key(d: Dict[str, int], k: str) -> None:
    if k not in d:
        d[k] = 0


def _per_class_from_counts(
    *,
    class_keys_clean: List[str],
    correct: Dict[str, int],
    total: Dict[str, int],
    clean_to_orig: Dict[str, str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c_clean in class_keys_clean:
        orig = clean_to_orig.get(c_clean, c_clean)
        c = int(correct.get(c_clean, 0))
        t = int(total.get(c_clean, 0))
        out[orig] = {"correct": c, "total": t, "accuracy": (float(c) / float(t) if t > 0 else None)}
    return out


def _hybrid_per_class_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build a wide per-class table containing both OOD and OOV splits.

    This is used when eval_mode=neuron_classifier.
    """

    per_all = metrics.get("per_class", {}) or {}
    per_inv = metrics.get("per_class_in_vocab_target", {}) or {}
    per_oov = metrics.get("per_class_oov_target", {}) or {}
    per_cvcl = metrics.get("per_class_cvcl_branch", {}) or {}
    per_neuron = metrics.get("per_class_neuron_branch", {}) or {}

    class_names = sorted(set(per_all) | set(per_inv) | set(per_oov) | set(per_cvcl) | set(per_neuron))

    rows: List[Dict[str, Any]] = []
    for name in class_names:
        row: Dict[str, Any] = {"class": name}

        def add(prefix: str, d: Dict[str, Any]) -> None:
            m = d.get(name, None)
            if m is None:
                row[f"{prefix}_correct"] = 0
                row[f"{prefix}_total"] = 0
                row[f"{prefix}_accuracy"] = None
            else:
                row[f"{prefix}_correct"] = m.get("correct", 0)
                row[f"{prefix}_total"] = m.get("total", 0)
                row[f"{prefix}_accuracy"] = m.get("accuracy", None)

        add("all", per_all)
        add("in_vocab_target", per_inv)
        add("oov_target", per_oov)
        add("cvcl_branch", per_cvcl)
        add("neuron_branch", per_neuron)

        rows.append(row)

    return rows


# -----------------------------------------------------------------------------
# Evaluation loops
# -----------------------------------------------------------------------------


def _cvcl_only_eval_loop(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    eval_data: List[Any],
    classes: List[str],
    vocab: Optional[Dict[str, int]],
    args: argparse.Namespace,
    checkpoint_name: str,
    config: Dict[str, Any],
) -> Tuple[List[dict], Dict[str, Any]]:
    """Pure CVCL / CLIP evaluation (original behavior)."""

    if args.use_kitty_label and not args.clip_eval:
        if "cat" in classes:
            classes = [c for c in classes if c != "cat"]
        if "kitty" not in classes:
            classes = classes + ["kitty"]

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    results: List[dict] = []

    for i, batch in enumerate(dataloader):
        img, label, label_len, raw_label = batch
        class_label = raw_label[0][0]

        # kitty patch
        if args.use_kitty_label and class_label == "cat" and not args.clip_eval:
            class_label = "kitty"
            if vocab is None:
                raise RuntimeError("--use_kitty_label requires a vocab")
            if args.eval_type == "image":
                new_label = [vocab[class_label]]
                if args.eval_include_sos_eos:
                    new_label = [SOS_TOKEN_ID] + new_label + [EOS_TOKEN_ID]
                label = torch.LongTensor([new_label])
            elif args.eval_type == "text":
                label[0][0] = vocab[class_label]

        if args.eval_type == "image":
            img_ = img.squeeze(0).to(device)

            with torch.no_grad():
                if args.clip_eval:
                    label_ = label.squeeze(0)
                    _, logits_per_text = model(img_, label_)
                else:
                    label_ = label.to(device)
                    label_len_ = label_len.to(device)
                    _, logits_per_text = _cvcl_forward(model, img_, label_, label_len_)

                logits = logits_per_text[0]
                logits_soft = torch.softmax(logits, dim=-1)
                logits_list = logits_soft.detach().cpu().numpy().tolist()

                pred = int(torch.argmax(logits))
                ground_truth = 0

        elif args.eval_type == "text":
            img_ = img.squeeze(0).to(device)

            with torch.no_grad():
                if args.clip_eval:
                    logits_per_image, _ = model(img_, label.squeeze(0).to(device))
                else:
                    labels_ = label.squeeze(0).to(device)
                    label_len_ = label_len.squeeze(0).to(device)
                    logits_per_image, _ = _cvcl_forward(model, img_, labels_, label_len_)

                logits = logits_per_image[0]
                logits_soft = torch.softmax(logits, dim=-1)
                logits_list = logits_soft.detach().cpu().numpy().tolist()

                pred = int(torch.argmax(logits))
                ground_truth = 0
        else:
            raise ValueError(f"Unknown eval_type: {args.eval_type}")

        correct = pred == ground_truth
        correct_pred[class_label] += int(correct)
        total_pred[class_label] += 1

        curr_eval_categories = _trial_categories(eval_data, i)

        curr_results = {
            "checkpoint": checkpoint_name,
            "model": config.get("model", None),
            "seed": config.get("seed", None),
            "shuffle_utterances": config.get("shuffle_utterances", None),
            "augment_frames": config.get("augment_frames", None),
            "multiple_frames": config.get("multiple_frames", None),
            "cnn": config.get("cnn", None),
            "eval_mode": "cvcl",
            "eval_type": args.eval_type,
            "eval_dataset": args.eval_dataset,
            "stage": args.stage,
            "trial_idx": i,
            "categories": curr_eval_categories,
            "logits": logits_list,
            "pred": pred,
            "correct": bool(correct),
        }
        results.append(curr_results)

    total_correct = sum(correct_pred.values())
    total = sum(total_pred.values())
    overall_accuracy = float(total_correct) / float(total) if total > 0 else 0.0

    per_class_metrics: Dict[str, Any] = {}
    for classname in classes:
        t = int(total_pred.get(classname, 0))
        c = int(correct_pred.get(classname, 0))
        acc = float(c) / t if t > 0 else None
        per_class_metrics[classname] = {"correct": c, "total": t, "accuracy": acc}

    valid_accs = [m["accuracy"] for m in per_class_metrics.values() if m["accuracy"] is not None]
    macro_accuracy = float(np.mean(valid_accs)) if valid_accs else None

    metrics = {
        "overall_accuracy": overall_accuracy,
        "macro_accuracy": macro_accuracy,
        "per_class": per_class_metrics,
        "n_trials": int(total),
    }

    return results, metrics


def _hybrid_eval_loop(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    eval_data: List[Any],
    classes: List[str],
    vocab: Optional[Dict[str, int]],
    args: argparse.Namespace,
    checkpoint_name: str,
    config: Dict[str, Any],
) -> Tuple[List[dict], Dict[str, Any]]:
    """Hybrid evaluation.

    - CVCL branch for in-vocab trials (OOD-style).
    - Neuron-classifier branch for OOV trials.
    """

    if args.clip_eval:
        raise ValueError("[eval] Hybrid neuron evaluation is intended for CVCL checkpoints, not --clip_eval")

    if vocab is None:
        raise RuntimeError("[eval] Hybrid eval requires a CVCL vocab dict")

    vocab_set = set(vocab.keys())

    # Optional kitty relabeling: keep exact old behavior (but now applies to hybrid too).
    if args.use_kitty_label and "kitty" in vocab:
        if "cat" in classes:
            classes = [c for c in classes if c != "cat"]
        if "kitty" not in classes:
            classes = classes + ["kitty"]

    clean_classes = [_clean_label_for_text(c) for c in classes]
    class_clean_to_orig = {_clean_label_for_text(c): c for c in classes}

    # Per-class counts (clean label keys)
    correct_all: Dict[str, int] = {c: 0 for c in clean_classes}
    total_all: Dict[str, int] = {c: 0 for c in clean_classes}

    correct_in_vocab_target: Dict[str, int] = {c: 0 for c in clean_classes}
    total_in_vocab_target: Dict[str, int] = {c: 0 for c in clean_classes}

    correct_oov_target: Dict[str, int] = {c: 0 for c in clean_classes}
    total_oov_target: Dict[str, int] = {c: 0 for c in clean_classes}

    correct_cvcl_branch: Dict[str, int] = {c: 0 for c in clean_classes}
    total_cvcl_branch: Dict[str, int] = {c: 0 for c in clean_classes}

    correct_neuron_branch: Dict[str, int] = {c: 0 for c in clean_classes}
    total_neuron_branch: Dict[str, int] = {c: 0 for c in clean_classes}

    # Branch summary metrics
    branch_correct = {"cvcl": 0, "neuron": 0}
    branch_total = {"cvcl": 0, "neuron": 0}

    # Target-label in-vocab vs OOV summary metrics
    inv_correct = 0
    inv_total = 0
    oov_correct = 0
    oov_total = 0

    # Determine whether we will ever need neuron maps.
    mapped_labels_for_check = [
        _maybe_map_cat_to_kitty(c, use_kitty_label=args.use_kitty_label, vocab=vocab) for c in classes
    ]
    has_any_oov_label = any((c not in vocab_set) for c in mapped_labels_for_check)
    need_neuron = bool(args.nc_all_labels) or bool(has_any_oov_label)

    label_to_neuron: Dict[str, int] = {}
    neuron_map_meta: Optional[Dict[str, Any]] = None

    resolved_hook_path: Optional[str] = None
    hook_handle = None
    catcher: Optional[_ActivationCatcher] = None

    if need_neuron:
        label_to_neuron, neuron_map_meta = _build_neuron_classifier_maps(
            model=model,
            datamodule=args._datamodule,
            dataset_labels=classes,
            vocab=vocab,
            common_words_txt=Path(args.nc_common_words_txt).expanduser() if args.nc_common_words_txt else None,
            hook_path=args.nc_hook,
            clip_model_name=args.nc_clip_model,
            clip_download_root=args.nc_clip_download_root,
            topk=args.nc_topk,
            probe_max_images=args.nc_probe_max_images,
            activation_reduce=args.nc_activation_reduce,
            prompt=args.nc_prompt,
            cache_dir=Path(args.nc_cache_dir),
            regenerate=args.nc_regenerate,
        )

        resolved_hook_path, hook_module = _resolve_hook_module(model, args.nc_hook)
        catcher = _ActivationCatcher()
        hook_handle = hook_module.register_forward_hook(catcher)

    results: List[dict] = []

    for i, batch in enumerate(dataloader):
        img, _label_from_ds, _label_len_from_ds, raw_label = batch

        target_label_orig = raw_label[0][0]
        target_label = _maybe_map_cat_to_kitty(
            target_label_orig, use_kitty_label=args.use_kitty_label, vocab=vocab
        )

        curr_eval_categories = _trial_categories(eval_data, i)
        if curr_eval_categories is not None:
            curr_eval_categories = [
                _maybe_map_cat_to_kitty(c, use_kitty_label=args.use_kitty_label, vocab=vocab)
                for c in curr_eval_categories
            ]

        target_in_vocab = bool(target_label in vocab_set)
        if target_in_vocab:
            inv_total += 1
        else:
            oov_total += 1

        # Decide branch.
        use_cvcl = _should_use_cvcl_branch(
            args=args,
            vocab=vocab,
            target_label=target_label,
            curr_categories=curr_eval_categories,
        )
        branch = "cvcl" if use_cvcl else "neuron"

        pred: int
        logits_list: List[float]

        if use_cvcl:
            # CVCL branch
            if args.eval_type == "image":
                imgs = img.squeeze(0).to(device)

                label_ids, label_len = _encode_cvcl_label_tensor(
                    vocab=vocab,
                    labels=[target_label],
                    include_sos_eos=args.eval_include_sos_eos,
                )
                label_ids = label_ids.to(device)
                label_len = label_len.to(device)

                with torch.no_grad():
                    _, logits_per_text = _cvcl_forward(model, imgs, label_ids, label_len)
                    logits = logits_per_text[0]
                    logits_soft = torch.softmax(logits, dim=-1)
                    logits_list = logits_soft.detach().cpu().numpy().tolist()
                    pred = int(torch.argmax(logits).item())
                    ground_truth = 0

            elif args.eval_type == "text":
                if curr_eval_categories is None:
                    raise RuntimeError("[eval] eval_type=text requires eval_data with foil_categories")

                img1 = img.squeeze(0).to(device)

                label_ids, label_len = _encode_cvcl_label_tensor(
                    vocab=vocab,
                    labels=curr_eval_categories,
                    include_sos_eos=args.eval_include_sos_eos,
                )
                label_ids = label_ids.to(device)
                label_len = label_len.to(device)

                with torch.no_grad():
                    logits_per_image, _ = _cvcl_forward(model, img1, label_ids, label_len)
                    logits = logits_per_image[0]
                    logits_soft = torch.softmax(logits, dim=-1)
                    logits_list = logits_soft.detach().cpu().numpy().tolist()
                    pred = int(torch.argmax(logits).item())
                    ground_truth = 0
            else:
                raise ValueError(f"Unknown eval_type: {args.eval_type}")

            correct = pred == ground_truth
            branch_correct[branch] += int(correct)
            branch_total[branch] += 1

        else:
            # Neuron branch
            if not need_neuron:
                raise RuntimeError("[eval] Internal error: need_neuron=False but neuron branch selected")
            assert catcher is not None
            assert resolved_hook_path is not None

            tgt_clean = _clean_label_for_text(target_label)
            if tgt_clean not in label_to_neuron:
                # This should not happen (we build maps for all labels), but skip safely.
                print(f"[neuron_eval] WARNING: target label missing neuron map: '{target_label}'")
                continue

            if args.eval_type == "image":
                imgs = img.squeeze(0).to(device)
                catcher.out = None
                with torch.no_grad():
                    _forward_images_for_hook(model, imgs)
                act_bc = _reduce_activation_map(catcher.out, reduce=args.nc_activation_reduce)

                neuron_idx = int(label_to_neuron[tgt_clean])
                scores = act_bc[:, neuron_idx]
                pred = int(torch.argmax(scores).item())
                ground_truth = 0
                logits_list = scores.detach().cpu().numpy().tolist()

            elif args.eval_type == "text":
                if curr_eval_categories is None:
                    raise RuntimeError("[neuron_eval] eval_type=text requires eval_data with foil_categories")

                img1 = img.squeeze(0).to(device)
                catcher.out = None
                with torch.no_grad():
                    _forward_images_for_hook(model, img1)
                act_bc = _reduce_activation_map(catcher.out, reduce=args.nc_activation_reduce)
                act_c = act_bc[0]

                cand_clean = [_clean_label_for_text(c) for c in curr_eval_categories]
                missing_cands = [c for c in cand_clean if c not in label_to_neuron]
                if missing_cands:
                    print(
                        f"[neuron_eval] WARNING: {len(missing_cands)} candidate labels missing neuron map; "
                        "skipping trial. Examples: "
                        f"{missing_cands[:5]}"
                    )
                    continue

                cand_neurons = [int(label_to_neuron[c]) for c in cand_clean]
                scores = act_c[cand_neurons]
                pred = int(torch.argmax(scores).item())
                ground_truth = 0
                logits_list = scores.detach().cpu().numpy().tolist()
            else:
                raise ValueError(f"Unknown eval_type: {args.eval_type}")

            correct = pred == ground_truth
            branch_correct[branch] += int(correct)
            branch_total[branch] += 1

        # Update inv/oov correctness (target-based)
        if target_in_vocab:
            inv_correct += int(correct)
        else:
            oov_correct += int(correct)

        # Update per-class counts (using clean mapped target)
        tgt_clean_key = _clean_label_for_text(target_label)

        # Ensure keys exist (robust against unexpected labels)
        for d in [
            correct_all,
            total_all,
            correct_in_vocab_target,
            total_in_vocab_target,
            correct_oov_target,
            total_oov_target,
            correct_cvcl_branch,
            total_cvcl_branch,
            correct_neuron_branch,
            total_neuron_branch,
        ]:
            _ensure_key(d, tgt_clean_key)

        correct_all[tgt_clean_key] += int(correct)
        total_all[tgt_clean_key] += 1

        if target_in_vocab:
            correct_in_vocab_target[tgt_clean_key] += int(correct)
            total_in_vocab_target[tgt_clean_key] += 1
        else:
            correct_oov_target[tgt_clean_key] += int(correct)
            total_oov_target[tgt_clean_key] += 1

        if branch == "cvcl":
            correct_cvcl_branch[tgt_clean_key] += int(correct)
            total_cvcl_branch[tgt_clean_key] += 1
        else:
            correct_neuron_branch[tgt_clean_key] += int(correct)
            total_neuron_branch[tgt_clean_key] += 1

        curr_results = {
            "checkpoint": checkpoint_name,
            "model": config.get("model", None),
            "seed": config.get("seed", None),
            "shuffle_utterances": config.get("shuffle_utterances", None),
            "augment_frames": config.get("augment_frames", None),
            "multiple_frames": config.get("multiple_frames", None),
            "cnn": config.get("cnn", None),
            "eval_mode": "neuron_classifier",
            "eval_branch": branch,
            "nc_all_labels": bool(args.nc_all_labels),
            "nc_hook": resolved_hook_path,
            "nc_hook_requested": args.nc_hook,
            "nc_clip_model": args.nc_clip_model,
            "nc_topk": int(args.nc_topk),
            "nc_probe_max_images": int(args.nc_probe_max_images),
            "eval_type": args.eval_type,
            "eval_dataset": args.eval_dataset,
            "stage": args.stage,
            "trial_idx": i,
            "target_label": target_label_orig,
            "target_label_mapped": target_label,
            "target_in_vocab": target_in_vocab,
            "categories": curr_eval_categories,
            "logits": logits_list,
            "logits_kind": "softmax" if branch == "cvcl" else "activation",
            "pred": pred,
            "correct": bool(correct),
        }
        results.append(curr_results)

    if hook_handle is not None:
        hook_handle.remove()

    total_correct = sum(correct_all.values())
    total = sum(total_all.values())
    overall_accuracy = float(total_correct) / float(total) if total > 0 else 0.0

    class_keys_clean = sorted(set(total_all.keys()) | set(correct_all.keys()))

    per_class_all = _per_class_from_counts(
        class_keys_clean=class_keys_clean,
        correct=correct_all,
        total=total_all,
        clean_to_orig=class_clean_to_orig,
    )
    per_class_inv_target = _per_class_from_counts(
        class_keys_clean=class_keys_clean,
        correct=correct_in_vocab_target,
        total=total_in_vocab_target,
        clean_to_orig=class_clean_to_orig,
    )
    per_class_oov_target = _per_class_from_counts(
        class_keys_clean=class_keys_clean,
        correct=correct_oov_target,
        total=total_oov_target,
        clean_to_orig=class_clean_to_orig,
    )
    per_class_cvcl = _per_class_from_counts(
        class_keys_clean=class_keys_clean,
        correct=correct_cvcl_branch,
        total=total_cvcl_branch,
        clean_to_orig=class_clean_to_orig,
    )
    per_class_neuron = _per_class_from_counts(
        class_keys_clean=class_keys_clean,
        correct=correct_neuron_branch,
        total=total_neuron_branch,
        clean_to_orig=class_clean_to_orig,
    )

    valid_accs = [m["accuracy"] for m in per_class_all.values() if m["accuracy"] is not None]
    macro_accuracy = float(np.mean(valid_accs)) if valid_accs else None

    metrics: Dict[str, Any] = {
        "overall_accuracy": overall_accuracy,
        "macro_accuracy": macro_accuracy,
        "per_class": per_class_all,
        # Extra per-class splits (requested)
        "per_class_in_vocab_target": per_class_inv_target,
        "per_class_oov_target": per_class_oov_target,
        "per_class_cvcl_branch": per_class_cvcl,
        "per_class_neuron_branch": per_class_neuron,
        "n_trials": int(total),
        "branches": {
            "cvcl": {
                "correct": int(branch_correct["cvcl"]),
                "total": int(branch_total["cvcl"]),
                "accuracy": float(branch_correct["cvcl"]) / float(branch_total["cvcl"])
                if branch_total["cvcl"] > 0
                else None,
            },
            "neuron": {
                "correct": int(branch_correct["neuron"]),
                "total": int(branch_total["neuron"]),
                "accuracy": float(branch_correct["neuron"]) / float(branch_total["neuron"])
                if branch_total["neuron"] > 0
                else None,
            },
        },
        "oov": {
            "correct": int(oov_correct),
            "total": int(oov_total),
            "accuracy": float(oov_correct) / float(oov_total) if oov_total > 0 else None,
        },
        "in_vocab": {
            "correct": int(inv_correct),
            "total": int(inv_total),
            "accuracy": float(inv_correct) / float(inv_total) if inv_total > 0 else None,
        },
    }

    if neuron_map_meta is not None:
        metrics["neuron_map_meta"] = neuron_map_meta

    return results, metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    _seed_everything(0)

    # ---------------- model and config loading ----------------
    if args.clip_eval:
        if args.eval_mode != "cvcl":
            raise ValueError("--clip_eval only supports --eval_mode cvcl")

        print("[eval] Loading CLIP ViT-L/14")
        checkpoint_name = "clip_vitl_14"
        run_tag = checkpoint_name
        model, _ = clip.load("ViT-L/14", device=device)
        model.eval()
        print("[eval] CLIP loaded")

        parser = _setup_parser()
        data_args = parser.parse_args([])

        config = {
            "model": "clip",
            "seed": None,
            "shuffle_utterances": None,
            "cnn": "clip",
            "augment_frames": None,
            "multiple_frames": None,
        }
    else:
        ckpt_path = _resolve_ckpt_path(args.checkpoint)
        checkpoint_name = ckpt_path.name
        run_tag = _infer_run_tag(ckpt_path)

        print(f"[eval] Using checkpoint: {ckpt_path}")
        print(f"[eval] run_tag: {run_tag}")
        print("[eval] Forcing vm_enable=False during checkpoint load")

        lit = _load_lit_disable_vm(ckpt_path, map_location=device)
        model = lit

        _ckpt_key_sanity_report(ckpt_path, lit)

        hp = getattr(lit, "args", {}) or getattr(lit, "hparams", {}) or {}

        pretrained = bool(hp.get("pretrained_cnn", False))
        finetune = bool(hp.get("finetune_cnn", False))
        if pretrained and finetune:
            cnn_str = "finetune_pretrained"
        elif (not pretrained) and finetune:
            cnn_str = "finetune_random_init"
        elif (not pretrained) and (not finetune):
            cnn_str = "frozen_random_init"
        else:
            cnn_str = "frozen_pretrained"

        config = {
            "model": hp.get("text_encoder", "embedding"),
            "seed": hp.get("seed", None),
            "shuffle_utterances": bool(hp.get("shuffle_utterances", False)),
            "augment_frames": bool(hp.get("augment_frames", True)),
            "multiple_frames": bool(hp.get("multiple_frames", True)),
            "cnn": cnn_str,
        }

        parser = _setup_parser()
        data_args = parser.parse_args([])

        for k, v in hp.items():
            if hasattr(data_args, k):
                setattr(data_args, k, v)

    # ---------------- data module and eval split ----------------
    data_args.augment_frames = False
    data_args.eval_include_sos_eos = args.eval_include_sos_eos
    data_args.eval_type = args.eval_type
    data_args.eval_metadata_filename = args.eval_metadata_filename

    # Pass through dataset-specific knobs if present
    for k in [
        "coco_data_dir",
        "coco_images_dir",
        "coco_instances_json",
        "coco_n_foils",
        "coco_n_repeats",
        "coco_max_instances_per_label",
        "coco_min_box_area",
        "coco_min_box_side",
        "coco_seed",
        "coco_regenerate_trials",
        "coco_regenerate_label_map",
        "coco_label_map_path",
        "coco_label_overrides_json",
        "coco_label_mode",
        "vocab_filename",
        "imagenet_data_dir",
        "imagenet_images_dir",
        "imagenet_bbox_dir",
        "imagenet_words_txt",
        "imagenet_n_foils",
        "imagenet_n_repeats",
        "imagenet_max_images_per_label",
        "imagenet_min_box_area",
        "imagenet_min_box_side",
        "imagenet_seed",
        "imagenet_regenerate_trials",
        "imagenet_regenerate_label_map",
        "imagenet_label_map_path",
        "imagenet_label_overrides_json",
        "imagenet_no_bboxes",
        "imagenet_label_mode",
        "cifar_data_dir",
        "cifar_dataset",
        "cifar_n_foils",
        "cifar_n_repeats",
        "cifar_max_images_per_label",
        "cifar_seed",
        "cifar_regenerate_trials",
        "cifar_regenerate_label_map",
        "cifar_label_map_path",
        "cifar_label_overrides_json",
        "cifar_label_mode",
        "konkle_data_dir",
        "konkle_categories_dir",
        "konkle_label_mode",
        "konkle_regenerate_trials",
        "konkle_regenerate_label_map",
        "konkle_label_map_path",
        "konkle_label_overrides_json",
        "konkle_n_foils",
        "konkle_n_repeats",
        "konkle_max_images_per_label",
        "konkle_seed",
    ]:
        if hasattr(args, k):
            setattr(data_args, k, getattr(args, k))

    if args.clip_eval:
        data_args.clip_eval = True

    # In neuron_classifier mode, default to a label_mode that preserves OOV labels when supported.
    # This makes it possible to do OOV evaluation without changing the CLI defaults.
    if args.eval_mode == "neuron_classifier" and not getattr(args, "nc_keep_label_mode", False):
        for attr in [
            "konkle_label_mode",
            "coco_label_mode",
            "imagenet_label_mode",
            "cifar_label_mode",
        ]:
            if hasattr(data_args, attr) and getattr(data_args, attr) == "vocab":
                setattr(data_args, attr, "canonical")

    # ---------------- select dataset ----------------
    datamodule = None

    if args.eval_dataset == "saycam":
        datamodule = MultiModalSAYCamDataModule(data_args)
        datamodule.setup()

        vocab = datamodule.read_vocab()
        eval_data = load_data(EVAL_DATA_DIR / data_args.eval_metadata_filename)

        dl_fn = {"dev": datamodule.val_dataloader, "test": datamodule.test_dataloader}[args.stage]
        dataloader = dl_fn()[1]

        if getattr(dataloader, "batch_size", 1) != 1:
            raise RuntimeError(f"[eval] This eval loop assumes batch_size=1, got {dataloader.batch_size}.")

        if data_args.eval_metadata_filename == "eval_manual_filtered_test.json":
            classes = sorted(os.listdir(DATA_DIR / "eval_manual_filtered" / "test"))
        else:
            split_dir = EVAL_FRAMES_DIRNAME / args.stage
            if split_dir.exists():
                classes = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
            else:
                classes = sorted(os.listdir(EVAL_FRAMES_DIRNAME / "dev"))

    elif args.eval_dataset == "object_categories":
        datamodule = KonkleObjectCategoriesDataModule(data_args)
        datamodule.setup()

        dataloader = datamodule.test_dataloader()
        if getattr(dataloader, "batch_size", 1) != 1:
            raise RuntimeError(f"[eval] This eval loop assumes batch_size=1, got {dataloader.batch_size}.")

        vocab = datamodule.read_vocab()
        eval_data = load_data(EVAL_DATA_DIR / data_args.eval_metadata_filename)
        classes = datamodule.labels

    elif args.eval_dataset == "coco_instances":
        datamodule = COCOInstancesDataModule(data_args)
        datamodule.setup()

        dataloader = datamodule.test_dataloader()
        if getattr(dataloader, "batch_size", 1) != 1:
            raise RuntimeError(f"[eval] This eval loop assumes batch_size=1, got {dataloader.batch_size}.")

        vocab = datamodule.vocab
        eval_data = load_data(EVAL_DATA_DIR / data_args.eval_metadata_filename)
        classes = datamodule.labels

    elif args.eval_dataset == "imagenet_val":
        datamodule = ImageNetValDataModule(data_args)
        datamodule.setup()

        dataloader = datamodule.test_dataloader()
        if getattr(dataloader, "batch_size", 1) != 1:
            raise RuntimeError(f"[eval] This eval loop assumes batch_size=1, got {dataloader.batch_size}.")

        vocab = datamodule.vocab
        eval_data = load_data(EVAL_DATA_DIR / data_args.eval_metadata_filename)
        classes = datamodule.labels

    elif args.eval_dataset in ("cifar10", "cifar100"):
        if getattr(data_args, "cifar_dataset", None) is None:
            data_args.cifar_dataset = args.eval_dataset

        datamodule = CIFARForcedChoiceDataModule(data_args)
        datamodule.setup()

        dataloader = datamodule.test_dataloader()
        vocab = datamodule.vocab
        eval_data = load_data(EVAL_DATA_DIR / data_args.eval_metadata_filename)
        classes = datamodule.labels

    else:
        raise ValueError(f"Unknown eval_dataset: {args.eval_dataset}")

    if len(eval_data) != len(dataloader):
        print(
            f"[eval] WARNING: eval_data length ({len(eval_data)}) != dataloader length ({len(dataloader)}). "
            "Result logging that uses eval_data[i] may be misaligned."
        )

    args._datamodule = datamodule

    # ---------------- evaluation ----------------
    if args.eval_mode == "cvcl":
        if getattr(args, "labeled_t", False):
            if args.eval_dataset != "saycam":
                raise ValueError("[labeled_t] Only implemented for --eval_dataset saycam.")
            results, metrics = _cvcl_labeled_t_eval_loop(
                model=model,
                eval_data=eval_data,
                classes=classes,
                vocab=vocab,
                args=args,
                checkpoint_name=checkpoint_name,
                config=config,
            )
        else:
            results, metrics = _cvcl_only_eval_loop(
                model=model,
                dataloader=dataloader,
                eval_data=eval_data,
                classes=classes,
                vocab=vocab,
                args=args,
                checkpoint_name=checkpoint_name,
                config=config,
            )

        print(f"\n[eval] Total accuracy: {metrics['overall_accuracy']:%}")

    elif args.eval_mode == "neuron_classifier":
        results, metrics = _hybrid_eval_loop(
            model=model,
            dataloader=dataloader,
            eval_data=eval_data,
            classes=classes,
            vocab=vocab,
            args=args,
            checkpoint_name=checkpoint_name,
            config=config,
        )

        print(f"\n[hybrid_eval] Total accuracy: {metrics['overall_accuracy']:%}")
        b = metrics.get("branches", {})
        if b.get("cvcl", {}).get("accuracy") is not None:
            print(
                f"[hybrid_eval] CVCL-branch accuracy: {b['cvcl']['accuracy']:%} (n={b['cvcl']['total']})"
            )
        if b.get("neuron", {}).get("accuracy") is not None:
            print(
                f"[hybrid_eval] Neuron-branch accuracy: {b['neuron']['accuracy']:%} (n={b['neuron']['total']})"
            )

        if metrics["oov"]["accuracy"] is not None:
            print(
                f"[hybrid_eval] OOV(target) accuracy: {metrics['oov']['accuracy']:%} (n={metrics['oov']['total']})"
            )
        if metrics["in_vocab"]["accuracy"] is not None:
            print(
                f"[hybrid_eval] In-vocab(target) accuracy: {metrics['in_vocab']['accuracy']:%} "
                f"(n={metrics['in_vocab']['total']})"
            )
    else:
        raise ValueError(f"Unknown eval_mode: {args.eval_mode}")

    # ---------------- save predictions ----------------
    if args.save_predictions:
        results_dict = {"data": results, "metrics": metrics}

        base_dir = Path("results") / args.eval_dataset / run_tag
        base_dir.mkdir(parents=True, exist_ok=True)

        if args.clip_eval:
            leaf = f"clip_{args.eval_type}_{args.eval_dataset}_{args.stage}_{args.eval_mode}_eval_predictions.json"
        else:
            suffix = args.eval_mode
            if getattr(args, "labeled_t", False):
                suffix = suffix + "_labeledT"
            if args.eval_mode == "neuron_classifier" and args.nc_all_labels:
                suffix = "neuron_classifier_all"
            ckpt_stem = Path(args.checkpoint).name.replace(".ckpt", "") if args.checkpoint else "none"
            leaf = (
                f"{config.get('model','cvcl')}_{config.get('cnn','cnn')}_ckpt_{ckpt_stem}_seed_{config.get('seed',None)}_"
                f"{args.eval_type}_{args.eval_dataset}_{args.stage}_{suffix}_eval_predictions.json"
            )

        results_filename = base_dir / leaf
        print(f"\nSaving predictions to {results_filename}")
        with open(results_filename, "w") as f:
            json.dump(results_dict, f)
        try:
            base = os.path.splitext(str(results_filename))[0]

            if args.eval_mode == "neuron_classifier":
                # Save TWO CSVs, one for OOD (in-vocab targets) and one for OOV (out-of-vocab targets).
                per_ood = metrics.get("per_class_in_vocab_target", {}) or {}
                per_oov = metrics.get("per_class_oov_target", {}) or {}

                def _rows(per_class_dict):
                    rows = []
                    for name in sorted(per_class_dict.keys()):
                        m = per_class_dict.get(name, {}) or {}
                        total = int(m.get("total", 0) or 0)
                        if total <= 0:
                            continue
                        rows.append({
                            "class": name,
                            "correct": int(m.get("correct", 0) or 0),
                            "total": total,
                            "accuracy": m.get("accuracy", None),
                        })
                    return rows

                ood_csv = base + "_OOD_class_metrics.csv"
                oov_csv = base + "_OOV_class_metrics.csv"

                pd.DataFrame(_rows(per_ood)).to_csv(ood_csv, index=False)
                pd.DataFrame(_rows(per_oov)).to_csv(oov_csv, index=False)

                print(f"Saved OOD per-class metrics CSV to {ood_csv}")
                print(f"Saved OOV per-class metrics CSV to {oov_csv}")

            else:
                per_class_rows = [{"class": k, **v} for k, v in metrics.get("per_class", {}).items()]
                metrics_csv = base + "_class_metrics.csv"
                pd.DataFrame(per_class_rows).to_csv(metrics_csv, index=False)
                print(f"Saved per-class metrics CSV to {metrics_csv}")

        except Exception as e:
            print(f"Could not write per-class metrics CSV(s): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=False, default=None)
    parser.add_argument("--clip_eval", action="store_true")
    parser.add_argument("--stage", type=str, default="test", choices=["dev", "test"])
    parser.add_argument("--eval_include_sos_eos", action="store_true")
    parser.add_argument("--eval_type", type=str, default="image", choices=["image", "text"])
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="saycam",
        choices=["saycam", "object_categories", "coco_instances", "imagenet_val", "cifar10", "cifar100"],
    )
    parser.add_argument("--eval_metadata_filename", type=str, default="eval_test.json")
    parser.add_argument("--use_kitty_label", action="store_true")
    parser.add_argument("--save_predictions", action="store_true")

    parser.add_argument("--eval_mode", type=str, default="cvcl", choices=["cvcl", "neuron_classifier"])

    parser.add_argument(
        "--vocab_filename",
        type=str,
        default=None,
        help="Path to the CVCL training vocab.json (must match the checkpoint).",
    )

    # Neuron-classifier options
    parser.add_argument(
        "--nc_all_labels",
        action="store_true",
        help="If set, run all trials with neuron-classifier (instead of only OOV trials).",
    )
    parser.add_argument(
        "--nc_keep_label_mode",
        action="store_true",
        help="If set, do not auto-switch label_mode from vocab->canonical in neuron_classifier mode.",
    )
    parser.add_argument("--nc_hook", type=str, default="auto")
    parser.add_argument("--nc_clip_model", type=str, default="ViT-L/14")
    parser.add_argument("--nc_clip_download_root", type=str, default=str(Path("~/.cache/clip").expanduser()))
    parser.add_argument("--nc_topk", type=int, default=20)
    parser.add_argument("--nc_probe_max_images", type=int, default=2000)
    parser.add_argument("--nc_activation_reduce", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--nc_prompt", type=str, default="a photo of a {}")
    parser.add_argument("--nc_common_words_txt", type=str, default=None)
    parser.add_argument("--nc_cache_dir", type=str, default=str(EVAL_DATA_DIR / "neuron_cache"))
    parser.add_argument("--nc_regenerate", action="store_true")

    # Konkle objects
    parser.add_argument("--konkle_data_dir", type=str, default="eval_datasets/17-objects")
    parser.add_argument("--konkle_categories_dir", type=str, default=None)
    parser.add_argument("--konkle_label_mode", type=str, default="vocab", choices=["vocab", "canonical", "raw"])
    parser.add_argument("--konkle_regenerate_trials", action="store_true")
    parser.add_argument("--konkle_regenerate_label_map", action="store_true")
    parser.add_argument("--konkle_label_map_path", type=str, default=None)
    parser.add_argument("--konkle_label_overrides_json", type=str, default=None)
    parser.add_argument("--konkle_n_foils", type=int, default=3)
    parser.add_argument("--konkle_n_repeats", type=int, default=5)
    parser.add_argument("--konkle_max_images_per_label", type=int, default=200)
    parser.add_argument("--konkle_seed", type=int, default=0)

    # COCO
    parser.add_argument("--coco_data_dir", type=str, default="eval_datasets/coco")
    parser.add_argument("--coco_images_dir", type=str, default=None)
    parser.add_argument("--coco_instances_json", type=str, default=None)
    parser.add_argument("--coco_n_foils", type=int, default=3)
    parser.add_argument("--coco_n_repeats", type=int, default=1)
    parser.add_argument("--coco_max_instances_per_label", type=int, default=200)
    parser.add_argument("--coco_min_box_area", type=float, default=32 * 32)
    parser.add_argument("--coco_min_box_side", type=float, default=16)
    parser.add_argument("--coco_seed", type=int, default=0)
    parser.add_argument("--coco_regenerate_trials", action="store_true")
    parser.add_argument("--coco_regenerate_label_map", action="store_true")
    parser.add_argument("--coco_label_map_path", type=str, default=None)
    parser.add_argument("--coco_label_overrides_json", type=str, default=None)
    parser.add_argument("--coco_label_mode", type=str, default="vocab", choices=["vocab", "canonical", "raw"])

    # ImageNet
    parser.add_argument("--imagenet_data_dir", type=str, default="eval_datasets/imagenet")
    parser.add_argument("--imagenet_images_dir", type=str, default="eval_datasets/imagenet/imgs")
    parser.add_argument("--imagenet_bbox_dir", type=str, default=None)
    parser.add_argument("--imagenet_words_txt", type=str, default=None)
    parser.add_argument("--imagenet_n_foils", type=int, default=3)
    parser.add_argument("--imagenet_n_repeats", type=int, default=1)
    parser.add_argument("--imagenet_max_images_per_label", type=int, default=50)
    parser.add_argument("--imagenet_min_box_area", type=float, default=32 * 32)
    parser.add_argument("--imagenet_min_box_side", type=float, default=16)
    parser.add_argument("--imagenet_seed", type=int, default=0)
    parser.add_argument("--imagenet_regenerate_trials", action="store_true")
    parser.add_argument("--imagenet_regenerate_label_map", action="store_true")
    parser.add_argument("--imagenet_label_map_path", type=str, default=None)
    parser.add_argument("--imagenet_label_overrides_json", type=str, default=None)
    parser.add_argument("--imagenet_no_bboxes", action="store_true")
    parser.add_argument("--imagenet_label_mode", type=str, default="vocab", choices=["vocab", "canonical", "raw"])

    # CIFAR
    parser.add_argument("--cifar_data_dir", type=str, default="eval_datasets")
    parser.add_argument("--cifar_dataset", type=str, default=None, choices=["cifar10", "cifar100"])
    parser.add_argument("--cifar_n_foils", type=int, default=3)
    parser.add_argument("--cifar_n_repeats", type=int, default=1)
    parser.add_argument("--cifar_max_images_per_label", type=int, default=100)
    parser.add_argument("--cifar_seed", type=int, default=0)
    parser.add_argument("--cifar_regenerate_trials", action="store_true")
    parser.add_argument("--cifar_regenerate_label_map", action="store_true")
    parser.add_argument("--cifar_label_map_path", type=str, default=None)
    parser.add_argument("--cifar_label_overrides_json", type=str, default=None)
    parser.add_argument("--cifar_label_mode", type=str, default="vocab", choices=["vocab", "canonical", "raw"])

    # ---------------- Labeled-T (temporal) eval ----------------
    parser.add_argument(
        "--labeled_t",
        action="store_true",
        help="If set, run temporal Labeled-T evaluation: each candidate image becomes a clip (bag of frames).",
    )
    parser.add_argument(
        "--labeled_t_frames_root",
        type=str,
        default=str(EXTRACTED_FRAMES_DIRNAME),
        help="Root dir of 5fps frames to build temporal clips from (default: expt_saycam/train_5fps).",
    )
    parser.add_argument("--labeled_t_num_frames", type=int, default=5)
    parser.add_argument("--labeled_t_stride", type=int, default=1)
    parser.add_argument(
        "--labeled_t_combine",
        type=str,
        default="alpha",
        choices=["alpha", "sum", "max"],
        help="How to combine global and object scores.",
    )
    parser.add_argument("--labeled_t_alpha", type=float, default=0.5, help="Alpha for combine=alpha.")
    parser.add_argument(
        "--labeled_t_exclude_null",
        action="store_true",
        help="Exclude the packed null object candidate when computing object score.",
    )
    parser.set_defaults(labeled_t_exclude_null=True)

    parser.add_argument(
        "--labeled_t_mask_source",
        type=str,
        default="patch",
        choices=["patch", "sam"],
        help="Force MIL mask source during Labeled-T eval (default patch, since no SAM on test).",
    )

    args = parser.parse_args()

    if not args.clip_eval and not args.checkpoint:
        raise ValueError("Provide --checkpoint /path/to/last.ckpt (or use --clip_eval).")

    main(args)
