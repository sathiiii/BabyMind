#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CVCL training on BabyMind-style data with:
- default cache paths in data_splits/ (train_pairs.jsonl, test_pairs.jsonl)
- skipping augmented events (is_aug==True) by default
- cache-first loading when --save_pairs_cache is NOT given
- cache writing when --save_pairs_cache is given
- verbose logging + smoke-test option
- safe default for TextEncoder.embedding_dim

Quick build cache + exit:
  python train_cvcl_babymind.py \
    --frames_root data_splits/5fps_images \
    --train_json  data_splits/train.json \
    --save_pairs_cache \
    --verbose \
    --build_pairs_only

Typical train (auto-load cache if it exists; else build on the fly):
  python train_cvcl_babymind.py \
    --frames_root data_splits/5fps_images \
    --train_json  data_splits/train.json \
    --embedding_dim 256 \
    --batch_size 128 --num_workers 8 \
    --verbose
"""

from __future__ import annotations
import argparse
import os
import re
import json
import glob
import math
import time
import random
import collections
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# ---- CVCL encoders / Lightning Module (from your repo) ----
from multimodal.multimodal import VisionEncoder, TextEncoder
from multimodal.multimodal_lit import MultiModalLitModel


# =========================
# Utilities and constants
# =========================
def _now() -> str:
    return time.strftime("%H:%M:%S")

def _natural_sort_key(x: str):
    name = os.path.basename(x)
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p for p in parts]

def _list_images(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if not os.path.exists(folder):
        return []
    try:
        allf = [os.path.join(folder, f) for f in os.listdir(folder)]
    except Exception:
        return []
    imgs = [p for p in allf if os.path.isfile(p) and p.lower().endswith(exts)]
    return sorted(imgs, key=_natural_sort_key)

def _split_sents(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in re.split(r'(?<=[.!?])\s+', s.strip()) if x.strip()]

def _basic_tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    return re.findall(r"[a-zA-Z0-9']+|[.,!?;]", s)

def _uniform_pick(seq: List[str], n: int) -> List[str]:
    if not seq:
        return []
    if n <= 1:
        return [seq[(len(seq)-1)//2]]
    if n >= len(seq):
        return list(seq)
    idxs = [int(round(i * (len(seq)-1) / float(n-1))) for i in range(n)]
    return [seq[i] for i in idxs]

def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


# =========================
# Text special tokens
# =========================
PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN = "<pad>", "<unk>", "<s>", "</s>"
PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3


# =========================
# Vocab builder (verbose, skips augmented if requested)
# =========================
def build_vocab_from_json(
    json_path: str,
    max_size: int = 30000,
    min_freq: int = 1,
    include_caption: bool = True,
    include_narration: bool = True,
    skip_augmented: bool = True,
    verbose: bool = False,
    progress_every: int = 500,
) -> Dict[str, int]:
    t0 = time.time()
    if verbose:
        print(f"[{_now()}] [vocab] Reading JSON: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    events = data["events"] if isinstance(data, dict) and "events" in data else data
    if not isinstance(events, list):
        raise RuntimeError("Unsupported JSON for vocab: expected list or {'events': [...]}")

    cnt = collections.Counter()
    n = len(events)
    skipped_aug = 0
    if verbose:
        print(f"[{_now()}] [vocab] Events: {n} | skip_augmented={skip_augmented}")

    for i, ev in enumerate(events, 1):
        if skip_augmented and _as_bool(ev.get("is_aug", False)):
            skipped_aug += 1
            continue

        for t in (ev.get("transcript") or ev.get("utterances") or ev.get("captions") or []):
            if isinstance(t, dict):
                txt = (t.get("text") or t.get("caption") or t.get("utterance") or "").strip()
            else:
                txt = str(t)
            if txt:
                cnt.update(_basic_tokenize(txt))
        if include_caption:
            cnt.update(_basic_tokenize(ev.get("caption") or ev.get("description") or ""))
        if include_narration:
            cnt.update(_basic_tokenize(ev.get("narration") or ""))

        if verbose and (i % progress_every == 0):
            print(f"[{_now()}] [vocab] processed {i}/{n} … uniq_tokens={len(cnt)}")

    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID, SOS_TOKEN: SOS_ID, EOS_TOKEN: EOS_ID}
    for tok, c in cnt.most_common():
        if c < min_freq: break
        if tok in vocab: continue
        if len(vocab) >= max_size: break
        vocab[tok] = len(vocab)

    if verbose:
        print(f"[{_now()}] [vocab] DONE in {time.time()-t0:.2f}s | size={len(vocab)} | skipped_aug={skipped_aug}")
    return vocab

def encode_text(text: str, vocab: Dict[str, int]) -> Tuple[torch.Tensor, int]:
    toks = [SOS_TOKEN] + _basic_tokenize(text) + [EOS_TOKEN]
    ids = [vocab.get(t, UNK_ID) for t in toks]
    return torch.tensor(ids, dtype=torch.long), len(ids)


# =========================
# CVCL dataset (from filesystem) — can DUMP cache
# =========================
class CVCLBabyMindPairs(Dataset):
    """
    Emits: (image_tensor, token_ids, token_len, [raw_text])
    Strong pairs: utterances with timestamps (1 sampled frame from utterance window)
    Weak pairs (optional): narration/caption sentences (uniform frame sampling)
    Can dump a JSONL cache with up to K frame candidates per pair.
    """
    def __init__(
        self,
        splits_json: str,
        frames_root: str,
        split: str = "train",
        vocab: Optional[Dict[str, int]] = None,
        img_size: int = 224,
        sample_fps: float = 5.0,
        include_utterances: bool = True,
        include_narration_sentences: bool = False,
        include_caption_sentences: bool = False,
        max_global_sent_per_event: int = 4,
        skip_augmented: bool = True,
        train_augs: bool = True,
        rng_seed: int = 0,
        verbose: bool = False,
        progress_every: int = 500,
    ):
        t0 = time.time()
        self.verbose = bool(verbose)
        self.progress_every = int(progress_every)
        self.skip_augmented = bool(skip_augmented)
        if self.verbose:
            print(f"[{_now()}] [ds] INIT split={split} | frames_root={frames_root} | json={splits_json} | skip_augmented={self.skip_augmented}")

        assert os.path.exists(splits_json), f"splits json not found: {splits_json}"
        assert os.path.exists(frames_root), f"frames root not found: {frames_root}"
        self.split = split
        self.sample_fps = float(sample_fps)
        self.include_utts = bool(include_utterances)
        self.include_narr = bool(include_narration_sentences)
        self.include_caps = bool(include_caption_sentences)
        self.max_global = int(max_global_sent_per_event)
        self.vocab = vocab or {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID, SOS_TOKEN: SOS_ID, EOS_TOKEN: EOS_ID}
        random.seed(rng_seed)

        # allow frames_root/<split> if present
        candidate = os.path.join(frames_root, split)
        self.frames_root = candidate if os.path.exists(candidate) else frames_root
        if self.verbose:
            print(f"[{_now()}] [ds] Using frames root: {self.frames_root}")

        # read JSON
        with open(splits_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        events_all: List[Dict[str, Any]] = data["events"] if isinstance(data, dict) and "events" in data else data
        if not isinstance(events_all, list):
            raise RuntimeError("Unsupported JSON format: expected list or {'events': [...]}")

        # filter augments up front
        if self.skip_augmented:
            kept, skipped = [], 0
            for ev in events_all:
                if _as_bool(ev.get("is_aug", False)):
                    skipped += 1
                    continue
                kept.append(ev)
            self.events = kept
            if self.verbose:
                print(f"[{_now()}] [ds] filtered augmented: kept={len(self.events)} / total={len(events_all)} (skipped_aug={skipped})")
        else:
            self.events = events_all
            if self.verbose:
                print(f"[{_now()}] [ds] events loaded: {len(self.events)} (including augmented)")

        # transforms
        if train_augs and split == "train":
            self.tf = T.Compose([
                T.Resize(int(img_size * 1.15), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomCrop(img_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
            ])
        else:
            self.tf = T.Compose([
                T.Resize(int(img_size * 1.15), interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
            ])

        # build (frame_candidates, text) pairs
        self.pairs: List[Dict[str, Any]] = []
        cnt_frames_total = 0
        cnt_events_with_frames = 0
        n = len(self.events)

        for i, ev in enumerate(self.events, 1):
            vid = str(ev.get("video_id") or ev.get("video") or ev.get("vid") or "")
            eid = ev.get("event_id") or ev.get("id") or ev.get("event")
            frames = self._get_event_frames(vid, eid, ev)

            if frames:
                cnt_events_with_frames += 1
                cnt_frames_total += len(frames)

            is_aug_flag = _as_bool(ev.get("is_aug", False))

            # utterance-based pairs (strong)
            if frames and self.include_utts:
                ev_start = ev.get("start")
                for u in (ev.get("transcript") or ev.get("utterances") or ev.get("captions") or []):
                    if isinstance(u, dict):
                        s = u.get("start"); e = u.get("end")
                        txt = (u.get("text") or u.get("caption") or u.get("utterance") or "").strip()
                    else:
                        s = e = None
                        txt = str(u)
                    if not txt:
                        continue
                    cand = frames
                    if (s is not None) and (e is not None) and (ev_start is not None):
                        rel_s = max(0.0, float(s) - float(ev_start))
                        rel_e = max(0.0, float(e) - float(ev_start))
                        s_idx = int(math.floor(rel_s * self.sample_fps))
                        e_idx = max(s_idx, int(math.ceil(rel_e * self.sample_fps)))
                        L = len(frames)
                        s_idx = max(0, min(s_idx, L - 1))
                        e_idx = max(0, min(e_idx, L - 1))
                        cand = frames[s_idx:e_idx + 1] if e_idx >= s_idx else [frames[(s_idx + e_idx) // 2]]
                    self.pairs.append(dict(
                        video_id=vid, event_id=eid, text=txt,
                        frame_candidates=cand, kind="utter",
                        is_aug=is_aug_flag,
                    ))

            # narration sentences (weak)
            if frames and self.include_narr:
                for s in _split_sents(ev.get("narration") or "")[:self.max_global]:
                    self.pairs.append(dict(
                        video_id=vid, event_id=eid, text=s,
                        frame_candidates=frames, kind="narr",
                        is_aug=is_aug_flag,
                    ))

            # caption sentences (weak)
            if frames and self.include_caps:
                cap = ev.get("caption") or ev.get("description") or ""
                for s in _split_sents(cap)[:self.max_global]:
                    self.pairs.append(dict(
                        video_id=vid, event_id=eid, text=s,
                        frame_candidates=frames, kind="cap",
                        is_aug=is_aug_flag,
                    ))

            if self.verbose and (i % self.progress_every == 0):
                print(f"[{_now()}] [ds] processed {i}/{n} events … pairs={len(self.pairs)}")

        if len(self.pairs) == 0:
            raise RuntimeError("No (frame, text) pairs could be constructed. "
                               "Check frames_root and JSON fields: 'video_id','event_id','transcript'.")

        if self.verbose:
            kinds = collections.Counter([p["kind"] for p in self.pairs])
            avg_frames = (cnt_frames_total / max(1, cnt_events_with_frames))
            print(f"[{_now()}] [ds] DONE pairs build in {time.time()-t0:.2f}s | pairs={len(self.pairs)} "
                  f"| kinds={dict(kinds)} | events_with_frames={cnt_events_with_frames}/{n} "
                  f"| avg_frames/event≈{avg_frames:.1f}")

    def _get_event_frames(self, video_id: str, event_id: Any, ev: Dict[str, Any]) -> List[str]:
        cand1 = os.path.join(self.frames_root, str(video_id), f"event_{event_id}")
        cand2 = os.path.join(self.frames_root, str(video_id), str(event_id))
        if os.path.exists(cand1):
            fps = _list_images(cand1)
            if fps:
                return fps
        if os.path.exists(cand2):
            fps = _list_images(cand2)
            if fps:
                return fps

        # fall back to video-level frames
        video_dir = os.path.join(self.frames_root, str(video_id))
        fps = []
        if os.path.exists(video_dir):
            cand_flat = sorted(glob.glob(os.path.join(video_dir, "*", "*")), key=_natural_sort_key)
            cand_flat = [p for p in cand_flat if os.path.isfile(p) and p.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))]
            if not cand_flat:
                cand_flat = _list_images(video_dir)
            if cand_flat:
                s = ev.get("start"); e = ev.get("end")
                if s is None or e is None:
                    fps = cand_flat
                else:
                    s_idx = int(math.floor(float(s) * self.sample_fps))
                    e_idx = int(math.ceil (float(e) * self.sample_fps)) - 1
                    L = len(cand_flat)
                    s_idx = max(0, min(s_idx, L - 1)); e_idx = max(0, min(e_idx, L - 1))
                    fps = cand_flat[s_idx:e_idx + 1] if e_idx >= s_idx else [cand_flat[(s_idx + e_idx) // 2]]
        return fps

    # ---- runtime sampling ----
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        d = self.pairs[idx]
        frame_path = random.choice(d["frame_candidates"])
        try:
            img = Image.open(frame_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        image = self.tf(img)
        tok_ids, tok_len = encode_text(d["text"], self.vocab)
        return image, tok_ids, tok_len, [d["text"]]

    # ---- cache dump (JSONL) ----
    def dump_pairs_cache(self, jsonl_path: str, max_candidates_per_pair: int = 1, verbose: bool = False):
        Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
        K = max(1, int(max_candidates_per_pair))
        t0 = time.time()
        n = len(self.pairs)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i, p in enumerate(self.pairs, 1):
                cands = p["frame_candidates"]
                sel = _uniform_pick(cands, K)
                rec = {
                    "image_paths": sel,
                    "text": p["text"],
                    "kind": p["kind"],
                    "video_id": p["video_id"],
                    "event_id": str(p["event_id"]),
                    "is_aug": bool(p.get("is_aug", False)),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if verbose and (i % 50000 == 0):
                    print(f"[{_now()}] [cache] wrote {i}/{n} lines …")
        if verbose:
            print(f"[{_now()}] [cache] Saved {n} pairs to {jsonl_path} in {time.time()-t0:.2f}s")
        return jsonl_path


# =========================
# CVCL dataset (from JSONL cache)
# =========================
class CVCLPairsFromCache(Dataset):
    """
    Loads pairs from a JSONL file produced by CVCLBabyMindPairs.dump_pairs_cache.
    Each item: picks one image path at random from 'image_paths', tokenizes 'text'.
    Can skip augmented pairs at load time.
    """
    def __init__(
        self,
        jsonl_path: str,
        vocab: Dict[str, int],
        split: str = "train",
        img_size: int = 224,
        train_augs: bool = True,
        skip_augmented: bool = True,
        verbose: bool = False,
    ):
        assert os.path.exists(jsonl_path), f"pairs cache not found: {jsonl_path}"
        self.vocab = vocab
        self.verbose = bool(verbose)
        self.split = split
        self.skip_augmented = bool(skip_augmented)

        # transforms
        if train_augs and split == "train":
            self.tf = T.Compose([
                T.Resize(int(img_size * 1.15), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomCrop(img_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
            ])
        else:
            self.tf = T.Compose([
                T.Resize(int(img_size * 1.15), interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
            ])

        # load JSONL
        t0 = time.time()
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not rec.get("image_paths"):
                    continue
                if self.skip_augmented and _as_bool(rec.get("is_aug", False)):
                    continue
                self.rows.append(rec)
                if self.verbose and (i % 100000 == 0):
                    print(f"[{_now()}] [cache] read {i} lines … kept={len(self.rows)}")
        if self.verbose:
            print(f"[{_now()}] [cache] Loaded {len(self.rows)} pairs from {jsonl_path} in {time.time()-t0:.2f}s "
                  f"(skip_augmented={self.skip_augmented})")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        p = random.choice(r["image_paths"])
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        image = self.tf(img)
        tok_ids, tok_len = encode_text(r["text"], self.vocab)
        return image, tok_ids, tok_len, [r["text"]]


# =========================
# Collate (pad token seqs)
# =========================
def pad_collate(batch):
    imgs, seqs, lens, raw = zip(*batch)
    imgs = torch.stack(imgs, 0)
    maxL = max(int(L) for L in lens)
    padded = torch.full((len(seqs), maxL), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        L = s.numel()
        padded[i, :L] = s
    return imgs, padded, torch.tensor(lens, dtype=torch.long), list(raw)


# =========================
# CLI + Trainer builder
# =========================
def parse_args():
    p = argparse.ArgumentParser()

    # Data (no validation by default)
    p.add_argument("--frames_root", type=str, required=True)
    p.add_argument("--train_json", type=str, required=True)
    p.add_argument("--test_json", type=str, default=None)
    p.add_argument("--vocab_json", type=str, default=None)

    # Pairs cache files (now optional; defaults are derived)
    p.add_argument("--pairs_cache_train", type=str, default=None,
                   help="JSONL to load/save train pairs. Defaults to data_splits/<train_stem>_pairs.jsonl")
    p.add_argument("--pairs_cache_test", type=str, default=None,
                   help="JSONL to load/save test pairs.  Defaults to data_splits/<test_stem>_pairs.jsonl")
    p.add_argument("--pairs_candidates", type=int, default=4,
                   help="How many candidate frame paths to store per pair (1..K).")
    p.add_argument("--save_pairs_cache", action="store_true",
                   help="If set, will save cache JSONL when building from filesystem.")
    p.add_argument("--force_rebuild_pairs_cache", action="store_true",
                   help="If set, ignore existing cache and rebuild from filesystem.")
    p.add_argument("--build_pairs_only", action="store_true",
                   help="Build cache(s) then exit without training.")

    # Dataset knobs
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--sample_fps", type=float, default=5.0)
    p.add_argument("--include_narration_sentences", action="store_true")
    p.add_argument("--include_caption_sentences", action="store_true")
    p.add_argument("--max_global_sent_per_event", type=int, default=4)
    p.add_argument("--include_utterances", dest="include_utterances", action="store_true")
    p.add_argument("--no_include_utterances", dest="include_utterances", action="store_false")
    p.set_defaults(include_utterances=True)

    # Skip augmented events
    p.add_argument("--skip_augmented", dest="skip_augmented", action="store_true",
                   help="Skip events with is_aug=True (default).")
    p.add_argument("--dont_skip_augmented", dest="skip_augmented", action="store_false",
                   help="Include augmented events.")
    p.set_defaults(skip_augmented=True)

    # Vocab
    p.add_argument("--max_vocab_size", type=int, default=30000)
    p.add_argument("--min_token_freq", type=int, default=1)

    # Loader
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)

    # Training / logging
    p.add_argument("--exp_name", type=str, default="cvcl_babymind")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_top_k", type=int, default=1)
    p.add_argument("--resume_ckpt", type=str, default=None)
    p.add_argument("--logger", action="store_true")

    # Trainer core
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--accelerator", type=str, default="auto", choices=["auto","cpu","gpu","cuda","mps","tpu"])
    p.add_argument("--devices", type=str, default="auto")
    p.add_argument("--precision", type=str, default="32")
    p.add_argument("--strategy", type=str, default="auto")
    p.add_argument("--gradient_clip_val", type=float, default=0.0)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--log_every_n_steps", type=int, default=50)
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches", type=float, default=1.0)
    p.add_argument("--val_check_interval", type=float, default=1.0)
    p.add_argument("--check_val_every_n_epoch", type=int, default=1)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--fast_dev_run", action="store_true")

    # Verbose/debug
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--progress_every", type=int, default=500)
    p.add_argument("--debug_first_batch", action="store_true")

    # Forward model args
    VisionEncoder.add_to_argparse(p)
    TextEncoder.add_to_argparse(p)
    MultiModalLitModel.add_to_argparse(p)

    return p.parse_args()

def _default_cache_path(json_path: Optional[str]) -> Optional[str]:
    """Derive data_splits/<stem>_pairs.jsonl from a json path; else None."""
    if not json_path:
        return None
    j = Path(json_path)
    stem = j.stem  # e.g., 'train'
    # Put caches under the *same* directory OR under data_splits/ if it's already that.
    if j.parent.name == "data_splits":
        return str(j.parent / f"{stem}_pairs.jsonl")
    # fallback: always put in data_splits/
    return str(Path("data_splits") / f"{stem}_pairs.jsonl")

def build_trainer(args, callbacks, logger, has_val_loader: bool):
    devices = args.devices
    if isinstance(devices, str) and devices not in ("auto",):
        if "," in devices:
            devices = [int(x) for x in devices.split(",") if x.strip() != ""]
        else:
            try:
                devices = int(devices)
            except ValueError:
                pass

    num_sanity = 0 if not has_val_loader else 2

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=devices,
        precision=args.precision,
        strategy=args.strategy,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        deterministic=args.deterministic,
        benchmark=args.benchmark,
        fast_dev_run=args.fast_dev_run,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=num_sanity,
        enable_checkpointing=True,
    )
    return trainer


# =========================
# Pipeline helpers
# =========================
def maybe_make_vocab(args) -> dict:
    if args.vocab_json and Path(args.vocab_json).exists():
        if args.verbose:
            print(f"[{_now()}] [main] Loading vocab from {args.vocab_json}")
        with open(args.vocab_json, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        if args.verbose:
            print(f"[{_now()}] [main] Vocab loaded: size={len(vocab)}")
        return vocab

    if args.verbose:
        print(f"[{_now()}] [main] Building vocab from {args.train_json} …")

    t0 = time.time()
    vocab = build_vocab_from_json(
        args.train_json,
        max_size=args.max_vocab_size,
        min_freq=args.min_token_freq,
        include_caption=True,
        include_narration=True,
        skip_augmented=args.skip_augmented,
        verbose=args.verbose,
        progress_every=args.progress_every,
    )
    if args.vocab_json:
        Path(args.vocab_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.vocab_json, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        if args.verbose:
            print(f"[{_now()}] [main] Vocab saved to {args.vocab_json}")
    if args.verbose:
        print(f"[{_now()}] [main] Vocab ready in {time.time()-t0:.2f}s | size={len(vocab)}")
    return vocab

def _build_or_load_split(args, json_path: Optional[str], cache_path: Optional[str],
                         split: str, vocab: dict, train_augs: bool):
    """
    Returns (dataset, used_cache: bool).
    Policy:
      - If a cache path exists (defaulted or user-given) and not forcing rebuild -> load cache.
      - Else build from filesystem.
      - If --save_pairs_cache, write cache (to that path) after building.
    """
    if json_path is None:
        return None, False

    # Resolve default cache path if none provided
    cache_path = cache_path or _default_cache_path(json_path)
    if args.verbose:
        print(f"[{_now()}] [main] split={split} cache path => {cache_path}")

    use_cache = False
    if cache_path and os.path.exists(cache_path) and (not args.force_rebuild_pairs_cache):
        if args.verbose:
            print(f"[{_now()}] [main] Loading split={split} from cache: {cache_path}")
        ds = CVCLPairsFromCache(cache_path, vocab, split=split, img_size=args.img_size,
                                train_augs=train_augs, skip_augmented=args.skip_augmented,
                                verbose=args.verbose)
        use_cache = True
        return ds, use_cache

    # Build from filesystem
    if args.verbose:
        why = "no cache found" if (cache_path and not os.path.exists(cache_path)) else "force rebuild"
        print(f"[{_now()}] [main] Building split={split} from filesystem ({why}): {json_path}")
    ds_fs = CVCLBabyMindPairs(
        splits_json=json_path,
        frames_root=args.frames_root,
        split=split,
        vocab=vocab,
        img_size=args.img_size,
        sample_fps=args.sample_fps,
        include_utterances=args.include_utterances,
        include_narration_sentences=args.include_narration_sentences,
        include_caption_sentences=args.include_caption_sentences,
        max_global_sent_per_event=args.max_global_sent_per_event,
        skip_augmented=args.skip_augmented,
        train_augs=train_augs,
        rng_seed=args.seed,
        verbose=args.verbose,
        progress_every=args.progress_every,
    )

    # Save cache if requested
    if cache_path and args.save_pairs_cache:
        if args.verbose:
            print(f"[{_now()}] [main] Saving cache for split={split} to {cache_path} "
                  f"(candidates per pair={args.pairs_candidates}) …")
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        ds_fs.dump_pairs_cache(cache_path, max_candidates_per_pair=args.pairs_candidates, verbose=args.verbose)
        # re-open from cache (leaner memory & uniform behavior)
        ds = CVCLPairsFromCache(cache_path, vocab, split=split, img_size=args.img_size,
                                train_augs=train_augs, skip_augmented=args.skip_augmented,
                                verbose=args.verbose)
        use_cache = True
        return ds, use_cache

    # Otherwise, return filesystem dataset
    if args.verbose and cache_path and not os.path.exists(cache_path) and not args.save_pairs_cache:
        print(f"[{_now()}] [main] Note: cache {cache_path} not found and --save_pairs_cache not set; "
              f"training will read from filesystem directly.")
    return ds_fs, use_cache

def make_loader(ds: Dataset, split: str, args):
    if ds is None:
        return None
    if args.verbose:
        print(f"[{_now()}] [main] DataLoader for split={split} | batch_size={args.batch_size} | num_workers={args.num_workers}")
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        collate_fn=pad_collate
    )


# =========================
# Main
# =========================
def main():
    args = parse_args()
    if args.verbose:
        print(f"[{_now()}] [main] Args parsed. Setting seed={args.seed}")
    pl.seed_everything(args.seed)

    if args.verbose:
        print(f"[{_now()}] [env] Torch {torch.__version__} | CUDA: {torch.cuda.is_available()} "
              f"| GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    # Derive default cache paths up-front so logs are clear
    if args.pairs_cache_train is None:
        args.pairs_cache_train = _default_cache_path(args.train_json)
        if args.verbose:
            print(f"[{_now()}] [main] Default train cache => {args.pairs_cache_train}")
    if args.test_json and args.pairs_cache_test is None:
        args.pairs_cache_test = _default_cache_path(args.test_json)
        if args.verbose:
            print(f"[{_now()}] [main] Default test cache  => {args.pairs_cache_test}")

    ckpt_dir = Path("checkpoints") / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_ckpt == "last":
        args.resume_ckpt = ckpt_dir / "last.ckpt"

    # --------- Vocab ----------
    vocab = maybe_make_vocab(args)

    # --------- Build/load pairs ----------
    train_ds, _ = _build_or_load_split(
        args, args.train_json, args.pairs_cache_train, "train", vocab, train_augs=True
    )
    # No validation set by default.
    val_ds = None

    # Optionally cache test (not used for training)
    if args.test_json and args.pairs_cache_test and (args.save_pairs_cache or args.force_rebuild_pairs_cache):
        _ = _build_or_load_split(args, args.test_json, args.pairs_cache_test, "test", vocab, train_augs=False)

    if args.build_pairs_only:
        if args.verbose:
            print(f"[{_now()}] [main] build_pairs_only requested — exiting before training.")
        return

    # --------- DataLoaders ----------
    train_loader = make_loader(train_ds, "train", args)
    val_loader   = make_loader(val_ds,   "val",   args)  # remains None

    # First-batch smoke test
    if args.debug_first_batch:
        if args.verbose:
            print(f"[{_now()}] [main] Debugging first batch …")
        try:
            batch = next(iter(train_loader))
            imgs, ids, lens, raw = batch
            print(f"[{_now()}] [main] First batch shapes: images={tuple(imgs.shape)}, ids={tuple(ids.shape)}, lens={tuple(lens.shape)}")
            print(f"[{_now()}] [main] Sample raw text[0]: {raw[0][0][:120] if raw and raw[0] else 'N/A'}")
        except Exception as e:
            print(f"[{_now()}] [main][ERROR] Failed to fetch first batch: {e}")
            print("[hint] Try --num_workers 0 to debug DataLoader issues.")
            raise

    # --------- Model ----------
    if args.verbose:
        print(f"[{_now()}] [main] Initializing encoders …")
    if not getattr(args, "embedding_type", None):
        args.embedding_type = "flat"
    if not hasattr(args, "embedding_dim") or args.embedding_dim is None:
        args.embedding_dim = 256
    if args.verbose:
        print(f"[{_now()}] [main] Text config: vocab_size={len(vocab)} | embedding_dim={args.embedding_dim}")

    vision_encoder = VisionEncoder(args=args)
    text_encoder = TextEncoder(vocab, image_feature_map_dim=vision_encoder.last_cnn_out_dim, args=args)
    lit_model = MultiModalLitModel(vision_encoder, text_encoder, args)
    if args.verbose:
        n_params = sum(p.numel() for p in lit_model.parameters())
        print(f"[{_now()}] [main] Model ready | total params={n_params/1e6:.2f}M")

    # --------- Checkpointing & logging ----------
    callbacks = [ModelCheckpoint(save_last=True, save_top_k=0, dirpath=ckpt_dir, filename="{epoch}")]
    logger = WandbLogger(project="multimodal-babymind", name=args.exp_name, log_model=True) if args.logger else None
    trainer = build_trainer(args, callbacks=callbacks, logger=logger, has_val_loader=False)

    # --------- Train ----------
    if args.verbose:
        print(f"[{_now()}] [main] Starting training … (no validation)")
        print(args)

    trainer.fit(
        lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=None,
        ckpt_path=str(args.resume_ckpt) if args.resume_ckpt else None
    )

    if args.verbose:
        print(f"[{_now()}] [main] Training finished.")


if __name__ == "__main__":
    main()
