# cvcl_babymind_pairs.py
from __future__ import annotations
import os, re, json, glob, math, random, collections
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

# ---------------- Special tokens (TextEncoder typically expects these) ----------------
PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN = "<pad>", "<unk>", "<s>", "</s>"
PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3

# ---------------- Small helpers ----------------
def _natural_sort_key(x: str):
    name = os.path.basename(x)
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p for p in parts]

def _list_images(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if not os.path.exists(folder): return []
    try:
        allf = [os.path.join(folder, f) for f in os.listdir(folder)]
    except Exception:
        return []
    imgs = [p for p in allf if os.path.isfile(p) and p.lower().endswith(exts)]
    return sorted(imgs, key=_natural_sort_key)

def _split_sents(s: str) -> List[str]:
    if not s: return []
    return [x.strip() for x in re.split(r'(?<=[.!?])\s+', s.strip()) if x.strip()]

def _basic_tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    return re.findall(r"[a-zA-Z0-9']+|[.,!?;]", s)

def build_vocab_from_json(
    json_path: str,
    max_size: int = 30000,
    min_freq: int = 1,
    include_caption: bool = True,
    include_narration: bool = True
) -> Dict[str, int]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    events = data["events"] if isinstance(data, dict) and "events" in data else data
    cnt = collections.Counter()
    for ev in events:
        # transcripts / utterances
        for t in (ev.get("transcript") or ev.get("utterances") or ev.get("captions") or []):
            if isinstance(t, dict):
                txt = (t.get("text") or t.get("caption") or t.get("utterance") or "").strip()
            else:
                txt = str(t)
            if txt: cnt.update(_basic_tokenize(txt))
        if include_caption:  cnt.update(_basic_tokenize(ev.get("caption") or ev.get("description") or ""))
        if include_narration: cnt.update(_basic_tokenize(ev.get("narration") or ""))

    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID, SOS_TOKEN: SOS_ID, EOS_TOKEN: EOS_ID}
    for tok, c in cnt.most_common():
        if c < min_freq: break
        if tok in vocab: continue
        if len(vocab) >= max_size: break
        vocab[tok] = len(vocab)
    return vocab

def encode_text(text: str, vocab: Dict[str, int]) -> Tuple[torch.Tensor, int]:
    toks = [SOS_TOKEN] + _basic_tokenize(text) + [EOS_TOKEN]
    ids = [vocab.get(t, UNK_ID) for t in toks]
    return torch.tensor(ids, dtype=torch.long), len(ids)

# ---------------- The dataset for CVCL ----------------
class CVCLBabyMindPairs(Dataset):
    """
    Emits tuples expected by the CVCL training loop:
       (image_tensor, token_ids, token_len, [raw_text_string])

    • Primary supervision: utterances with timestamps (1 frame sampled from the utterance window).
    • Optional weak supervision: narration/caption sentences (uniform frame sampling across event).
    • Does NOT require both caption & narration. Uses what's available.

    Folder structure supported (relative to frames_root or frames_root/split):
      frames_root/
          <split>/ (optional)
              <video_id>/
                  event_<event_id>/   # images...
                or <video_id>/<event_id>/
              # If event_* folders are missing, it will fallback to scanning video-level frames.

    JSON supports either:
      - top-level list of events
      - {"events": [...]} wrapper

    Each event ideally includes:
      {
        "video_id": str, "event_id": int/str,
        "start": float, "end": float,
        "transcript": [{"start": float, "end": float, "text": str}, ...],
        "narration": str,  # optional
        "caption": str     # optional
      }
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
        train_augs: bool = True,
        rng_seed: int = 0
    ):
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

        # allow frames_root/<split> override if present
        base_root = frames_root
        candidate = os.path.join(base_root, split)
        self.frames_root = candidate if os.path.exists(candidate) else base_root

        # read JSON
        with open(splits_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.events: List[Dict[str, Any]] = data["events"] if isinstance(data, dict) and "events" in data else data
        if not isinstance(self.events, list):
            raise RuntimeError("Unsupported JSON format: expected list or {'events': [...]}")

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
        for ev in self.events:
            vid = str(ev.get("video_id") or ev.get("video") or ev.get("vid") or "")
            eid = ev.get("event_id") or ev.get("id") or ev.get("event")
            frames = self._get_event_frames(vid, eid, ev)
            if not frames:  # skip if no images
                continue

            # Utterance pairs (strong supervision)
            if self.include_utts:
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
                    self.pairs.append(dict(video_id=vid, event_id=eid, text=txt, frame_candidates=cand, kind="utter"))

            # Narration (weak)
            if self.include_narr:
                for s in _split_sents(ev.get("narration") or "")[:self.max_global]:
                    self.pairs.append(dict(video_id=vid, event_id=eid, text=s, frame_candidates=frames, kind="narr"))

            # Caption (weak)
            if self.include_caps:
                cap = ev.get("caption") or ev.get("description") or ""
                for s in _split_sents(cap)[:self.max_global]:
                    self.pairs.append(dict(video_id=vid, event_id=eid, text=s, frame_candidates=frames, kind="cap"))

        if len(self.pairs) == 0:
            raise RuntimeError("No (frame, text) pairs could be constructed. "
                               "Check frames_root and your JSON fields: 'video_id', 'event_id', 'transcript'/texts.")

    # ---------- frame resolving ----------
    def _get_event_frames(self, video_id: str, event_id: Any, ev: Dict[str, Any]) -> List[str]:
        # event-level folders
        cand1 = os.path.join(self.frames_root, str(video_id), f"event_{event_id}")
        cand2 = os.path.join(self.frames_root, str(video_id), str(event_id))
        if os.path.exists(cand1):
            fps = _list_images(cand1)
            if fps: return fps
        if os.path.exists(cand2):
            fps = _list_images(cand2)
            if fps: return fps

        # fallbacks: flattened video frames
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

    # ---------- Dataset API ----------
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        d = self.pairs[idx]
        frame_path = random.choice(d["frame_candidates"])  # 1 frame per pair (CVCL-style)
        try:
            img = Image.open(frame_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        image = self.tf(img)

        tok_ids, tok_len = encode_text(d["text"], self.vocab)
        # Tuple expected by CVCL Lightning code: (image, token_ids, token_len, [raw_text])
        return image, tok_ids, tok_len, [d["text"]]

# ---------------- Collate: pad token sequences to the batch max ----------------
def pad_collate(batch):
    imgs, seqs, lens, raw = zip(*batch)
    imgs = torch.stack(imgs, 0)
    maxL = max(int(L) for L in lens)
    padded = torch.full((len(seqs), maxL), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        L = s.numel()
        padded[i, :L] = s
    return imgs, padded, torch.tensor(lens, dtype=torch.long), list(raw)
