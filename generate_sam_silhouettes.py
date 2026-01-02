#!/usr/bin/env python
import argparse
import inspect
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

import contextlib
import gc

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    CLIPProcessor,
    CLIPModel,
)

from segment_anything import sam_model_registry, SamPredictor


# -------------------------------------------------------------------
# Labeled-S concepts (for coverage accounting / canonicalization)
# -------------------------------------------------------------------
LABELED_S_OBJECT_CONCEPTS: List[str] = [
    "ball",
    "basket",
    "car",
    "cat",
    "chair",
    "computer",
    "crib",
    "door",
    "foot",
    "ground",  # treated as texture / background below
    "hand",
    "kitchen",  # scene, handled as texture below
    "paper",
    "puzzle",
    "road",     # scene, handled as texture below
    "room",     # scene, handled as texture below
    "sand",     # texture below
    "stairs",
    "table",
    "toy",
    "window",
]

LABELED_S_TEXTURE_CONCEPTS: List[str] = [
    "floor",
    "ground",
    "road",
    "sand",
    "room",
    "kitchen",
]

LABELED_S_ALL_CONCEPTS: List[str] = sorted(
    set(LABELED_S_OBJECT_CONCEPTS + LABELED_S_TEXTURE_CONCEPTS)
)
LABELED_S_CANONICAL_BY_LOWER: Dict[str, str] = {
    c.lower(): c for c in LABELED_S_ALL_CONCEPTS
}


# -------------------------------------------------------------------
# Priority / extra object concepts (for DINO + CLIP)
# -------------------------------------------------------------------
PRIORITY_CLASSES: List[str] = [
    "ball",
    "basket",
    "car",
    "cat",
    "chair",
    "computer",
    "crib",
    "door",
    "foot",
    "hand",
    "paper",
    "puzzle",
    "stairs",
    "table",
    "window",
]
PRIORITY_CLASSES_LOWER = {c.lower() for c in PRIORITY_CLASSES}

DEFAULT_EXTRA_CONCEPTS: List[str] = [
    # household / tableware / food
    "cup", "bottle", "mug", "bowl", "plate", "spoon", "fork", "knife",

    # furniture / fixtures
    "chair", "table", "couch", "sofa", "bed", "crib", "high chair", "stool",
    "shelf", "cushion", "pillow", "lamp", "light",

    # clothing / body-related items
    "shirt", "pants", "dress", "shorts", "jacket", "hat", "sock", "shoe",
    "diaper",

    # toys / baby gear
    "toy", "ball", "block", "book", "puzzle", "doll", "stroller", "swing",

    # vehicles
    "car", "truck", "bus", "train", "bike", "wagon",

    # animals
    "dog", "cat", "bird",

    # people / body parts
    "baby", "child",
    "face", "head",
    "hand", "foot", "leg", "arm",

    # appliances / fixtures
    "fridge", "refrigerator", "oven", "stove", "microwave", "dishwasher",
    "sink", "fan",

    # furniture / storage
    "bookshelf", "dresser", "drawer", "cabinet", "closet",
    "wardrobe", "desk", "rug", "carpet", "curtain", "blinds", "highchair",

    # electronics
    "phone", "cell phone", "smartphone",
    "television", "tv",
    "laptop", "computer",

    # extra toys
    "stuffed animal", "teddy bear", "rattle", "blocks", "toy car",
]

GENERIC_HUMAN_CONCEPTS = {
    "person",
    "people",
    "adult",
    "human",
}


# -------------------------------------------------------------------
# Texture and background concepts
# -------------------------------------------------------------------
TEXTURE_CONCEPTS = {
    "floor",
    "ground",
    "road",
    "sand",
    "room",
    "kitchen",
    "grass",
    "sky",
    "cloud",
    "street",
    "sidewalk",
    "path",
    "carpet",
    "rug",
    "yard",
    "park",
    "curtain",
}
TEXTURE_CONCEPTS_LOWER = {c.lower() for c in TEXTURE_CONCEPTS}

DEFAULT_BACKGROUND_CONCEPTS = {
    "wall",
    "floor",
    "ground",
    "ceiling",
    "room",
    "street",
    "road",
    "path",
    "sidewalk",
    "yard",
    "park",
    "sky",
    "cloud",
    "grass",
    "carpet",
    "rug",
    "sand",
    "background",
}
DEFAULT_BACKGROUND_CONCEPTS_LOWER = {c.lower() for c in DEFAULT_BACKGROUND_CONCEPTS}

BACKGROUND_ONLY_CONCEPTS = [
    c
    for c in DEFAULT_BACKGROUND_CONCEPTS
    if c.lower() not in TEXTURE_CONCEPTS_LOWER
]
BACKGROUND_ONLY_CONCEPTS_LOWER = {c.lower() for c in BACKGROUND_ONLY_CONCEPTS}

# Masks larger than this fraction of the frame are treated as near whole frame
NEAR_FULL_MASK_AREA_FRAC = 0.995


# -------------------------------------------------------------------
# CLIP prompt templates and context overrides
# -------------------------------------------------------------------
CLIP_TEMPLATES: List[str] = [
    "a photo of a {}",
    "a close-up photo of a {}",
    "a small {}",
    "a toy {}",
]

CLIP_CONTEXT_OVERRIDES: Dict[str, str] = {
    "leg": "human leg",
    "arm": "human arm",
    "hand": "human hand",
    "foot": "human foot",
    "face": "human face",
    "head": "human head",
    "baby": "baby person",
    "child": "child person",
}


# -------------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi GPU Grounding DINO + SAM + CLIP mask extraction for SAYCam frames "
            "with separate passes for objects, textures, and backgrounds."
        )
    )

    # Data paths
    parser.add_argument(
        "--frames-root",
        type=str,
        required=True,
        help="Root directory containing training frames (e.g. expt_saycam/train_5fps).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where mask npz files and index JSON will be saved.",
    )

    # Optional concept expansion
    parser.add_argument(
        "--extra-concepts",
        type=str,
        default=None,
        help=(
            "Optional JSON file with extra concept names (list or dict keys) "
            "to merge with the built in concept list."
        ),
    )
    parser.add_argument(
        "--no-default-extra-concepts",
        action="store_true",
        help="If set, do not add DEFAULT_EXTRA_CONCEPTS as extra candidates.",
    )

    # Grounding DINO
    parser.add_argument(
        "--dino-model",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="Hugging Face repo id for Grounding DINO.",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.30,
        help="Box confidence threshold for Grounding DINO.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.75,
        help="Text matching threshold for Grounding DINO.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=30,
        help=(
            "If set, keep at most this many Grounding DINO detections per frame "
            "with highest scores after thresholding (per category pass)."
        ),
    )
    parser.add_argument(
        "--concepts-per-prompt",
        type=int,
        default=20,
        help=(
            "Number of concept names per Grounding DINO text prompt. "
            "The script runs DINO multiple times over chunks."
        ),
    )

    # SAM model config
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        required=True,
        help="Path to SAM checkpoint (e.g. sam_vit_h_4b8939.pth).",
    )
    parser.add_argument(
        "--sam-model-type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM backbone type.",
    )

    # Precision for GPU models
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help=(
            "Computation dtype for SAM, and autocast for DINO/CLIP on GPU if set to fp16. "
            "fp16 usually speeds up inference on modern GPUs."
        ),
    )

    # SAM thresholds and geometry filters
    parser.add_argument(
        "--sam-score-threshold",
        type=float,
        default=0.87,
        help="Minimum SAM mask score to accept a mask for a DINO box.",
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=3000,
        help="Minimum mask area used for filtering (in pixels).",
    )
    parser.add_argument(
        "--max-mask-area-frac",
        type=float,
        default=0.6,
        help=(
            "Maximum allowed mask area as a fraction of the full frame "
            "for the object pass. Larger masks are treated as surfaces "
            "and dropped there (textures/backgrounds handle them)."
        ),
    )
    parser.add_argument(
        "--min-mask-fill-frac",
        type=float,
        default=0.25,
        help=(
            "Minimum ratio of mask area to its bounding box area. "
            "Very thin or noisy masks are dropped."
        ),
    )
    parser.add_argument(
        "--max-mask-aspect-ratio",
        type=float,
        default=8.0,
        help=(
            "Maximum width/height or height/width of a mask bounding box. "
            "Extremely elongated masks (shadows, edges) are dropped."
        ),
    )

    # Background filtering (only affects the object pass)
    parser.add_argument(
        "--disable-background-filter",
        action="store_true",
        help="If set, do not drop background-like labels in the object pass.",
    )

    # Whether to mine textures and backgrounds
    parser.add_argument(
        "--no-mine-textures",
        action="store_false",
        dest="mine_textures",
        help="Disable texture mining pass.",
    )
    parser.add_argument(
        "--no-mine-backgrounds",
        action="store_false",
        dest="mine_backgrounds",
        help="Disable background mining pass.",
    )
    parser.set_defaults(mine_textures=True, mine_backgrounds=True)

    # CLIP configuration
    parser.add_argument(
        "--clip-model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Hugging Face repo id for CLIP model used for mask-level labeling.",
    )
    parser.add_argument(
        "--clip-sim-threshold",
        type=float,
        default=0.3,
        help=(
            "Minimum CLIP cosine similarity required to trust a label. "
            "Masks with lower similarity are kept but get clip_label=None "
            "and are not counted as Labeled-S hits."
        ),
    )
    parser.add_argument(
        "--clip-batch-size",
        type=int,
        default=64,
        help="Batch size for CLIP image feature computation.",
    )
    parser.add_argument(
        "--clip-on-cpu",
        action="store_true",
        help=(
            "Force CLIP to run on CPU even if CUDA is available. "
            "By default CLIP runs on the same GPU as DINO/SAM for speed."
        ),
    )

    args = parser.parse_args()
    return args


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------
def get_rank_and_world_size() -> Tuple[int, int]:
    """Read rank/world_size from torchrun environment, or default to single process."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return local_rank, world_size


def normalize_label_for_logic(label: str) -> str:
    """Lowercase and strip spaces/dots for logic checks."""
    return str(label).lower().strip(" .")


def sanitize_label_for_filename(label: str) -> str:
    """
    Turn a label into a safe token for filenames.
    Uses a simple normalization, then strips weird chars.
    """
    s = str(label).strip().lower()
    s = s.replace(" ", "_").replace("/", "_")
    s = "".join(ch for ch in s if ch.isalnum() or ch == "_")
    if not s:
        s = "unknown"
    return s


def load_extra_concepts(extra_json_path: Path) -> List[str]:
    """Optional extra concepts from a JSON file (dict keys or list)."""
    if not extra_concepts_path_is_valid(extra_json_path):
        return []
    with open(extra_json_path, "r") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        concepts = list(obj.keys())
    elif isinstance(obj, list):
        concepts = obj
    else:
        raise ValueError("extra_concepts JSON must be a dict or a list")

    concepts = [str(c).strip() for c in concepts if str(c).strip()]
    return concepts


def extra_concepts_path_is_valid(extra_json_path: Path) -> bool:
    if not extra_json_path.is_file():
        print(f"[WARN] extra_concepts path does not exist: {extra_json_path}")
        return False
    return True


def list_all_frames(frames_root: Path) -> List[str]:
    """Return all .jpg frames under frames_root, as relative paths."""
    rel_paths = [
        str(p.relative_to(frames_root))
        for p in sorted(frames_root.rglob("*.jpg"))
    ]
    return rel_paths


def chunk_list(lst: List[str], chunk_size: int) -> List[List[str]]:
    """Split a list into chunks of at most chunk_size."""
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]


def load_processed_frames_from_jsonl(index_file_path: Path) -> Set[str]:
    """
    Read frame_bases that have already been indexed from the rank JSONL file.

    This lets us skip frames instantly on resume without touching the GPU
    or scanning all NPZ directories.
    """
    processed: Set[str] = set()
    if not index_file_path.is_file():
        return processed

    with open(index_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            rel = entry.get("frame_relpath")
            if not rel:
                continue
            frame_base = rel.replace(os.sep, "_").split(".")[0]
            processed.add(frame_base)
    return processed


# -------------------------------------------------------------------
# Labeled-S counting helpers
# -------------------------------------------------------------------
def init_labeled_s_counts(csv_path: Path) -> Dict[str, int]:
    """
    Initialize Labeled-S counts from an existing CSV if present,
    otherwise start from zeros.
    """
    counts: Dict[str, int] = {c: 0 for c in LABELED_S_ALL_CONCEPTS}
    if not csv_path.is_file():
        return counts

    try:
        import csv

        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            _ = next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                concept, value = row[0], row[1]
                if concept in counts:
                    try:
                        counts[concept] = int(value)
                    except ValueError:
                        pass
    except Exception as e:
        print(f"[WARN] Failed to read existing Labeled-S counts from {csv_path}: {e}")

    return counts


def save_labeled_s_counts(csv_path: Path, counts: Dict[str, int]) -> None:
    """Write current Labeled-S counts to CSV (overwrites)."""
    try:
        import csv

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["concept", "count"])
            for concept in LABELED_S_ALL_CONCEPTS:
                writer.writerow([concept, counts.get(concept, 0)])
    except Exception as e:
        print(f"[WARN] Failed to write Labeled-S counts CSV at {csv_path}: {e}")


# -------------------------------------------------------------------
# Model setup
# -------------------------------------------------------------------
def build_grounding_dino(model_name: str, device: torch.device):
    print(f"Loading Grounding DINO model: {model_name} on {device}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, processor


def build_sam_predictor(args: argparse.Namespace, device: torch.device) -> SamPredictor:
    print(f"Loading SAM model: {args.sam_model_type} from {args.sam_checkpoint} on {device}")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    if args.precision == "fp16" and device.type == "cuda":
        sam.to(device=device, dtype=torch.float16)
    else:
        sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    return predictor


def build_clip(model_name: str, device: torch.device):
    """
    Build CLIP model and processor.

    Important: use_safetensors=True so that transformers loads model.safetensors
    rather than pytorch_model.bin.
    """
    print(f"Loading CLIP model: {model_name} on {device}")
    processor = CLIPProcessor.from_pretrained(model_name)
    try:
        model = CLIPModel.from_pretrained(
            model_name,
            use_safetensors=True,
        )
    except ValueError as e:
        raise RuntimeError(
            f"Failed to load CLIP model '{model_name}' with safetensors. "
            "Make sure this repo has model.safetensors or upgrade torch."
        ) from e

    model.to(device)
    model.eval()
    return model, processor


def safe_post_process_grounded_od(
    processor,
    outputs,
    input_ids,
    target_sizes,
    box_threshold: float,
    text_threshold: float,
):
    """
    Call GroundingDinoProcessor.post_process_grounded_object_detection in a way
    that works across different transformers versions.
    """
    fn = processor.post_process_grounded_object_detection
    sig = inspect.signature(fn)
    param_names = set(sig.parameters.keys())

    kwargs = {
        "outputs": outputs,
        "input_ids": input_ids,
    }

    if "box_threshold" in param_names:
        kwargs["box_threshold"] = box_threshold
    if "text_threshold" in param_names:
        kwargs["text_threshold"] = text_threshold
    if "target_sizes" in param_names:
        kwargs["target_sizes"] = target_sizes

    return fn(**kwargs)


# -------------------------------------------------------------------
# CLIP helpers
# -------------------------------------------------------------------
def precompute_clip_text_features(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
    class_names: List[str],
) -> Tuple[torch.Tensor, List[str]]:
    """
    Precompute CLIP text features for the given class names using templates.
    Returns:
      text_features: (T, D) normalized
      class_names: same order as text_features
    """
    if not class_names:
        raise ValueError("No CLIP class names provided")

    all_feats: List[torch.Tensor] = []

    for name in class_names:
        ctx_name = CLIP_CONTEXT_OVERRIDES.get(name.lower(), name)
        prompts = [tpl.format(ctx_name) for tpl in CLIP_TEMPLATES]
        with torch.no_grad():
            text_inputs = clip_processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            feats = clip_model.get_text_features(**text_inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        mean_feat = feats.mean(dim=0, keepdim=True)
        mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)
        all_feats.append(mean_feat)

    text_features = torch.cat(all_feats, dim=0)
    return text_features, class_names


def clip_annotate_masks(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
    image: np.ndarray,
    masks: List[np.ndarray],
    priority_text_features: torch.Tensor,
    priority_names: List[str],
    extra_text_features: torch.Tensor,
    extra_names: List[str],
    sim_threshold: float,
    batch_size: int,
) -> List[Tuple[Optional[str], float, bool]]:
    """
    CLIP labeling for masks.

    For each mask:
      1. Compute CLIP features on both the raw box crop and the masked crop,
         then fuse them.
      2. Compute similarity to priority and extra concepts.
      3. Take the single best match across all concepts.
      4. If its similarity is below sim_threshold, return clip_label=None
         but still report the similarity so masks are never dropped here.
    """
    if not masks:
        return []

    H, W = image.shape[:2]

    crops_box: List[Image.Image] = []
    crops_masked: List[Image.Image] = []
    valid_indices: List[int] = []

    for idx, mask in enumerate(masks):
        seg = mask.astype(bool)
        if not seg.any():
            continue

        ys, xs = np.where(seg)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        y_min = max(0, y_min)
        x_min = max(0, x_min)
        y_max = min(H - 1, y_max)
        x_max = min(W - 1, x_max)

        crop_full = image[y_min: y_max + 1, x_min: x_max + 1, :]
        seg_crop = seg[y_min: y_max + 1, x_min: x_max + 1]

        crop_masked = crop_full.copy()
        crop_masked[~seg_crop] = 0

        crops_box.append(Image.fromarray(crop_full))
        crops_masked.append(Image.fromarray(crop_masked))
        valid_indices.append(idx)

    if not crops_box:
        return [(None, 0.0, False) for _ in masks]

    all_image_features: List[torch.Tensor] = []

    use_amp = (device.type == "cuda")
    amp_ctx = torch.cuda.amp.autocast if use_amp else contextlib.nullcontext

    for start in range(0, len(crops_box), batch_size):
        batch_box = crops_box[start: start + batch_size]
        batch_masked = crops_masked[start: start + batch_size]

        with torch.no_grad(), amp_ctx():
            inputs_box = clip_processor(
                images=batch_box,
                return_tensors="pt",
            )
            inputs_box = {k: v.to(device) for k, v in inputs_box.items()}
            feats_box = clip_model.get_image_features(**inputs_box)
            feats_box = feats_box / feats_box.norm(dim=-1, keepdim=True)

            inputs_mask = clip_processor(
                images=batch_masked,
                return_tensors="pt",
            )
            inputs_mask = {k: v.to(device) for k, v in inputs_mask.items()}
            feats_mask = clip_model.get_image_features(**inputs_mask)
            feats_mask = feats_mask / feats_mask.norm(dim=-1, keepdim=True)

        fused = feats_box + feats_mask
        fused = fused / fused.norm(dim=-1, keepdim=True)
        all_image_features.append(fused)

    image_features = torch.cat(all_image_features, dim=0)

    out: List[Tuple[Optional[str], float, bool]] = [(None, 0.0, False) for _ in masks]

    has_priority = (
        priority_text_features is not None and priority_text_features.numel() > 0
    )
    has_extras = (
        extra_text_features is not None and extra_text_features.numel() > 0
    )

    sims_pri_all: Optional[torch.Tensor] = None
    sims_ex_all: Optional[torch.Tensor] = None

    if has_priority:
        sims_pri_all = image_features @ priority_text_features.T
    if has_extras:
        sims_ex_all = image_features @ extra_text_features.T

    for local_i, mask_idx in enumerate(valid_indices):
        best_label: Optional[str] = None
        best_sim = -1.0
        is_priority = False

        pri_best_sim = -1.0
        ex_best_sim = -1.0
        pri_idx: Optional[int] = None
        ex_idx: Optional[int] = None

        if has_priority and sims_pri_all is not None:
            sims_pri = sims_pri_all[local_i]
            val, idx = sims_pri.max(dim=0)
            pri_best_sim = float(val.item())
            pri_idx = int(idx.item())

        if has_extras and sims_ex_all is not None:
            sims_ex = sims_ex_all[local_i]
            val, idx = sims_ex.max(dim=0)
            ex_best_sim = float(val.item())
            ex_idx = int(idx.item())

        if pri_best_sim >= ex_best_sim:
            best_sim = pri_best_sim
            if pri_idx is not None:
                best_label = priority_names[pri_idx]
            is_priority = True
        else:
            best_sim = ex_best_sim
            if ex_idx is not None:
                best_label = extra_names[ex_idx]
            is_priority = False

        if best_sim < sim_threshold:
            best_label_out: Optional[str] = None
        else:
            best_label_out = best_label

        out[mask_idx] = (best_label_out, float(best_sim), bool(is_priority))

    return out


# -------------------------------------------------------------------
# Category pass: DINO -> SAM -> CLIP for a concept set
# -------------------------------------------------------------------
def run_category_pass(
    category_name: str,
    image: np.ndarray,
    image_pil: Image.Image,
    H: int,
    W: int,
    concepts: List[str],
    gdino_model,
    gdino_processor,
    sam_predictor: SamPredictor,
    clip_model,
    clip_processor,
    priority_text_features: torch.Tensor,
    priority_names: List[str],
    extra_text_features: torch.Tensor,
    extra_names: List[str],
    args: argparse.Namespace,
    dino_device: torch.device,
    clip_device: torch.device,
    filter_background_in_sam: bool,
    enforce_object_area_limit: bool,
    box_scale: float,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    """
    Run a full pass for a concept category (objects, textures, backgrounds).
    Returns final_dets, final_masks for that pass.
    """
    if not concepts:
        return [], []

    # Step 1: Grounding DINO
    concept_chunks = chunk_list(concepts, args.concepts_per_prompt)

    all_raw_dets: List[Dict[str, Any]] = []

    use_amp_dino = (dino_device.type == "cuda" and args.precision == "fp16")
    dino_amp_ctx = torch.cuda.amp.autocast if use_amp_dino else contextlib.nullcontext

    for chunk in concept_chunks:
        text_prompt = " . ".join(chunk)

        inputs = gdino_processor(
            images=image_pil,
            text=text_prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(dino_device) for k, v in inputs.items()}

        with torch.no_grad(), dino_amp_ctx():
            outputs = gdino_model(**inputs)

        target_sizes = torch.tensor([[H, W]], device=dino_device)

        results_list = safe_post_process_grounded_od(
            gdino_processor,
            outputs=outputs,
            input_ids=inputs["input_ids"],
            target_sizes=target_sizes,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )

        if not results_list:
            continue
        results = results_list[0]

        if "text_labels" in results:
            raw_labels = list(results["text_labels"])
        else:
            raw_labels = list(results["labels"])

        gdino_boxes = results["boxes"].detach().cpu().numpy()  # (N, 4) xyxy
        gdino_scores = results["scores"].detach().cpu().numpy()  # (N,)

        keep_mask = gdino_scores >= args.box_threshold
        gdino_boxes = gdino_boxes[keep_mask]
        gdino_scores = gdino_scores[keep_mask]
        gdino_labels = [
            str(lab) for lab, k in zip(raw_labels, keep_mask.tolist()) if k
        ]

        for box, score, label in zip(gdino_boxes, gdino_scores, gdino_labels):
            all_raw_dets.append(
                {
                    "box": box.tolist(),
                    "score": float(score),
                    "label": label,
                }
            )

    if not all_raw_dets:
        return [], []

    all_raw_dets.sort(key=lambda d: d["score"], reverse=True)

    if args.max_detections is not None and args.max_detections > 0:
        all_raw_dets = all_raw_dets[: args.max_detections]

    # Step 2: SAM refinement + geometric filtering
    sam_predictor.set_image(image)

    sam_kept_dets: List[Dict[str, Any]] = []
    sam_kept_masks: List[np.ndarray] = []

    full_area = H * W

    for det in all_raw_dets:
        label = det["label"]
        score = det["score"]

        label_norm = normalize_label_for_logic(label)

        if filter_background_in_sam and label_norm in DEFAULT_BACKGROUND_CONCEPTS_LOWER:
            continue

        x0, y0, x1, y1 = det["box"]
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        w = x1 - x0
        h = y1 - y0
        scale = box_scale
        nw = w * scale
        nh = h * scale

        nx0 = max(0.0, cx - nw / 2.0)
        ny0 = max(0.0, cy - nh / 2.0)
        nx1 = min(float(W - 1), cx + nw / 2.0)
        ny1 = min(float(H - 1), cy + nh / 2.0)

        box = np.array([[nx0, ny0, nx1, ny1]], dtype=np.float32)

        with torch.no_grad():
            masks, sam_scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True,
            )

        best_idx = int(np.argmax(sam_scores))
        mask = masks[best_idx]  # (H, W) bool
        sam_score = float(sam_scores[best_idx])

        if sam_score < args.sam_score_threshold:
            continue

        area = int(mask.sum())
        if area < args.min_mask_area:
            continue

        area_frac = area / float(full_area)
        if enforce_object_area_limit and area_frac > args.max_mask_area_frac:
            continue

        if area_frac >= NEAR_FULL_MASK_AREA_FRAC:
            continue

        ys, xs = np.where(mask)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        bbox_w = x_max - x_min + 1
        bbox_h = y_max - y_min + 1
        bbox_area = bbox_w * bbox_h

        fill_frac = area / float(bbox_area)
        if fill_frac < args.min_mask_fill_frac:
            continue

        aspect = max(bbox_w, bbox_h) / max(1.0, float(min(bbox_w, bbox_h)))
        if aspect > args.max_mask_aspect_ratio:
            continue

        det_out = {
            "box": det["box"],
            "score": score,
            "label": label,
            "mask_area": area,
            "sam_score": sam_score,
        }
        sam_kept_dets.append(det_out)
        sam_kept_masks.append(mask)

    if not sam_kept_dets:
        return [], []

    # Step 3: CLIP annotation (on clip_device)
    clip_results = clip_annotate_masks(
        clip_model=clip_model,
        clip_processor=clip_processor,
        device=clip_device,
        image=image,
        masks=sam_kept_masks,
        priority_text_features=priority_text_features,
        priority_names=priority_names,
        extra_text_features=extra_text_features,
        extra_names=extra_names,
        sim_threshold=args.clip_sim_threshold,
        batch_size=args.clip_batch_size,
    )

    final_dets: List[Dict[str, Any]] = []
    final_masks: List[np.ndarray] = []

    for det, mask, (clip_label, clip_sim, is_priority) in zip(
        sam_kept_dets, sam_kept_masks, clip_results
    ):
        det_out = dict(det)
        det_out["clip_label"] = clip_label
        det_out["clip_sim"] = float(clip_sim)
        det_out["is_priority"] = bool(is_priority)
        final_dets.append(det_out)
        final_masks.append(mask)

    if not final_dets:
        return [], []

    order = sorted(
        range(len(final_dets)),
        key=lambda i: (
            final_dets[i]["is_priority"],
            final_dets[i]["score"],
            final_dets[i]["sam_score"],
            final_dets[i]["clip_sim"],
        ),
        reverse=True,
    )
    final_dets = [final_dets[i] for i in order]
    final_masks = [final_masks[i] for i in order]

    return final_dets, final_masks


# -------------------------------------------------------------------
# Main worker
# -------------------------------------------------------------------
def run_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    torch.set_grad_enabled(False)

    frames_root = Path(args.frames_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Subdirectories that do not depend on GPU
    masked_images_root = output_dir / f"masked_images_rank{rank}"
    masked_images_root.mkdir(parents=True, exist_ok=True)
    masked_images_obj_dir = masked_images_root / "obj"
    masked_images_tex_dir = masked_images_root / "tex"
    masked_images_bg_dir = masked_images_root / "bg"
    masked_images_obj_dir.mkdir(parents=True, exist_ok=True)
    masked_images_tex_dir.mkdir(parents=True, exist_ok=True)
    masked_images_bg_dir.mkdir(parents=True, exist_ok=True)

    npz_root = output_dir / "npz"
    npz_root.mkdir(parents=True, exist_ok=True)
    npz_obj_dir = npz_root / "obj"
    npz_tex_dir = npz_root / "tex"
    npz_bg_dir = npz_root / "bg"
    npz_obj_dir.mkdir(parents=True, exist_ok=True)
    npz_tex_dir.mkdir(parents=True, exist_ok=True)
    npz_bg_dir.mkdir(parents=True, exist_ok=True)

    # Counts CSV (per rank) for Labeled-S concepts
    labeled_s_counts_csv = output_dir / f"labeled_s_counts_rank{rank}.csv"
    labeled_s_counts = init_labeled_s_counts(labeled_s_counts_csv)

    # Stream index entries line by line to JSONL to avoid RAM blowup
    index_file_path = output_dir / f"gdino_sam_clip_multi_index_rank{rank}.jsonl"

    # Read which frames this rank has already indexed so we can skip them instantly
    processed_frame_bases = load_processed_frames_from_jsonl(index_file_path)

    # List all frames and shard across ranks
    all_frames = list_all_frames(frames_root)
    if rank == 0:
        print(f"Total frames found : {len(all_frames)}")

    shard_frames = [
        rel_path
        for idx, rel_path in enumerate(all_frames)
        if idx % world_size == rank
    ]

    # Filter out frames already indexed in this rank's JSONL
    shard_unprocessed = [
        rel_path
        for rel_path in shard_frames
        if rel_path.replace(os.sep, "_").split(".")[0] not in processed_frame_bases
    ]

    print(
        f"[Rank {rank}] {len(shard_frames)} frames assigned, "
        f"{len(shard_unprocessed)} unprocessed for this run."
    )

    if not shard_unprocessed:
        print(f"[Rank {rank}] nothing to do, exiting without touching GPU.")
        return

    # Build concept lists
    extra_concepts_from_json: List[str] = []
    if args.extra_concepts is not None:
        extra_path = Path(args.extra_concepts)
        extra_concepts_from_json = load_extra_concepts(extra_path)

    # Base Labeled-S concepts (for accounting only)
    base_concepts = list(LABELED_S_ALL_CONCEPTS)

    # Full concept list (for metadata)
    full_concepts_set = set(base_concepts)
    if not args.no_default_extra_concepts:
        full_concepts_set.update(DEFAULT_EXTRA_CONCEPTS)
    full_concepts_set.update(extra_concepts_from_json)
    full_concepts = sorted(full_concepts_set)

    # Object concepts (DINO vocabulary for object pass)
    object_concepts: List[str] = []
    seen_obj = set()

    def add_object(name: str) -> None:
        n = str(name).strip()
        ln = n.lower()
        if not n:
            return
        if ln in GENERIC_HUMAN_CONCEPTS:
            return
        if ln in TEXTURE_CONCEPTS_LOWER:
            return
        if ln in DEFAULT_BACKGROUND_CONCEPTS_LOWER:
            return
        if ln in seen_obj:
            return
        seen_obj.add(ln)
        object_concepts.append(n)

    # Labeled-S objects first
    for c in LABELED_S_OBJECT_CONCEPTS:
        add_object(c)

    # Built-in extras
    if not args.no_default_extra_concepts:
        for c in DEFAULT_EXTRA_CONCEPTS:
            add_object(c)

    # Extra concepts from JSON
    for c in extra_concepts_from_json:
        add_object(c)

    # Texture and background concepts for separate passes
    texture_concepts = sorted(TEXTURE_CONCEPTS)
    background_concepts = sorted(BACKGROUND_ONLY_CONCEPTS)

    # Devices: DINO + SAM on GPU if available, CLIP on same GPU by default
    if torch.cuda.is_available():
        dino_device = torch.device(f"cuda:{rank}")
        if args.clip_on_cpu:
            clip_device = torch.device("cpu")
        else:
            clip_device = dino_device
    else:
        dino_device = torch.device("cpu")
        clip_device = torch.device("cpu")

    if rank == 0:
        print(f"Frames root        : {frames_root}")
        print(f"Output dir         : {output_dir}")
        print(f"NPZ root           : {npz_root}")
        print(f"Base Labeled-S     : {len(base_concepts)}")
        print(f"Full concept list  : {len(full_concepts)} (metadata only)")
        print(f"Object concepts    : {len(object_concepts)} for DINO")
        print(f"Texture concepts   : {len(texture_concepts)} for DINO")
        print(f"Background concepts: {len(background_concepts)} for DINO")
        print(f"World size         : {world_size}")
        print(f"DINO model         : {args.dino_model}")
        print(f"DINO thresholds    : box={args.box_threshold}, text={args.text_threshold}")
        print(f"Concepts/prompt    : {args.concepts_per_prompt}")
        print(f"SAM model          : {args.sam_model_type}, precision={args.precision}")
        print(f"SAM score thr      : {args.sam_score_threshold}")
        print(f"min_mask_area      : {args.min_mask_area}")
        print(f"background filter  : {'off' if args.disable_background_filter else 'on'}")
        print(f"CLIP model         : {args.clip_model}")
        print(f"CLIP sim threshold : {args.clip_sim_threshold}")
        print(f"CLIP batch size    : {args.clip_batch_size}")
        print(f"DINO/SAM device    : {dino_device}")
        print(f"CLIP device        : {clip_device}")

    # Now that we know there is work to do, build models on the proper devices
    gdino_model, gdino_processor = build_grounding_dino(args.dino_model, dino_device)
    sam_predictor = build_sam_predictor(args, dino_device)
    clip_model, clip_processor = build_clip(args.clip_model, clip_device)

    # CLIP text features:
    # object pass: priority vs extras
    object_concepts_lower = {c.lower() for c in object_concepts}
    obj_priority_concepts = [
        c for c in PRIORITY_CLASSES if c.lower() in object_concepts_lower
    ]
    obj_extra_concepts = [
        c for c in object_concepts if c.lower() not in PRIORITY_CLASSES_LOWER
    ]

    if obj_priority_concepts:
        obj_priority_feats, obj_priority_names = precompute_clip_text_features(
            clip_model,
            clip_processor,
            clip_device,
            obj_priority_concepts,
        )
    else:
        obj_priority_feats = torch.empty(0, 0, device=clip_device)
        obj_priority_names = []

    if obj_extra_concepts:
        obj_extra_feats, obj_extra_names = precompute_clip_text_features(
            clip_model,
            clip_processor,
            clip_device,
            obj_extra_concepts,
        )
    else:
        obj_extra_feats = torch.empty(0, 0, device=clip_device)
        obj_extra_names = []

    # texture pass: all textures as "priority"
    if args.mine_textures and texture_concepts:
        tex_feats, tex_names = precompute_clip_text_features(
            clip_model,
            clip_processor,
            clip_device,
            texture_concepts,
        )
    else:
        tex_feats = torch.empty(0, 0, device=clip_device)
        tex_names = []

    # background pass: all background-only as "priority"
    if args.mine_backgrounds and background_concepts:
        bg_feats, bg_names = precompute_clip_text_features(
            clip_model,
            clip_processor,
            clip_device,
            background_concepts,
        )
    else:
        bg_feats = torch.empty(0, 0, device=clip_device)
        bg_names = []

    # Open index JSONL for appending
    index_file = open(index_file_path, "a")

    total_masks = 0
    frames_with_masks = 0

    pbar = tqdm(
        shard_unprocessed,
        desc=f"rank{rank}",
        position=rank,
    )

    for rel_path in pbar:
        frame_base = rel_path.replace(os.sep, "_").split(".")[0]

        img_path = frames_root / rel_path
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"[Rank {rank}] Failed to open {img_path}: {e}")
            continue

        H, W = image.shape[:2]
        image_pil = Image.fromarray(image)

        # Per-frame counts
        concept_instance_counts: Dict[str, int] = defaultdict(int)
        saved_for_frame = 0

        # Pass 1: objects
        obj_dets, obj_masks = run_category_pass(
            category_name="objects",
            image=image,
            image_pil=image_pil,
            H=H,
            W=W,
            concepts=object_concepts,
            gdino_model=gdino_model,
            gdino_processor=gdino_processor,
            sam_predictor=sam_predictor,
            clip_model=clip_model,
            clip_processor=clip_processor,
            priority_text_features=obj_priority_feats,
            priority_names=obj_priority_names,
            extra_text_features=obj_extra_feats,
            extra_names=obj_extra_names,
            args=args,
            dino_device=dino_device,
            clip_device=clip_device,
            filter_background_in_sam=not args.disable_background_filter,
            enforce_object_area_limit=True,
            box_scale=1.2,
        )

        # Pass 2: textures
        tex_dets: List[Dict[str, Any]] = []
        tex_masks: List[np.ndarray] = []
        if args.mine_textures:
            tex_dets, tex_masks = run_category_pass(
                category_name="textures",
                image=image,
                image_pil=image_pil,
                H=H,
                W=W,
                concepts=texture_concepts,
                gdino_model=gdino_model,
                gdino_processor=gdino_processor,
                sam_predictor=sam_predictor,
                clip_model=clip_model,
                clip_processor=clip_processor,
                priority_text_features=tex_feats,
                priority_names=tex_names,
                extra_text_features=torch.empty(0, 0, device=clip_device),
                extra_names=[],
                args=args,
                dino_device=dino_device,
                clip_device=clip_device,
                filter_background_in_sam=False,
                enforce_object_area_limit=False,
                box_scale=1.0,
            )

        # Pass 3: backgrounds
        bg_dets: List[Dict[str, Any]] = []
        bg_masks: List[np.ndarray] = []
        if args.mine_backgrounds:
            bg_dets, bg_masks = run_category_pass(
                category_name="backgrounds",
                image=image,
                image_pil=image_pil,
                H=H,
                W=W,
                concepts=background_concepts,
                gdino_model=gdino_model,
                gdino_processor=gdino_processor,
                sam_predictor=sam_predictor,
                clip_model=clip_model,
                clip_processor=clip_processor,
                priority_text_features=bg_feats,
                priority_names=bg_names,
                extra_text_features=torch.empty(0, 0, device=clip_device),
                extra_names=[],
                args=args,
                dino_device=dino_device,
                clip_device=clip_device,
                filter_background_in_sam=False,
                enforce_object_area_limit=False,
                box_scale=1.0,
            )

        # Helper to save one category
        def save_category(
            category_prefix: str,
            concept_kind: str,
            dets: List[Dict[str, Any]],
            masks: List[np.ndarray],
            npz_dir: Path,
            masked_dir: Path,
        ) -> int:
            nonlocal total_masks, saved_for_frame, labeled_s_counts
            saved = 0

            for det, mask in zip(dets, masks):
                clip_label = det.get("clip_label")
                if clip_label is None:
                    # keep unlabeled masks out of the dataset index
                    continue

                seg = mask.astype(np.uint8)
                area = det["mask_area"]
                bbox = list(map(float, det["box"]))
                sam_score = det["sam_score"]
                dino_score = det["score"]
                clip_sim = det["clip_sim"]
                dino_label = det["label"]
                dino_label_norm = normalize_label_for_logic(dino_label)

                # Update Labeled-S counts if this CLIP label corresponds to a Labeled-S concept
                lower_label = clip_label.lower()
                if lower_label in LABELED_S_CANONICAL_BY_LOWER:
                    canonical = LABELED_S_CANONICAL_BY_LOWER[lower_label]
                    labeled_s_counts[canonical] = labeled_s_counts.get(canonical, 0) + 1

                key_for_counts = f"{category_prefix}:{clip_label}"
                instance_id = concept_instance_counts[key_for_counts]
                concept_instance_counts[key_for_counts] += 1

                label_clean = sanitize_label_for_filename(clip_label)
                base_name = f"{frame_base}_{category_prefix}_{label_clean}_{instance_id:03d}"

                mask_path = npz_dir / f"{base_name}.npz"
                np.savez_compressed(mask_path, mask=seg)

                masked_img = image.copy()
                masked_img[seg == 0] = 0
                masked_pil = Image.fromarray(masked_img)
                masked_img_path = masked_dir / f"{base_name}.png"
                masked_pil.save(masked_img_path)

                entry: Dict[str, Any] = {
                    "frame_relpath": rel_path,
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "masked_image_path": str(masked_img_path),
                    "height": H,
                    "width": W,
                    "concept_clip": clip_label,
                    "concept_kind": concept_kind,
                    "category_prefix": category_prefix,
                    "instance_id": instance_id,
                    "dino_label_raw": dino_label,
                    "dino_label_norm": dino_label_norm,
                    "dino_score": dino_score,
                    "sam_score": sam_score,
                    "clip_sim": clip_sim,
                    "mask_area": area,
                    "mask_bbox_xyxy": bbox,
                    "is_priority": bool(det.get("is_priority", False)),
                }
                # Stream entry to JSONL file to avoid large in-memory list
                index_file.write(json.dumps(entry) + "\n")

                total_masks += 1
                saved_for_frame += 1
                saved += 1

            return saved

        save_category("obj", "object", obj_dets, obj_masks, npz_obj_dir, masked_images_obj_dir)
        if args.mine_textures:
            save_category("tex", "texture", tex_dets, tex_masks, npz_tex_dir, masked_images_tex_dir)
        if args.mine_backgrounds:
            save_category("bg", "background", bg_dets, bg_masks, npz_bg_dir, masked_images_bg_dir)

        if saved_for_frame > 0:
            frames_with_masks += 1

        # Update Labeled-S coverage CSV after each frame for safety
        save_labeled_s_counts(labeled_s_counts_csv, labeled_s_counts)

        pbar.set_postfix(
            masks=total_masks,
            frames_with_masks=frames_with_masks,
        )

        # Free per-frame tensors/arrays
        del image, image_pil, obj_dets, obj_masks, tex_dets, tex_masks, bg_dets, bg_masks

        try:
            sam_predictor.reset_image()
        except AttributeError:
            pass

        gc.collect()

        if dino_device.type == "cuda":
            torch.cuda.empty_cache()

    pbar.close()
    index_file.close()

    print(
        f"[Rank {rank}] saved {total_masks} masks with CLIP labels in "
        f"{frames_with_masks} frames (out of {len(shard_unprocessed)} newly processed frames)."
    )

    partial_index: Dict[str, Any] = {
        "frames_root": str(frames_root),
        "output_dir": str(output_dir),
        "npz_root": str(npz_root),
        "npz_dirs": {
            "obj": str(npz_obj_dir),
            "tex": str(npz_tex_dir),
            "bg": str(npz_bg_dir),
        },
        "masked_images_root": str(masked_images_root),
        "base_concepts": base_concepts,
        "concept_list": full_concepts,
        "object_concepts": object_concepts,
        "texture_concepts": texture_concepts,
        "background_concepts": background_concepts,
        "world_size": world_size,
        "rank": rank,
        "index_file": str(index_file_path),
        "num_masks": total_masks,
        "num_frames_with_masks": frames_with_masks,
        "num_frames_processed": len(shard_unprocessed),
    }

    partial_index_path = (
        output_dir / f"gdino_sam_clip_multi_index_rank{rank}.json"
    )
    with open(partial_index_path, "w") as f:
        json.dump(partial_index, f)
    print(f"[Rank {rank}] wrote summary to {partial_index_path}")
    print(f"[Rank {rank}] stream index entries in {index_file_path}")

    if world_size == 1 and rank == 0:
        final_index_path = output_dir / "gdino_sam_clip_multi_index.json"
        with open(final_index_path, "w") as f:
            json.dump(
                {
                    "frames_root": str(frames_root),
                    "output_dir": str(output_dir),
                    "npz_root": str(npz_root),
                    "npz_dirs": {
                        "obj": str(npz_obj_dir),
                        "tex": str(npz_tex_dir),
                        "bg": str(npz_bg_dir),
                    },
                    "masked_images_root": str(masked_images_root),
                    "base_concepts": base_concepts,
                    "concept_list": full_concepts,
                    "object_concepts": object_concepts,
                    "texture_concepts": texture_concepts,
                    "background_concepts": background_concepts,
                    "index_file": str(index_file_path),
                    "num_masks": total_masks,
                    "num_frames_with_masks": frames_with_masks,
                    "num_frames_processed": len(shard_unprocessed),
                },
                f,
            )
        print(f"[Rank 0] wrote combined summary to {final_index_path}")


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    rank, world_size = get_rank_and_world_size()
    run_worker(rank, world_size, args)


if __name__ == "__main__":
    main()
