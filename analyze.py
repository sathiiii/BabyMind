#!/usr/bin/env python
import argparse
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    CLIPProcessor,
    CLIPModel,
)

from segment_anything import sam_model_registry, SamPredictor

import matplotlib
matplotlib.use("Agg")  # safe on headless servers
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Global debug switch
# -------------------------------------------------------------------
DEBUG = False


def bp(step_name: str) -> None:
    if DEBUG:
        print(f"\n[DEBUG] Reached step: {step_name}. Entering debugger...\n")
        breakpoint()


# -------------------------------------------------------------------
# Priority concepts (objects we care about most)
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


# -------------------------------------------------------------------
# Extra object-like concepts (for DINO and CLIP)
# -------------------------------------------------------------------
DEFAULT_EXTRA_CONCEPTS: List[str] = [
    # furniture / fixtures
    "chair", "table", "couch", "sofa", "bed", "crib", "high chair", "stool",
    "shelf",
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
    "person", "baby", "child",
    "face", "head",
    "hand", "foot", "leg", "arm",
    # furniture / storage
    "bookshelf", "dresser", "drawer", "cabinet", "closet",
    "wardrobe", "desk", "rug", "carpet", "curtain", "blinds", "highchair",
    # electronics
    "phone", "cell phone", "smartphone", "remote",
    "television", "tv",
    "laptop", "computer", "keyboard", "mouse", "tablet", "monitor", "camera",
    # bathroom
    "bathtub", "toilet", "mirror",
    "toothbrush", "toothpaste", "soap", "shampoo",
    # extra toys
    "stuffed animal", "teddy bear", "rattle", "blocks", "toy car",
]

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
# CLI
# -------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grounding DINO + SAM + CLIP on a single image frame, "
            "with separate passes for objects, textures, and backgrounds."
        )
    )

    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to a single SAYCam frame (jpg or png).",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save debug visualizations and masks.",
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
        default=None,
        help=(
            "If set, keep at most this many Grounding DINO detections "
            "with highest scores after thresholding (per pass)."
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

    # SAM
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        required=True,
        help="Path to SAM checkpoint (sam_vit_h_4b8939.pth etc.).",
    )
    parser.add_argument(
        "--sam-model-type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM backbone type.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help="Computation dtype for SAM (fp16 saves memory).",
    )
    parser.add_argument(
        "--sam-score-threshold",
        type=float,
        default=0.87,
        help="Minimum SAM mask score to accept a mask for a DINO box.",
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=350,
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

    # Concept options
    parser.add_argument(
        "--no-default-extra-concepts",
        action="store_true",
        help="If set, do not add DEFAULT_EXTRA_CONCEPTS as extra DINO/CLIP candidates.",
    )

    # Background filtering (only affects the object pass)
    parser.add_argument(
        "--disable-background-filter",
        action="store_true",
        help="If set, do not drop background-like labels in the object pass.",
    )

    # Whether to mine textures and backgrounds (flags)
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
        default="openai/clip-vit-large-patch14",
        help="Hugging Face repo id for CLIP model used for mask-level labeling.",
    )
    parser.add_argument(
        "--clip-sim-threshold",
        type=float,
        default=0.18,
        help=(
            "Minimum CLIP cosine similarity required to trust a label. "
            "Masks with lower similarity are kept but get clip_label=None."
        ),
    )
    parser.add_argument(
        "--clip-batch-size",
        type=int,
        default=16,
        help="Batch size for CLIP image feature computation.",
    )

    args = parser.parse_args()
    return args


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def build_grounding_dino(model_name: str, device: torch.device):
    print(f"Loading Grounding DINO model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, processor


def build_sam_predictor(args: argparse.Namespace, device: torch.device) -> SamPredictor:
    print(f"Loading SAM model: {args.sam_model_type} from {args.sam_checkpoint}")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    if args.precision == "fp16" and device.type == "cuda":
        sam.to(device=device, dtype=torch.float16)
    else:
        sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    return predictor


def build_clip(model_name: str, device: torch.device):
    print(f"Loading CLIP model: {model_name}")
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


def chunk_list(lst: List[str], chunk_size: int) -> List[List[str]]:
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]


# -------------------------------------------------------------------
# CLIP helpers
# -------------------------------------------------------------------
def precompute_clip_text_features(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
    class_names: List[str],
) -> Tuple[torch.Tensor, List[str]]:
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

    for start in range(0, len(crops_box), batch_size):
        batch_box = crops_box[start: start + batch_size]
        batch_masked = crops_masked[start: start + batch_size]

        with torch.no_grad():
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

    has_priority = priority_text_features is not None and priority_text_features.numel() > 0
    has_extras = extra_text_features is not None and extra_text_features.numel() > 0

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
# Visualizations
# -------------------------------------------------------------------
def vis_original(image: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.set_title("Original image")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def vis_boxes(
    image: np.ndarray,
    dets: List[Dict[str, Any]],
    out_path: Path,
    title: str,
    label_key: str = "label",
    score_key: str = "score",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    for det in dets:
        x0, y0, x1, y1 = det["box"]
        label = det.get(label_key, "")
        score = det.get(score_key, 0.0)
        w = x1 - x0
        h = y1 - y0
        rect = plt.Rectangle(
            (x0, y0),
            w,
            h,
            fill=False,
            edgecolor="r",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            x0,
            y0,
            f"{label} {score:.2f}",
            fontsize=8,
            color="yellow",
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def vis_masks(
    image: np.ndarray,
    dets: List[Dict[str, Any]],
    masks: List[np.ndarray],
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)

    for det, mask in zip(dets, masks):
        x0, y0, x1, y1 = det["box"]
        clip_label = det.get("clip_label") or det.get("label")
        clip_sim = det.get("clip_sim", 0.0)

        colored = np.zeros((*mask.shape, 4), dtype=float)
        colored[mask] = [1.0, 0.0, 0.0, 0.35]
        ax.imshow(colored)

        w = x1 - x0
        h = y1 - y0
        rect = plt.Rectangle(
            (x0, y0),
            w,
            h,
            fill=False,
            edgecolor="white",
            linewidth=1.0,
        )
        ax.add_patch(rect)
        ax.text(
            x0,
            y0,
            f"{clip_label} {clip_sim:.2f}",
            fontsize=8,
            color="white",
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )

    ax.set_title(title)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------
# One full pass: DINO -> SAM -> CLIP for one concept set
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
    out_dir: Path,
    filter_background_in_sam: bool,
    enforce_object_area_limit: bool,
    device: torch.device,
    box_scale: float,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    if not concepts:
        print(f"\n[{category_name}] No concepts; skipping this pass.")
        return [], []

    print(f"\n========== {category_name.upper()} PASS ==========")
    print(f"{category_name}: {len(concepts)} concepts")

    # ---------------------------
    # Step 1: Grounding DINO
    # ---------------------------
    concept_chunks = chunk_list(concepts, args.concepts_per_prompt)
    print(
        f"[{category_name}] Number of concept chunks: {len(concept_chunks)} "
        f"(chunk size = {args.concepts_per_prompt})"
    )

    all_raw_dets: List[Dict[str, Any]] = []

    for chunk_idx, chunk in enumerate(concept_chunks):
        text_prompt = " . ".join(chunk)
        print(
            f"[{category_name}] Chunk {chunk_idx + 1}/{len(concept_chunks)} "
            f"({len(chunk)} concepts, prompt length = {len(text_prompt)} chars)"
        )

        inputs = gdino_processor(
            images=image_pil,
            text=text_prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if chunk_idx == 0:
            print(
                f"[{category_name}] Grounding DINO input_ids shape:",
                inputs["input_ids"].shape,
            )

        with torch.no_grad():
            outputs = gdino_model(**inputs)

        target_sizes = torch.tensor([[H, W]], device=device)

        results_list = safe_post_process_grounded_od(
            gdino_processor,
            outputs=outputs,
            input_ids=inputs["input_ids"],
            target_sizes=target_sizes,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )

        results = results_list[0]

        if "text_labels" in results:
            raw_labels = list(results["text_labels"])
        else:
            raw_labels = list(results["labels"])

        gdino_boxes = results["boxes"].detach().cpu().numpy()
        gdino_scores = results["scores"].detach().cpu().numpy()

        keep_mask = gdino_scores >= args.box_threshold
        gdino_boxes = gdino_boxes[keep_mask]
        gdino_scores = gdino_scores[keep_mask]
        gdino_labels = [
            str(lab) for lab, k in zip(raw_labels, keep_mask.tolist()) if k
        ]

        print(
            f"[{category_name}]  DINO chunk produced {len(gdino_boxes)} "
            f"detections after box_threshold={args.box_threshold}."
        )

        for box, score, label in zip(gdino_boxes, gdino_scores, gdino_labels):
            all_raw_dets.append(
                {
                    "box": box.tolist(),
                    "score": float(score),
                    "label": label,
                }
            )

    print(
        f"[{category_name}] Total DINO detections after threshold: "
        f"{len(all_raw_dets)}"
    )

    if not all_raw_dets:
        print(f"[{category_name}] No detections from Grounding DINO.")
        return [], []

    all_raw_dets.sort(key=lambda d: d["score"], reverse=True)

    if args.max_detections is not None:
        all_raw_dets = all_raw_dets[: args.max_detections]
        print(
            f"[{category_name}] Keeping top {args.max_detections} detections "
            "by DINO score."
        )

    vis_boxes(
        image,
        all_raw_dets,
        out_dir / f"{category_name}_dino_boxes.png",
        title=f"{category_name.capitalize()} DINO detections (sorted by DINO score)",
        label_key="label",
        score_key="score",
    )

    bp(f"after_grounding_dino_{category_name}")

    # ---------------------------
    # Step 2: SAM refinement
    # ---------------------------
    print(f"[{category_name}] Running SAM predictor for each detection...")
    sam_predictor.set_image(image)

    sam_kept_dets: List[Dict[str, Any]] = []
    sam_kept_masks: List[np.ndarray] = []

    full_area = H * W

    for det in all_raw_dets:
        label = det["label"]
        score = det["score"]

        label_norm = label.lower().strip(" .")

        if filter_background_in_sam and label_norm in DEFAULT_BACKGROUND_CONCEPTS_LOWER:
            print(
                f"[{category_name}] Skipping background detection '{label}' "
                f"(score={score:.2f})."
            )
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
        mask = masks[best_idx]
        sam_score = float(sam_scores[best_idx])

        if sam_score < args.sam_score_threshold:
            print(
                f"[{category_name}] Skipping low SAM score mask for '{label}' "
                f"(sam_score={sam_score:.3f} < {args.sam_score_threshold})."
            )
            continue

        area = int(mask.sum())
        if area < args.min_mask_area:
            print(
                f"[{category_name}] Skipping small mask for '{label}' "
                f"(area={area} < {args.min_mask_area})."
            )
            continue

        area_frac = area / float(full_area)
        if enforce_object_area_limit and area_frac > args.max_mask_area_frac:
            print(
                f"[{category_name}] Skipping large surface-like mask for '{label}' "
                f"(area_frac={area_frac:.3f} > {args.max_mask_area_frac})."
            )
            continue

        if area_frac >= NEAR_FULL_MASK_AREA_FRAC:
            print(
                f"[{category_name}] Skipping near-whole-frame mask for '{label}' "
                f"(area_frac={area_frac:.3f})."
            )
            continue

        ys, xs = np.where(mask)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        bbox_w = x_max - x_min + 1
        bbox_h = y_max - y_min + 1
        bbox_area = bbox_w * bbox_h

        fill_frac = area / float(bbox_area)
        if fill_frac < args.min_mask_fill_frac:
            print(
                f"[{category_name}] Skipping low-fill mask for '{label}' "
                f"(fill_frac={fill_frac:.3f} < {args.min_mask_fill_frac})."
            )
            continue

        aspect = max(bbox_w, bbox_h) / max(1.0, float(min(bbox_w, bbox_h)))
        if aspect > args.max_mask_aspect_ratio:
            print(
                f"[{category_name}] Skipping elongated mask for '{label}' "
                f"(aspect={aspect:.2f} > {args.max_mask_aspect_ratio})."
            )
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

    print(
        f"[{category_name}] After SAM + geometric filtering, kept "
        f"{len(sam_kept_dets)} detections/masks (pre-CLIP)."
    )

    if not sam_kept_dets:
        print(f"[{category_name}] No detections survived SAM filtering.")
        return [], []

    # ---------------------------
    # Step 3: CLIP annotation
    # ---------------------------
    clip_results = clip_annotate_masks(
        clip_model=clip_model,
        clip_processor=clip_processor,
        device=device,
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

    print(
        f"[{category_name}] After CLIP annotation, kept {len(final_dets)} masks "
        "(all masks that passed SAM filters)."
    )

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

    vis_boxes(
        image,
        final_dets,
        out_dir / f"{category_name}_clip_boxes.png",
        title=f"{category_name.capitalize()} final detections (CLIP labels)",
        label_key="clip_label",
        score_key="clip_sim",
    )

    vis_masks(
        image,
        final_dets,
        final_masks,
        out_dir / f"{category_name}_sam_masks_clip.png",
        title=f"{category_name.capitalize()} SAM masks + CLIP labels",
    )

    return final_dets, final_masks


# -------------------------------------------------------------------
# Main debug path
# -------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    image_path = Path(args.image_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image = np.array(Image.open(image_path).convert("RGB"))
    H, W = image.shape[:2]
    print(f"Loaded image {image_path} with shape {H}x{W}")

    vis_original(image, out_dir / "step1_original.png")

    # ----------------------------------------------------------------
    # Build concept lists for each pass
    # ----------------------------------------------------------------
    object_concepts: List[str] = []
    seen_obj = set()

    def add_object(name: str) -> None:
        n = str(name).strip()
        ln = n.lower()
        if not n:
            return
        if ln in TEXTURE_CONCEPTS_LOWER:
            return
        if ln in DEFAULT_BACKGROUND_CONCEPTS_LOWER:
            return
        if ln in seen_obj:
            return
        seen_obj.add(ln)
        object_concepts.append(n)

    for c in PRIORITY_CLASSES:
        add_object(c)
    if not args.no_default_extra_concepts:
        for c in DEFAULT_EXTRA_CONCEPTS:
            add_object(c)

    print(f"Object concepts used for DINO: {len(object_concepts)}")

    texture_concepts = sorted(TEXTURE_CONCEPTS)
    print(f"Texture concepts used for DINO: {len(texture_concepts)}")

    background_concepts = sorted(BACKGROUND_ONLY_CONCEPTS)
    print(f"Background concepts used for DINO: {len(background_concepts)}")

    # ----------------------------------------------------------------
    # Build models
    # ----------------------------------------------------------------
    gdino_model, gdino_processor = build_grounding_dino(args.dino_model, device)
    sam_predictor = build_sam_predictor(args, device)
    clip_model, clip_processor = build_clip(args.clip_model, device)
    print("Models ready.")

    # ----------------------------------------------------------------
    # Precompute CLIP text features for each pass
    # ----------------------------------------------------------------
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
            device,
            obj_priority_concepts,
        )
    else:
        obj_priority_feats = torch.empty(0, 0, device=device)
        obj_priority_names = []

    if obj_extra_concepts:
        obj_extra_feats, obj_extra_names = precompute_clip_text_features(
            clip_model,
            clip_processor,
            device,
            obj_extra_concepts,
        )
    else:
        obj_extra_feats = torch.empty(0, 0, device=device)
        obj_extra_names = []

    if args.mine_textures and texture_concepts:
        tex_feats, tex_names = precompute_clip_text_features(
            clip_model,
            clip_processor,
            device,
            texture_concepts,
        )
    else:
        tex_feats = torch.empty(0, 0, device=device)
        tex_names = []

    if args.mine_backgrounds and background_concepts:
        bg_feats, bg_names = precompute_clip_text_features(
            clip_model,
            clip_processor,
            device,
            background_concepts,
        )
    else:
        bg_feats = torch.empty(0, 0, device=device)
        bg_names = []

    image_pil = Image.fromarray(image)

    # ----------------------------------------------------------------
    # Pass 1: objects
    # ----------------------------------------------------------------
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
        out_dir=out_dir,
        filter_background_in_sam=not args.disable_background_filter,
        enforce_object_area_limit=True,
        device=device,
        box_scale=1.2,
    )

    # ----------------------------------------------------------------
    # Pass 2: textures
    # ----------------------------------------------------------------
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
            extra_text_features=torch.empty(0, 0, device=device),
            extra_names=[],
            args=args,
            out_dir=out_dir,
            filter_background_in_sam=False,
            enforce_object_area_limit=False,
            device=device,
            box_scale=1.0,
        )

    # ----------------------------------------------------------------
    # Pass 3: backgrounds
    # ----------------------------------------------------------------
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
            extra_text_features=torch.empty(0, 0, device=device),
            extra_names=[],
            args=args,
            out_dir=out_dir,
            filter_background_in_sam=False,
            enforce_object_area_limit=False,
            device=device,
            box_scale=1.0,
        )

    # ----------------------------------------------------------------
    # Save masks separately for each category
    # ----------------------------------------------------------------
    def save_category_masks(
        prefix: str,
        dets: List[Dict[str, Any]],
        masks: List[np.ndarray],
    ) -> None:
        for idx, (det, mask) in enumerate(zip(dets, masks)):
            primary_label = det.get("clip_label") or det["label"]
            label_clean = str(primary_label).replace(" ", "_").replace(".", "")
            mask_path = out_dir / f"mask_{prefix}_{idx:03d}_{label_clean}.npz"
            np.savez_compressed(mask_path, mask=mask.astype(np.uint8))
            print(
                f"[save-{prefix}] Saved mask {idx} for CLIP='{primary_label}' "
                f"(area={det['mask_area']}, "
                f"DINO score={det['score']:.3f}, SAM score={det['sam_score']:.3f}, "
                f"CLIP sim={det['clip_sim']:.3f}, "
                f"is_priority={det['is_priority']}) "
                f"to {mask_path}"
            )

    save_category_masks("obj", obj_dets, obj_masks)
    if args.mine_textures:
        save_category_masks("tex", tex_dets, tex_masks)
    if args.mine_backgrounds:
        save_category_masks("bg", bg_dets, bg_masks)

    print(
        "\nSummary:"
        f"\n  objects     : {len(obj_dets)} masks"
        f"\n  textures    : {len(tex_dets) if args.mine_textures else 0} masks"
        f"\n  backgrounds : {len(bg_dets) if args.mine_backgrounds else 0} masks"
    )

    bp("final")


if __name__ == "__main__":
    main()
