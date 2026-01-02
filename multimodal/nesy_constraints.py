# multimodal/neurosym_constraints.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F

EPS = 1e-7

ALIAS = {
    # --- exact inflections/child-speak we want to collapse ---
    "feet": "foot", "toes": "foot", "toe": "foot",
    "hands": "hand", "fingers": "hand", "finger": "hand", "thumb": "hand",

    "kitty": "cat", "kitties": "cat", "kitten": "cat", "kittie": "cat", "kittys": "cat",

    # stairs
    "stair": "stairs", "steps": "stairs",

    # window/door plain plurals handled by fallback, but keep common variants
    "windows": "window", "doors": "door",

    # --- location/object variants that are strong signals ---
    "fridge": "kitchen", "microwave": "kitchen", "oven": "kitchen",
    "sink": "kitchen", "pantry": "kitchen", "dishwasher": "kitchen",

    # road-like surfaces → road
    "street": "road", "driveway": "road", "path": "road",
    "track": "road", "tracks": "road",

    # sand contexts
    "sandpit": "sand", "sandy": "sand", "beach": "sand",

    # ground synonyms
    "grass": "ground", "dirt": "ground", "cement": "ground",

    # room variants
    "bathroom": "room",

    # --- furniture/object near-synonyms ---
    "seat": "chair", "stool": "chair", "sofa": "chair", "couch": "chair", "bench": "chair",

    # horizontal surfaces -> table
    "desk": "table", "counter": "table",

    # paper-like items
    "papers": "paper", "page": "paper", "pages": "paper",
    "magazine": "paper", "magazines": "paper", "kleenex": "paper",

    # toys
    "teddy": "toy", "doll": "toy", "dolly": "toy", "puppet": "toy",
    "robot": "toy", "block": "toy", "blocks": "toy",

    # computer
    "computers": "computer", "monitor": "computer",

    # basket
    "baskets": "basket",
}

def _soft_or(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Probabilistic OR: 1 - Π (1 - x)."""
    x = torch.clamp(x, 0.0, 1.0)
    return 1.0 - torch.prod(1.0 - x + EPS, dim=dim)

def _canon_tokens(text_or_tokens):
    """
    Accepts a string or a (possibly nested) sequence; returns canonical tokens:
    lowercase words, plural stripping, ALIAS mapping.
    Safely handles list/tuple/set, torch.Tensor, and numpy arrays.
    """
    import re
    import numpy as np
    import torch

    def _yield_items(x):
        if isinstance(x, str):
            # split a sentence into word tokens
            for w in re.findall(r"[A-Za-z]+", x.lower()):
                yield w
        elif isinstance(x, (list, tuple, set)):
            for y in x:
                yield from _yield_items(y)
        elif torch.is_tensor(x):
            # id tensors -> just yield their str form; caller should pass words,
            # but we won't crash; numbers will be ignored below.
            for y in x.reshape(-1).tolist():
                yield str(y).lower()
        elif isinstance(x, np.ndarray):
            for y in x.reshape(-1).tolist():
                yield str(y).lower()
        else:
            yield str(x).lower()

    out = []
    for t in _yield_items(text_or_tokens):
        # skip pure numbers (these came from ID tensors)
        if t.isdigit():
            continue
        t0 = ALIAS.get(t, t)
        if t0 == t:  # fallback plural handling
            if len(t0) > 3 and t0.endswith("es"):
                t0 = t0[:-2]
            elif len(t0) > 2 and t0.endswith("s"):
                t0 = t0[:-1]
        out.append(t0)
    return out

def build_targets(raw_utterances, concept_list):
    """
    raw_utterances: list of strings OR list of token lists.
    concept_list:   your 22 concepts (lowercased).
    Returns a (B, C) float tensor mask where 1 indicates the utterance
    mentions that concept explicitly (after canonicalization).
    """
    import torch
    names = [c.lower() for c in concept_list]
    name_set = set(names)
    name2idx = {n: i for i, n in enumerate(names)}

    B, C = len(raw_utterances), len(names)
    mask = torch.zeros(B, C, dtype=torch.float32)
    for b, utt in enumerate(raw_utterances):
        toks = set(_canon_tokens(utt))
        # intersect with our concept names
        for t in toks:
            if t in name_set:
                mask[b, name2idx[t]] = 1.0
    return mask

@dataclass
class NSLossWeights:
    lambda_exist: float = 0.5
    lambda_hier: float = 0.1

def existential_soft_or_loss(
    logits: torch.Tensor,
    pos_mask: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Encourage that for any utterance that mentions one of our concepts,
    the OR over those concepts' probabilities is 1.
    If class_weights is provided (C,), the loss is reweighted per sample by
    the average weight of the mentioned classes.
    """
    if logits.dim() == 2:
        probs = torch.sigmoid(logits)              # (B, C)
        masked = probs * pos_mask
        or_over_c = _soft_or(masked, dim=1)        # (B)
    elif logits.dim() == 3:
        B, T, C = logits.shape
        probs = torch.sigmoid(logits)              # (B, T, C)
        masked = probs * pos_mask.unsqueeze(1)     # (B, T, C)
        or_over_c = _soft_or(_soft_or(masked, 2), 1)  # (B)
    else:
        raise ValueError("logits must be (B,C) or (B,T,C).")

    has_pos = pos_mask.sum(dim=1) > 0
    if not has_pos.any():
        return logits.new_tensor(0.0, requires_grad=True)

    # Per-sample existential loss
    losses = -torch.log(torch.clamp(or_over_c, EPS, 1 - EPS))  # (B)

    if class_weights is None:
        return losses[has_pos].mean()

    # Average the class weights over the mentioned concepts in each sample.
    cw = class_weights.to(logits.device)  # (C,)
    denom = pos_mask.sum(dim=1).clamp_min(1.0)     # (B)
    sample_w = (pos_mask * cw).sum(dim=1) / denom  # (B)
    sample_w = sample_w[has_pos]
    return (losses[has_pos] * sample_w).sum() / (sample_w.sum() + EPS)

def implication_hinge_loss(
    logits: torch.Tensor,
    edges: List[Tuple[int, int]],
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Soft A -> B per example: penalize p(A) > p(B) with ReLU(pA - pB).
    Optionally weighted per-edge with edge_weights (E,).
    """
    if not edges:
        return logits.new_tensor(0.0, requires_grad=True)
    if logits.dim() == 3:
        probs = torch.sigmoid(logits).amax(dim=1)  # (B,C)
    else:
        probs = torch.sigmoid(logits)              # (B,C)

    per_edge = []  # list of scalar losses per edge (averaged across batch)
    for i, (a, b) in enumerate(edges):
        penalty = F.relu(probs[:, a] - probs[:, b]).mean()  # scalar
        if edge_weights is None:
            per_edge.append(penalty)
        else:
            per_edge.append(penalty * edge_weights[i].to(probs.device))

    if edge_weights is None:
        return torch.stack(per_edge, dim=0).mean()

    # Normalize by total weight to keep the scale stable
    Z = edge_weights.sum().to(probs.device).clamp_min(EPS)
    return torch.stack(per_edge, dim=0).sum() / Z

def build_default_rules(concepts: List[str]) -> List[Tuple[int, int]]:
    idx = {n.lower(): i for i, n in enumerate(concepts)}
    core = [
        ("kitchen", "room"), ("floor",   "room"), ("door",  "room"),
        ("window",  "room"), ("crib",    "room"), ("computer","room"),
        ("sand", "ground"), ("road", "ground"),
        ("ball", "toy"), ("puzzle", "toy"),
    ]
    edges = []
    for a, b in core:
        if a in idx and b in idx:
            edges.append((idx[a], idx[b]))
    return edges

def build_edge_weights(
    class_weights: torch.Tensor,
    edges: List[Tuple[int, int]],
    scheme: str = "mean",
) -> torch.Tensor:
    """
    Produce a weight per edge from class_weights.
    scheme in {"a", "b", "mean", "geomean"}:
        - "a": weight = w[a]
        - "b": weight = w[b]
        - "mean": (w[a] + w[b]) / 2
        - "geomean": sqrt(w[a] * w[b])
    """
    cw = class_weights
    wa = torch.tensor([cw[a].item() for a, _ in edges], dtype=torch.float32)
    wb = torch.tensor([cw[b].item() for _, b in edges], dtype=torch.float32)
    if scheme == "a":
        ew = wa
    elif scheme == "b":
        ew = wb
    elif scheme == "geomean":
        ew = (wa * wb).sqrt()
    else:
        ew = 0.5 * (wa + wb)
    return ew