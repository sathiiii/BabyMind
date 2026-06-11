#!/usr/bin/env python3
# tools/concept_sparsity_viz.py
# -*- coding: utf-8 -*-

"""
Concept sparsity + co-occurrence visualization for SAYCam utterances (CVCL-style pipeline).

Exports publication-friendly figures (no titles):
  (1) Per-concept mention counts (sorted), with log-scale on (count+1)
  (2) Lorenz curve + equality line, with Gini + coverage stats annotated
  (3) Utterance-level sparsity: histogram of #concepts mentioned per utterance
  (4) Co-occurrence heatmap (log-counts / Jaccard / PMI)
  (5) UpSet plot of intersection frequencies (top subsets, aesthetically spaced)

Also exports:
  - CSV of per-class counts/frequencies
  - JSON with summary stats
  - CSV of full co-occurrence matrix + top co-occurring pairs
  - CSV of intersection counts used for UpSet

All args after `--` are forwarded to train._setup_parser, like training.
"""

import argparse
import collections
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Headless-safe backend (cluster friendly)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- repo imports (same pipeline as training) ---
from multimodal.multimodal_saycam_data_module import MultiModalSAYCamDataModule
from multimodal.nesy_constraints import build_targets  # canonicalization + targets
from train import _setup_parser as build_repo_parser


DEFAULT_22 = [
    "ball", "basket", "car", "cat", "chair", "computer", "crib", "door", "floor", "foot",
    "ground", "hand", "kitchen", "paper", "puzzle", "road", "room", "sand", "stairs",
    "table", "toy", "window",
]


# ----------------------------
# io helpers
# ----------------------------

def _with_split(outdir: Path, split: str, leaf: str) -> Path:
    return outdir / f"{split}_{leaf}"


def save_both(fig: plt.Figure, out_pdf: Path, out_png: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    # A bit more padding to feel less cramped in papers
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


# ----------------------------
# shared plotting style
# ----------------------------

def set_pub_style() -> None:
    """Consistent, compact paper style with a touch more whitespace."""
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.2,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def soften_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(axis="both", which="both", direction="out")


# ----------------------------
# decoding / utterance gather
# ----------------------------

def build_id_maps_and_ignores(dm):
    vocab = dm.read_vocab()
    if isinstance(vocab, dict):
        id2tok = {int(i): tok for tok, i in vocab.items()}
    elif hasattr(vocab, "itos"):
        itos = list(vocab.itos)
        id2tok = {i: tok for i, tok in enumerate(itos)}
    else:
        raise ValueError("Unsupported vocab type returned by DataModule.read_vocab()")

    ignore_tokens = {"<pad>", "<unk>", "<sos>", "<eos>", ".", ",", "?", "!", "...", "..", "...."}
    ignore_ids = set()
    if isinstance(vocab, dict):
        for t in ignore_tokens:
            if t in vocab:
                ignore_ids.add(int(vocab[t]))
    return vocab, id2tok, ignore_ids


def token_to_words(tok: str) -> List[str]:
    tok = tok.replace("▁", " ")
    tok = tok.replace("Ġ", " ")
    tok = tok.replace("##", "")
    return [w for w in re.split(r"[^a-zA-Z]+", tok.lower()) if w]


def decode_any(obj, id2tok, ignore_ids):
    import torch

    def _flatten(x):
        if torch.is_tensor(x):
            for v in x.reshape(-1).tolist():
                yield v
        elif isinstance(x, (list, tuple)):
            for y in x:
                yield from _flatten(y)
        else:
            yield x

    toks: List[str] = []
    for item in _flatten(obj):
        if isinstance(item, int):
            if item in ignore_ids:
                continue
            tok = id2tok.get(int(item), "<unk>")
            toks.extend(token_to_words(tok))
        elif isinstance(item, str):
            toks.extend(token_to_words(item))
        else:
            toks.extend(token_to_words(str(item)))
    return toks


def extract_utterances_from_batch(batch, id2tok, ignore_ids):
    """Return List[List[str]] of length B for this batch."""
    import torch

    B = None
    if isinstance(batch, (list, tuple)) and len(batch) > 0 and torch.is_tensor(batch[0]) and batch[0].ndim >= 1:
        B = int(batch[0].shape[0])

    # Prefer explicit raw utterances if present (list/tuple len B)
    if isinstance(batch, (list, tuple)):
        for item in batch:
            if isinstance(item, (list, tuple)) and B is not None and len(item) == B:
                return [decode_any(u, id2tok, ignore_ids) for u in item]

    # Next: y token IDs tensor (B, L), optional lengths at batch[2]
    y = None
    y_len = None
    if isinstance(batch, (list, tuple)) and len(batch) >= 2 and hasattr(batch[1], "ndim") and batch[1].ndim == 2:
        y = batch[1]
    if y is not None and isinstance(batch, (list, tuple)) and len(batch) >= 3 and hasattr(batch[2], "ndim") and batch[2].ndim == 1:
        if int(batch[2].shape[0]) == int(y.shape[0]):
            y_len = batch[2]

    if y is not None:
        out = []
        for i in range(int(y.shape[0])):
            L = int(y_len[i].item()) if y_len is not None else None
            row = y[i][:L] if L is not None else y[i]
            out.append(decode_any(row, id2tok, ignore_ids))
        return out

    # Fallback
    if isinstance(batch, (list, tuple)) and len(batch) > 0:
        return [decode_any(batch[0], id2tok, ignore_ids)]
    return []


def gather_raw_utterances(loader, id2tok, ignore_ids, max_utts: int = 0) -> List[List[str]]:
    raws: List[List[str]] = []
    for batch in loader:
        raws.extend(extract_utterances_from_batch(batch, id2tok, ignore_ids))
        if max_utts and len(raws) >= max_utts:
            raws = raws[:max_utts]
            break
    return raws


# ----------------------------
# stats
# ----------------------------

def gini_from_counts(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    g = (n + 1 - 2.0 * np.sum(cumx) / cumx[-1]) / n
    return float(g)


def lorenz_curve(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(counts, dtype=np.float64)
    x = np.sort(x)
    n = x.size
    if n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    total = x.sum()
    pop = np.linspace(0.0, 1.0, n + 1)
    if total <= 0:
        return pop, np.zeros_like(pop)
    cum = np.concatenate([[0.0], np.cumsum(x) / total])
    return pop, cum


def compute_sparsity_stats(raws: List[List[str]], concepts: List[str]) -> Dict[str, Any]:
    names = [c.lower() for c in concepts]
    mask_t = build_targets(raws, names)  # (B,C), torch tensor

    B, C = int(mask_t.shape[0]), int(mask_t.shape[1])
    mask = mask_t.cpu().numpy().astype(np.uint8)

    mentions_per_utt = mask.sum(axis=1).astype(int)
    per_class_counts = mask.sum(axis=0).astype(int)

    coverage = float((mentions_per_utt > 0).mean()) if B else 0.0
    hist = collections.Counter(mentions_per_utt.tolist())
    size_hist = {int(k): int(v) for k, v in sorted(hist.items(), key=lambda kv: int(kv[0]))}

    gini = gini_from_counts(per_class_counts)
    pop, share = lorenz_curve(per_class_counts)

    total_mentions = float(per_class_counts.sum())
    if total_mentions > 0:
        p = per_class_counts / total_mentions
        eff = float(1.0 / np.sum(p ** 2))
    else:
        eff = 0.0

    return dict(
        B=B,
        C=C,
        concepts=names,
        mask=mask,  # (B,C) uint8
        mentions_per_utt=mentions_per_utt,
        per_class_counts=per_class_counts,
        per_class_freq=per_class_counts / max(1, B),
        coverage=coverage,
        size_hist=size_hist,
        gini=gini,
        lorenz_pop=pop,
        lorenz_share=share,
        total_mentions=total_mentions,
        effective_num_concepts=eff,
    )


# ----------------------------
# co-occurrence helpers
# ----------------------------

def cooc_counts_from_mask(mask: np.ndarray) -> np.ndarray:
    """Return CxC co-occurrence counts: number of utterances where i and j co-occur."""
    m = mask.astype(np.int32)  # (B,C)
    return (m.T @ m).astype(np.int64)


def cooc_jaccard(cooc: np.ndarray, marg: np.ndarray) -> np.ndarray:
    """Jaccard(i,j) = cooc / (marg_i + marg_j - cooc)."""
    marg = marg.astype(np.float64)
    denom = (marg[:, None] + marg[None, :] - cooc.astype(np.float64))
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(cooc, np.maximum(denom, 1.0), out=out, where=(denom > 0))
    return out


def cooc_pmi(cooc: np.ndarray, marg: np.ndarray, B: int, eps: float = 1e-12) -> np.ndarray:
    """PMI(i,j) = log( p(i,j) / (p(i)p(j)) )."""
    if B <= 0:
        return np.zeros_like(cooc, dtype=np.float64)
    p_ij = np.maximum(cooc.astype(np.float64) / float(B), eps)
    p_i = np.maximum(marg.astype(np.float64) / float(B), eps)
    denom = np.maximum(p_i[:, None] * p_i[None, :], eps)
    return np.log(p_ij / denom)


def intersection_counts_series(
    mask: np.ndarray,
    concept_names: List[str],
    drop_empty: bool = True,
) -> pd.Series:
    """
    Compute counts per intersection pattern (UpSet input) efficiently.

    Returns a Series indexed by a MultiIndex of booleans (one level per concept),
    with integer counts.
    """
    B, C = mask.shape
    if B == 0 or C == 0:
        mi = pd.MultiIndex.from_arrays([[] for _ in range(C)], names=concept_names)
        return pd.Series([], index=mi, dtype=np.int64)

    packed = np.packbits(mask.astype(np.uint8), axis=1, bitorder="little")
    uniq, counts = np.unique(packed, axis=0, return_counts=True)

    bits = np.unpackbits(uniq, axis=1, count=C, bitorder="little").astype(bool)  # (U,C)
    mi = pd.MultiIndex.from_arrays([bits[:, i] for i in range(C)], names=concept_names)
    s = pd.Series(counts.astype(np.int64), index=mi).sort_values(ascending=False)

    if drop_empty:
        # empty set = all False across C levels
        if len(s) > 0:
            empty_key = tuple([False] * C)
            if empty_key in s.index:
                s = s.drop(index=empty_key)
    return s


# ----------------------------
# plotting
# ----------------------------

def plot_per_concept_counts(stats: Dict[str, Any], out_pdf: Path, out_png: Path) -> None:
    names = stats["concepts"]
    counts = stats["per_class_counts"].astype(int)

    order = np.argsort(-counts)
    names_s = [names[i] for i in order]
    counts_s = counts[order]

    fig, ax = plt.subplots(figsize=(6.6, 2.8), constrained_layout=True)
    idx = np.arange(len(names_s))
    ax.bar(idx, counts_s + 1)
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels(names_s, rotation=45, ha="right")
    ax.set_ylabel("Mentions (log count+1)")
    ax.margins(x=0.01)
    soften_axes(ax)
    save_both(fig, out_pdf, out_png)


def plot_lorenz_gini(stats: Dict[str, Any], out_pdf: Path, out_png: Path) -> None:
    pop = stats["lorenz_pop"]
    share = stats["lorenz_share"]

    fig, ax = plt.subplots(figsize=(3.9, 2.6), constrained_layout=True)
    ax.plot(pop, share, marker="o", markersize=2.6, label="Lorenz")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="Equality")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Fraction of concepts")
    ax.set_ylabel("Fraction of mentions")
    ax.legend(frameon=False, loc="lower right", handlelength=1.4)

    ax.text(
        0.05, 0.95,
        f"Gini = {stats['gini']:.3f}\n"
        f"Effective # = {stats['effective_num_concepts']:.2f}/{stats['C']}\n"
        f"Coverage = {stats['coverage']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
    )
    soften_axes(ax)
    save_both(fig, out_pdf, out_png)


def plot_utterance_setsize_hist(stats: Dict[str, Any], out_pdf: Path, out_png: Path) -> None:
    size_hist = stats["size_hist"]
    xs = np.array(sorted(size_hist.keys()), dtype=int)
    ys = np.array([size_hist[k] for k in xs], dtype=int)

    fig, ax = plt.subplots(figsize=(4.4, 2.6), constrained_layout=True)
    ax.bar(xs.astype(str), ys)
    ax.set_xlabel("# concepts per utterance")
    ax.set_ylabel("Utterances")
    ax.margins(x=0.01)
    soften_axes(ax)
    save_both(fig, out_pdf, out_png)


def plot_cooccurrence_heatmap(
    concept_names: List[str],
    cooc: np.ndarray,
    metric: str,
    B: int,
    out_pdf: Path,
    out_png: Path,
    clip_neg_pmi: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - df_matrix: full CxC matrix in the chosen metric space (float)
      - df_top_pairs: top co-occurring pairs (i<j) by the chosen metric (or by count if metric != count)
    """
    C = len(concept_names)
    marg = np.diag(cooc).astype(np.int64)

    if metric == "count":
        mat = np.log1p(cooc.astype(np.float64))
        mat_label = "log(1 + co-occurrence count)"
        score_for_ranking = cooc.astype(np.float64)
    elif metric == "jaccard":
        mat = cooc_jaccard(cooc, marg=marg)
        mat_label = "Jaccard similarity"
        score_for_ranking = mat
    elif metric == "pmi":
        mat = cooc_pmi(cooc, marg=marg, B=B)
        if clip_neg_pmi:
            mat = np.maximum(mat, 0.0)
            mat_label = "PMI (clipped at 0)"
        else:
            mat_label = "PMI"
        score_for_ranking = mat
    else:
        raise ValueError(f"Unknown --cooc-metric {metric}")

    df_matrix = pd.DataFrame(mat, index=concept_names, columns=concept_names)

    pairs = []
    for i in range(C):
        for j in range(i + 1, C):
            pairs.append({
                "concept_i": concept_names[i],
                "concept_j": concept_names[j],
                "cooc_count": int(cooc[i, j]),
                "score": float(score_for_ranking[i, j]),
            })
    df_pairs = pd.DataFrame(pairs).sort_values("score", ascending=False)

    fig, ax = plt.subplots(figsize=(6.3, 5.8), constrained_layout=True)
    im = ax.imshow(mat, aspect="equal")

    ax.set_xticks(np.arange(C))
    ax.set_yticks(np.arange(C))
    ax.set_xticklabels(concept_names, rotation=45, ha="right")
    ax.set_yticklabels(concept_names)

    ax.set_xlabel("Concept")
    ax.set_ylabel("Concept")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(mat_label, rotation=90, va="bottom")

    soften_axes(ax)
    save_both(fig, out_pdf, out_png)
    return df_matrix, df_pairs


def plot_upset(
    subset_counts: pd.Series,
    out_pdf: Path,
    out_png: Path,
    *,
    max_subsets: int,
    min_count: int,
    with_lines: bool,
) -> None:
    """
    A cleaner-looking UpSet plot.

    subset_counts:
      Series indexed by boolean MultiIndex levels (concept names), values are counts.
    """
    try:
        from upsetplot import UpSet
    except Exception:
        print("[WARN] upsetplot not installed. Skipping UpSet plot. Try: pip install upsetplot")
        return

    s = subset_counts.copy()
    if min_count > 1:
        s = s[s >= int(min_count)]
    s = s.sort_values(ascending=False)
    if max_subsets > 0 and len(s) > max_subsets:
        s = s.iloc[:max_subsets]

    # More whitespace: larger figure + constrained layout
    fig = plt.figure(figsize=(8.4, 5.2), constrained_layout=True)

    upset = UpSet(
        s,
        sort_by="cardinality",
        sort_categories_by="cardinality",
        show_counts="%d",
        show_percentages=False,
        facecolor="black",
        other_dots_color=0.22,   # lighter inactive dots
        shading_color=0.06,      # very subtle row shading
        with_lines=bool(with_lines),
        element_size=16,         # dot size in pt, helps readability
        intersection_plot_elements=10,
        totals_plot_elements=3,
    )
    axes = upset.plot(fig=fig)

    # Gentle cleanup on axes
    for ax in axes.values():
        if hasattr(ax, "spines"):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="both", direction="out")

    save_both(fig, out_pdf, out_png)


# ----------------------------
# main
# ----------------------------

def main():
    p = argparse.ArgumentParser("Concept sparsity + co-occurrence visualization (SAYCam, CVCL-style)")
    p.add_argument("--concepts", type=str, default=None)
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--outdir", type=str, default="concept_sparsity_out")
    p.add_argument("--debug_show_tokens", type=int, default=0)
    p.add_argument("--max_utterances", type=int, default=0, help="0 = no cap. Useful for quick iteration.")

    # Co-occurrence heatmap controls
    p.add_argument("--cooc-metric", type=str, default="count", choices=["count", "jaccard", "pmi"])
    p.add_argument("--no-clip-neg-pmi", action="store_true", help="If set, allow negative PMI values in heatmap.")

    # UpSet controls
    p.add_argument("--upset-max-subsets", type=int, default=28, help="Show at most this many intersections.")
    p.add_argument("--upset-min-count", type=int, default=6, help="Only show intersections with at least this many utterances.")
    p.add_argument("--upset-drop-empty", action="store_true", help="Drop empty-set (no concept) intersection.")
    p.add_argument("--upset-no-lines", action="store_true", help="Disable connector lines between dots.")

    known, unknown = p.parse_known_args()
    args = known

    repo_parser = build_repo_parser()
    data_args = repo_parser.parse_args(unknown if unknown is not None else [])
    data_args.dataset = "saycam"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.concepts:
        with open(args.concepts, "r") as f:
            concepts = json.load(f)
        if not isinstance(concepts, list):
            raise ValueError("--concepts must be a JSON list of strings")
    else:
        concepts = DEFAULT_22

    dm = MultiModalSAYCamDataModule(data_args)
    dm.prepare_data()
    dm.setup()

    if args.split == "train":
        loader = dm.train_dataloader()
    elif args.split == "val":
        loader = dm.val_dataloader()[0]
    else:
        loader = dm.test_dataloader()[0]

    _, id2tok, ignore_ids = build_id_maps_and_ignores(dm)
    raws = gather_raw_utterances(
        loader,
        id2tok=id2tok,
        ignore_ids=ignore_ids,
        max_utts=int(args.max_utterances) if args.max_utterances else 0,
    )

    if args.debug_show_tokens > 0:
        for i in range(min(args.debug_show_tokens, len(raws))):
            print(f"[DEBUG] utt #{i}:", raws[i][:50])

    stats = compute_sparsity_stats(raws, concepts)

    # Save per-class CSV
    per_class_csv = _with_split(outdir, args.split, "per_class_counts.csv")
    pd.DataFrame({
        "concept": stats["concepts"],
        "count": stats["per_class_counts"],
        "freq": stats["per_class_freq"],
    }).sort_values("count", ascending=False).to_csv(per_class_csv, index=False)

    # Save summary JSON
    summary = {
        "split": args.split,
        "utterances_analyzed": int(stats["B"]),
        "num_concepts": int(stats["C"]),
        "coverage_ge1_mention": float(stats["coverage"]),
        "total_mentions": float(stats["total_mentions"]),
        "gini_over_concepts": float(stats["gini"]),
        "effective_num_concepts": float(stats["effective_num_concepts"]),
        "size_hist": stats["size_hist"],
    }
    summary_path = _with_split(outdir, args.split, "sparsity_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    set_pub_style()

    # Existing figures
    plot_per_concept_counts(
        stats,
        _with_split(outdir, args.split, "per_concept_counts.pdf"),
        _with_split(outdir, args.split, "per_concept_counts.png"),
    )
    plot_lorenz_gini(
        stats,
        _with_split(outdir, args.split, "lorenz_gini.pdf"),
        _with_split(outdir, args.split, "lorenz_gini.png"),
    )
    plot_utterance_setsize_hist(
        stats,
        _with_split(outdir, args.split, "utterance_setsize_hist.pdf"),
        _with_split(outdir, args.split, "utterance_setsize_hist.png"),
    )

    # Co-occurrence
    mask = stats["mask"]
    cooc = cooc_counts_from_mask(mask)
    df_cooc_count = pd.DataFrame(cooc, index=stats["concepts"], columns=stats["concepts"])
    cooc_csv = _with_split(outdir, args.split, "cooccurrence_counts_matrix.csv")
    df_cooc_count.to_csv(cooc_csv)

    df_metric, df_pairs = plot_cooccurrence_heatmap(
        concept_names=stats["concepts"],
        cooc=cooc,
        metric=str(args.cooc_metric),
        B=int(stats["B"]),
        out_pdf=_with_split(outdir, args.split, f"cooccurrence_heatmap_{args.cooc_metric}.pdf"),
        out_png=_with_split(outdir, args.split, f"cooccurrence_heatmap_{args.cooc_metric}.png"),
        clip_neg_pmi=(not bool(args.no_clip_neg_pmi)),
    )

    top_pairs_csv = _with_split(outdir, args.split, f"cooccurrence_top_pairs_{args.cooc_metric}.csv")
    df_pairs.head(200).to_csv(top_pairs_csv, index=False)

    metric_matrix_csv = _with_split(outdir, args.split, f"cooccurrence_matrix_{args.cooc_metric}.csv")
    df_metric.to_csv(metric_matrix_csv)

    # UpSet (intersection co-occurrence)
    subset_counts = intersection_counts_series(
        mask=mask,
        concept_names=stats["concepts"],
        drop_empty=bool(args.upset_drop_empty),
    )

    upset_counts_csv = _with_split(outdir, args.split, "upset_intersection_counts.csv")
    subset_counts.to_csv(upset_counts_csv, header=["count"])

    plot_upset(
        subset_counts=subset_counts,
        out_pdf=_with_split(outdir, args.split, "upset.pdf"),
        out_png=_with_split(outdir, args.split, "upset.png"),
        max_subsets=int(args.upset_max_subsets),
        min_count=int(args.upset_min_count),
        with_lines=(not bool(args.upset_no_lines)),
    )

    print(f"Utterances analyzed: {stats['B']}")
    print(f"Coverage (≥1 mention): {stats['coverage']:.3f}")
    print(f"Gini (concept imbalance): {stats['gini']:.3f}")
    print(f"Effective #concepts: {stats['effective_num_concepts']:.2f}/{stats['C']}")
    print(f"Wrote: {outdir}")


if __name__ == "__main__":
    main()
