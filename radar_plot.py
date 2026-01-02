#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------- CSV loading helpers ----------------------

_CATEGORY_CANDIDATES = ["category", "class", "label", "name", "cat"]
_ACCURACY_CANDIDATES = ["accuracy", "acc", "score", "value", "percent"]

def _find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in cols:
            return cols[k]
    # fallback: if exactly one numeric column for accuracy, or non-numeric for category
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if candidates is _ACCURACY_CANDIDATES and len(num_cols) == 1:
        return num_cols[0]
    non_num = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if candidates is _CATEGORY_CANDIDATES and non_num:
        return non_num[0]
    raise ValueError(f"Could not infer column from candidates {candidates}. Columns: {list(df.columns)}")

def _normalize_accuracy_series(s: pd.Series) -> pd.Series:
    """
    Return accuracies in 0..100 range.
    If values look like 0..1, multiply by 100.
    """
    v = pd.to_numeric(s, errors="coerce")
    # Heuristic: if <=1.5 and >=0 (typical for fractions), convert to %
    if v.dropna().between(0, 1.5).all():
        return v * 100.0
    return v

def load_scores_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cat_col = _find_col(df, _CATEGORY_CANDIDATES)
    acc_col = _find_col(df, _ACCURACY_CANDIDATES)
    out = pd.DataFrame({
        "category": df[cat_col].astype(str),
        "accuracy": _normalize_accuracy_series(df[acc_col])
    })
    # drop rows where accuracy is NaN
    out = out.dropna(subset=["accuracy"]).copy()
    return out


# ---------------------- metric helpers ----------------------

def macro_avg(acc: pd.Series) -> float:
    values = acc.dropna().values
    return float(np.mean(values)) if len(values) else float("nan")

def micro_avg(correct: pd.Series, total: pd.Series) -> float:
    # If counts present, use them; otherwise returns NaN
    c = pd.to_numeric(correct, errors="coerce").fillna(0)
    t = pd.to_numeric(total,   errors="coerce").fillna(0)
    denom = t.sum()
    return float(c.sum() / denom * 100.0) if denom > 0 else float("nan")


# ---------------------- plotting ----------------------

def radar_plot(categories: List[str],
              baseline_vals: np.ndarray,
              exp_vals: np.ndarray,
              title: str,
              baseline_label: str,
              exp_label: str,
              out_base: Path) -> None:
    """
    Draws a radar plot (0..100%) comparing baseline vs experiment across categories.
    Saves <out_base>.png and <out_base>.pdf.
    """
    # Angles
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    # Values (close the loop)
    b = np.array(baseline_vals, dtype=float)
    e = np.array(exp_vals, dtype=float)
    b_closed = np.concatenate([b, b[:1]])
    e_closed = np.concatenate([e, e[:1]])

    # Figure
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Radial bounds and ticks (0..100)
    ax.set_ylim(0, 100)
    rticks = [0, 20, 40, 60, 80, 100]
    ax.set_yticks(rticks)
    ax.set_yticklabels([f"{t}%" for t in rticks])
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.grid(True, linestyle=":", alpha=0.6)

    # Category labels
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=10)

    # Plot series
    # (No explicit colors set—let Matplotlib choose defaults. Thicker line for experiment.)
    ax.plot(angles_closed, b_closed, linewidth=2, label=baseline_label)
    ax.fill(angles_closed, b_closed, alpha=0.10)
    ax.plot(angles_closed, e_closed, linewidth=3, label=exp_label)
    ax.fill(angles_closed, e_closed, alpha=0.10)

    # Averages for legend (computed outside; pass preformatted labels if desired)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))

    # Title & subtitle
    ax.set_title(title, va="bottom", fontsize=14, pad=20)

    # Save
    out_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_base.with_suffix(".png")
    pdf_path = out_base.with_suffix(".pdf")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    plt.close(fig)


# ---------------------- main ----------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compare baseline vs experiment CSVs and plot a radar chart."
    )
    ap.add_argument("--baseline_csv", required=True, type=Path, help="CSV with per-category baseline accuracies")
    ap.add_argument("--experiment_csv", required=True, type=Path, help="CSV with per-category experiment accuracies")
    ap.add_argument("--title", default="Per-Category Accuracy (Baseline vs Experiment)")
    ap.add_argument("--sort_by", choices=["alpha", "baseline", "experiment", "delta"], default="alpha",
                    help="Ordering of categories on the radar")
    ap.add_argument("--out", type=Path, default=Path("radar_comparison"),
                    help="Output path *without* extension (e.g., results/plots/radar)")
    args = ap.parse_args()

    # Load
    base_df = load_scores_csv(args.baseline_csv)
    exp_df  = load_scores_csv(args.experiment_csv)

    # Merge on category
    merged = pd.merge(base_df, exp_df, on="category", suffixes=("_baseline", "_experiment"), how="inner")
    if merged.empty:
        raise ValueError("No overlapping categories between the two CSVs.")

    # Optional: if counts exist, compute micro averages; else NaN
    # (If your CSVs include 'correct' and 'total', merge them above and compute real micro.)
    merged["delta"] = merged["accuracy_experiment"] - merged["accuracy_baseline"]

    # Sort as requested
    if args.sort_by == "alpha":
        merged = merged.sort_values("category")
    elif args.sort_by == "baseline":
        merged = merged.sort_values("accuracy_baseline", ascending=False)
    elif args.sort_by == "experiment":
        merged = merged.sort_values("accuracy_experiment", ascending=False)
    elif args.sort_by == "delta":
        merged = merged.sort_values("delta", ascending=False)

    # Averages
    macro_base = macro_avg(merged["accuracy_baseline"])
    macro_exp  = macro_avg(merged["accuracy_experiment"])

    # Create human-friendly labels with averages
    baseline_label  = f"Baseline (macro={macro_base:.1f}%)"
    experiment_label = f"Experiment (macro={macro_exp:.1f}%)"

    # Save merged table for reference
    out_base = args.out
    out_base.parent.mkdir(parents=True, exist_ok=True)
    merged_out = out_base.parent / (out_base.name + "_merged.csv")
    merged.to_csv(merged_out, index=False)
    print(f"Saved merged CSV: {merged_out}")

    # Radar plot
    categories = merged["category"].tolist()
    bvals = merged["accuracy_baseline"].to_numpy()
    evals = merged["accuracy_experiment"].to_numpy()

    radar_plot(
        categories=categories,
        baseline_vals=bvals,
        exp_vals=evals,
        title=args.title,
        baseline_label=baseline_label,
        exp_label=experiment_label,
        out_base=out_base
    )

if __name__ == "__main__":
    main()
