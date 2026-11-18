"""Model-comparison charts from results/experiments.csv.

Usage:
    python viz/plot_model_comparison.py --csv results/experiments.csv --out results/plots
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def grouped(df, cols, labels, title, ylabel, fname, out, ylim=None):
    x = np.arange(len(df))
    w = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (c, lab) in enumerate(zip(cols, labels)):
        bars = ax.bar(x + (i - (len(cols) - 1) / 2) * w, df[c], w, label=lab)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, df["model"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out) / fname, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.csv)

    grouped(df, ["test_top1_accuracy", "test_macro_f1"], ["Top-1 accuracy", "Macro F1"],
            "Student-test: Top-1 accuracy and macro F1", "score", "st_top1_macro_f1.png", a.out, (0.8, 0.95))
    grouped(df, ["cv_top1_accuracy", "test_top1_accuracy"], ["CV (train, GroupKFold)", "Student-test"],
            "Top-1 accuracy: cross-validation vs student-test", "Top-1 accuracy",
            "top1_cv_vs_test.png", a.out, (0.84, 0.90))
    print("saved charts to", a.out)


if __name__ == "__main__":
    main()
