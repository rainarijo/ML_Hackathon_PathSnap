"""Train and benchmark 5 ML classifiers for PathSnap Part B.

Classifiers implemented:
1. Logistic Regression
2. K-Nearest Neighbors (KNN) - 200,000 sampled training rows (seed 42)
3. Random Forest
4. LightGBM
5. MLP Classifier (Neural Network)

Key features:
- GroupKFold (5 splits) grouped by trip_id (zero data leakage between trips)
- Filters out GPS outlier trips '00000033' and '00000054'
- Top-1 candidate ranking accuracy computed on candidate sets per GPS point
- Student-test evaluation
- Generates results/experiments.csv and 5 comparison plots in results/
"""
import os
import sys

# Silence warnings and configure cache paths before imports
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["MPLCONFIGDIR"] = "C:/Users/rajat/AppData/Local/Temp"

import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Add current directory to path to import evaluate_models
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
from evaluate_models import evaluate_predictions, format_metrics  # noqa: E402

FEATURE_COLS = [
    "dist",
    "t_frac",
    "arc_len",
    "speed",
    "dt_prev",
    "heading_diff",
    "heading_reliable",
    "n_cand",
    "dist_rank",
    "dist_minus_min",
    "dist_ratio",
    "same_as_prev_nearest",
    "connected_to_prev_nearest",
]

EXCLUDE_COLS = ["trip_id", "point_idx", "arc_id", "label"]


def load_dataset(
    train_path: str,
    test_path: str,
    drop_outlier_trips: bool = True,
) -> Tuple[Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame], Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]]:
    """Load train and student-test parquet files, applying the outlier filter."""
    logger.info("Loading training data from: %s", train_path)
    df_train = pd.read_parquet(train_path)

    if drop_outlier_trips:
        outlier_trips = ["00000033", "00000054"]
        n_before = len(df_train)
        df_train = df_train[~df_train["trip_id"].isin(outlier_trips)].reset_index(drop=True)
        n_after = len(df_train)
        logger.info(
            "Dropped outlier trips %s: %d -> %d rows (%d dropped)",
            outlier_trips,
            n_before,
            n_after,
            n_before - n_after,
        )

    logger.info("Loading student-test data from: %s", test_path)
    df_test = pd.read_parquet(test_path)

    # Extract metadata, features, targets, and groups
    meta_train = df_train[["trip_id", "point_idx", "arc_id", "label"]].copy()
    X_train = df_train[FEATURE_COLS].copy()
    y_train = df_train["label"].to_numpy(dtype=np.int64)
    groups_train = df_train["trip_id"].to_numpy()

    meta_test = df_test[["trip_id", "point_idx", "arc_id", "label"]].copy()
    X_test = df_test[FEATURE_COLS].copy()
    y_test = df_test["label"].to_numpy(dtype=np.int64)
    groups_test = df_test["trip_id"].to_numpy()

    logger.info(
        "Train features: %s | Target balance: pos=%.2f%% | Unique trips: %d",
        X_train.shape,
        100.0 * np.mean(y_train),
        len(np.unique(groups_train)),
    )
    logger.info(
        "Test features: %s | Target balance: pos=%.2f%% | Unique trips: %d",
        X_test.shape,
        100.0 * np.mean(y_test),
        len(np.unique(groups_test)),
    )

    return (X_train, y_train, groups_train, meta_train), (X_test, y_test, groups_test, meta_test)


def get_model(name: str):
    """Instantiate classifier pipeline with appropriate scaling and hyperparameters."""
    name = name.lower()
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        C=1.0,
                        class_weight="balanced",
                        random_state=42,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    elif name == "knn":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    KNeighborsClassifier(
                        n_neighbors=15,
                        weights="distance",
                        algorithm="brute",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    elif name == "random_forest":
        return RandomForestClassifier(
            n_estimators=60,
            max_depth=14,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    elif name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=7,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    elif name == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-4,
                        batch_size=1024,
                        learning_rate_init=1e-3,
                        max_iter=20,
                        early_stopping=True,
                        n_iter_no_change=3,
                        random_state=42,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


def train_and_eval_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    meta_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    meta_test: pd.DataFrame,
    n_splits: int = 5,
    knn_sample_size: int = 200000,
) -> Dict[str, Any]:
    """Train model using GroupKFold and evaluate on CV and Student-Test."""
    logger.info("=" * 60)
    logger.info("Training Model: %s", model_name.upper())
    logger.info("=" * 60)

    gkf = GroupKFold(n_splits=n_splits)
    oof_probas = np.zeros(len(X_train), dtype=np.float64)
    total_train_time = 0.0
    total_val_infer_time = 0.0

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train), 1):
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
        meta_va = meta_train.iloc[val_idx]

        # For KNN: subsample training rows to prevent computational bottleneck
        if model_name.lower() == "knn":
            if len(X_tr) > knn_sample_size:
                logger.info(
                    "Fold %d: Subsampling KNN training set from %d to %d rows (seed=42, stratified)",
                    fold,
                    len(X_tr),
                    knn_sample_size,
                )
                idx_sub, _ = train_test_split(
                    np.arange(len(X_tr)),
                    train_size=knn_sample_size,
                    stratify=y_tr,
                    random_state=42 + fold,
                )
                X_tr = X_tr.iloc[idx_sub]
                y_tr = y_tr[idx_sub]

        model = get_model(model_name)

        # Train fold
        t0 = time.time()
        model.fit(X_tr, y_tr)
        t_fit = time.time() - t0
        total_train_time += t_fit

        # Predict fold
        t0 = time.time()
        val_probs = model.predict_proba(X_va)[:, 1]
        t_infer = time.time() - t0
        total_val_infer_time += t_infer

        oof_probas[val_idx] = val_probs

        fold_eval = evaluate_predictions(meta_va, y_va, val_probs)
        fold_metrics.append(fold_eval)
        logger.info(
            "Fold %d/%d (fit: %.1fs, infer: %.1fs) -> %s",
            fold,
            n_splits,
            t_fit,
            t_infer,
            format_metrics(fold_eval),
        )

    # Aggregated CV evaluation
    cv_overall = evaluate_predictions(meta_train, y_train, oof_probas)
    logger.info("-" * 60)
    logger.info("OVERALL CV for %s: %s", model_name, format_metrics(cv_overall))
    logger.info("-" * 60)

    # Train on full training data and test on student-test set
    logger.info("Refitting %s on full training dataset for Student-Test evaluation...", model_name)
    full_model = get_model(model_name)
    X_full, y_full = X_train, y_train

    if model_name.lower() == "knn" and len(X_full) > knn_sample_size:
        idx_sub, _ = train_test_split(
            np.arange(len(X_full)),
            train_size=knn_sample_size,
            stratify=y_full,
            random_state=42,
        )
        X_full = X_full.iloc[idx_sub]
        y_full = y_full[idx_sub]

    t0 = time.time()
    full_model.fit(X_full, y_full)
    refit_time = time.time() - t0

    # Student-Test inference
    t0 = time.time()
    test_probs = full_model.predict_proba(X_test)[:, 1]
    test_infer_time = time.time() - t0

    test_eval = evaluate_predictions(meta_test, y_test, test_probs)
    logger.info("STUDENT-TEST for %s: %s", model_name, format_metrics(test_eval))
    logger.info("Student-Test inference time: %.2fs for %d rows", test_infer_time, len(X_test))

    result = {
        "model": model_name,
        # CV metrics
        "cv_top1_accuracy": cv_overall["top1_accuracy"],
        "cv_macro_f1": cv_overall["macro_f1"],
        "cv_f1": cv_overall["f1"],
        "cv_precision": cv_overall["precision"],
        "cv_recall": cv_overall["recall"],
        "cv_log_loss": cv_overall["log_loss"],
        "cv_row_accuracy": cv_overall["row_accuracy"],
        # Student Test metrics
        "test_top1_accuracy": test_eval["top1_accuracy"],
        "test_macro_f1": test_eval["macro_f1"],
        "test_f1": test_eval["f1"],
        "test_precision": test_eval["precision"],
        "test_recall": test_eval["recall"],
        "test_log_loss": test_eval["log_loss"],
        "test_row_accuracy": test_eval["row_accuracy"],
        # Timings
        "total_train_time_s": round(total_train_time + refit_time, 2),
        "test_inference_time_s": round(test_infer_time, 2),
        "inference_latency_ms_per_point": round(1000.0 * test_infer_time / len(X_test), 4),
    }
    return result


def plot_comparisons(df_results: pd.DataFrame, output_dir: Path):
    """Generate and save the 5 required comparison charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. CV Macro F1
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=df_results, x="model", y="cv_macro_f1", hue="model", legend=False, palette="viridis")
    plt.title("5-Fold Grouped Cross-Validation: Macro F1-Score", fontsize=13, weight="bold")
    plt.xlabel("Model", fontsize=11)
    plt.ylabel("CV Macro F1", fontsize=11)
    plt.ylim(0, 1.0)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2.0, p.get_height() + 0.02),
                    ha="center", fontsize=10, weight="bold")
    plt.tight_layout()
    cv_f1_path = output_dir / "cv_macro_f1.png"
    plt.savefig(cv_f1_path, dpi=300)
    plt.close()
    logger.info("Saved plot: %s", cv_f1_path)

    # 2. Student-test Top-1 accuracy
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=df_results, x="model", y="test_top1_accuracy", hue="model", legend=False, palette="magma")
    plt.title("Student-Test: Top-1 Candidate Ranking Accuracy", fontsize=13, weight="bold")
    plt.xlabel("Model", fontsize=11)
    plt.ylabel("Top-1 Accuracy", fontsize=11)
    plt.ylim(0, 1.0)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2.0, p.get_height() + 0.02),
                    ha="center", fontsize=10, weight="bold")
    plt.tight_layout()
    top1_path = output_dir / "student_test_top1_accuracy.png"
    plt.savefig(top1_path, dpi=300)
    plt.close()
    logger.info("Saved plot: %s", top1_path)

    # 3. Student-test F1
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=df_results, x="model", y="test_f1", hue="model", legend=False, palette="crest")
    plt.title("Student-Test: Positive Class (label=1) F1-Score", fontsize=13, weight="bold")
    plt.xlabel("Model", fontsize=11)
    plt.ylabel("Test F1 (Positive Class)", fontsize=11)
    plt.ylim(0, 1.0)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2.0, p.get_height() + 0.02),
                    ha="center", fontsize=10, weight="bold")
    plt.tight_layout()
    test_f1_path = output_dir / "student_test_f1.png"
    plt.savefig(test_f1_path, dpi=300)
    plt.close()
    logger.info("Saved plot: %s", test_f1_path)

    # 4. Log Loss
    plt.figure(figsize=(9, 5))
    df_loss = pd.melt(
        df_results,
        id_vars=["model"],
        value_vars=["cv_log_loss", "test_log_loss"],
        var_name="Dataset",
        value_name="Log Loss",
    )
    df_loss["Dataset"] = df_loss["Dataset"].map({"cv_log_loss": "Grouped CV", "test_log_loss": "Student-Test"})
    ax = sns.barplot(data=df_loss, x="model", y="Log Loss", hue="Dataset", palette="Blues_d")
    plt.title("Model Calibration: Log Loss (CV vs. Student-Test)", fontsize=13, weight="bold")
    plt.xlabel("Model", fontsize=11)
    plt.ylabel("Log Loss (Lower is Better)", fontsize=11)
    plt.tight_layout()
    loss_path = output_dir / "log_loss.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()
    logger.info("Saved plot: %s", loss_path)

    # 5. Training Time and Inference Time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=df_results, x="model", y="total_train_time_s", hue="model", legend=False, ax=ax1, palette="rocket")
    ax1.set_title("Total Training Time (seconds)", fontsize=12, weight="bold")
    ax1.set_ylabel("Seconds")
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.1f}s", (p.get_x() + p.get_width() / 2.0, p.get_height() + 1.0),
                     ha="center", fontsize=9)

    sns.barplot(data=df_results, x="model", y="inference_latency_ms_per_point", hue="model", legend=False, ax=ax2, palette="flare")
    ax2.set_title("Inference Latency per Candidate (ms)", fontsize=12, weight="bold")
    ax2.set_ylabel("Milliseconds / Row")
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.4f}ms", (p.get_x() + p.get_width() / 2.0, p.get_height() + 0.001),
                     ha="center", fontsize=9)
    plt.tight_layout()
    time_path = output_dir / "training_inference_time.png"
    plt.savefig(time_path, dpi=300)
    plt.close()
    logger.info("Saved plot: %s", time_path)


def main():
    parser = argparse.ArgumentParser(description="PathSnap Model Training & Benchmarking")
    parser.add_argument("--train-data", default="results/features/features_train.parquet")
    parser.add_argument("--test-data", default="results/features/features_studenttest.parquet")
    parser.add_argument("--out-csv", default="results/experiments.csv")
    parser.add_argument("--plots-dir", default="results")
    parser.add_argument("--models", nargs="+", default=["logistic_regression", "knn", "random_forest", "lightgbm", "mlp"])
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--knn-sample", type=int, default=200000)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / args.train_data
    test_path = project_root / args.test_data
    out_csv = project_root / args.out_csv
    plots_dir = project_root / args.plots_dir

    train_pack, test_pack = load_dataset(str(train_path), str(test_path), drop_outlier_trips=True)
    X_train, y_train, groups_train, meta_train = train_pack
    X_test, y_test, groups_test, meta_test = test_pack

    results = []
    # If experiments.csv already exists, load existing rows if not re-running
    if out_csv.exists():
        try:
            df_existing = pd.read_csv(out_csv)
            existing_dict = {row["model"]: row.to_dict() for _, row in df_existing.iterrows()}
        except Exception:
            existing_dict = {}
    else:
        existing_dict = {}

    for model_name in args.models:
        res = train_and_eval_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            meta_train=meta_train,
            X_test=X_test,
            y_test=y_test,
            meta_test=meta_test,
            n_splits=args.splits,
            knn_sample_size=args.knn_sample,
        )
        existing_dict[model_name] = res

    # Combine all results
    df_res = pd.DataFrame(list(existing_dict.values()))
    # Sort in standard order
    order = ["logistic_regression", "knn", "random_forest", "lightgbm", "mlp"]
    df_res["order"] = df_res["model"].map(lambda m: order.index(m) if m in order else 99)
    df_res = df_res.sort_values("order").drop(columns=["order"]).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_csv, index=False)
    logger.info("Saved benchmark results to: %s", out_csv)

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    cols_to_print = ["model", "cv_top1_accuracy", "cv_macro_f1", "test_top1_accuracy", "test_macro_f1", "cv_log_loss", "total_train_time_s", "test_inference_time_s"]
    print(df_res[cols_to_print].to_string(index=False))
    print("=" * 80 + "\n")

    plot_comparisons(df_res, plots_dir)
    logger.info("All comparison plots generated successfully in: %s", plots_dir)


if __name__ == "__main__":
    main()
