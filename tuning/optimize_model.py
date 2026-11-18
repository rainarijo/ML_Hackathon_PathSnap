"""Bayesian hyperparameter optimization using Optuna for PathSnap Part B.

Optimizes the best classifier (LightGBM) using GroupKFold cross-validation
grouped strictly by trip_id to prevent data leakage.

Key Outputs:
- tuning/optuna_trials.csv
- tuning/best_parameters.json
- tuning/optuna_study.pkl
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

# Fix loky/joblib cpu detection warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, os.cpu_count() or 4))
# Suppress Optuna verbose logging by default
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Add models directory to path to import evaluate_models
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "models"))
from evaluate_models import evaluate_predictions  # noqa: E402
from train_models import load_dataset, FEATURE_COLS  # noqa: E402


def create_objective(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    meta_train: pd.DataFrame,
    n_splits: int = 5,
):
    """Factory creating an Optuna objective function targeting Grouped CV Macro F1."""
    gkf = GroupKFold(n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        # Define hyperparameter search space
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "num_leaves": trial.suggest_int("num_leaves", 20, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        oof_probas = np.zeros(len(X_train), dtype=np.float64)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
            X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_tr, y_tr)
            val_probs = clf.predict_proba(X_va)[:, 1]
            oof_probas[val_idx] = val_probs

        metrics = evaluate_predictions(meta_train, y_train, oof_probas)
        score = metrics["macro_f1"]

        # Track additional auxiliary metrics as user attributes
        trial.set_user_attr("top1_accuracy", metrics["top1_accuracy"])
        trial.set_user_attr("positive_f1", metrics["f1"])
        trial.set_user_attr("log_loss", metrics["log_loss"])

        logger.info(
            "Trial %d finished: Macro F1 = %.4f | Top-1 Acc = %.4f | Log Loss = %.4f",
            trial.number,
            score,
            metrics["top1_accuracy"],
            metrics["log_loss"],
        )
        return score

    return objective


def run_tuning(
    n_trials: int = 50,
    n_splits: int = 5,
    train_path: str = "results/features/features_train.parquet",
    output_dir: str = "tuning",
):
    """Run Optuna optimization study and persist all artifacts."""
    out_dir = ROOT_DIR / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset with trips 33 & 54 filtered
    train_file = ROOT_DIR / train_path
    logger.info("Loading train dataset from: %s", train_file)
    df_train = pd.read_parquet(train_file)
    df_train = df_train[~df_train["trip_id"].isin(["00000033", "00000054"])].reset_index(drop=True)

    meta_train = df_train[["trip_id", "point_idx", "arc_id", "label"]].copy()
    X_train = df_train[FEATURE_COLS].copy()
    y_train = df_train["label"].to_numpy(dtype=np.int64)
    groups_train = df_train["trip_id"].to_numpy()

    logger.info("Starting Bayesian Optimization (%d trials, %d-fold GroupKFold)", n_trials, n_splits)
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="pathsnap_lgbm_study")

    objective_func = create_objective(X_train, y_train, groups_train, meta_train, n_splits=n_splits)

    # Record trial results dynamically
    def save_callback(st, tr):
        df_trials = st.trials_dataframe()
        df_trials.to_csv(out_dir / "optuna_trials.csv", index=False)

    study.optimize(objective_func, n_trials=n_trials, callbacks=[save_callback])

    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE!")
    logger.info("Best Trial #%d: Macro F1 = %.4f", study.best_trial.number, study.best_value)
    logger.info("Best Hyperparameters:")
    for k, v in study.best_params.items():
        logger.info("  %s: %s", k, v)
    logger.info("=" * 60)

    # Save artifacts
    best_params_path = out_dir / "best_parameters.json"
    study_pkl_path = out_dir / "optuna_study.pkl"
    trials_csv_path = out_dir / "optuna_trials.csv"

    best_record = {
        "best_trial_number": study.best_trial.number,
        "best_macro_f1": study.best_value,
        "best_user_attrs": study.best_trial.user_attrs,
        "best_params": study.best_params,
    }

    with open(best_params_path, "w") as f:
        json.dump(best_record, f, indent=4)
    logger.info("Saved best parameters to: %s", best_params_path)

    joblib.dump(study, study_pkl_path)
    logger.info("Saved Optuna study to: %s", study_pkl_path)

    df_trials = study.trials_dataframe()
    df_trials.to_csv(trials_csv_path, index=False)
    logger.info("Saved complete trial history to: %s", trials_csv_path)

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PathSnap Hyperparameter Optimization")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--output-dir", default="tuning")
    args = parser.parse_args()

    run_tuning(n_trials=args.trials, n_splits=args.splits, output_dir=args.output_dir)
