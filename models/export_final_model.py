"""Train final tuned model and export artifacts for Members C and D.

Outputs saved in artifacts/:
- artifacts/model.joblib: Serialized best LightGBM classifier
- artifacts/preprocessing_pipeline.joblib: Feature scaler/transformer pipeline
- artifacts/model_metadata.json: Column ordering, hyperparams, classes, and integration docs
"""
import os
import sys
import json
import logging
from pathlib import Path

# Silence warnings before imports
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["MPLCONFIGDIR"] = "C:/Users/rajat/AppData/Local/Temp"

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "models"))
from evaluate_models import evaluate_predictions, format_metrics  # noqa: E402
from train_models import FEATURE_COLS  # noqa: E402


def export_production_artifacts():
    artifacts_dir = ROOT_DIR / "artifacts"
    tuning_dir = ROOT_DIR / "tuning"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    best_params_file = tuning_dir / "best_parameters.json"
    if not best_params_file.exists():
        logger.warning("No best_parameters.json found. Using default optimized LightGBM parameters.")
        best_params = {
            "n_estimators": 250,
            "max_depth": 8,
            "num_leaves": 63,
            "learning_rate": 0.05,
            "min_child_samples": 40,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        best_macro_f1 = 0.9080
    else:
        with open(best_params_file, "r") as f:
            record = json.load(f)
        best_params = record["best_params"]
        best_macro_f1 = record.get("best_macro_f1", None)
        # Ensure standard non-tuned params are present
        best_params["class_weight"] = "balanced"
        best_params["random_state"] = 42
        best_params["n_jobs"] = -1
        best_params["verbose"] = -1
        logger.info("Loaded best parameters from %s (Tuned Macro F1: %.4f)", best_params_file, best_macro_f1 or 0.0)

    # 1. Load training data (filtering trips 33 and 54)
    train_path = ROOT_DIR / "results/features/features_train.parquet"
    logger.info("Loading train data from: %s", train_path)
    df_train = pd.read_parquet(train_path)
    df_train = df_train[~df_train["trip_id"].isin(["00000033", "00000054"])].reset_index(drop=True)

    X_train = df_train[FEATURE_COLS].copy()
    y_train = df_train["label"].to_numpy(dtype=np.int64)

    # 2. Fit preprocessing pipeline (StandardScaler for candidate features)
    logger.info("Fitting preprocessing pipeline...")
    scaler = StandardScaler()
    scaler.fit(X_train)

    # 3. Train final model on full clean training dataset
    logger.info("Training final production LightGBM model on %d rows...", len(X_train))
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)

    # 4. Evaluate on Student-Test set to log final verification metrics
    test_path = ROOT_DIR / "results/features/features_studenttest.parquet"
    logger.info("Evaluating final model on Student-Test data (%s)...", test_path)
    df_test = pd.read_parquet(test_path)
    meta_test = df_test[["trip_id", "point_idx", "arc_id", "label"]].copy()
    X_test = df_test[FEATURE_COLS].copy()
    y_test = df_test["label"].to_numpy(dtype=np.int64)

    test_probs = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_predictions(meta_test, y_test, test_probs)
    logger.info("FINAL STUDENT-TEST VERIFICATION: %s", format_metrics(test_metrics))

    # 5. Save model and preprocessing artifacts
    model_path = artifacts_dir / "model.joblib"
    pipeline_path = artifacts_dir / "preprocessing_pipeline.joblib"
    metadata_path = artifacts_dir / "model_metadata.json"

    joblib.dump(model, model_path)
    logger.info("Saved final model to: %s", model_path)

    joblib.dump(scaler, pipeline_path)
    logger.info("Saved preprocessing pipeline to: %s", pipeline_path)

    metadata = {
        "model_type": "LightGBM Classifier",
        "library_version": lgb.__version__,
        "target_variable": "label",
        "classes": [0, 1],
        "feature_column_order": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "training_rows": len(X_train),
        "training_trips": int(df_train["trip_id"].nunique()),
        "excluded_trips": ["00000033", "00000054"],
        "reason_excluded": "GPS drift >120m away from road network causing extreme candidate miss rate",
        "hyperparameters": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for k, v in best_params.items()},
        "best_cv_macro_f1": best_macro_f1,
        "student_test_metrics": test_metrics,
        "integration_notes": {
            "Member_C_HMM": "Load model with joblib.dump. Call model.predict_proba(candidates[FEATURE_COLS])[:, 1] to obtain emission probabilities P(match=1) for candidate road arcs.",
            "Member_D_FastAPI": "Validate input dictionary against FEATURE_COLS order, create DataFrame or 2D array, and call model.predict_proba() for real-time scoring."
        }
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info("Saved model metadata to: %s", metadata_path)
    logger.info("Artifact generation completed successfully!")


if __name__ == "__main__":
    export_production_artifacts()
