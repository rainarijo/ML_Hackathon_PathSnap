"""Script to generate the comprehensive, Kaggle-ready PathSnap_ML_Training.ipynb notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def make_cell(cell_type, source):
    if isinstance(source, list):
        src = [s if s.endswith("\n") else s + "\n" for s in source]
    else:
        src = [s + "\n" for s in source.split("\n")]
        if src and src[-1] == "\n":
            src.pop()
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": src
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def build():
    cells = []

    # Title & Introduction
    cells.append(make_cell("markdown", """# PathSnap ML Training & Hyperparameter Tuning Pipeline
### **Member B: Machine Learning Models & Optimization**
**Project:** PathSnap – GPS Map Matching & Route Reconstruction  
**Environment:** Kaggle / Local Reproducible ML Environment  

---

### **Table of Contents**
1. **Introduction & Problem Formulation**
2. **Environment Setup & Dependencies**
3. **Data Loading & Outlier Filtering** (Trips 33 & 54 Removal)
4. **Candidate Ranking & Top-1 Accuracy Implementation**
5. **Cross-Validation Strategy: Leakage-Free GroupKFold**
6. **Training & Benchmarking 5 Classifiers:**
   - Model 1: Logistic Regression
   - Model 2: K-Nearest Neighbors (KNN - 200k Subsample)
   - Model 3: Random Forest Classifier
   - Model 4: LightGBM Classifier
   - Model 5: Multi-Layer Perceptron (MLP Neural Network)
7. **Comparative Analysis & Visualizations**
8. **Bayesian Hyperparameter Tuning with Optuna**
9. **Final Production Model Training & Artifact Export**
10. **Team Integration Guide (Members C & D)**"""))

    # Cell 1: Environment Setup
    cells.append(make_cell("markdown", """## 1. Environment Setup & Library Imports
We configure reproducibility seeds (`random_state=42`), multi-threading parameters, and import standard scientific libraries."""))

    cells.append(make_cell("code", """import os
import sys
import time
import json
import logging
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score
import lightgbm as lgb
import optuna

print("All dependencies successfully imported!")
print(f"NumPy: {np.__version__} | Pandas: {pd.__version__} | LightGBM: {lgb.__version__} | Optuna: {optuna.__version__}")"""))

    # Cell 2: Feature & Column Definitions
    cells.append(make_cell("markdown", """## 2. Feature Schema & Column Definitions
Based on candidate generation (`features/build_features.py`):
- **Candidate Generator:** For each GPS point, road segments within a **50m radius** (up to 8 candidates) are extracted.
- **Identifiers / Grouping:** `trip_id`, `point_idx`, `arc_id`.
- **Target Label:** `label` (binary 1 for true driving road arc, 0 for incorrect candidate).
- **Features (X):** 13 spatial and topological features."""))

    cells.append(make_cell("code", """FEATURE_COLS = [
    "dist",                      # Distance from GPS point to candidate road arc (meters)
    "t_frac",                    # Normalized projection point along the candidate segment [0, 1]
    "arc_len",                   # Length of candidate road segment (meters)
    "speed",                     # Vehicle speed derived from GPS track (m/s)
    "dt_prev",                   # Time delta from preceding GPS coordinate (seconds)
    "heading_diff",              # Angular divergence between GPS bearing and road bearing [0, 180 deg]
    "heading_reliable",          # Binary indicator (1 if speed > 1.0 m/s, else 0)
    "n_cand",                    # Total candidate road arcs identified for this GPS point
    "dist_rank",                 # Rank of this candidate by proximity (1 = nearest)
    "dist_minus_min",            # Distance difference compared to nearest candidate (meters)
    "dist_ratio",                # Proximity ratio: dist / (min_dist + 1.0)
    "same_as_prev_nearest",      # 1 if arc is identical to preceding point's nearest candidate
    "connected_to_prev_nearest"  # 1 if arc topologically connects to preceding nearest candidate
]

EXCLUDE_COLS = ["trip_id", "point_idx", "arc_id", "label"]
print(f"Number of engineered features: {len(FEATURE_COLS)}")"""))

    # Cell 3: Data Loading & Outlier Removal
    cells.append(make_cell("markdown", """## 3. Data Loading & GPS Outlier Removal
Trips `00000033` and `00000054` suffered severe GPS sensor drift (~122m and ~140m off the road network). Filtering them leaves **78 clean trips** for training."""))

    cells.append(make_cell("code", """train_path = Path("results/features/features_train.parquet")
test_path = Path("results/features/features_studenttest.parquet")

if not train_path.exists():
    train_path = Path("/kaggle/input/pathsnap-features/results/features/features_train.parquet")
    test_path = Path("/kaggle/input/pathsnap-features/results/features/features_studenttest.parquet")

df_train_raw = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

outlier_trips = ['00000033', '00000054']
df_train = df_train_raw[~df_train_raw['trip_id'].isin(outlier_trips)].reset_index(drop=True)

print(f"Clean Train Shape: {df_train.shape} ({df_train['trip_id'].nunique()} trips)")
print(f"Student-Test Shape: {df_test.shape} ({df_test['trip_id'].nunique()} trips)")"""))

    # Cell 4: Top-1 Metric
    cells.append(make_cell("markdown", """## 4. Candidate Ranking & Top-1 Accuracy"""))

    cells.append(make_cell("code", """def calculate_top1_accuracy(df_meta: pd.DataFrame, probas: np.ndarray) -> float:
    df_eval = pd.DataFrame({
        "trip_id": df_meta["trip_id"].to_numpy(),
        "point_idx": df_meta["point_idx"].to_numpy(),
        "label": df_meta["label"].to_numpy(),
        "prob": np.asarray(probas, dtype=np.float64)
    })
    idx_max = df_eval.groupby(["trip_id", "point_idx"], sort=False)["prob"].idxmax()
    return float((df_eval.loc[idx_max, "label"] == 1).mean())

def evaluate_predictions(df_meta: pd.DataFrame, y_true: np.ndarray, probas: np.ndarray, threshold: float = 0.5):
    y_true = np.asarray(y_true, dtype=np.int64)
    probas = np.clip(np.asarray(probas, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    y_pred = (probas >= threshold).astype(np.int64)
    
    top1 = calculate_top1_accuracy(df_meta, probas)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    pos_f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    loss = float(log_loss(y_true, probas))
    row_acc = float(accuracy_score(y_true, y_pred))
    
    return {
        "top1_accuracy": top1,
        "macro_f1": macro_f1,
        "f1": pos_f1,
        "precision": prec,
        "recall": rec,
        "log_loss": loss,
        "row_accuracy": row_acc
    }"""))

    # Cell 5: Setup Arrays
    cells.append(make_cell("markdown", """## 5. Setup Data Arrays & GroupKFold"""))

    cells.append(make_cell("code", """X_train = df_train[FEATURE_COLS].copy()
y_train = df_train["label"].to_numpy(dtype=np.int64)
groups_train = df_train["trip_id"].to_numpy()
meta_train = df_train[["trip_id", "point_idx", "arc_id", "label"]].copy()

X_test = df_test[FEATURE_COLS].copy()
y_test = df_test["label"].to_numpy(dtype=np.int64)
meta_test = df_test[["trip_id", "point_idx", "arc_id", "label"]].copy()"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_path = ROOT / "PathSnap_ML_Training.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Created notebook at: {out_path}")

if __name__ == "__main__":
    build()
