"""Compare classifier-only vs classifier + HMM/Viterbi on a partition.

Without a model (distance-only emissions):
    python hmm/run_hmm.py --data data/student_test --out results/hmm/hmm_distance_only.csv
With the chosen model (after Rajat saves artifacts/model.joblib and the feature list):
    python hmm/run_hmm.py --data data/student_test --model artifacts/model.joblib --out results/hmm/hmm_model.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_processing"))
sys.path.insert(0, str(ROOT / "features"))
sys.path.insert(0, str(ROOT / "hmm"))
from loader import list_trips, load_trip, load_route, project_trip  # noqa: E402
from labels import label_points  # noqa: E402
from build_features import generate_candidates, add_features  # noqa: E402
from viterbi import (viterbi, classifier_only, top1_accuracy, route_continuity,  # noqa: E402
                     route_precision_recall)

ID_COLS = ["trip_id", "point_idx", "arc_id", "label"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="joblib file of the chosen classifier")
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--sigma", type=float, default=10.0)
    ap.add_argument("--radius", type=float, default=50.0)
    ap.add_argument("--max-cand", type=int, default=8)
    ap.add_argument("--trips", nargs="*", default=None, help="optional subset of trip ids")
    a = ap.parse_args()

    model = None
    if a.model:
        import joblib
        model = joblib.load(a.model)

    rows = []
    for tid in (a.trips or list_trips(a.data)):
        t0 = time.time()
        track, nodes, arcs = load_trip(a.data, tid)
        track, nodes, _ = project_trip(track, nodes)
        cand = generate_candidates(track, nodes, arcs, a.radius, a.max_cand)
        if model is not None:
            cand = add_features(cand, track, nodes, arcs)
            feats = getattr(model, "feature_names_in_", None)
            X = cand[list(feats)] if feats is not None else cand.drop(columns=["point_idx", "arc_id"])
            cand["prob"] = model.predict_proba(X)[:, 1]

        base = classifier_only(cand, len(track), a.sigma)
        hmm = viterbi(track, nodes, arcs, cand, beta=a.beta, sigma=a.sigma)
        secs = time.time() - t0

        # ground truth only used here, AFTER prediction
        lab, _, _ = label_points(track, nodes, arcs, load_route(a.data, tid))
        route = load_route(a.data, tid)
        for name, pred in [("classifier_only", base), ("classifier+hmm", hmm)]:
            p, r = route_precision_recall(pred, route)
            rows.append({"trip_id": tid, "method": name,
                         "top1_acc": top1_accuracy(pred, lab),
                         "route_continuity": route_continuity(pred, arcs),
                         "route_precision": p, "route_recall": r,
                         "seconds": round(secs, 1)})
        print(tid, f"{secs:.1f}s",
              f"top1 {rows[-2]['top1_acc']:.3f} -> {rows[-1]['top1_acc']:.3f}",
              f"continuity {rows[-2]['route_continuity']:.3f} -> {rows[-1]['route_continuity']:.3f}")

    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print("\nMean over trips:")
    print(df.groupby("method")[["top1_acc", "route_continuity", "route_precision",
                                "route_recall"]].mean().round(4))


if __name__ == "__main__":
    main()
