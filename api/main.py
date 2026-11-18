
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from data_processing.loader import (
    list_trips,
    load_trip,
    load_route,
    project_trip,
)
from data_processing.labels import label_points
from features.build_features import generate_candidates, add_features
from hmm.viterbi import viterbi, route_continuity


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "Student folder" / "student_test"
MODEL_PATH = ROOT / "artifacts" / "model.joblib"

model = None

app = FastAPI(
    title="PathSnap API",
    description="GPS trajectory map-matching API using LightGBM and HMM",
    version="2.0",
    swagger_ui_parameters={"syntaxHighlight": False}
)


# --------------------------------------------------
# LOAD MODEL ONCE WHEN API STARTS
# --------------------------------------------------

@app.on_event("startup")
def load_model():
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    print(f"[MODEL] Loaded LightGBM model from {MODEL_PATH}", flush=True)
    print(
        f"[MODEL] Number of features: "
        f"{len(model.feature_names_in_)}",
        flush=True
    )


# --------------------------------------------------
# REQUEST FORMAT
# --------------------------------------------------

class PredictRequest(BaseModel):
    trip_id: str


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    print("[REQUEST] Home endpoint called", flush=True)

    return {
        "message": "Welcome to PathSnap API",
        "status": "running",
        "model": "LightGBM + HMM"
    }


# --------------------------------------------------
# GET AVAILABLE TRIPS
# --------------------------------------------------

@app.get("/trips")
def get_trips():
    print("[REQUEST] GET /trips", flush=True)

    if not DATA_DIR.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Dataset folder not found: {DATA_DIR}"
        )

    trips = list_trips(DATA_DIR)

    return {
        "trips": trips,
        "total": len(trips)
    }


# --------------------------------------------------
# PREDICT ROUTE
# --------------------------------------------------

@app.post("/predict", response_model=dict[str, Any])
def predict_route(request: PredictRequest):

    trip_id = request.trip_id

    print(f"[REQUEST] POST /predict | trip_id={trip_id}", flush=True)

    available_trips = list_trips(DATA_DIR)

    if trip_id not in available_trips:
        raise HTTPException(
            status_code=404,
            detail="Trip not found"
        )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model has not been loaded"
        )

    try:
        # 1. Load GPS track and road network.
        #    Ground-truth route is NOT loaded here.
        track, nodes, arcs = load_trip(DATA_DIR, trip_id)

        # 2. Convert coordinates to metres.
        track, nodes, epsg = project_trip(track, nodes)

        # 3. Generate nearby road candidates.
        candidates = generate_candidates(
            track, nodes, arcs, 50, 8
        )

        # 4. Extract candidate features.
        candidates = add_features(
            candidates, track, nodes, arcs
        )

        # 5. LightGBM: calculate candidate probabilities.
        if not candidates.empty:
            feature_names = list(model.feature_names_in_)

            missing_features = [
                col for col in feature_names
                if col not in candidates.columns
            ]

            if missing_features:
                raise ValueError(
                    f"Missing model features: {missing_features}"
                )

            candidates["prob"] = model.predict_proba(
                candidates[feature_names]
            )[:, 1]

        # 6. HMM + Viterbi: predict the most likely road sequence.
        pred = viterbi(
            track,
            nodes,
            arcs,
            candidates,
            beta=10.0
        )

        # 7. Prepare GPS coordinates for the map.
        gps_points = [
            {
                "lat": float(row.lat),
                "lon": float(row.lon)
            }
            for row in track.itertuples()
        ]

        # 8. Prepare predicted road segments for the map.
        #    -1 means no predicted arc for that GPS point.
        candidate_distances = {
            (int(row.point_idx), int(row.arc_id)): float(row.dist)
            for row in candidates.itertuples()
        }

        predicted_arcs = []

        for point_idx, arc_id in enumerate(pred):
            arc_id = int(arc_id)

            if arc_id < 0:
                continue

            arc = arcs.iloc[arc_id]

            start_node = nodes.iloc[int(arc["u"])]
            end_node = nodes.iloc[int(arc["v"])]

            predicted_arcs.append({
                "point_idx": point_idx,
                "arc_id": arc_id,
                "distance_m": candidate_distances.get(
                    (point_idx, arc_id)
                ),
                "coordinates": [
                    [
                        float(start_node["lon"]),
                        float(start_node["lat"])
                    ],
                    [
                        float(end_node["lon"]),
                        float(end_node["lat"])
                    ]
                ]
            })

        # 9. Evaluate route continuity from predictions.
        continuity = route_continuity(pred, arcs)

        # 10. Load ground truth ONLY AFTER prediction.
        route = load_route(DATA_DIR, trip_id)

        true_arcs, label_distances, _ = label_points(
            track,
            nodes,
            arcs,
            route
        )

        # Missing predictions (-1) count as incorrect.
        correct = sum(
            int(predicted) == int(actual)
            for predicted, actual in zip(pred, true_arcs)
        )

        total_points = len(track)

        accuracy = (
            correct / total_points * 100
            if total_points else 0.0
        )

        coverage = (
            np.count_nonzero(pred >= 0) / total_points * 100
            if total_points else 0.0
        )

        median_distance = (
            float(np.median(label_distances))
            if len(label_distances) else None
        )

        print(
            f"[RESULT] trip={trip_id}, "
            f"accuracy={accuracy:.2f}%, "
            f"coverage={coverage:.2f}%, "
            f"continuity={continuity:.4f}",
            flush=True
        )

        # 11. Return results to Streamlit.
        return {
            "trip_id": trip_id,
            "total_points": total_points,
            "accuracy_percent": round(accuracy, 2),
            "correct_points": correct,
            "candidate_coverage_percent": round(coverage, 2),
            "median_label_distance_m": (
                round(median_distance, 2)
                if median_distance is not None else None
            ),
            "route_continuity": round(continuity, 4),
            "gps_points": gps_points,
            "predicted_arcs": predicted_arcs,
            "prediction_method": "LightGBM + HMM (Viterbi)",
            "epsg": int(epsg)
        }

    except HTTPException:
        raise

    except Exception as error:
        print(
            f"[ERROR] trip={trip_id}: {error}",
            flush=True
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
