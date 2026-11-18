
from pathlib import Path

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


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "Student folder" / "student_test"

app = FastAPI(
    title="PathSnap API",
    description="GPS trajectory map-matching API",
    version="1.0"
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
        "status": "running"
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

@app.post("/predict")
def predict_route(request: PredictRequest):

    trip_id = request.trip_id

    print(f"[REQUEST] POST /predict | trip_id={trip_id}", flush=True)

    # Validate trip ID
    available_trips = list_trips(DATA_DIR)

    if trip_id not in available_trips:
        raise HTTPException(
            status_code=404,
            detail="Trip not found"
        )

    try:
        # 1. Load GPS track and road network.
        #    This does NOT load the ground-truth route.
        track, nodes, arcs = load_trip(DATA_DIR, trip_id)

        # 2. Convert coordinates to metres.
        track, nodes, epsg = project_trip(track, nodes)

        # 3. Generate nearby road candidates.
        candidates = generate_candidates(
            track,
            nodes,
            arcs,
            radius=50.0,
            max_cand=8
        )

        # 4. Calculate candidate features.
        candidates = add_features(
            candidates,
            track,
            nodes,
            arcs
        )

        # 5. Select the nearest candidate for each GPS point.
        nearest = candidates[
            candidates["dist_rank"] == 1
        ].set_index("point_idx")

        predictions = [None] * len(track)

        for point_idx, row in nearest.iterrows():
            predictions[int(point_idx)] = int(row["arc_id"])

        # 6. Prepare GPS coordinates for the map.
        gps_points = [
            {
                "lat": float(row.lat),
                "lon": float(row.lon)
            }
            for row in track.itertuples()
        ]

        # 7. Prepare predicted road segments for the map.
        predicted_arcs = []

        for point_idx, arc_id in enumerate(predictions):

            if arc_id is None:
                continue

            arc = arcs.iloc[arc_id]

            start_node = nodes.iloc[int(arc["u"])]
            end_node = nodes.iloc[int(arc["v"])]

            predicted_arcs.append({
                "point_idx": point_idx,
                "arc_id": int(arc_id),
                "distance_m": float(
                    nearest.loc[point_idx, "dist"]
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

        # 8. Evaluation: load ground truth separately.
        route = load_route(DATA_DIR, trip_id)

        true_arcs, label_distances, _ = label_points(
            track,
            nodes,
            arcs,
            route
        )

        # Missing predictions count as incorrect.
        correct = sum(
            prediction == int(true_arc)
            for prediction, true_arc in zip(predictions, true_arcs)
            if prediction is not None
        )

        total_points = len(track)

        accuracy = (
            correct / total_points * 100
            if total_points else 0.0
        )

        coverage = (
            sum(p is not None for p in predictions)
            / total_points * 100
            if total_points else 0.0
        )

        print(
            f"[RESULT] trip={trip_id}, "
            f"accuracy={accuracy:.2f}%, "
            f"coverage={coverage:.2f}%",
            flush=True
        )

        # 9. Return results to the UI.
        return {
            "trip_id": trip_id,
            "total_points": total_points,
            "accuracy_percent": round(accuracy, 2),
            "correct_points": correct,
            "candidate_coverage_percent": round(coverage, 2),
            "median_label_distance_m": round(
                float(np.median(label_distances)), 2
            ),
            "gps_points": gps_points,
            "predicted_arcs": predicted_arcs,
            "prediction_method": "Nearest candidate baseline",
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
