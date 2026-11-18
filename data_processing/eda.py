"""Dataset exploration for one partition.

Usage:
    python data_processing/eda.py --data data/train --out results/eda/train
    python data_processing/eda.py --data data/student_test --out results/eda/student_test
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loader import list_trips, load_trip, load_route  # noqa: E402


def trip_stats(data_dir, trip_id):
    track, nodes, arcs = load_trip(data_dir, trip_id)
    route = load_route(data_dir, trip_id)
    dt = np.diff(track["t"].to_numpy())

    G = nx.DiGraph()
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from(arcs[["u", "v"]].itertuples(index=False, name=None))
    weak = nx.number_weakly_connected_components(G)
    strong = nx.number_strongly_connected_components(G)
    isolated = sum(1 for n in G.nodes if G.degree(n) == 0)

    pairs = set(arcs[["u", "v"]].itertuples(index=False, name=None))
    has_reverse = np.mean([(v, u) in pairs for u, v in pairs])

    # does consecutive route arc i end where arc i+1 starts?
    r_u = arcs["u"].to_numpy()[route]
    r_v = arcs["v"].to_numpy()[route]
    route_connected = np.mean(r_v[:-1] == r_u[1:]) if len(route) > 1 else np.nan

    return {
        "trip_id": trip_id,
        "n_gps": len(track),
        "n_nodes": len(nodes),
        "n_arcs": len(arcs),
        "n_route_arcs": len(route),
        "missing_values": int(track.isna().sum().sum()),
        "duplicate_timestamps": int((dt == 0).sum()),
        "non_increasing_time": int((dt < 0).sum()),
        "pct_dt_1s": float(np.mean(dt == 1) * 100),
        "max_gap_s": float(dt.max()) if len(dt) else np.nan,
        "gaps_over_10s": int((dt > 10).sum()),
        "duration_s": float(track["t"].iloc[-1] - track["t"].iloc[0]),
        "lon_min": track["lon"].min(), "lon_max": track["lon"].max(),
        "lat_min": track["lat"].min(), "lat_max": track["lat"].max(),
        "coords_plausible": bool(track["lon"].between(-180, 180).all()
                                 and track["lat"].between(-90, 90).all()),
        "weak_components": weak,
        "strong_components": strong,
        "isolated_nodes": isolated,
        "pct_arcs_with_reverse": float(has_reverse * 100),
        "pct_route_steps_connected": float(route_connected * 100),
        "route_ids_valid": bool((route >= 0).all() and (route < len(arcs)).all()),
    }, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, all_dt = [], []
    for tid in list_trips(a.data):
        s, dt = trip_stats(a.data, tid)
        rows.append(s)
        all_dt.append(dt)
        print(f"{tid}: {s['n_gps']} pts, {s['n_arcs']} arcs, max gap {s['max_gap_s']:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out / "inventory.csv", index=False)
    df.describe().T.to_csv(out / "summary_stats.csv")

    all_dt = np.concatenate(all_dt)
    plt.figure(figsize=(7, 4))
    plt.hist(np.clip(all_dt, 0, 20), bins=40)
    plt.yscale("log")
    plt.xlabel("time gap between GPS points (s, clipped at 20)")
    plt.ylabel("count (log)")
    plt.title(f"Sampling intervals - {Path(a.data).name}")
    plt.tight_layout()
    plt.savefig(out / "time_gaps.png", dpi=150)

    plt.figure(figsize=(7, 4))
    plt.hist(df["n_gps"], bins=20)
    plt.xlabel("GPS points per trip")
    plt.ylabel("trips")
    plt.title(f"Trajectory length - {Path(a.data).name}")
    plt.tight_layout()
    plt.savefig(out / "points_per_trip.png", dpi=150)

    plt.figure(figsize=(8, 4))
    plt.scatter(df["lon_min"], df["lat_min"], s=15)
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.title(f"Trip locations - {Path(a.data).name}")
    plt.tight_layout()
    plt.savefig(out / "trip_locations.png", dpi=150)

    print(f"\nSaved to {out}")
    print(df[["n_gps", "n_arcs", "max_gap_s", "weak_components",
              "pct_arcs_with_reverse", "pct_route_steps_connected"]].describe().round(1))


if __name__ == "__main__":
    main()
