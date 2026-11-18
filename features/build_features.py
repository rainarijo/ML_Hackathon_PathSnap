"""Candidate generation + feature engineering for PathSnap.

One output row = one (GPS point, candidate arc) pair. label = 1 if the candidate is the
reference arc for that point (only when a .route exists), else 0.

Usage:
    python features/build_features.py --data data/train        --out results/features/features_train.parquet
    python features/build_features.py --data data/student_test --out results/features/features_studenttest.parquet
Options: --radius 50 --max-cand 8
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_processing"))
from loader import list_trips, load_trip, load_route, project_trip, arc_geometry  # noqa: E402
from labels import label_points, point_segment_dist  # noqa: E402


def gps_motion(track, win=2):
    """Heading (deg, 0=north, clockwise), speed (m/s) and dt to previous point."""
    x, y, t = track["x"].to_numpy(), track["y"].to_numpy(), track["t"].to_numpy()
    n = len(x)
    i0 = np.clip(np.arange(n) - win, 0, n - 1)
    i1 = np.clip(np.arange(n) + win, 0, n - 1)
    dx, dy = x[i1] - x[i0], y[i1] - y[i0]
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    heading = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    speed = np.hypot(dx, dy) / dt
    dt_prev = np.r_[0.0, np.diff(t)]
    return heading, speed, dt_prev


def generate_candidates(track, nodes, arcs, radius=50.0, max_cand=8):
    """All arcs within `radius` m of each point, keeping the `max_cand` closest.

    Returns DataFrame with point_idx, arc_id, dist, t_frac.
    """
    x1, y1, x2, y2 = arc_geometry(nodes, arcs)
    lines = shapely.linestrings(np.stack([np.c_[x1, y1], np.c_[x2, y2]], axis=1))
    tree = STRtree(lines)
    pts = shapely.points(track["x"].to_numpy(), track["y"].to_numpy())
    pi, ai = tree.query(pts, predicate="dwithin", distance=radius)

    px, py = track["x"].to_numpy()[pi], track["y"].to_numpy()[pi]
    d, tf = point_segment_dist(px, py, x1[ai], y1[ai], x2[ai], y2[ai])
    c = pd.DataFrame({"point_idx": pi, "arc_id": ai, "dist": d, "t_frac": tf})
    c = c.sort_values(["point_idx", "dist"], kind="mergesort")
    c = c.groupby("point_idx", sort=False).head(max_cand).reset_index(drop=True)
    return c


def add_features(c, track, nodes, arcs):
    """Add per-candidate features. Uses ONLY track + road network (no ground truth)."""
    x1, y1, x2, y2 = arc_geometry(nodes, arcs)
    arc_len = np.hypot(x2 - x1, y2 - y1)
    arc_bearing = (np.degrees(np.arctan2(x2 - x1, y2 - y1)) + 360) % 360

    heading, speed, dt_prev = gps_motion(track)
    p = c["point_idx"].to_numpy()
    a = c["arc_id"].to_numpy()

    c["arc_len"] = arc_len[a]
    c["speed"] = speed[p]
    c["dt_prev"] = dt_prev[p]
    hd = np.abs(heading[p] - arc_bearing[a]) % 360
    c["heading_diff"] = np.minimum(hd, 360 - hd)          # 0 = same direction, 180 = opposite
    c["heading_reliable"] = (speed[p] > 1.0).astype(int)  # heading is noise when nearly stopped
    
    # turn rate: how fast the GPS heading is changing (deg/s); 0 when nearly stopped
    t = track["t"].to_numpy()
    dh = np.abs(np.diff(heading))
    dh = np.minimum(dh, 360 - dh)
    turn = np.r_[0.0, dh / np.maximum(np.diff(t), 1e-6)]
    turn[speed < 1.0] = 0.0
    c["turn_rate"] = turn[p]

    g = c.groupby("point_idx")["dist"]
    c["n_cand"] = g.transform("size")
    c["dist_rank"] = g.rank(method="first").astype(int)
    c["dist_minus_min"] = c["dist"] - g.transform("min")
    c["dist_ratio"] = c["dist"] / (g.transform("min") + 1.0)

    # connectivity with the previous point's nearest candidate
    nearest = c[c["dist_rank"] == 1].set_index("point_idx")["arc_id"]
    prev_near = pd.Series(p - 1).map(nearest).to_numpy()
    u, v = arcs["u"].to_numpy(), arcs["v"].to_numpy()
    ok = ~pd.isna(prev_near)
    pn = np.where(ok, prev_near, 0).astype(int)
    same = ok & (pn == a)
    follows = ok & (v[pn] == u[a])
    c["same_as_prev_nearest"] = same.astype(int)
    c["connected_to_prev_nearest"] = (same | follows).astype(int)
    return c


def build_trip(data_dir, tid, radius, max_cand, with_labels=True):
    track, nodes, arcs = load_trip(data_dir, tid)
    track, nodes, epsg = project_trip(track, nodes)
    c = generate_candidates(track, nodes, arcs, radius, max_cand)
    c = add_features(c, track, nodes, arcs)
    c.insert(0, "trip_id", tid)
    info = {"trip_id": tid, "epsg": epsg, "n_points": len(track),
            "points_with_cand": c["point_idx"].nunique()}
    if with_labels:
        route = load_route(data_dir, tid)
        lab, lab_d, _ = label_points(track, nodes, arcs, route)
        c["label"] = (c["arc_id"].to_numpy() == lab[c["point_idx"].to_numpy()]).astype(int)
        hit = c.groupby("point_idx")["label"].max()
        info["candidate_miss_rate"] = 1 - hit.reindex(range(len(track)), fill_value=0).mean()
        info["label_dist_median"] = float(np.median(lab_d))
        info["label_dist_p95"] = float(np.percentile(lab_d, 95))
    return c, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=float, default=50.0)
    ap.add_argument("--max-cand", type=int, default=8)
    ap.add_argument("--no-labels", action="store_true")
    a = ap.parse_args()

    frames, infos = [], []
    t0 = time.time()
    for tid in list_trips(a.data):
        c, info = build_trip(a.data, tid, a.radius, a.max_cand, not a.no_labels)
        frames.append(c)
        infos.append(info)
        print(info)
    df = pd.concat(frames, ignore_index=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    pd.DataFrame(infos).to_csv(out.with_suffix(".trip_info.csv"), index=False)
    print(f"\n{len(df)} rows, {df['trip_id'].nunique()} trips, {time.time()-t0:.0f}s -> {out}")
    if "label" in df:
        print("positive rate:", round(df["label"].mean(), 3))
        print("mean candidate miss rate:", round(pd.DataFrame(infos)["candidate_miss_rate"].mean(), 4))


if __name__ == "__main__":
    main()
