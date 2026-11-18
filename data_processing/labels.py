"""Derive a reference arc for every GPS point from the trip's .route (arc sequence).

The .route file lists arcs in driving order but does NOT say which GPS point is on which arc.
We assign each point to one route position so that:
  * the total point-to-arc distance is minimal, and
  * positions never go backwards in time (monotone along the route).
This is solved exactly with dynamic programming (O(points x route_len)).

Limitations to state in the report: points near junctions are ambiguous; if the route itself
is wrong, labels inherit the error; a point far from every route arc gets a large label distance.
"""
import numpy as np


def point_segment_dist(px, py, x1, y1, x2, y2):
    """Distance from point(s) to segment(s) and projection fraction t in [0,1]. Broadcasts."""
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - x1) * dx + (py - y1) * dy) / L2
    t = np.where(L2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    qx, qy = x1 + t * dx, y1 + t * dy
    return np.hypot(px - qx, py - qy), t


def label_points(track, nodes, arcs, route):
    """Return (label_arc_id per point, label_distance_m per point, route_position per point).

    track/nodes must already have x, y columns (metres) from loader.project_trip.
    """
    px, py = track["x"].to_numpy(), track["y"].to_numpy()
    u = arcs["u"].to_numpy()[route]
    v = arcs["v"].to_numpy()[route]
    nx_, ny_ = nodes["x"].to_numpy(), nodes["y"].to_numpy()
    x1, y1, x2, y2 = nx_[u], ny_[u], nx_[v], ny_[v]

    n, m = len(px), len(route)
    back = np.empty((n, m), dtype=np.int32)       # argmin pointer for backtracking
    prev = None
    for i in range(n):
        c, _ = point_segment_dist(px[i], py[i], x1, y1, x2, y2)
        if prev is None:
            cur = c
            back[i] = np.arange(m)
        else:
            # best previous cost among positions <= j  (monotone constraint)
            cm = np.minimum.accumulate(prev)
            # index of that minimum
            idx = np.arange(m)
            is_new_min = np.r_[True, prev[1:] < cm[:-1]]
            arg = np.maximum.accumulate(np.where(is_new_min, idx, 0))
            cur = c + cm
            back[i] = arg
        prev = cur

    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]

    lab_arc = route[pos]
    d, _ = point_segment_dist(px, py, x1[pos], y1[pos], x2[pos], y2[pos])
    return lab_arc, d, pos
