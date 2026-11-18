"""HMM + Viterbi route reconstruction for PathSnap.

Hidden states  : the candidate arcs of each GPS point (from features/build_features.generate_candidates).
Emission       : log P(candidate | point). Either the classifier's probability (column `prob`)
                 or, without a model, a Gaussian on the point-to-arc distance.
Transition     : how plausible it is to drive from candidate i (point t-1) to candidate j (point t):
                 exp(-|road_distance - gps_distance| / beta)   (Newson & Krumm style)
                 road_distance uses shortest paths on the directed road graph.
                 Unreachable pairs get a tiny probability instead of 0.
Breaks         : if no transition is possible (or a point has no candidates) the route is split
                 and decoding restarts from the emission scores.
"""
import numpy as np
import networkx as nx

LOG_EPS = -50.0  # log-probability used for "impossible" transitions


def build_graph(nodes, arcs):
    """Directed road graph with edge length in metres (needs nodes x, y)."""
    x, y = nodes["x"].to_numpy(), nodes["y"].to_numpy()
    u, v = arcs["u"].to_numpy(), arcs["v"].to_numpy()
    length = np.hypot(x[v] - x[u], y[v] - y[u])
    G = nx.DiGraph()
    G.add_weighted_edges_from(zip(u.tolist(), v.tolist(), length.tolist()))
    return G, length


class RoadDistance:
    """Cached shortest-path distances (metres) between road nodes, with a search cutoff."""

    def __init__(self, G):
        self.G = G
        self.cache = {}  # node -> (cutoff, {target: dist})

    def from_node(self, s, cutoff):
        hit = self.cache.get(s)
        if hit is None or hit[0] < cutoff:
            d = nx.single_source_dijkstra_path_length(self.G, s, cutoff=cutoff, weight="weight")
            self.cache[s] = (cutoff, d)
            return d
        return hit[1]


def road_distance(rd, arcs_u, arcs_v, arc_len, a, ta, b, tb, cutoff):
    """Driving distance from position ta on arc a to position tb on arc b (np.inf if unreachable)."""
    if a == b:
        fwd = (tb - ta) * arc_len[a]
        # small backward moves on the same arc are GPS noise, allow them
        return abs(fwd) if fwd >= -15.0 else np.inf
    rest_a = (1.0 - ta) * arc_len[a]
    into_b = tb * arc_len[b]
    d = rd.from_node(arcs_v[a], cutoff).get(arcs_u[b])
    if d is None:
        return np.inf
    return rest_a + d + into_b


def emission_logp(cand, sigma=10.0):
    """Log emission per candidate row. Uses classifier `prob` if present, else distance Gaussian."""
    if "prob" in cand.columns:
        p = cand["prob"].to_numpy()
        # normalise within each point so candidates of one point sum to 1
        s = cand.groupby("point_idx")["prob"].transform("sum").to_numpy()
        return np.log(np.clip(p / np.maximum(s, 1e-12), 1e-9, 1.0))
    d = cand["dist"].to_numpy()
    return -0.5 * (d / sigma) ** 2


def viterbi(track, nodes, arcs, cand, beta=5.0, sigma=10.0, emission_weight=1.0,
            gap_seconds=30.0):
    """Decode the most likely arc for every GPS point.

    cand: DataFrame with point_idx, arc_id, dist, t_frac (+ optional prob), from generate_candidates.
    Returns np.array of predicted arc_id per point (-1 where no candidate).
    """
    G, arc_len = build_graph(nodes, arcs)
    rd = RoadDistance(G)
    au, av = arcs["u"].to_numpy(), arcs["v"].to_numpy()

    cand = cand.sort_values(["point_idx", "dist"]).reset_index(drop=True)
    cand["logE"] = emission_weight * emission_logp(cand, sigma)
    groups = {p: g for p, g in cand.groupby("point_idx")}

    x, y, t = track["x"].to_numpy(), track["y"].to_numpy(), track["t"].to_numpy()
    n = len(track)
    pred = np.full(n, -1, dtype=np.int64)

    # Viterbi over runs of points; a run ends at a break
    run_pts, run_arcs, run_back = [], [], []
    prev_score = None
    prev_p = None

    def flush():
        if not run_pts:
            return
        k = int(np.argmax(prev_score))
        for i in range(len(run_pts) - 1, -1, -1):
            pred[run_pts[i]] = run_arcs[i][k]
            k = run_back[i][k]
        run_pts.clear(); run_arcs.clear(); run_back.clear()

    for p in range(n):
        g = groups.get(p)
        if g is None:                      # no candidates -> break the route here
            flush()
            prev_score, prev_p = None, None
            continue
        arcs_p = g["arc_id"].to_numpy()
        tf_p = g["t_frac"].to_numpy()
        logE = g["logE"].to_numpy()

        if prev_score is None:
            score = logE.copy()
            back = np.zeros(len(arcs_p), dtype=np.int64)
        else:
            gp = groups[prev_p]
            arcs_q, tf_q = gp["arc_id"].to_numpy(), gp["t_frac"].to_numpy()
            d_gps = np.hypot(x[p] - x[prev_p], y[p] - y[prev_p])
            dt = t[p] - t[prev_p]
            b = beta * max(1.0, dt / 5.0)              # be more tolerant across longer gaps
            cutoff = 3.0 * d_gps + 200.0
            logT = np.full((len(arcs_q), len(arcs_p)), LOG_EPS)
            for i in range(len(arcs_q)):
                for j in range(len(arcs_p)):
                    d_road = road_distance(rd, au, av, arc_len, arcs_q[i], tf_q[i],
                                           arcs_p[j], tf_p[j], cutoff)
                    if np.isfinite(d_road):
                        logT[i, j] = -abs(d_road - d_gps) / b
            total = prev_score[:, None] + logT
            back = np.argmax(total, axis=0)
            best = total[back, np.arange(len(arcs_p))]
            if dt > gap_seconds or np.all(logT <= LOG_EPS):   # big gap / disconnected -> break
                flush()
                score = logE.copy()
                back = np.zeros(len(arcs_p), dtype=np.int64)
            else:
                score = best + logE

        run_pts.append(p); run_arcs.append(arcs_p); run_back.append(back)
        prev_score, prev_p = score, p
    flush()
    return pred


def classifier_only(cand, n_points, sigma=10.0):
    """Baseline: take the best-scoring candidate of each point independently."""
    c = cand.copy()
    c["logE"] = emission_logp(c, sigma)
    best = c.loc[c.groupby("point_idx")["logE"].idxmax()]
    pred = np.full(n_points, -1, dtype=np.int64)
    pred[best["point_idx"].to_numpy()] = best["arc_id"].to_numpy()
    return pred


# ---------------- evaluation ----------------
def top1_accuracy(pred, label_arc):
    return float(np.mean(pred == label_arc))


def route_continuity(pred, arcs):
    """% of consecutive (changing) predicted arcs that are directly connected (v_a == u_b)."""
    seq = [a for a in pred if a >= 0]
    seq = [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]   # collapse repeats
    if len(seq) < 2:
        return 1.0
    u, v = arcs["u"].to_numpy(), arcs["v"].to_numpy()
    seq = np.array(seq)
    return float(np.mean(v[seq[:-1]] == u[seq[1:]]))


def route_precision_recall(pred, route):
    """Sequence-level: overlap between the set of predicted arcs and the reference route arcs."""
    P = set(int(a) for a in pred if a >= 0)
    R = set(int(a) for a in route)
    if not P or not R:
        return 0.0, 0.0
    inter = len(P & R)
    return inter / len(P), inter / len(R)
