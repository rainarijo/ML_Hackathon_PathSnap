"""Folium map of one trip: GPS, road network, reference route, classifier-only and HMM predictions.

Usage (student-test trip, final model):
    python viz/route_map.py --data data/student_test --trip 00000076 --model artifacts/model.joblib \
        --beta 10 --out results/maps/route_00000076.html
"""
import argparse
import sys
from pathlib import Path

import folium
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ["data_processing", "features", "hmm"]:
    sys.path.insert(0, str(ROOT / sub))
from loader import load_trip, load_route, project_trip  # noqa: E402
from labels import label_points  # noqa: E402
from build_features import generate_candidates, add_features  # noqa: E402
from viterbi import viterbi, classifier_only, top1_accuracy, route_continuity  # noqa: E402


def arc_lines(arc_ids, nodes, arcs):
    """Collapse consecutive repeats and return [(lat, lon), (lat, lon)] per arc."""
    seq = [a for i, a in enumerate(arc_ids) if a >= 0 and (i == 0 or a != arc_ids[i - 1])]
    out = []
    for a in dict.fromkeys(seq):  # unique, order kept
        u, v = arcs.u[a], arcs.v[a]
        out.append([(nodes.lat[u], nodes.lon[u]), (nodes.lat[v], nodes.lon[v])])
    return out


def add_lines(group, lines, color, weight, opacity=1.0, dash=None):
    """Draw all segments as ONE multi-polyline (keeps the HTML file small)."""
    if lines:
        lines = [[(round(la, 6), round(lo, 6)) for la, lo in ln] for ln in lines]
        folium.PolyLine(lines, color=color, weight=weight, opacity=opacity,
                        dash_array=dash).add_to(group)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--trip", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    track, nodes, arcs = load_trip(a.data, a.trip)
    track, nodes, _ = project_trip(track, nodes)
    model = joblib.load(a.model)

    cand = generate_candidates(track, nodes, arcs, 50.0, 8)
    cand = add_features(cand, track, nodes, arcs)
    cand["prob"] = model.predict_proba(cand[list(model.feature_names_in_)])[:, 1]
    base = classifier_only(cand, len(track))
    hmm = viterbi(track, nodes, arcs, cand, beta=a.beta)

    route = load_route(a.data, a.trip)            # reference used only for display + scores
    lab, _, _ = label_points(track, nodes, arcs, route)
    s_base = (top1_accuracy(base, lab), route_continuity(base, arcs))
    s_hmm = (top1_accuracy(hmm, lab), route_continuity(hmm, arcs))

    m = folium.Map(location=[track.lat.mean(), track.lon.mean()], tiles="OpenStreetMap")
    m.fit_bounds([[track.lat.min(), track.lon.min()], [track.lat.max(), track.lon.max()]])

    pad = 0.003
    inside = (nodes.lon.between(track.lon.min() - pad, track.lon.max() + pad)
              & nodes.lat.between(track.lat.min() - pad, track.lat.max() + pad)).to_numpy()
    keep = arcs[inside[arcs.u] & inside[arcs.v]]
    g_road = folium.FeatureGroup(name="Road network", show=True)
    add_lines(g_road, [[(nodes.lat[u], nodes.lon[u]), (nodes.lat[v], nodes.lon[v])]
                       for u, v in keep.itertuples(index=False)], "#888888", 1, 0.5)

    g_ref = folium.FeatureGroup(name="Reference route")
    add_lines(g_ref, arc_lines(list(route), nodes, arcs), "#2ca02c", 7, 0.6)

    g_base = folium.FeatureGroup(name=f"Classifier only (Top-1 {s_base[0]:.1%})", show=False)
    add_lines(g_base, arc_lines(list(base), nodes, arcs), "#ff7f0e", 3, 0.9, "6,6")

    g_hmm = folium.FeatureGroup(name=f"Classifier + HMM (Top-1 {s_hmm[0]:.1%})")
    add_lines(g_hmm, arc_lines(list(hmm), nodes, arcs), "#d62728", 3, 1.0)

    g_gps = folium.FeatureGroup(name="GPS observations")
    step = max(1, len(track) // 1500)
    for lat, lon, t in track[["lat", "lon", "t"]].iloc[::step].itertuples(index=False):
        folium.CircleMarker((lat, lon), radius=2, color="#1f77b4", fill=True, fill_opacity=0.8,
                            weight=0, popup=f"t={t:.0f}s").add_to(g_gps)
    folium.Marker((track.lat.iloc[0], track.lon.iloc[0]), tooltip="Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(g_gps)
    folium.Marker((track.lat.iloc[-1], track.lon.iloc[-1]), tooltip="End",
                  icon=folium.Icon(color="red", icon="stop")).add_to(g_gps)

    for g in [g_road, g_ref, g_base, g_hmm, g_gps]:
        g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    legend = f"""
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999; background: white;
         padding: 10px 12px; border: 1px solid #999; border-radius: 6px; font: 13px Arial;">
      <b>Trip {a.trip}</b><br>
      <span style="color:#1f77b4">&#9679;</span> GPS observations<br>
      <span style="color:#888">&#9472;&#9472;</span> Road network<br>
      <span style="color:#2ca02c">&#9473;&#9473;</span> Reference route<br>
      <span style="color:#ff7f0e">- - -</span> Classifier only: Top-1 {s_base[0]:.1%}, continuity {s_base[1]:.1%}<br>
      <span style="color:#d62728">&#9473;&#9473;</span> Classifier + HMM: Top-1 {s_hmm[0]:.1%}, continuity {s_hmm[1]:.1%}
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    m.save(a.out)
    print(f"saved {a.out} | classifier-only top1 {s_base[0]:.3f} cont {s_base[1]:.3f} "
          f"| HMM top1 {s_hmm[0]:.3f} cont {s_hmm[1]:.3f}")


if __name__ == "__main__":
    main()
