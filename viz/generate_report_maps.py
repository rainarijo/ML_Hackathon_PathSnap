"""Generate Folium interactive map and publication-grade static visualization
for Section 6 of the PathSnap Technical Report.
"""
import sys
from pathlib import Path
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_processing"))
sys.path.insert(0, str(ROOT / "features"))
sys.path.insert(0, str(ROOT / "hmm"))

from loader import load_trip, load_route, project_trip
from build_features import generate_candidates, add_features
from viterbi import viterbi, classifier_only, top1_accuracy, route_continuity, route_precision_recall

def main():
    trip_id = "00000015"
    data_dir = ROOT / "data" / "evaluator_test"
    out_dir = ROOT / "results" / "maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading trip {trip_id} from {data_dir}...")
    track, nodes, arcs = load_trip(data_dir, trip_id)
    route = load_route(data_dir, trip_id)
    track, nodes, _ = project_trip(track, nodes)
    
    print("Generating candidates & computing features...")
    cand = generate_candidates(track, nodes, arcs, radius=50.0, max_cand=8)
    cand = add_features(cand, track, nodes, arcs)
    
    model_path = ROOT / "artifacts" / "model.joblib"
    print(f"Loading trained LightGBM model from {model_path}...")
    model = joblib.load(model_path)
    
    feats = getattr(model, "feature_names_in_", None)
    X = cand[list(feats)] if feats is not None else cand.drop(columns=["point_idx", "arc_id"])
    cand["prob"] = model.predict_proba(X)[:, 1]
    
    print("Running HMM Viterbi decoding...")
    pred_arcs = viterbi(track, nodes, arcs, cand, beta=5.0, sigma=10.0)
    base_arcs = classifier_only(cand, len(track), sigma=10.0)
    
    from labels import label_points
    lab, _, _ = label_points(track, nodes, arcs, route)
    
    top1 = top1_accuracy(pred_arcs, lab)
    cont = route_continuity(pred_arcs, arcs)
    prec, rec = route_precision_recall(pred_arcs, route)
    print(f"Trip {trip_id} Performance: Top-1={top1:.4f}, Continuity={cont:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")
    
    # -------------------------------------------------------------------------
    # 1. BUILD INTERACTIVE FOLIUM MAP (results/maps/representative_trip_00000015.html)
    # -------------------------------------------------------------------------
    m = folium.Map(
        location=[track["lat"].mean(), track["lon"].mean()],
        zoom_start=14,
        tiles="OpenStreetMap"
    )
    
    # Bounding box road network
    pad = 0.005
    inside = (nodes["lon"].between(track["lon"].min() - pad, track["lon"].max() + pad) &
              nodes["lat"].between(track["lat"].min() - pad, track["lat"].max() + pad)).to_numpy()
    keep = arcs[inside[arcs["u"]] & inside[arcs["v"]]]
    
    roads_group = folium.FeatureGroup(name=f"Road Network ({len(keep)} arcs)", show=True)
    for u, v in keep.itertuples(index=False):
        folium.PolyLine(
            [(nodes.lat[u], nodes.lon[u]), (nodes.lat[v], nodes.lon[v])],
            color="#94a3b8", weight=1.5, opacity=0.6
        ).add_to(roads_group)
    roads_group.add_to(m)
    
    # Ground truth reference route (Green)
    ref_group = folium.FeatureGroup(name="Reference Ground Truth (Green)", show=True)
    for aid in route:
        u, v = arcs.u[aid], arcs.v[aid]
        folium.PolyLine(
            [(nodes.lat[u], nodes.lon[u]), (nodes.lat[v], nodes.lon[v])],
            color="#16a34a", weight=5, opacity=0.8, popup=f"Ground Truth Arc {aid}"
        ).add_to(ref_group)
    ref_group.add_to(m)
    
    # Predicted route (Red dashed / orange)
    pred_group = folium.FeatureGroup(name="LightGBM + HMM Predicted Route (Red)", show=True)
    for aid in pred_arcs:
        if aid < 0:
            continue
        u, v = arcs.u[aid], arcs.v[aid]
        folium.PolyLine(
            [(nodes.lat[u], nodes.lon[u]), (nodes.lat[v], nodes.lon[v])],
            color="#dc2626", weight=3, opacity=0.9, dash_array="6", popup=f"Predicted Arc {aid}"
        ).add_to(pred_group)
    pred_group.add_to(m)
    
    # GPS observations (Blue markers)
    gps_group = folium.FeatureGroup(name="GPS Observations (Blue Points)", show=True)
    step = max(1, len(track) // 500)
    for row in track.iloc[::step].itertuples(index=False):
        folium.CircleMarker(
            (row.lat, row.lon), radius=3, color="#2563eb", fill=True, fill_color="#3b82f6",
            fill_opacity=0.8, popup=f"GPS t={row.t:.1f}s"
        ).add_to(gps_group)
    gps_group.add_to(m)
    
    folium.LayerControl(collapsed=False).add_to(m)
    
    # HTML Legend
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 25px; left: 25px; width: 280px; height: 160px; 
                background-color: white; z-index:9999; font-size:12px;
                border:2px solid #cbd5e1; border-radius: 8px; padding: 12px;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.2);">
        <b style="font-size:13px; color:#1e293b;">PathSnap Trajectory Map ({trip_id})</b><br>
        <span style="color:#2563eb; font-weight:bold;">&#9679; GPS Observations (1 Hz)</span><br>
        <span style="color:#94a3b8; font-weight:bold;">&#9472; Road Network Arcs</span><br>
        <span style="color:#16a34a; font-weight:bold;">&#9473;&#9473; Ground Truth Route</span><br>
        <span style="color:#dc2626; font-weight:bold;">- - LightGBM + HMM Route</span><hr style="margin:6px 0;">
        <span style="color:#475569; font-size:11px;">Top-1 Acc: <b>{top1*100:.1f}%</b> | Continuity: <b>{cont*100:.1f}%</b></span>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    html_path = out_dir / f"representative_trip_{trip_id}.html"
    m.save(str(html_path))
    print(f"Saved interactive Folium map to {html_path}")
    
    # -------------------------------------------------------------------------
    # 2. GENERATE HIGH-RES STATIC PLOT FOR REPORT PDF
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=300)
    
    # Panel A: Macro Trajectory View
    ax1 = axes[0]
    ax1.set_title(f"Trajectory {trip_id}: Macro Overview", fontsize=14, fontweight="bold", pad=12)
    # Plot road network
    for u, v in keep.itertuples(index=False):
        ax1.plot([nodes.lon[u], nodes.lon[v]], [nodes.lat[u], nodes.lat[v]],
                 color="#cbd5e1", lw=0.8, zorder=1)
    
    # Plot Reference Route
    for aid in route:
        u, v = arcs.u[aid], arcs.v[aid]
        ax1.plot([nodes.lon[u], nodes.lon[v]], [nodes.lat[u], nodes.lat[v]],
                 color="#16a34a", lw=4.5, alpha=0.75, zorder=2, label="Reference Route" if aid == route[0] else "")
                 
    # Plot Predicted Route
    for aid in pred_arcs:
        if aid < 0:
            continue
        u, v = arcs.u[aid], arcs.v[aid]
        ax1.plot([nodes.lon[u], nodes.lon[v]], [nodes.lat[u], nodes.lat[v]],
                 color="#dc2626", lw=2.2, linestyle="--", zorder=3, label="Reconstructed Route" if aid == pred_arcs[0] else "")
                 
    # Plot GPS Points
    step_plot = max(1, len(track) // 400)
    ax1.scatter(track.lon.iloc[::step_plot], track.lat.iloc[::step_plot],
                color="#2563eb", s=12, alpha=0.6, zorder=4, label="Raw GPS Points")
                
    ax1.set_xlabel("Longitude (°)", fontsize=11)
    ax1.set_ylabel("Latitude (°)", fontsize=11)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Panel B: Micro Zoom-In Detail (shows GPS drift & topological correction)
    ax2 = axes[1]
    ax2.set_title("Micro Detail: GPS Noise vs Reconstructed Topological Match", fontsize=14, fontweight="bold", pad=12)
    
    # Zoom in on center 20% of track
    mid_idx = len(track) // 2
    sub_track = track.iloc[mid_idx - 60 : mid_idx + 60]
    min_lon, max_lon = sub_track.lon.min() - 0.001, sub_track.lon.max() + 0.001
    min_lat, max_lat = sub_track.lat.min() - 0.001, sub_track.lat.max() + 0.001
    
    sub_inside = (nodes["lon"].between(min_lon, max_lon) & nodes["lat"].between(min_lat, max_lat)).to_numpy()
    sub_arcs = arcs[sub_inside[arcs["u"]] & sub_inside[arcs["v"]]]
    
    for u, v in sub_arcs.itertuples(index=False):
        ax2.plot([nodes.lon[u], nodes.lon[v]], [nodes.lat[u], nodes.lat[v]],
                 color="#94a3b8", lw=1.5, zorder=1)
                 
    for aid in route:
        u, v = arcs.u[aid], arcs.v[aid]
        if sub_inside[u] or sub_inside[v]:
            ax2.plot([nodes.lon[u], nodes.lon[v]], [nodes.lat[u], nodes.lat[v]],
                     color="#16a34a", lw=5.0, alpha=0.75, zorder=2, label="Reference Route" if "ref_labeled" not in locals() else "")
            ref_labeled = True
            
    for aid in pred_arcs:
        if aid < 0:
            continue
        u, v = arcs.u[aid], arcs.v[aid]
        if sub_inside[u] or sub_inside[v]:
            ax2.plot([nodes.lon[u], nodes.lon[v]], [nodes.lat[u], nodes.lat[v]],
                     color="#dc2626", lw=2.5, linestyle="--", zorder=3, label="Predicted Match" if "pred_labeled" not in locals() else "")
            pred_labeled = True
            
    ax2.scatter(sub_track.lon, sub_track.lat, color="#2563eb", s=35, edgecolor="black", lw=0.5,
                alpha=0.85, zorder=4, label="Noisy GPS Observations")
                
    # Add metric box
    stats_text = (
        f"Trip ID: {trip_id}\n"
        f"Observation Top-1: {top1*100:.2f}%\n"
        f"Route Continuity: {cont*100:.2f}%\n"
        f"Route Precision: {prec*100:.2f}%\n"
        f"Route Recall: {rec*100:.2f}%"
    )
    ax2.text(0.04, 0.05, stats_text, transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#94a3b8", alpha=0.95),
             verticalalignment="bottom")
             
    ax2.set_xlim(min_lon, max_lon)
    ax2.set_ylim(min_lat, max_lat)
    ax2.set_xlabel("Longitude (°)", fontsize=11)
    ax2.set_ylabel("Latitude (°)", fontsize=11)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    static_png_path = out_dir / f"representative_trip_{trip_id}.png"
    plt.savefig(str(static_png_path), dpi=300)
    plt.close()
    print(f"Saved publication-grade static visualization to {static_png_path}")

if __name__ == "__main__":
    main()
