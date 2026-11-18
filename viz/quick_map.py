"""Quick Folium sanity map: GPS points, nearby road network, reference route.

Usage:
    python viz/quick_map.py --data data/train --trip 00000054 --out results/maps/train_00000054.html
"""
import argparse
import sys
from pathlib import Path

import folium
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_processing"))
from loader import load_trip, load_route  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--trip", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    track, nodes, arcs = load_trip(a.data, a.trip)
    route = load_route(a.data, a.trip)

    m = folium.Map(location=[track["lat"].mean(), track["lon"].mean()], zoom_start=14,
                   tiles="OpenStreetMap")

    # road network: only arcs inside the GPS bounding box (+margin), else the map is huge
    pad = 0.005
    inside = (nodes["lon"].between(track["lon"].min() - pad, track["lon"].max() + pad)
              & nodes["lat"].between(track["lat"].min() - pad, track["lat"].max() + pad)).to_numpy()
    keep = arcs[inside[arcs["u"]] & inside[arcs["v"]]]
    roads = folium.FeatureGroup(name=f"Road network ({len(keep)} arcs)")
    for u, v in keep.itertuples(index=False):
        folium.PolyLine([(nodes.lat[u], nodes.lon[u]), (nodes.lat[v], nodes.lon[v])],
                        color="#999999", weight=1, opacity=0.6).add_to(roads)
    roads.add_to(m)

    ref = folium.FeatureGroup(name="Reference route")
    for aid in route:
        u, v = arcs.u[aid], arcs.v[aid]
        folium.PolyLine([(nodes.lat[u], nodes.lon[u]), (nodes.lat[v], nodes.lon[v])],
                        color="green", weight=4).add_to(ref)
    ref.add_to(m)

    gps = folium.FeatureGroup(name="GPS points")
    step = max(1, len(track) // 2000)  # thin very long trips
    for lon, lat, t in track.iloc[::step].itertuples(index=False):
        folium.CircleMarker((lat, lon), radius=2, color="blue", fill=True,
                            popup=f"t={t:.0f}s").add_to(gps)
    gps.add_to(m)

    folium.LayerControl().add_to(m)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    m.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
