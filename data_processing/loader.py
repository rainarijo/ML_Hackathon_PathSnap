"""Read raw PathSnap trips and project them to metres.

Each trip folder <id>/ contains <id>.track, .nodes, .arcs and (train / student_test only) .route.
Route loading is kept in a separate function so prediction code never touches it.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


def list_trips(partition_dir):
    """Return sorted trip ids (folder names) in a partition folder."""
    return sorted(p.name for p in Path(partition_dir).iterdir() if p.is_dir())


def load_trip(partition_dir, trip_id):
    """Load GPS track and road network for one trip (NO ground truth)."""
    d = Path(partition_dir) / trip_id
    track = pd.read_csv(d / f"{trip_id}.track", sep=r"\s+", header=None,
                        names=["lon", "lat", "t"])
    nodes = pd.read_csv(d / f"{trip_id}.nodes", sep=r"\s+", header=None,
                        names=["lon", "lat"])
    arcs = pd.read_csv(d / f"{trip_id}.arcs", sep=r"\s+", header=None,
                       names=["u", "v"])
    # row index IS the id (0-based) -> make it explicit
    nodes.index.name = "node_id"
    arcs.index.name = "arc_id"
    return track, nodes, arcs


def load_route(partition_dir, trip_id):
    """Ground-truth arc sequence. Only for labelling / evaluation, never for prediction."""
    d = Path(partition_dir) / trip_id
    return pd.read_csv(d / f"{trip_id}.route", header=None, names=["arc_id"])["arc_id"].to_numpy()


def utm_crs_for(lon, lat):
    """Pick the UTM zone CRS for a lon/lat point (trips are spread worldwide)."""
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def project_trip(track, nodes):
    """Add x, y (metres) columns to track and nodes using one local UTM zone per trip."""
    lon0, lat0 = track["lon"].median(), track["lat"].median()
    crs = utm_crs_for(lon0, lat0)
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)  # always_xy -> (lon, lat) order
    track = track.copy()
    nodes = nodes.copy()
    track["x"], track["y"] = tf.transform(track["lon"].to_numpy(), track["lat"].to_numpy())
    nodes["x"], nodes["y"] = tf.transform(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    return track, nodes, crs.to_epsg()


def arc_geometry(nodes, arcs):
    """Return arrays of arc start/end coordinates in metres: (x1, y1, x2, y2)."""
    x1 = nodes["x"].to_numpy()[arcs["u"].to_numpy()]
    y1 = nodes["y"].to_numpy()[arcs["u"].to_numpy()]
    x2 = nodes["x"].to_numpy()[arcs["v"].to_numpy()]
    y2 = nodes["y"].to_numpy()[arcs["v"].to_numpy()]
    return x1, y1, x2, y2
