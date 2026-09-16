"""Snap floor-plan projections to discrete seats.

Pairs with:
- ``tools/calibrate_lecture_seats.py`` (produces the seat CSV)
- ``lib/lecture_cv.py::project_to_floor_plan`` (produces the floating points
  this module snaps)

The snap is *greedy* (closest-fit first) and applies a distance threshold so
that out-of-quadrilateral detections (e.g. the lecturer pacing in front) get
dropped instead of arbitrarily snapping to whatever seat happens to be nearby.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial import cKDTree


@st.cache_data
def load_seats(path: str = "data/lecture_144_seats.csv") -> pd.DataFrame:
    """Load the seat map. Columns: ``seat_id``, ``x_px``, ``y_px``."""
    df = pd.read_csv(path)
    # Defensive cast: the calibration tool writes floats, but humans editing CSVs
    # sometimes round to ints. Either way, KDTree wants floats.
    df["x_px"] = df["x_px"].astype(float)
    df["y_px"] = df["y_px"].astype(float)
    return df


@st.cache_resource
def build_seat_tree(seats: pd.DataFrame) -> cKDTree:
    """Build a cKDTree over the seat coordinates.

    Cached across reruns, invalidated when the seats DataFrame identity
    changes (i.e. the CSV is reloaded or edited).
    """
    return cKDTree(seats[["x_px", "y_px"]].to_numpy())


def snap_detections(
    points: Sequence[tuple[float, float]],
    seats: pd.DataFrame,
    tree: cKDTree,
    max_dist: float = 80.0,
    k_candidates: int = 5,
) -> tuple[set[str], list[dict]]:
    """Greedy nearest-seat assignment with a distance threshold.

    Args:
        points: floor-plan-space ``(x, y)`` projections (from YOLO + homography).
        seats: DataFrame returned by :func:`load_seats`.
        tree: KD-tree returned by :func:`build_seat_tree`.
        max_dist: drop detections whose nearest seat is further than this (px).
        k_candidates: how many ranked neighbours to consider when the
            preferred seat is already claimed by an earlier (closer) detection.

    Returns:
        occupied: set of ``seat_id`` strings claimed this tick.
        debug:    one dict per detection, ordered by input index, with keys
                  ``detection_index``, ``x``, ``y``, ``seat_id`` (or ``None``),
                  ``distance_px`` (or ``None``), and ``reason``.
    """
    if not points:
        return set(), []

    arr = np.asarray(points, dtype=np.float64)
    # cKDTree.query accepts k>=1 and returns scalar shapes when k==1; force 2-D.
    k = max(1, min(k_candidates, len(seats)))
    dists, idxs = tree.query(arr, k=k)
    if k == 1:
        dists = dists.reshape(-1, 1)
        idxs = idxs.reshape(-1, 1)

    seat_ids = seats["seat_id"].tolist()
    n = len(points)
    debug: list[dict | None] = [None] * n

    # Sort detections by their best (k=1) distance ascending so the closest
    # fits get first pick of seats.
    order = sorted(range(n), key=lambda i: dists[i, 0])
    occupied: set[str] = set()

    for i in order:
        best_d = float(dists[i, 0])
        chosen_sid: str | None = None
        chosen_d: float | None = None
        reason: str

        if best_d > max_dist:
            reason = f"nearest seat {seat_ids[idxs[i, 0]]} is {best_d:.1f}px > {max_dist:.0f}px"
        else:
            for j in range(k):
                d = float(dists[i, j])
                if d > max_dist:
                    break
                sid = seat_ids[idxs[i, j]]
                if sid not in occupied:
                    occupied.add(sid)
                    chosen_sid = sid
                    chosen_d = d
                    break
            if chosen_sid is None:
                reason = f"top {k} candidates all claimed (best {best_d:.1f}px)"
            else:
                reason = "ok"

        debug[i] = {
            "detection_index": i,
            "x": round(float(points[i][0]), 1),
            "y": round(float(points[i][1]), 1),
            "seat_id": chosen_sid,
            "distance_px": round(chosen_d, 2) if chosen_d is not None else None,
            "reason": reason,
        }

    return occupied, debug  # type: ignore[return-value]
