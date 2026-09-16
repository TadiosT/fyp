"""Zone-based seating heuristic for Lecture Hall 144 (discrete-seat version).

Zones are horizontal y-bands across the top-down floor plan (1110×1330). List
order is **front-to-back** since the "front-most band with free seats" logic
relies on iteration order. In the lecture-hall image the *front* of the hall
sits at large y (bottom of the image), so Front has the largest y-range.

Capacities are no longer hardcoded; they're derived per-tick from the seat
CSV. Edit the seat map (rerun ``tools/calibrate_lecture_seats.py``) and
capacities update automatically.
"""
from __future__ import annotations

import copy

import pandas as pd

# Floor plan is 1110×1330. Splits at ⅓ / ⅔ of the height. Tune by eye once
# the seat CSV exists by widening / narrowing these bands.
FLOOR_PLAN_W = 1110
FLOOR_PLAN_H = 1330

ZONES: list[dict] = [
    {"name": "Front",  "y_min": 887, "y_max": FLOOR_PLAN_H},
    {"name": "Middle", "y_min": 443, "y_max": 887},
    {"name": "Back",   "y_min":   0, "y_max": 443},
]


def _band_of(y: float, zones: list[dict]) -> dict | None:
    for z in zones:
        if z["y_min"] <= y < z["y_max"]:
            return z
    return None


def assess_zones(
    seats: pd.DataFrame,
    occupied: set[str],
    zones: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """Bucket seats into front/middle/back bands and count occupancy.

    Args:
        seats: DataFrame with at least ``seat_id`` and ``y_px`` columns.
        occupied: set of seat_ids currently in use this tick.
        zones: optional override of the module-level ``ZONES`` list.

    Returns:
        ``(zones_enriched, recommended)``: each zone dict gains ``capacity``
        (total seats in band), ``count`` (occupied seats in band), and
        ``free`` (capacity - count). ``recommended`` is the front-most zone
        with ``free > 0``, falling back to the last zone if everything is full.
    """
    if zones is None:
        zones = copy.deepcopy(ZONES)

    # Reset counters
    for z in zones:
        z["capacity"] = 0
        z["count"] = 0
        z["free"] = 0

    # Single pass over seats: assign each to its band, increment capacity and
    # (if occupied) count.
    for row in seats.itertuples(index=False):
        z = _band_of(float(row.y_px), zones)
        if z is None:
            continue
        z["capacity"] += 1
        if row.seat_id in occupied:
            z["count"] += 1

    for z in zones:
        z["free"] = max(0, z["capacity"] - z["count"])

    recommended = next((z for z in zones if z["free"] > 0), zones[-1])
    return zones, recommended


def zone_shapes(recommended: dict) -> list[dict]:
    """Return a single translucent gold rectangle highlighting *recommended*."""
    return [
        dict(
            type="rect",
            xref="x",
            yref="y",
            x0=0,
            x1=FLOOR_PLAN_W,
            y0=recommended["y_min"],
            y1=recommended["y_max"],
            fillcolor="rgba(255, 215, 0, 0.18)",
            line=dict(color="rgba(255, 215, 0, 0.6)", width=2),
            layer="above",
        )
    ]
