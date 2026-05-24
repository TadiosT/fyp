"""Zone-based seating heuristic for Lecture Hall 144.

The floor plan is 1864×1048. Zones are horizontal bands in floor-plan pixel
space; list order is front-to-back (which drives the "front-most free zone"
recommendation). NOTE: in the lecture-hall image the *front* of the hall is at
the bottom (large y) and the *back* is at the top (small y), so Front has the
largest y-range.
"""
from __future__ import annotations

ZONES: list[dict] = [
    {"name": "Front",  "y_min": 699, "y_max": 1048, "capacity": 30},
    {"name": "Middle", "y_min": 349, "y_max":  699, "capacity": 40},
    {"name": "Back",   "y_min":   0, "y_max":  349, "capacity": 50},
]


def assess_zones(
    points: list[tuple[float, float]],
    zones: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """Count detections per zone; return enriched zone list and the recommended zone.

    Each returned zone dict gains two new keys:
    - ``count``  — number of detected people whose y_px falls in the band
    - ``free``   — max(0, capacity - count)

    The recommended zone is the front-most zone with at least one free seat.
    Falls back to the last zone (Back) when all zones are full.
    """
    if zones is None:
        import copy
        zones = copy.deepcopy(ZONES)

    for z in zones:
        z["count"] = sum(1 for (_, y) in points if z["y_min"] <= y < z["y_max"])
        z["free"] = max(0, z["capacity"] - z["count"])

    recommended = next((z for z in zones if z["free"] > 0), zones[-1])
    return zones, recommended


def zone_shapes(recommended: dict) -> list[dict]:
    """Return a list containing one Plotly shape: a gold highlight for *recommended*."""
    return [
        dict(
            type="rect",
            xref="x",
            yref="y",
            x0=0,
            x1=1864,
            y0=recommended["y_min"],
            y1=recommended["y_max"],
            fillcolor="rgba(255, 215, 0, 0.18)",
            line=dict(color="rgba(255, 215, 0, 0.6)", width=2),
            layer="above",
        )
    ]
