"""Synthetic data builders shared by the test modules."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from campus_occupancy.simulation.wc_sim import CSV_COLUMNS

# A fixed weekday (Wednesday) and weekend day (Saturday) so tests don't depend on today.
WED = datetime(2026, 9, 2, 0, 0)
SAT = datetime(2026, 9, 5, 0, 0)


def grid_desks(n_cols: int = 5, n_rows: int = 4, zone: str = "1a", prefix: str | None = None,
               spacing: float = 50.0, origin: tuple[float, float] = (0.0, 0.0),
               with_zone: bool = True) -> pd.DataFrame:
    prefix = prefix or f"WC{zone.upper()}"
    rows = []
    for r in range(n_rows):
        for c in range(n_cols):
            i = r * n_cols + c + 1
            rows.append({"pc_id": f"{prefix}-{i:03d}", "x_px": origin[0] + c * spacing,
                         "y_px": origin[1] + r * spacing, "zone": zone})
    df = pd.DataFrame(rows)
    return df if with_zone else df.drop(columns=["zone"])


def sensors_df(points: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"sensor_id": f"WC-S{i + 1:02d}", "x_px": x, "y_px": y}
                         for i, (x, y) in enumerate(points)])


def write_log(path: str | Path, pc_ids, ts: datetime, state: str = "Offline",
              user: str = "N/A", ttl="N/A", append: bool = False) -> None:
    mode = "a" if append else "w"
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not append:
            w.writeheader()
        for pc in pc_ids:
            w.writerow({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "pc_id": pc,
                        "state": state, "user_id": user, "session_ttl_remaining": ttl})


def tiny_image(path: str | Path, w: int = 200, h: int = 300) -> None:
    Image.new("RGB", (w, h), (240, 235, 220)).save(path)


def synthetic_frame(w: int = 64, h: int = 48, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def tiny_video(path: str | Path, w: int = 64, h: int = 48, n_frames: int = 10, fps: float = 10.0) -> None:
    import cv2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for i in range(n_frames):
            writer.write(synthetic_frame(w, h, seed=i))
    finally:
        writer.release()


def consistent_reading(desks: pd.DataFrame, zone: str, noise_extra: int = 0) -> dict:
    """A sensor reading consistent with the zone's current occupancy (wc_sim model)."""
    g = desks[desks["zone"] == zone]
    r = float(g["in_use_now"].mean()) if len(g) else 0.0
    return {"co2": int(420 + 900 * r), "noise": int(32 + 25 * r) + noise_extra,
            "temperature": round(20 + 3 * r, 1)}
