"""Simulation engine for the White City labs: desks and virtual Netatmo sensors.

Pure functions, no Streamlit and no I/O, shared by the seed generator
(``tools/generate_lab_seed.py``), the live worker (``workers/lab_simulator.py``) and the
dashboard's always-on sensor readings (``lib/lab_dashboard.py``).

Desk state machine mirrors ``simulator.py`` (Huxley): sessions with a 15–120
minute TTL, 90/10 In Use / Idle flips while a session is active, an anonymous
user pool. Arrivals (and early departures) are steered so the In-Use fraction
tracks a time-of-day curve instead of a flat 5 %/min arrival rate.

Virtual sensors take the desks within ``RADIUS_PX`` and turn the local
occupancy ratio into Netatmo-shaped readings.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CSV_COLUMNS = ["timestamp", "pc_id", "state", "user_id", "session_ttl_remaining"]

# Target fraction of desks with an active session, by hour of day.
OCCUPANCY_CURVE = {
    0: .02, 1: .02, 2: .02, 3: .02, 4: .02, 5: .02, 6: .03, 7: .06,
    8: .15, 9: .35, 10: .55, 11: .65, 12: .55, 13: .60, 14: .70, 15: .65,
    16: .50, 17: .35, 18: .20, 19: .12, 20: .10, 21: .08, 22: .05, 23: .03,
}
# Monday=0 … Sunday=6
WEEKDAY_FACTOR = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.9, 5: 0.25, 6: 0.15}

MEAN_SESSION_MIN = 67.5
GAIN_ARRIVE = 0.30
GAIN_DEPART = 0.15
RADIUS_PX = 300
MIN_NEARBY = 20


def target_fraction(ts: datetime) -> float:
    h, frac = ts.hour, ts.minute / 60.0
    a, b = OCCUPANCY_CURVE[h], OCCUPANCY_CURVE[(h + 1) % 24]
    return (a + (b - a) * frac) * WEEKDAY_FACTOR[ts.weekday()]


def make_initial_state(pc_ids) -> dict:
    return {pc: {"state": "Offline", "user_id": "N/A", "ttl": 0} for pc in pc_ids}


def make_user_pool(n: int) -> list[str]:
    return [f"AnonUser_{i:04d}" for i in range(1, n + 1)]


def step_desks(state, current_time, available_users, rng=None):
    """Advance one minute; mutates state/available_users; returns (rows, new_time)."""
    rng = rng or random
    current_time = current_time + timedelta(minutes=1)
    target = target_fraction(current_time)

    n = len(state)
    active = sum(1 for d in state.values() if d["ttl"] > 0)
    ratio = active / n if n else 0.0
    empty = n - active

    p_hold = (target * n / MEAN_SESSION_MIN) / max(empty, 1)
    p_arrive = min(1.0, max(0.0, p_hold + GAIN_ARRIVE * max(0.0, target - ratio)))
    p_depart = min(1.0, GAIN_DEPART * max(0.0, ratio - target))

    entries = []
    for pc_id, data in state.items():
        if data["ttl"] > 0:
            data["ttl"] -= 1
            if data["ttl"] <= 0 or rng.random() < p_depart:
                available_users.append(data["user_id"])
                data["state"], data["user_id"], data["ttl"] = "Offline", "N/A", 0
            else:
                data["state"] = "In Use" if rng.random() < 0.90 else "Idle"
        else:
            if available_users and rng.random() < p_arrive:
                user = available_users.pop(rng.randrange(len(available_users)))
                data["state"], data["user_id"] = "In Use", user
                data["ttl"] = rng.randint(15, 120)
            else:
                data["state"] = "Offline"
        entries.append({
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "pc_id": pc_id,
            "state": data["state"],
            "user_id": data["user_id"],
            "session_ttl_remaining": data["ttl"] if data["user_id"] != "N/A" else "N/A",
        })
    return entries, current_time


def state_counts(state: dict) -> dict:
    c = {"in_use": 0, "idle": 0, "offline": 0}
    for d in state.values():
        c["in_use" if d["state"] == "In Use" else "idle" if d["state"] == "Idle" else "offline"] += 1
    return c


def nearby_desks(sensors_df: pd.DataFrame, desks_df: pd.DataFrame) -> dict[str, list[str]]:
    dx = desks_df[["x_px", "y_px"]].to_numpy(dtype=float)
    out = {}
    for s in sensors_df.itertuples(index=False):
        d = np.hypot(dx[:, 0] - float(s.x_px), dx[:, 1] - float(s.y_px))
        idx = np.where(d <= RADIUS_PX)[0]
        if len(idx) < MIN_NEARBY:
            idx = np.argsort(d)[:MIN_NEARBY]
        out[s.sensor_id] = desks_df["pc_id"].iloc[idx].tolist()
    return out


def sensor_zone_map(sensors_df: pd.DataFrame, desks_df: pd.DataFrame) -> dict[str, str]:
    """sensor_id → the zone most of its nearby desks belong to."""
    nb = nearby_desks(sensors_df, desks_df)
    zone_of = dict(zip(desks_df["pc_id"], desks_df.get("zone", pd.Series(["?"] * len(desks_df)))))
    out = {}
    for sid, pcs in nb.items():
        zones = pd.Series([zone_of.get(pc, "?") for pc in pcs])
        out[sid] = zones.mode().iloc[0] if not zones.empty else "?"
    return out


def reading_from_ratio(sensor_id: str, r: float, ts: datetime, rng=None) -> dict:
    rng = rng or random
    daily = math.sin(2 * math.pi * (ts.hour + ts.minute / 60 - 9) / 24)
    return {
        "ok": True,
        "error": None,
        "device_name": sensor_id,
        "temperature": round(20.0 + 3.0 * r + 0.8 * daily + rng.gauss(0, 0.15), 1),
        "humidity": int(round(45 + 10 * r + rng.gauss(0, 2))),
        "co2": int(round(420 + 900 * r + rng.gauss(0, 25))),
        "noise": int(round(32 + 25 * r + rng.gauss(0, 1.5))),
        "time_utc": int(ts.timestamp()),
    }


def sensor_snapshots(nearby: dict[str, list[str]], state: dict, ts: datetime, rng=None) -> list[dict]:
    snaps = []
    for sensor_id, pcs in nearby.items():
        r = (sum(1 for pc in pcs if state.get(pc, {}).get("ttl", 0) > 0) / len(pcs)) if pcs else 0.0
        snaps.append(reading_from_ratio(sensor_id, r, ts, rng))
    return snaps
