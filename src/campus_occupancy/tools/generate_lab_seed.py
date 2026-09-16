"""Seed generator for a lab (White City or Huxley).

Produces:
1. the lab's per-desk, per-minute log from ``--days`` ago at 00:00 up to now
   (default 2 days; the live simulator keeps a rolling 48 h anyway);
2. the lab's hourly history (``--history-days``, default 90) generated
   statistically from the occupancy curve with a weekday/weekend factor,
   which drives the Week / Month / 3-Month charts;
3. (labs with virtual sensors) sensor rows in the shared SQLite DB: every
   5 minutes across the minute log, hourly across the history window.

Flags:
    --lab wc|huxley   which lab (paths come from campus_occupancy.config)
    --reset           delete the log, the hourly file and prior sensor rows first
    --sensors-only    only (re)build sensor rows from the *existing* minute log

Run:  python -m campus_occupancy.tools.generate_lab_seed --lab huxley --reset
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from datetime import datetime, timedelta

import pandas as pd

from campus_occupancy.config import SIM_CONFIGS, SimConfig
from campus_occupancy.data_io.database import delete_readings, init_db, save_readings_bulk
from campus_occupancy.paths import PROJECT_ROOT
from campus_occupancy.simulation.wc_sim import (
    CSV_COLUMNS, OCCUPANCY_CURVE, WEEKDAY_FACTOR, make_initial_state, make_user_pool,
    nearby_desks, reading_from_ratio, sensor_snapshots, state_counts, step_desks,
)

SENSOR_EVERY_MIN = 5


def _save(snaps: list[dict]) -> int:
    try:
        return save_readings_bulk(snaps)
    except Exception as e:  # never let a DB hiccup kill the seed
        print(f"  ! sensor write failed ({e.__class__.__name__}): {e}")
        return 0


def seed_minute_log(cfg: SimConfig, desks: pd.DataFrame, nearby: dict, start: datetime,
                    end: datetime, rng: random.Random) -> tuple[int, int]:
    pc_ids = sorted(desks["pc_id"].tolist())
    state = make_initial_state(pc_ids)
    users = make_user_pool(len(pc_ids))
    t = start
    total_minutes = int((end - start).total_seconds() // 60)
    n_rows = n_sensor = 0
    pending: list[dict] = []
    os.makedirs(os.path.dirname(cfg.log_csv) or ".", exist_ok=True)
    with open(cfg.log_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for _ in range(total_minutes):
            entries, t = step_desks(state, t, users, rng)
            w.writerows(entries)
            n_rows += len(entries)
            if nearby and t.minute % SENSOR_EVERY_MIN == 0:
                pending.extend(sensor_snapshots(nearby, state, t, rng))
            if t.minute == 0:
                if pending:
                    n_sensor += _save(pending)
                    pending = []
                c = state_counts(state)
                print(f"  {t:%Y-%m-%d %H:%M}  in use {c['in_use']}/{len(pc_ids)}")
    if pending:
        n_sensor += _save(pending)
    return n_rows, n_sensor


def seed_hourly_history(cfg: SimConfig, n_desks: int, sensors: pd.DataFrame | None,
                        start: datetime, end: datetime, rng: random.Random) -> tuple[int, int]:
    rows, snaps = [], []
    t = start
    day_noise: dict = {}
    while t < end:
        key = t.date()
        if key not in day_noise:
            day_noise[key] = rng.gauss(1.0, 0.08)
        expected = n_desks * OCCUPANCY_CURVE[t.hour] * WEEKDAY_FACTOR[t.weekday()] * day_noise[key]
        expected = max(0.0, min(n_desks, expected))
        in_use = int(round(expected * 0.9))
        idle = int(round(expected * 0.1))
        rows.append({"hour": t.strftime("%Y-%m-%d %H:%M:%S"), "in_use": in_use,
                     "idle": idle, "offline": n_desks - in_use - idle})
        if sensors is not None:
            r = expected / n_desks if n_desks else 0.0
            for sid in sensors["sensor_id"]:
                snaps.append(reading_from_ratio(sid, r, t, rng))
        t += timedelta(hours=1)
    pd.DataFrame(rows).to_csv(cfg.hourly_csv, index=False)
    return len(rows), (_save(snaps) if snaps else 0)


def backfill_sensors_from_log(cfg: SimConfig, nearby: dict, rng: random.Random) -> int:
    df = pd.read_csv(cfg.log_csv, usecols=["timestamp", "pc_id", "session_ttl_remaining"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["timestamp"].dt.minute % SENSOR_EVERY_MIN == 0]
    active = df[df["session_ttl_remaining"].astype(str) != "N/A"]
    snaps = []
    for ts, g in active.groupby("timestamp"):
        state = {pc: {"ttl": 1} for pc in g["pc_id"]}
        snaps.extend(sensor_snapshots(nearby, state, ts.to_pydatetime(), rng))
    for ts in set(df["timestamp"].unique()) - set(active["timestamp"].unique()):
        snaps.extend(sensor_snapshots(nearby, {}, pd.Timestamp(ts).to_pydatetime(), rng))
    return _save(snaps)


def run(cfg: SimConfig, days: int, history_days: int, seed: int, reset: bool, sensors_only: bool) -> int:
    if not os.path.exists(cfg.desks_csv):
        print(f"Missing {cfg.desks_csv}. Calibrate the lab's desks first.")
        return 1
    desks = pd.read_csv(cfg.desks_csv)
    sensors = None
    if cfg.sensors_csv and os.path.exists(cfg.sensors_csv):
        sensors = pd.read_csv(cfg.sensors_csv)
    nearby = nearby_desks(sensors, desks) if sensors is not None else {}
    if cfg.sensors_csv and sensors is None:
        print(f"No {cfg.sensors_csv}; sensor rows will be skipped.")

    init_db()
    rng = random.Random(seed)
    now = datetime.now().replace(second=0, microsecond=0)

    if sensors_only:
        if not os.path.exists(cfg.log_csv) or not nearby:
            print("Need both the minute log and sensor positions for --sensors-only.")
            return 1
        log_start = pd.to_datetime(pd.read_csv(cfg.log_csv, usecols=["timestamp"])["timestamp"]).min()
        # log timestamps are naive *local* time; make it aware so the DB layer converts to UTC
        deleted = delete_readings(list(nearby), since=log_start.to_pydatetime().astimezone())
        n = backfill_sensors_from_log(cfg, nearby, rng)
        print(f"Replaced {deleted} sensor rows with {n} rebuilt from {cfg.log_csv}.")
        return 0

    if reset:
        for p in (cfg.log_csv, cfg.hourly_csv):
            if os.path.exists(p):
                os.remove(p)
        if nearby:
            print(f"Deleted {delete_readings(list(nearby))} prior sensor rows.")
    elif os.path.exists(cfg.log_csv):
        print(f"{cfg.log_csv} already exists. Use --reset to regenerate, or --sensors-only.")
        return 1

    log_start = (now - timedelta(days=days)).replace(hour=0, minute=0)
    hist_start = (now - timedelta(days=history_days)).replace(hour=0, minute=0)

    print(f"[{cfg.name}] history: {hist_start:%Y-%m-%d} → {log_start:%Y-%m-%d} hourly")
    h_rows, h_snaps = seed_hourly_history(cfg, len(desks), sensors, hist_start, log_start, rng)
    print(f"  wrote {h_rows} hourly rows to {cfg.hourly_csv}, {h_snaps} sensor rows")

    print(f"[{cfg.name}] minute log: {len(desks)} desks, {len(nearby)} sensors, {log_start} → {now}")
    n_rows, n_snaps = seed_minute_log(cfg, desks, nearby, log_start, now, rng)
    print(f"  wrote {n_rows:,} rows to {cfg.log_csv}, {n_snaps} sensor rows")
    return 0


def main(argv: list[str] | None = None) -> int:
    os.chdir(PROJECT_ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", choices=sorted(SIM_CONFIGS), required=True)
    ap.add_argument("--days", type=int, default=2, help="minute-log span (from 00:00 that many days ago)")
    ap.add_argument("--history-days", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--sensors-only", action="store_true")
    a = ap.parse_args(argv)
    return run(SIM_CONFIGS[a.lab], a.days, a.history_days, a.seed, a.reset, a.sensors_only)


if __name__ == "__main__":
    sys.exit(main())
