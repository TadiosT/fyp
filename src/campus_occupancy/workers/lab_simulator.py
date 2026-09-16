"""Live simulator for a lab (``--lab wc`` or ``--lab huxley``).

* Resumes from the last row per desk in the lab's minute log.
* **Catches up to the wall clock** on boot (replays gaps ≤ 48 h minute by
  minute, rebuilds the last 48 h for larger gaps) and then keeps the log's
  latest timestamp equal to the current minute, including after a sleep.
* For labs with virtual sensors, writes a reading per sensor every 5
  simulated minutes to the shared DB.
* At every hour boundary appends the completed hour to the lab's hourly
  history file and trims the minute log to the last 48 h.

Run:  python -m campus_occupancy.workers.lab_simulator --lab wc
(the launcher does this for both labs).
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

from campus_occupancy.config import SIM_CONFIGS, SimConfig
from campus_occupancy.data_io.database import init_db, save_readings_bulk
from campus_occupancy.paths import PROJECT_ROOT
from campus_occupancy.simulation.wc_sim import (
    CSV_COLUMNS, make_initial_state, make_user_pool, nearby_desks,
    sensor_snapshots, state_counts, step_desks,
)

SENSOR_EVERY_MIN = 5
RETENTION_H = 48
CATCHUP_MAX_H = 48


def load_initial_state(log_csv: str):
    # keep_default_na: pandas would otherwise turn the literal "N/A" into NaN,
    # leaving resumed desks with user_id=NaN instead of "N/A".
    df = pd.read_csv(log_csv, keep_default_na=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    latest = df.sort_values("timestamp").groupby("pc_id").last().reset_index()
    state = {}
    for _, row in latest.iterrows():
        ttl = str(row["session_ttl_remaining"]).strip()
        user = str(row["user_id"]).strip() or "N/A"
        state[row["pc_id"]] = {
            "state": row["state"], "user_id": user,
            "ttl": int(float(ttl)) if ttl not in ("N/A", "", "nan") else 0,
        }
    return state, latest["timestamp"].max().to_pydatetime()


class Worker:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        desks = pd.read_csv(cfg.desks_csv)
        self.pc_ids = sorted(desks["pc_id"].tolist())
        self.nearby: dict = {}
        if cfg.sensors_csv and os.path.exists(cfg.sensors_csv):
            self.nearby = nearby_desks(pd.read_csv(cfg.sensors_csv), desks)
            init_db()
        self.rng = random.Random()
        self.hour_acc: list[dict] = []
        self.pending_snaps: list[dict] = []

        now = datetime.now().replace(second=0, microsecond=0)
        self.state, self.t = (load_initial_state(cfg.log_csv) if os.path.exists(cfg.log_csv)
                              else (None, None))
        if self.t is None or (now - self.t) > timedelta(hours=CATCHUP_MAX_H):
            start = now - timedelta(hours=RETENTION_H)
            self.log(f"gap too large or no log; rebuilding last {RETENTION_H} h from {start}")
            self.state = make_initial_state(self.pc_ids)
            self.t = start
            os.makedirs(os.path.dirname(cfg.log_csv) or ".", exist_ok=True)
            with open(cfg.log_csv, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()
        # Reconcile with the desk CSV: seats added via the calibrator since the
        # log was written start Offline; seats removed from the CSV are dropped.
        before = set(self.state)
        self.state = {pc: self.state.get(pc, {"state": "Offline", "user_id": "N/A", "ttl": 0})
                      for pc in self.pc_ids}
        added, removed = set(self.pc_ids) - before, before - set(self.pc_ids)
        if added or removed:
            self.log(f"desk list reconciled: +{len(added)} new, -{len(removed)} removed")
        users_all = set(make_user_pool(len(self.state)))
        users_active = {d["user_id"] for d in self.state.values() if d["user_id"] != "N/A"}
        self.users = list(users_all - users_active)
        self.log(f"{len(self.state)} desks, {len(self.nearby)} sensors; sim time {self.t}, now {now}")

    def log(self, msg: str) -> None:
        print(f"[{datetime.now():%H:%M:%S}] {self.cfg.name}_simulator: {msg}", flush=True)

    def step(self) -> list[dict]:
        entries, self.t = step_desks(self.state, self.t, self.users, self.rng)
        self.hour_acc.append(state_counts(self.state))
        if self.nearby and self.t.minute % SENSOR_EVERY_MIN == 0:
            self.pending_snaps.extend(sensor_snapshots(self.nearby, self.state, self.t, self.rng))
        if self.t.minute == 0:
            self.roll_hour()
        return entries

    def roll_hour(self) -> None:
        if not self.hour_acc:
            return
        hour_start = (self.t - timedelta(hours=1)).replace(minute=0, second=0)
        mean = pd.DataFrame(self.hour_acc).mean().round().astype(int)
        row = {"hour": hour_start.strftime("%Y-%m-%d %H:%M:%S"),
               "in_use": int(mean["in_use"]), "idle": int(mean["idle"]), "offline": int(mean["offline"])}
        new = not os.path.exists(self.cfg.hourly_csv)
        with open(self.cfg.hourly_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["hour", "in_use", "idle", "offline"])
            if new:
                w.writeheader()
            w.writerow(row)
        self.hour_acc = []

    def flush_snaps(self) -> None:
        if not self.pending_snaps:
            return
        try:
            n = save_readings_bulk(self.pending_snaps)
            self.pending_snaps = []
            self.log(f"stored {n} sensor readings")
        except Exception as e:
            self.log(f"sensor write failed ({e.__class__.__name__}): {e}; will retry next tick")

    def advance_to(self, target: datetime) -> int:
        n = 0
        buf: list[dict] = []
        with open(self.cfg.log_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            while self.t < target:
                buf.extend(self.step())
                n += 1
                if len(buf) >= 50_000:
                    w.writerows(buf)
                    buf = []
            if buf:
                w.writerows(buf)
        return n

    def trim_log(self) -> None:
        cutoff = self.t - timedelta(hours=RETENTION_H)
        df = pd.read_csv(self.cfg.log_csv)
        keep = df[pd.to_datetime(df["timestamp"]) >= cutoff]
        if len(keep) < len(df):
            keep.to_csv(self.cfg.log_csv, index=False)
            self.log(f"trimmed minute log to last {RETENTION_H} h ({len(keep):,} rows)")

    def run(self) -> None:
        now = datetime.now().replace(second=0, microsecond=0)
        if self.t < now:
            self.log(f"catching up {int((now - self.t).total_seconds() // 60)} minutes…")
            self.advance_to(now)
            self.flush_snaps()
            self.trim_log()
            self.log("caught up")
        while True:
            try:
                now = datetime.now().replace(second=0, microsecond=0)
                if self.t < now:
                    self.advance_to(now)
                    self.flush_snaps()
                    if self.t.minute == 0:
                        self.trim_log()
                    c = state_counts(self.state)
                    self.log(f"tick -> {self.t:%H:%M}: in use {c['in_use']}/{len(self.state)}")
                time.sleep(max(1.0, 60 - datetime.now().second))
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.log(f"tick failed: {e!r}; retrying in 10s")
                time.sleep(10)


def main(lab: str | None = None) -> int:
    os.chdir(PROJECT_ROOT)
    if lab is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--lab", choices=sorted(SIM_CONFIGS), required=True)
        lab = ap.parse_args().lab
    cfg = SIM_CONFIGS[lab]
    if not os.path.exists(cfg.desks_csv):
        print(f"[{lab}_simulator] {cfg.desks_csv} missing. Calibrate the lab's desks first.")
        return 1
    Worker(cfg).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
