"""Optimistic preloading of lab dashboard data.

A daemon thread, started once per Streamlit server process from ``app.py``,
builds a :class:`Bundle` for each registered lab as soon as the app boots and
refreshes it every ``refresh_s`` seconds. Pages call :func:`get_preloaded`
and render instantly from the bundle; if nothing is preloaded yet (or the
lab isn't registered, e.g. Huxley) they fall back to building a bundle on
demand.

The thread uses only pandas/PIL, with no ``st.*`` calls, so it needs no
ScriptRunContext. Bundles are immutable once published (readers must not
mutate the DataFrames); publishing is a single dict assignment, so readers
always see a complete old or new bundle.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from PIL import Image


@dataclass
class Bundle:
    data_file: str
    df: pd.DataFrame                       # full minute log
    latest: pd.DataFrame                   # last row per pc_id
    history: pd.DataFrame | None           # hourly counts (index hour) or None
    coords: pd.DataFrame | None
    sensors: pd.DataFrame | None
    nearby: dict = field(default_factory=dict)      # sensor_id -> [pc_id]
    zone_map: dict = field(default_factory=dict)    # sensor_id -> zone
    floor_plan: Image.Image | None = None
    loaded_at: datetime = field(default_factory=datetime.now)
    load_seconds: float = 0.0
    source: str = "on-demand"              # "preloaded" | "on-demand"
    log_mtime: float = 0.0


def build_bundle(data_file: str, coords_file: str, floor_plan: str,
                 sensor_positions: str | None, history_file: str | None,
                 source: str = "on-demand") -> Bundle:
    """Read everything a lab page needs. Raises FileNotFoundError if the log is missing."""
    t0 = time.perf_counter()
    df = pd.read_csv(data_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["session_ttl_remaining"] = pd.to_numeric(
        df["session_ttl_remaining"], errors="coerce").astype("Int64")
    latest = df.sort_values("timestamp").groupby("pc_id").tail(1).reset_index(drop=True)

    history = None
    if history_file and os.path.exists(history_file):
        h = pd.read_csv(history_file)
        h["hour"] = pd.to_datetime(h["hour"])
        history = h.set_index("hour").sort_index()

    coords = pd.read_csv(coords_file) if os.path.exists(coords_file) else None
    sensors = pd.read_csv(sensor_positions) if sensor_positions and os.path.exists(sensor_positions) else None

    nearby, zone_map = {}, {}
    if coords is not None and sensors is not None:
        from campus_occupancy.simulation.wc_sim import nearby_desks, sensor_zone_map
        nearby = nearby_desks(sensors, coords)
        if "zone" in coords.columns:
            zone_map = sensor_zone_map(sensors, coords)

    img = None
    if os.path.exists(floor_plan):
        img = Image.open(floor_plan)
        img.load()  # decode now, not lazily inside the page render

    return Bundle(
        data_file=data_file, df=df, latest=latest, history=history, coords=coords,
        sensors=sensors, nearby=nearby, zone_map=zone_map, floor_plan=img,
        loaded_at=datetime.now(), load_seconds=time.perf_counter() - t0, source=source,
        log_mtime=os.path.getmtime(data_file),
    )


class Preloader:
    def __init__(self, cfgs: list, refresh_s: int = 60):
        self.cfgs = list(cfgs)
        self.refresh_s = refresh_s
        self._bundles: dict[str, Bundle] = {}
        self._errors: dict[str, str] = {}
        self._started = datetime.now()
        self._thread = threading.Thread(target=self._loop, name="lab-preloader", daemon=True)
        self._thread.start()

    def _log(self, msg: str) -> None:
        print(f"[{datetime.now():%H:%M:%S}] preload: {msg}", flush=True)

    def _refresh_one(self, cfg) -> None:
        key = cfg.data_file
        try:
            if not os.path.exists(key):
                self._errors[key] = "data file missing"
                return
            prev = self._bundles.get(key)
            mtime = os.path.getmtime(key)
            if prev is not None and prev.log_mtime == mtime:
                return  # nothing new on disk, keep the current bundle
            b = build_bundle(cfg.data_file, cfg.coords_file, cfg.floor_plan,
                             cfg.sensor_positions, cfg.history_file, source="preloaded")
            self._bundles[key] = b
            self._errors.pop(key, None)
            self._log(f"{os.path.basename(key)}: {len(b.df):,} rows, {len(b.latest)} desks, "
                      f"history={'yes' if b.history is not None else 'no'} in {b.load_seconds:.1f}s")
        except Exception as e:
            self._errors[key] = f"{e.__class__.__name__}: {e}"
            self._log(f"{os.path.basename(key)} failed: {self._errors[key]}")

    def _loop(self) -> None:
        self._log(f"started for {[os.path.basename(c.data_file) for c in self.cfgs]}, "
                  f"refresh every {self.refresh_s}s")
        while True:
            for cfg in self.cfgs:
                self._refresh_one(cfg)
            time.sleep(self.refresh_s)

    def get(self, data_file: str) -> Bundle | None:
        return self._bundles.get(data_file)

    def status(self, data_file: str) -> dict:
        b = self._bundles.get(data_file)
        return {
            "ready": b is not None,
            "loaded_at": b.loaded_at if b else None,
            "age_s": (datetime.now() - b.loaded_at).total_seconds() if b else None,
            "load_seconds": b.load_seconds if b else None,
            "rows": len(b.df) if b else None,
            "error": self._errors.get(data_file),
            "since_boot_s": (datetime.now() - self._started).total_seconds(),
        }


_PRELOADER: Preloader | None = None
_LOCK = threading.Lock()


def start_preloader(cfgs: list, refresh_s: int = 60) -> Preloader:
    """Idempotent: the first call per server process starts the thread."""
    global _PRELOADER
    with _LOCK:
        if _PRELOADER is None:
            _PRELOADER = Preloader(cfgs, refresh_s)
    return _PRELOADER


def get_preloaded(data_file: str) -> Bundle | None:
    return _PRELOADER.get(data_file) if _PRELOADER else None


def preload_status(data_file: str) -> dict | None:
    return _PRELOADER.status(data_file) if _PRELOADER else None
