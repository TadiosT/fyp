from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:          # so `tests.support.helpers` imports from any cwd
    sys.path.insert(0, str(ROOT))

import campus_occupancy.data_io.database as db  # noqa: E402
import campus_occupancy.data_io.preload as preload  # noqa: E402
from campus_occupancy.config import SimConfig  # noqa: E402
from campus_occupancy.dashboard.lab_dashboard import LabConfig  # noqa: E402
from tests.support.helpers import grid_desks, sensors_df, tiny_image  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(autouse=True)
def restore_cwd():
    """Several modules os.chdir() at import; keep each test's cwd predictable."""
    cwd = os.getcwd()
    os.chdir(ROOT)
    yield
    os.chdir(cwd)


def _reset_engine() -> None:
    if db._engine is not None:
        db._engine.dispose()
    db._engine = None


@pytest.fixture(autouse=True)
def clear_st_caches():
    """Streamlit caches are process-global; keep tests independent."""
    import streamlit as st
    st.cache_data.clear()
    st.cache_resource.clear()
    preload._PRELOADER = None
    yield
    st.cache_data.clear()
    st.cache_resource.clear()
    preload._PRELOADER = None


@pytest.fixture
def scratch_db(tmp_path, monkeypatch):
    """A fresh SQLite file per test; the app's DB is never touched."""
    _reset_engine()
    path = str(tmp_path / "test.db")
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    yield path
    _reset_engine()


class _Secrets:
    def __init__(self, data: dict):
        self._d = data

    def get(self, key, default=None):
        return self._d.get(key, default)

    def __getitem__(self, key):
        return self._d[key]


@pytest.fixture
def fake_secrets(monkeypatch):
    """Factory: fake_secrets({'access_token': '...'}) installs a fake st.secrets."""
    import streamlit

    def _set(netatmo: dict | None):
        monkeypatch.setattr(streamlit, "secrets", _Secrets({} if netatmo is None else {"netatmo": netatmo}))

    return _set


@pytest.fixture(scope="session")
def synthetic_lab(tmp_path_factory):
    """A hermetic lab: 60 desks in two zones, 2 sensors, a floor plan, one day of
    minute log, 10 days of hourly history and sensor rows in its own DB."""
    d = tmp_path_factory.mktemp("lab")
    desks = pd.concat([
        grid_desks(6, 5, "1a", origin=(20, 20)),        # y 20..220
        grid_desks(6, 5, "2b", origin=(20, 1000)),      # y 1000..1200
    ], ignore_index=True)
    desks.to_csv(d / "desks.csv", index=False)
    sensors_df([(145.0, 120.0), (145.0, 1100.0)]).to_csv(d / "sensors.csv", index=False)
    tiny_image(d / "plan.png", 300, 1300)

    sim = SimConfig(name="synthetic", desks_csv=str(d / "desks.csv"), log_csv=str(d / "log.csv"),
                    hourly_csv=str(d / "hourly.csv"), sensors_csv=str(d / "sensors.csv"))
    db_path = str(d / "lab.db")

    prev = db.DB_PATH
    _reset_engine()
    db.DB_PATH = db_path
    try:
        from campus_occupancy.tools.generate_lab_seed import run
        assert run(sim, days=1, history_days=10, seed=1, reset=True, sensors_only=False) == 0
    finally:
        _reset_engine()
        db.DB_PATH = prev

    page = LabConfig(title="Synthetic Lab - Occupancy Dashboard", data_file=sim.log_csv,
                     coords_file=sim.desks_csv, floor_plan=str(d / "plan.png"),
                     calibrate_hint="n/a", map_height=400, sensor_positions=sim.sensors_csv,
                     history_file=sim.hourly_csv, missing_data_hint="Seed the synthetic lab.")
    return SimpleNamespace(dir=d, sim=sim, page=page, db_path=db_path, n_desks=60, n_sensors=2,
                           zones=("1a", "2b"))


@pytest.fixture
def lab_db(synthetic_lab, monkeypatch):
    """Point campus_occupancy.data_io.database at the synthetic lab's DB for the duration of a test."""
    _reset_engine()
    monkeypatch.setattr(db, "DB_PATH", synthetic_lab.db_path)
    yield synthetic_lab.db_path
    _reset_engine()


@pytest.fixture
def patched_lab_configs(synthetic_lab, lab_db, monkeypatch):
    """Re-point the app's LabConfigs at the synthetic lab (pages run in-process)."""
    import campus_occupancy.config as lc
    for cfg, sensors in ((lc.WC_CFG, synthetic_lab.page.sensor_positions), (lc.HUXLEY_CFG, None)):
        monkeypatch.setattr(cfg, "data_file", synthetic_lab.page.data_file)
        monkeypatch.setattr(cfg, "coords_file", synthetic_lab.page.coords_file)
        monkeypatch.setattr(cfg, "floor_plan", synthetic_lab.page.floor_plan)
        monkeypatch.setattr(cfg, "history_file", synthetic_lab.page.history_file)
        monkeypatch.setattr(cfg, "sensor_positions", sensors)
        monkeypatch.setattr(cfg, "map_height", 400)
    return synthetic_lab


@pytest.fixture
def tmp_sim(synthetic_lab, tmp_path) -> SimConfig:
    """A SimConfig sharing the synthetic desks/sensors but with fresh log/hourly paths."""
    return replace(synthetic_lab.sim, log_csv=str(tmp_path / "log.csv"),
                   hourly_csv=str(tmp_path / "hourly.csv"))
