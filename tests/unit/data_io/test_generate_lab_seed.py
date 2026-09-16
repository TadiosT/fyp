import runpy
import sqlite3
import sys
from dataclasses import replace
from datetime import datetime, timedelta

import pandas as pd
import pytest

from campus_occupancy.tools import generate_lab_seed as gls
from campus_occupancy.data_io.database import save_readings_bulk
from campus_occupancy.simulation.wc_sim import CSV_COLUMNS, reading_from_ratio


def test_refuses_without_reset_when_log_exists(tmp_sim, scratch_db):
    assert gls.run(tmp_sim, 1, 3, 1, reset=True, sensors_only=False) == 0
    assert gls.run(tmp_sim, 1, 3, 1, reset=False, sensors_only=False) == 1


def test_missing_desks_csv(tmp_sim, tmp_path):
    cfg = replace(tmp_sim, desks_csv=str(tmp_path / "nope.csv"))
    assert gls.run(cfg, 1, 3, 1, reset=True, sensors_only=False) == 1


def test_log_and_history_shape(tmp_sim, scratch_db):
    assert gls.run(tmp_sim, days=1, history_days=14, seed=3, reset=True, sensors_only=False) == 0
    log = pd.read_csv(tmp_sim.log_csv)
    desks = pd.read_csv(tmp_sim.desks_csv)
    now = datetime.now().replace(second=0, microsecond=0)
    start = (now - timedelta(days=1)).replace(hour=0, minute=0)
    assert list(log.columns) == CSV_COLUMNS
    assert set(log.pc_id) == set(desks.pc_id)
    assert pd.Timestamp(log.timestamp.min()) == pd.Timestamp(start + timedelta(minutes=1))
    assert pd.Timestamp(log.timestamp.max()) == pd.Timestamp(now)
    assert log.groupby("timestamp").size().eq(len(desks)).all()

    h = pd.read_csv(tmp_sim.hourly_csv)
    h["hour"] = pd.to_datetime(h.hour)
    assert len(h) == 24 * 13 or len(h) == 24 * 14                   # history_days-1 .. history_days
    assert (h.in_use + h.idle + h.offline == len(desks)).all()
    assert h.hour.max() < start
    assert h[h.hour.dt.weekday < 5].in_use.mean() > h[h.hour.dt.weekday >= 5].in_use.mean()


def test_sensor_rows_written_at_expected_cadence(tmp_sim, scratch_db):
    gls.run(tmp_sim, days=1, history_days=2, seed=1, reset=True, sensors_only=False)
    rows = dict(sqlite3.connect(scratch_db).execute(
        "select location_name, count(*) from sensorreading group by 1").fetchall())
    log = pd.read_csv(tmp_sim.log_csv, usecols=["timestamp"])
    ts = pd.to_datetime(log.timestamp.unique())
    expected_minutes = int((ts.minute % gls.SENSOR_EVERY_MIN == 0).sum())
    for sid in ("WC-S01", "WC-S02"):
        assert rows[sid] >= expected_minutes                         # + hourly history rows


def test_no_sensor_rows_for_sensorless_lab(tmp_sim, scratch_db):
    cfg = replace(tmp_sim, sensors_csv=None)
    gls.run(cfg, days=1, history_days=1, seed=1, reset=True, sensors_only=False)
    assert sqlite3.connect(scratch_db).execute("select count(*) from sensorreading").fetchone()[0] == 0


def test_sensors_only_rebuilds_log_span_only(tmp_sim, scratch_db):
    gls.run(tmp_sim, days=1, history_days=2, seed=1, reset=True, sensors_only=False)
    old = datetime.now() - timedelta(days=30)
    save_readings_bulk([reading_from_ratio("WC-S01", 0.3, old)])
    conn = sqlite3.connect(scratch_db)
    before = conn.execute("select count(*) from sensorreading").fetchone()[0]
    assert gls.run(tmp_sim, 1, 2, 1, reset=False, sensors_only=True) == 0
    after = conn.execute("select count(*) from sensorreading").fetchone()[0]
    cutoff = (old + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    kept_old = conn.execute("select count(*) from sensorreading where timestamp < ?", (cutoff,)).fetchone()[0]
    # exactly the log-span rows are replaced; history rows and the old row survive
    assert kept_old == 1 and after == before


def test_sensors_only_requires_log_and_sensors(tmp_sim, scratch_db):
    assert gls.run(tmp_sim, 1, 2, 1, reset=False, sensors_only=True) == 1        # no log yet
    gls.run(replace(tmp_sim, sensors_csv=None), 1, 1, 1, reset=True, sensors_only=False)
    assert gls.run(replace(tmp_sim, sensors_csv=None), 1, 1, 1, reset=False, sensors_only=True) == 1


def test_save_swallows_db_errors(monkeypatch, capsys):
    monkeypatch.setattr(gls, "save_readings_bulk", lambda s: (_ for _ in ()).throw(RuntimeError("disk I/O error")))
    assert gls._save([{"ok": True}]) == 0
    assert "sensor write failed" in capsys.readouterr().out


def test_main_dispatches_by_lab(monkeypatch):
    seen = {}

    def fake_run(cfg, days, history_days, seed, reset, sensors_only):
        seen.update(lab=cfg.name, days=days, history_days=history_days, seed=seed, reset=reset, so=sensors_only)
        return 0

    monkeypatch.setattr(gls, "run", fake_run)
    assert gls.main(["--lab", "huxley", "--days", "3", "--seed", "9", "--reset"]) == 0
    assert seen == {"lab": "huxley", "days": 3, "history_days": 90, "seed": 9, "reset": True, "so": False}
    with pytest.raises(SystemExit):
        gls.main(["--lab", "unknown"])
