import sqlite3
import time
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.exc import OperationalError

import campus_occupancy.data_io.database as db
from campus_occupancy.data_io.database import (
    SensorReading, _retry, delete_readings, latest_reading, prune_older_than, readings_between,
    save_reading, save_readings_bulk, utc,
)


def snap(loc="Bedroom", when=None, **over):
    when = when or datetime.now(timezone.utc)
    base = {"ok": True, "error": None, "device_name": loc, "temperature": 21.5, "humidity": 50,
            "co2": 700, "noise": 38, "time_utc": int(when.timestamp())}
    base.update(over)
    return base


def test_init_creates_table_in_override_path(scratch_db):
    conn = sqlite3.connect(scratch_db)
    tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    assert "sensorreading" in tables
    assert conn.execute("pragma journal_mode").fetchone()[0] == "delete"
    assert scratch_db.startswith("/private") or "test.db" in scratch_db


def test_save_reading_noops_on_unhealthy(scratch_db):
    save_reading({"ok": False, "time_utc": 1}, "x")
    save_reading(snap(time_utc=None), "x")
    assert latest_reading() is None


def test_round_trip_and_utc(scratch_db):
    when = datetime(2026, 9, 1, 12, 30, tzinfo=timezone.utc)
    save_reading(snap(when=when), "Bedroom")
    row = latest_reading("Bedroom")
    assert (row.device_name, row.temperature, row.humidity, row.co2, row.noise) == ("Bedroom", 21.5, 50, 700, 38)
    assert row.timestamp.tzinfo is None and row.timestamp == when.replace(tzinfo=None)
    assert utc(row.timestamp) == when
    assert utc(when) is when


def test_bulk_insert_counts_and_skips(scratch_db):
    n = save_readings_bulk([snap("A"), snap("B"), {"ok": False, "device_name": "C", "time_utc": 1},
                            snap("D", time_utc=None)])
    assert n == 2 and save_readings_bulk([]) == 0
    assert {latest_reading("A").location_name, latest_reading("B").location_name} == {"A", "B"}


def test_latest_reading_filters_and_orders(scratch_db):
    t0 = datetime.now(timezone.utc)
    save_reading(snap("A", when=t0 - timedelta(hours=2), co2=1), "A")
    save_reading(snap("A", when=t0 - timedelta(hours=1), co2=2), "A")
    save_reading(snap("B", when=t0, co2=3), "B")
    assert latest_reading("A").co2 == 2
    assert latest_reading().co2 == 3
    assert latest_reading("nope") is None


def test_readings_between_window(scratch_db):
    t0 = datetime.now(timezone.utc).replace(microsecond=0)
    for h in range(5):
        save_reading(snap("A", when=t0 - timedelta(hours=h), co2=h), "A")
    df = readings_between("A", t0 - timedelta(hours=2), t0)
    assert list(df.columns) == ["timestamp", "temperature", "humidity", "co2", "noise"]
    assert sorted(df.co2) == [0, 1, 2]
    assert df.timestamp.iloc[0].tzinfo is not None
    naive = readings_between("A", (t0 - timedelta(hours=2)).replace(tzinfo=None), t0.replace(tzinfo=None))
    assert len(naive) == 3
    assert readings_between("B", t0 - timedelta(days=1), t0).empty


def test_delete_readings_scoped_by_since(scratch_db):
    t0 = datetime.now(timezone.utc)
    save_reading(snap("A", when=t0 - timedelta(days=10)), "A")
    save_reading(snap("A", when=t0), "A")
    save_reading(snap("B", when=t0), "B")
    assert delete_readings(["A"], since=t0 - timedelta(days=1)) == 1
    assert latest_reading("A").timestamp < (t0 - timedelta(days=9)).replace(tzinfo=None)
    assert delete_readings(["A", "B"]) == 2 and latest_reading() is None


def test_prune_older_than(scratch_db):
    t0 = datetime.now(timezone.utc)
    save_reading(snap("A", when=t0 - timedelta(days=91)), "A")
    save_reading(snap("A", when=t0 - timedelta(days=89)), "A")
    assert prune_older_than(90) == 1
    assert prune_older_than(90) == 0 and latest_reading("A") is not None


def test_retry_recovers_then_gives_up(monkeypatch):
    monkeypatch.setattr(db.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise OperationalError("stmt", {}, Exception("locked"))
        return "ok"

    assert _retry(flaky) == "ok" and calls["n"] == 3

    def always():
        raise OperationalError("stmt", {}, Exception("locked"))

    with pytest.raises(OperationalError):
        _retry(always, attempts=2)


def test_engine_is_lazy_singleton(scratch_db):
    e1 = db.get_engine()
    assert db.get_engine() is e1


def test_env_override_is_read_at_import(tmp_path):
    """Reloading the module in-process would redefine the SQLModel table, so
    check the env override in a fresh interpreter instead."""
    import os
    import subprocess
    import sys
    env = {**os.environ, "FYP_DB_PATH": str(tmp_path / "elsewhere.db")}
    out = subprocess.run([sys.executable, "-c", "import campus_occupancy.data_io.database as d; print(d.DB_PATH)"],
                         capture_output=True, text=True, env=env, cwd=os.getcwd())
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == str(tmp_path / "elsewhere.db")


def test_utc_converts_aware_non_utc_input():
    from datetime import timedelta as td
    bst = timezone(td(hours=1))
    local = datetime(2026, 9, 6, 13, 0, tzinfo=bst)
    assert utc(local) == datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)


def test_delete_and_window_use_utc_for_aware_local_inputs(scratch_db):
    """Regression: aware local timestamps must be converted, not just stripped."""
    from datetime import timedelta as td
    bst = timezone(td(hours=1))
    stored = datetime(2026, 9, 6, 12, 30, tzinfo=timezone.utc)          # 13:30 BST
    save_reading(snap("A", when=stored), "A")
    assert len(readings_between("A", datetime(2026, 9, 6, 13, 0, tzinfo=bst), datetime(2026, 9, 6, 14, 0, tzinfo=bst))) == 1
    assert delete_readings(["A"], since=datetime(2026, 9, 6, 13, 0, tzinfo=bst)) == 1
