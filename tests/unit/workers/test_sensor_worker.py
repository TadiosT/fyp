import logging
from datetime import datetime, timezone

import pytest

from campus_occupancy.workers import sensor_worker as sw
from campus_occupancy.data_io.database import latest_reading


def healthy(name="Bedroom"):
    return {"ok": True, "error": None, "device_name": name, "temperature": 20.5, "humidity": 55,
            "co2": 640, "noise": 36, "time_utc": int(datetime.now(timezone.utc).timestamp())}


def test_resolve_location_prefers_secret(fake_secrets):
    fake_secrets({"location_name": " Living Room "})
    assert sw.resolve_location_name(healthy()) == "Living Room"


def test_resolve_location_falls_back_to_device_then_default(fake_secrets):
    fake_secrets({"location_name": ""})
    assert sw.resolve_location_name(healthy("Bedroom")) == "Bedroom"
    assert sw.resolve_location_name({"device_name": None}) == sw.LOCATION_FALLBACK


def test_resolve_location_when_secrets_unavailable(monkeypatch):
    import streamlit
    monkeypatch.setattr(streamlit, "secrets", None)
    assert sw.resolve_location_name(healthy("X")) == "X"


def test_tick_saves_healthy_reading(scratch_db, fake_secrets, monkeypatch, caplog):
    fake_secrets({"location_name": "Study"})
    monkeypatch.setattr(sw, "fetch_homecoach_snapshot", lambda: healthy("Bedroom"))
    caplog.set_level(logging.INFO, logger="sensor_worker")
    sw.tick()
    row = latest_reading("Study")
    assert row is not None and row.device_name == "Bedroom" and row.co2 == 640
    assert any("saved Study" in r.message for r in caplog.records)


def test_tick_logs_unhealthy_and_saves_nothing(scratch_db, fake_secrets, monkeypatch, caplog):
    fake_secrets({})
    monkeypatch.setattr(sw, "fetch_homecoach_snapshot",
                        lambda: {"ok": False, "error": "Netatmo API unreachable.", "time_utc": None})
    caplog.set_level(logging.WARNING, logger="sensor_worker")
    sw.tick()
    assert latest_reading() is None
    assert any("unhealthy" in r.message and "unreachable" in r.message for r in caplog.records)


def test_tick_reports_prune_count(scratch_db, fake_secrets, monkeypatch, caplog):
    fake_secrets({})
    monkeypatch.setattr(sw, "fetch_homecoach_snapshot", lambda: healthy())
    monkeypatch.setattr(sw, "prune_older_than", lambda days: 3)
    caplog.set_level(logging.INFO, logger="sensor_worker")
    sw.tick()
    assert any("pruned 3 rows" in r.message for r in caplog.records)


def test_tick_uses_unwrapped_fetch_when_available(scratch_db, fake_secrets, monkeypatch):
    fake_secrets({})
    calls = []

    def wrapped():
        calls.append("cached")
        return healthy()

    wrapped.__wrapped__ = lambda: (calls.append("raw") or healthy())
    monkeypatch.setattr(sw, "fetch_homecoach_snapshot", wrapped)
    sw.tick()
    assert calls == ["raw"]


def test_constants():
    assert sw.POLL_INTERVAL_S == 300 and sw.RETENTION_DAYS == 90


def test_main_loop_runs_tick_then_exits_on_interrupt(scratch_db, fake_secrets, monkeypatch, caplog):
    fake_secrets({})
    monkeypatch.setattr(sw, "fetch_homecoach_snapshot", lambda: healthy())
    monkeypatch.setattr(sw.time, "sleep", lambda s: (_ for _ in ()).throw(KeyboardInterrupt))
    caplog.set_level(logging.INFO, logger="sensor_worker")
    sw.main()
    assert latest_reading("Bedroom") is not None
    msgs = [r.message for r in caplog.records]
    assert any("starting" in m for m in msgs) and any("shutting down" in m for m in msgs)


def test_main_loop_survives_tick_exception(scratch_db, monkeypatch, caplog):
    monkeypatch.setattr(sw, "tick", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(sw.time, "sleep", lambda s: (_ for _ in ()).throw(KeyboardInterrupt))
    caplog.set_level(logging.INFO, logger="sensor_worker")
    sw.main()
    assert any("tick failed" in r.message for r in caplog.records)


def test_main_loop_interrupt_inside_tick(scratch_db, monkeypatch, caplog):
    monkeypatch.setattr(sw, "tick", lambda: (_ for _ in ()).throw(KeyboardInterrupt))
    caplog.set_level(logging.INFO, logger="sensor_worker")
    sw.main()
    assert any("interrupted; shutting down" in r.message for r in caplog.records)
