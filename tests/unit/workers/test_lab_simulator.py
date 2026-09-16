import csv
import os
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta

import pandas as pd
import pytest

from campus_occupancy.workers import lab_simulator as ls
from campus_occupancy.config import SIM_CONFIGS, SimConfig
from campus_occupancy.simulation.wc_sim import CSV_COLUMNS
from tests.support.helpers import write_log


def now_min():
    return datetime.now().replace(second=0, microsecond=0)


@pytest.fixture
def pcs(synthetic_lab):
    return sorted(pd.read_csv(synthetic_lab.sim.desks_csv).pc_id)


def test_load_initial_state_parses_last_row(tmp_path, pcs):
    log = tmp_path / "log.csv"
    t0 = now_min() - timedelta(minutes=5)
    write_log(log, pcs, t0)
    with open(log, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writerow({"timestamp": (t0 + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"), "pc_id": pcs[0],
                    "state": "In Use", "user_id": "AnonUser_0001", "session_ttl_remaining": 42})
    state, t = ls.load_initial_state(str(log))
    assert t == t0 + timedelta(minutes=1)
    assert state[pcs[0]] == {"state": "In Use", "user_id": "AnonUser_0001", "ttl": 42}
    # regression: pandas must not turn the literal "N/A" into NaN
    assert state[pcs[1]] == {"state": "Offline", "user_id": "N/A", "ttl": 0}
    assert all(isinstance(d["user_id"], str) for d in state.values())


def test_resume_keeps_user_pool_consistent(tmp_sim, pcs, scratch_db):
    """After a resume, seated users must be excluded from the pool and 'N/A' desks must not leak NaN."""
    t0 = now_min() - timedelta(minutes=1)
    write_log(tmp_sim.log_csv, pcs[1:], t0)
    write_log(tmp_sim.log_csv, pcs[:1], t0, state="In Use", user="AnonUser_0001", ttl=30, append=True)
    wk = ls.Worker(tmp_sim)
    assert "AnonUser_0001" not in wk.users and len(wk.users) == len(pcs) - 1
    entries = wk.step()
    na_rows = [e for e in entries if e["user_id"] == "N/A"]
    assert na_rows and all(e["session_ttl_remaining"] == "N/A" for e in na_rows)


def test_catch_up_to_wall_clock(tmp_sim, pcs, scratch_db):
    write_log(tmp_sim.log_csv, pcs, now_min() - timedelta(hours=3, minutes=5))
    wk = ls.Worker(tmp_sim)
    target = now_min()
    n = wk.advance_to(target)
    assert n == int((target - (now_min() - timedelta(hours=3, minutes=5))).total_seconds() // 60)
    df = pd.read_csv(tmp_sim.log_csv)
    assert pd.Timestamp(df.timestamp.max()) == pd.Timestamp(target)
    assert df.groupby("timestamp").size().eq(len(pcs)).all()
    assert len(wk.state) == len(pcs)


def test_large_gap_rebuilds_last_48h(tmp_sim, pcs, scratch_db):
    write_log(tmp_sim.log_csv, pcs, now_min() - timedelta(days=5))
    wk = ls.Worker(tmp_sim)
    assert wk.t == now_min() - timedelta(hours=ls.RETENTION_H)
    assert all(d["state"] == "Offline" for d in wk.state.values())
    assert pd.read_csv(tmp_sim.log_csv).empty                 # fresh header only


def test_no_log_starts_fresh(tmp_sim, scratch_db):
    wk = ls.Worker(tmp_sim)
    assert wk.t == now_min() - timedelta(hours=ls.RETENTION_H) and os.path.exists(tmp_sim.log_csv)


def test_reconciles_desk_list(tmp_sim, tmp_path, pcs, scratch_db):
    desks = pd.read_csv(tmp_sim.desks_csv)
    new = pd.concat([desks.iloc[:-3], pd.DataFrame({"pc_id": ["NEW-001", "NEW-002"], "x_px": [1, 2],
                                                     "y_px": [1, 2], "zone": ["1a", "1a"]})])
    new.to_csv(tmp_path / "desks2.csv", index=False)
    cfg = replace(tmp_sim, desks_csv=str(tmp_path / "desks2.csv"), sensors_csv=None)
    write_log(cfg.log_csv, pcs, now_min() - timedelta(minutes=2))
    wk = ls.Worker(cfg)
    assert set(wk.state) == set(new.pc_id) and len(wk.users) == len(new)
    assert wk.state["NEW-001"]["state"] == "Offline"
    wk.advance_to(now_min())
    assert set(pd.read_csv(cfg.log_csv).query("timestamp == @wk.t.strftime('%Y-%m-%d %H:%M:%S')").pc_id) == set(new.pc_id)


def test_sensor_lab_collects_and_flushes_snapshots(tmp_sim, pcs, scratch_db):
    t0 = now_min().replace(minute=0) - timedelta(hours=1)
    write_log(tmp_sim.log_csv, pcs, t0)
    wk = ls.Worker(tmp_sim)
    wk.advance_to(t0 + timedelta(minutes=10))
    assert len(wk.pending_snaps) == 2 * 2                       # minutes 5 and 10, two sensors
    wk.flush_snaps()
    assert wk.pending_snaps == []
    assert sqlite3.connect(scratch_db).execute("select count(*) from sensorreading").fetchone()[0] == 4


def test_sensorless_lab_never_touches_db(tmp_sim, pcs, scratch_db, monkeypatch):
    cfg = replace(tmp_sim, sensors_csv=None)
    write_log(cfg.log_csv, pcs, now_min() - timedelta(minutes=30))
    monkeypatch.setattr(ls, "save_readings_bulk", lambda snaps: (_ for _ in ()).throw(AssertionError("DB touched")))
    wk = ls.Worker(cfg)
    wk.advance_to(now_min())
    wk.flush_snaps()
    assert wk.nearby == {} and wk.pending_snaps == []


def test_flush_survives_db_failure(tmp_sim, pcs, scratch_db, monkeypatch):
    write_log(tmp_sim.log_csv, pcs, now_min().replace(minute=0) - timedelta(hours=1))
    wk = ls.Worker(tmp_sim)
    wk.advance_to(wk.t + timedelta(minutes=5))
    assert wk.pending_snaps
    monkeypatch.setattr(ls, "save_readings_bulk", lambda snaps: (_ for _ in ()).throw(RuntimeError("boom")))
    wk.flush_snaps()
    assert wk.pending_snaps                                      # kept for retry


def test_roll_hour_writes_hourly_row(tmp_sim, pcs, scratch_db):
    write_log(tmp_sim.log_csv, pcs, now_min() - timedelta(minutes=1))
    wk = ls.Worker(tmp_sim)
    wk.t = datetime(2026, 9, 2, 10, 0)
    wk.hour_acc = [{"in_use": 10, "idle": 2, "offline": 48}, {"in_use": 12, "idle": 0, "offline": 48}]
    wk.roll_hour()
    wk.hour_acc = [{"in_use": 1, "idle": 1, "offline": 58}]
    wk.t = datetime(2026, 9, 2, 11, 0)
    wk.roll_hour()
    h = pd.read_csv(tmp_sim.hourly_csv)
    assert list(h.columns) == ["hour", "in_use", "idle", "offline"]
    assert h.iloc[0].tolist() == ["2026-09-02 09:00:00", 11, 1, 48]
    assert len(h) == 2 and wk.hour_acc == []


def test_trim_log_keeps_retention_window(tmp_sim, pcs, scratch_db):
    old = now_min() - timedelta(hours=ls.RETENTION_H + 2)
    write_log(tmp_sim.log_csv, pcs, old)
    write_log(tmp_sim.log_csv, pcs, now_min() - timedelta(minutes=1), append=True)
    wk = ls.Worker(tmp_sim)
    wk.trim_log()
    df = pd.read_csv(tmp_sim.log_csv)
    assert pd.to_datetime(df.timestamp).min() > pd.Timestamp(old)
    assert len(df) == len(pcs)


def _interrupting_sleep(monkeypatch, after: int = 1):
    """Replace time.sleep so the worker's loop exits after `after` sleeps."""
    calls = {"n": 0}

    def fake_sleep(s):
        calls["n"] += 1
        if calls["n"] >= after:
            raise KeyboardInterrupt
    monkeypatch.setattr(ls.time, "sleep", fake_sleep)
    return calls


def test_run_catches_up_then_ticks_until_interrupted(tmp_sim, pcs, scratch_db, monkeypatch, capsys):
    write_log(tmp_sim.log_csv, pcs, now_min() - timedelta(minutes=3))
    wk = ls.Worker(tmp_sim)
    _interrupting_sleep(monkeypatch, after=1)
    wk.run()                                                     # returns on KeyboardInterrupt
    out = capsys.readouterr().out
    assert "catching up" in out and "caught up" in out
    assert pd.Timestamp(pd.read_csv(tmp_sim.log_csv).timestamp.max()) == pd.Timestamp(now_min())


def test_run_recovers_from_tick_failure(tmp_sim, pcs, scratch_db, monkeypatch, capsys):
    """A failure inside the tick loop is logged and retried; boot-time catch-up is not involved."""
    base = now_min()
    write_log(tmp_sim.log_csv, pcs, base)                        # current → no boot catch-up
    wk = ls.Worker(tmp_sim)
    calls = {"n": 0}

    class FakeDT(datetime):                                      # first now() == base, later == base+1min
        @classmethod
        def now(cls, tz=None):
            calls["n"] += 1
            return base if calls["n"] == 1 else base + timedelta(minutes=1)

    monkeypatch.setattr(ls, "datetime", FakeDT)
    real_advance = wk.advance_to
    state = {"failed": False}

    def flaky_advance(target):
        if not state["failed"]:
            state["failed"] = True
            raise RuntimeError("disk full")
        return real_advance(target)

    monkeypatch.setattr(wk, "advance_to", flaky_advance)
    _interrupting_sleep(monkeypatch, after=2)                    # 1st sleep = retry delay, 2nd = exit
    wk.run()
    out = capsys.readouterr().out
    assert "tick failed: RuntimeError('disk full')" in out and "tick ->" in out
    assert wk.t == base + timedelta(minutes=1)


def test_boot_catch_up_failure_is_fatal(tmp_sim, pcs, scratch_db, monkeypatch):
    """By design a failure during the boot-time catch-up propagates (the launcher notices)."""
    write_log(tmp_sim.log_csv, pcs, now_min() - timedelta(minutes=2))
    wk = ls.Worker(tmp_sim)
    monkeypatch.setattr(wk, "advance_to", lambda target: (_ for _ in ()).throw(RuntimeError("disk full")))
    with pytest.raises(RuntimeError):
        wk.run()


def test_main_errors(monkeypatch, tmp_path):
    with pytest.raises(KeyError):
        ls.main("nope")
    monkeypatch.setitem(SIM_CONFIGS, "ghost", SimConfig("ghost", str(tmp_path / "none.csv"),
                                                        str(tmp_path / "l.csv"), str(tmp_path / "h.csv")))
    assert ls.main("ghost") == 1
