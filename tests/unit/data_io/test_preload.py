import os
import time
from dataclasses import replace

import pandas as pd
import pytest

import campus_occupancy.data_io.preload as preload
from campus_occupancy.dashboard.lab_dashboard import LabConfig
from campus_occupancy.data_io.preload import Preloader, build_bundle, get_preloaded, preload_status, start_preloader
from tests.support.helpers import write_log


def _build(page: LabConfig, **over):
    kw = dict(data_file=page.data_file, coords_file=page.coords_file, floor_plan=page.floor_plan,
              sensor_positions=page.sensor_positions, history_file=page.history_file)
    kw.update(over)
    return build_bundle(**kw)


def wait_ready(p, data_file, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = p.status(data_file)
        if s["ready"] or s["error"]:
            return s
        time.sleep(0.1)
    raise AssertionError("preloader never became ready")


def test_build_bundle_contents(synthetic_lab):
    b = _build(synthetic_lab.page)
    assert len(b.latest) == synthetic_lab.n_desks and b.latest.pc_id.is_unique
    assert b.df.timestamp.dtype.kind == "M" and str(b.df.session_ttl_remaining.dtype) == "Int64"
    assert b.history is not None and b.history.index.name == "hour" and b.history.index.is_monotonic_increasing
    assert len(b.coords) == synthetic_lab.n_desks and len(b.sensors) == synthetic_lab.n_sensors
    assert set(b.nearby) == {"WC-S01", "WC-S02"} and b.zone_map == {"WC-S01": "1a", "WC-S02": "2b"}
    assert b.floor_plan.size == (300, 1300)
    assert b.load_seconds > 0 and b.log_mtime == os.path.getmtime(synthetic_lab.page.data_file)
    assert b.source == "on-demand"


def test_build_bundle_optional_parts(synthetic_lab, tmp_path):
    b = _build(synthetic_lab.page, history_file=None, sensor_positions=None, coords_file=str(tmp_path / "x.csv"),
               floor_plan=str(tmp_path / "x.png"))
    assert b.history is None and b.sensors is None and b.coords is None and b.floor_plan is None
    assert b.nearby == {} and b.zone_map == {}


def test_build_bundle_missing_log(synthetic_lab, tmp_path):
    with pytest.raises(FileNotFoundError):
        _build(synthetic_lab.page, data_file=str(tmp_path / "missing.csv"))


def test_preloader_becomes_ready_and_reports_status(synthetic_lab):
    p = Preloader([synthetic_lab.page], refresh_s=60)
    s = wait_ready(p, synthetic_lab.page.data_file)
    assert s["ready"] and s["error"] is None and s["rows"] == len(p.get(synthetic_lab.page.data_file).df)
    assert s["age_s"] >= 0 and s["load_seconds"] > 0 and s["since_boot_s"] >= 0
    assert p.get(synthetic_lab.page.data_file).source == "preloaded"
    assert set(s) == {"ready", "loaded_at", "age_s", "load_seconds", "rows", "error", "since_boot_s"}


def test_refresh_skips_unchanged_file_and_rebuilds_on_change(synthetic_lab, tmp_path):
    src = pd.read_csv(synthetic_lab.page.data_file)
    log = tmp_path / "log.csv"
    src.to_csv(log, index=False)
    page = replace(synthetic_lab.page, data_file=str(log))
    p = Preloader([page], refresh_s=3600)
    wait_ready(p, page.data_file)
    first = p.get(page.data_file)
    p._refresh_one(page)
    assert p.get(page.data_file) is first
    last_ts = pd.to_datetime(src.timestamp).max() + pd.Timedelta(minutes=1)
    write_log(log, src.pc_id.unique(), last_ts.to_pydatetime(), append=True)
    os.utime(log, (time.time() + 5, time.time() + 5))
    p._refresh_one(page)
    second = p.get(page.data_file)
    assert second is not first and len(second.df) == len(first.df) + src.pc_id.nunique()


def test_preloader_records_errors_without_crashing(tmp_path):
    page = LabConfig(title="x", data_file=str(tmp_path / "none.csv"), coords_file="", floor_plan="",
                     calibrate_hint="")
    p = Preloader([page], refresh_s=3600)
    s = wait_ready(p, page.data_file)
    assert not s["ready"] and s["error"] == "data file missing"


def test_start_preloader_idempotent_and_lookups(synthetic_lab):
    assert preload_status(synthetic_lab.page.data_file) is None and get_preloaded("x") is None
    p1 = start_preloader([synthetic_lab.page], refresh_s=3600)
    p2 = start_preloader([synthetic_lab.page], refresh_s=1)
    assert p1 is p2 and preload._PRELOADER is p1
    wait_ready(p1, synthetic_lab.page.data_file)
    assert preload_status(synthetic_lab.page.data_file)["ready"]
    assert get_preloaded(synthetic_lab.page.data_file) is p1.get(synthetic_lab.page.data_file)
    assert get_preloaded("unknown") is None


def test_bundle_fields_are_read_only_by_convention(synthetic_lab):
    b = _build(synthetic_lab.page)
    assert isinstance(b.df, pd.DataFrame) and isinstance(b.latest, pd.DataFrame)
    assert b.latest is not b.df
