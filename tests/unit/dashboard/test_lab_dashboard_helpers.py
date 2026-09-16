import math
import time

import pandas as pd
import pytest

import campus_occupancy.dashboard.lab_dashboard as ld
from campus_occupancy.dashboard.lab_dashboard import age_str, co2_band, current_desk_state, fmt, get_bundle, live_sensor_readings
from campus_occupancy.data_io.preload import build_bundle, preload_status, start_preloader


@pytest.mark.parametrize("co2,band", [(None, None), (0, "Good"), (799, "Good"), (800, "Fair"), (1199, "Fair"), (1200, "Poor")])
def test_co2_band(co2, band):
    assert co2_band(co2) == band


def test_fmt():
    assert fmt(None, "°C") == "—" and fmt(float("nan"), "x") == "—"
    assert fmt(21.456, "°C") == "21.5 °C" and fmt(63, "%") == "63 %"


@pytest.mark.parametrize("s,out", [(0, "0s"), (59, "59s"), (60, "1m"), (3599, "59m"), (3600, "1h 0m"), (3725, "1h 2m"), (-5, "0s")])
def test_age_str(s, out):
    assert age_str(s) == out


def test_current_desk_state():
    latest = pd.DataFrame({"pc_id": ["a", "b", "c"], "session_ttl_remaining": pd.array([5, 0, None], dtype="Int64")})
    assert current_desk_state(latest) == {"a": {"ttl": 1}, "b": {"ttl": 0}, "c": {"ttl": 0}}


def _bundle(page):
    return build_bundle(page.data_file, page.coords_file, page.floor_plan, page.sensor_positions, page.history_file)


def test_live_sensor_readings_stable_within_minute(synthetic_lab, monkeypatch):
    from datetime import datetime, timedelta
    fixed = datetime(2026, 9, 2, 14, 15, 30)

    class FakeDT(datetime):                     # pin the clock so both calls share a minute seed
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(ld, "datetime", FakeDT)
    b = _bundle(synthetic_lab.page)
    a, c = live_sensor_readings(b), live_sensor_readings(b)
    assert set(a) == {"WC-S01", "WC-S02"} and a == c
    assert all(r["ok"] and 400 <= r["co2"] <= 1400 for r in a.values())
    assert all(r["time_utc"] == int(fixed.replace(second=0).timestamp()) for r in a.values())

    fixed = fixed + timedelta(minutes=1)        # next minute → new seed → values may differ, keys same
    d = live_sensor_readings(b)
    assert set(d) == set(a)


def test_live_sensor_readings_empty_without_sensors(synthetic_lab):
    b = build_bundle(synthetic_lab.page.data_file, synthetic_lab.page.coords_file, synthetic_lab.page.floor_plan,
                     None, None)
    assert live_sensor_readings(b) == {}


def test_get_bundle_prefers_preloaded(synthetic_lab):
    on_demand = get_bundle(synthetic_lab.page)
    assert on_demand.source == "on-demand"
    assert get_bundle(synthetic_lab.page) is on_demand              # cached for 60 s
    start_preloader([synthetic_lab.page], refresh_s=3600)
    t0 = time.time()
    while not (preload_status(synthetic_lab.page.data_file) or {}).get("ready") and time.time() - t0 < 15:
        time.sleep(0.1)
    assert get_bundle(synthetic_lab.page).source == "preloaded"


def test_labconfig_defaults():
    cfg = ld.LabConfig(title="t", data_file="d", coords_file="c", floor_plan="f", calibrate_hint="h")
    assert cfg.map_height == 650 and cfg.sensor_positions is None and cfg.history_file is None
    assert "launcher.py" in cfg.missing_data_hint
