import random
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from campus_occupancy.simulation import wc_sim
from campus_occupancy.simulation.wc_sim import (
    CSV_COLUMNS, MIN_NEARBY, OCCUPANCY_CURVE, RADIUS_PX, WEEKDAY_FACTOR, make_initial_state,
    make_user_pool, nearby_desks, reading_from_ratio, sensor_snapshots, sensor_zone_map,
    state_counts, step_desks, target_fraction,
)
from tests.support.helpers import SAT, WED, grid_desks, sensors_df

VALID_STATES = {"In Use", "Idle", "Offline"}


# ─────────────────────── target_fraction ───────────────────────

def test_target_exact_on_the_hour():
    assert target_fraction(WED.replace(hour=14)) == pytest.approx(OCCUPANCY_CURVE[14])


def test_target_interpolates_within_the_hour():
    expected = (OCCUPANCY_CURVE[14] + OCCUPANCY_CURVE[15]) / 2
    assert target_fraction(WED.replace(hour=14, minute=30)) == pytest.approx(expected)


def test_target_wraps_from_23_to_0():
    expected = OCCUPANCY_CURVE[23] + (OCCUPANCY_CURVE[0] - OCCUPANCY_CURVE[23]) * 0.5
    assert target_fraction(WED.replace(hour=23, minute=30)) == pytest.approx(expected)


def test_target_applies_weekday_factor():
    assert target_fraction(SAT.replace(hour=14)) == pytest.approx(OCCUPANCY_CURVE[14] * WEEKDAY_FACTOR[5])
    assert WEEKDAY_FACTOR[5] < WEEKDAY_FACTOR[0]


# ─────────────────────── state / pool ───────────────────────

def test_initial_state_and_user_pool():
    state = make_initial_state(["A", "B"])
    assert state == {"A": {"state": "Offline", "user_id": "N/A", "ttl": 0},
                     "B": {"state": "Offline", "user_id": "N/A", "ttl": 0}}
    pool = make_user_pool(3)
    assert pool == ["AnonUser_0001", "AnonUser_0002", "AnonUser_0003"]


# ─────────────────────── step_desks invariants ───────────────────────

def _run(n_desks, start, minutes, seed=0, state=None, users=None):
    pcs = [f"PC{i:03d}" for i in range(n_desks)]
    state = state or make_initial_state(pcs)
    users = users if users is not None else make_user_pool(n_desks)
    rng = random.Random(seed)
    t = start
    rows = []
    for _ in range(minutes):
        entries, t = step_desks(state, t, users, rng)
        rows.append((t, entries))
    return state, users, rows


def test_step_invariants_hold_over_time():
    n = 100
    full_pool = set(make_user_pool(n))
    state, users, rows = _run(n, WED.replace(hour=9), 200, seed=1)
    prev_t = WED.replace(hour=9)
    for t, entries in rows:
        assert t == prev_t + timedelta(minutes=1)
        prev_t = t
        assert len(entries) == n
        assert set(entries[0]) == set(CSV_COLUMNS)
        for e in entries:
            assert e["state"] in VALID_STATES
            assert e["timestamp"] == t.strftime("%Y-%m-%d %H:%M:%S")
            assert (e["user_id"] == "N/A") == (e["session_ttl_remaining"] == "N/A")
            if e["user_id"] != "N/A":
                assert 1 <= e["session_ttl_remaining"] <= 120
            else:
                assert e["state"] == "Offline"
    # user conservation: every user is either seated exactly once or in the pool
    active = [d["user_id"] for d in state.values() if d["user_id"] != "N/A"]
    assert len(active) == len(set(active))
    assert set(active) | set(users) == full_pool
    assert not (set(active) & set(users))


def test_idle_only_while_session_active():
    _, _, rows = _run(80, WED.replace(hour=11), 120, seed=2)
    for _, entries in rows:
        for e in entries:
            if e["state"] == "Idle":
                assert e["user_id"] != "N/A" and e["session_ttl_remaining"] > 0


def test_arrivals_when_below_target():
    state, _, _ = _run(300, WED.replace(hour=14), 60, seed=3)
    assert sum(d["ttl"] > 0 for d in state.values()) / 300 > 0.3


def test_early_departures_when_above_target():
    pcs = [f"PC{i:03d}" for i in range(200)]
    users = make_user_pool(200)
    state = {pc: {"state": "In Use", "user_id": users[i], "ttl": 120} for i, pc in enumerate(pcs)}
    state, _, _ = _run(200, WED.replace(hour=3), 60, seed=4, state=state, users=[])
    assert sum(d["ttl"] > 0 for d in state.values()) / 200 < 0.5


def test_tracks_daily_curve():
    n = 300
    pcs = [f"PC{i:03d}" for i in range(n)]
    state, users, rng, t = make_initial_state(pcs), make_user_pool(n), random.Random(5), WED
    samples = {}
    for _ in range(1440):
        _, t = step_desks(state, t, users, rng)
        if t.minute == 0 and t.hour in (3, 10, 14, 22):
            samples[t.hour] = sum(d["ttl"] > 0 for d in state.values()) / n
    for h, ratio in samples.items():
        assert abs(ratio - target_fraction(WED.replace(hour=h))) < 0.12, (h, ratio)


def test_step_without_rng_uses_module_random():
    state = make_initial_state(["A"])
    entries, t = step_desks(state, WED, ["u1"])
    assert len(entries) == 1 and t == WED + timedelta(minutes=1)


def test_state_counts():
    state = {"a": {"state": "In Use"}, "b": {"state": "Idle"}, "c": {"state": "Offline"}, "d": {"state": "In Use"}}
    assert state_counts(state) == {"in_use": 2, "idle": 1, "offline": 1}


# ─────────────────────── sensors ───────────────────────

def test_nearby_desks_within_radius():
    desks = grid_desks(10, 10, spacing=100)          # x,y in 0..900
    sensors = sensors_df([(450.0, 450.0)])
    nb = nearby_desks(sensors, desks)
    d = np.hypot(desks.x_px - 450, desks.y_px - 450)
    assert set(nb["WC-S01"]) == set(desks.pc_id[d <= RADIUS_PX])
    assert len(nb["WC-S01"]) >= MIN_NEARBY


def test_nearby_desks_falls_back_to_nearest():
    desks = grid_desks(10, 10, spacing=100)
    nb = nearby_desks(sensors_df([(5000.0, 5000.0)]), desks)
    assert len(nb["WC-S01"]) == MIN_NEARBY
    d = np.hypot(desks.x_px - 5000, desks.y_px - 5000)
    assert set(nb["WC-S01"]) == set(desks.pc_id.iloc[np.argsort(d)[:MIN_NEARBY]])


def test_sensor_zone_map_majority_and_missing_zone():
    desks = pd.concat([grid_desks(6, 5, "A", origin=(0, 0)), grid_desks(6, 5, "B", origin=(0, 2000))],
                      ignore_index=True)
    zm = sensor_zone_map(sensors_df([(125.0, 100.0), (125.0, 2100.0)]), desks)
    assert zm == {"WC-S01": "A", "WC-S02": "B"}
    assert sensor_zone_map(sensors_df([(0.0, 0.0)]), desks.drop(columns=["zone"])) == {"WC-S01": "?"}


def test_reading_from_ratio_monotone_and_shape():
    lo = reading_from_ratio("S", 0.0, WED.replace(hour=12), random.Random(0))
    hi = reading_from_ratio("S", 1.0, WED.replace(hour=12), random.Random(0))
    assert set(lo) == {"ok", "error", "device_name", "temperature", "humidity", "co2", "noise", "time_utc"}
    assert lo["ok"] and lo["error"] is None and lo["device_name"] == "S"
    assert hi["co2"] - lo["co2"] == pytest.approx(900, abs=1)
    assert hi["noise"] - lo["noise"] == pytest.approx(25, abs=1)
    assert hi["temperature"] > lo["temperature"]
    assert lo["time_utc"] == int(WED.replace(hour=12).timestamp())


def test_sensor_snapshots_one_per_sensor_and_empty():
    desks = grid_desks(5, 4)
    nb = {"S1": list(desks.pc_id[:10]), "S2": list(desks.pc_id[10:])}
    state = {pc: {"ttl": 1} for pc in desks.pc_id[:10]}
    snaps = sensor_snapshots(nb, state, WED.replace(hour=10), random.Random(1))
    assert [s["device_name"] for s in snaps] == ["S1", "S2"]
    assert snaps[0]["co2"] > snaps[1]["co2"]          # S1's desks are all occupied
    assert sensor_snapshots({}, state, WED) == []


def test_module_constants_sane():
    assert set(OCCUPANCY_CURVE) == set(range(24)) and all(0 < v <= 1 for v in OCCUPANCY_CURVE.values())
    assert set(WEEKDAY_FACTOR) == set(range(7))
    assert wc_sim.MEAN_SESSION_MIN == pytest.approx((15 + 120) / 2)
