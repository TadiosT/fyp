import math
import random

import pandas as pd
import pytest

from campus_occupancy.simulation.wc_session import (
    CO2_LIMIT, DEFAULT_READING, LABELS, air_score, arrival_profile, assign_roles,
    build_zone_table, desk_states, simulate_session,
)
from tests.support.helpers import consistent_reading, grid_desks


@pytest.fixture
def desks():
    d = pd.concat([
        grid_desks(6, 5, "1a", origin=(0, 0)),         # 30
        grid_desks(6, 5, "1b", origin=(0, 500)),       # 30
        grid_desks(5, 4, "2a", origin=(0, 1000)),      # 20
        grid_desks(8, 5, "2b", origin=(0, 1500)),      # 40
    ], ignore_index=True)
    d["in_use_now"] = False
    return d


def readings(desks, **extra):
    return {z: consistent_reading(desks, z, extra.get(z, 0)) for z in desks["zone"].unique()}


# ─────────────────────── scoring & roles ───────────────────────

def test_air_score_baseline_and_monotone():
    assert air_score(DEFAULT_READING) == 0
    assert air_score({**DEFAULT_READING, "co2": 900}) > 0
    assert air_score({**DEFAULT_READING, "noise": 50}) > 0
    assert air_score({**DEFAULT_READING, "temperature": 25}) > 0
    assert air_score({**DEFAULT_READING, "co2": 300}) == 0    # below baseline clamps to 0


def test_build_zone_table(desks):
    desks.loc[desks.zone == "2b", "in_use_now"] = [i % 2 == 0 for i in range(40)]
    zt = build_zone_table(desks, {"2b": {"co2": 800, "noise": 40, "temperature": 22}}).set_index("zone")
    assert zt.loc["2b", "capacity"] == 40 and zt.loc["2b", "in_use_now"] == 20 and zt.loc["2b", "free_now"] == 20
    assert zt.loc["1a", "co2"] == DEFAULT_READING["co2"] and zt.loc["1a", "air_score"] == 0
    assert zt.loc["2b", "air_score"] > 0


def test_assign_roles_single_zone():
    zt = build_zone_table(grid_desks(3, 3, "solo").assign(in_use_now=False), {})
    assert assign_roles(zt) == {"teaching": "solo", "quiet": None, "overflow_order": []}


def test_assign_roles_prefers_best_air_largest_free(desks):
    rd = readings(desks)
    rd["2b"] = {"co2": 1300, "noise": 55, "temperature": 24}     # worst air, largest zone
    roles = assign_roles(build_zone_table(desks, rd))
    assert roles["teaching"] in ("1a", "1b")                       # 30-desk zones with clean air
    assert roles["quiet"] not in (roles["teaching"],)
    assert roles["overflow_order"][-1] == roles["quiet"]
    assert set(roles["overflow_order"]) | {roles["teaching"]} == set(desks.zone.unique())


def test_quiet_zone_is_quietest_of_the_rest(desks):
    rd = readings(desks, **{"1a": 8, "1b": 6, "2b": 4})           # 2a stays quietest
    roles = assign_roles(build_zone_table(desks, rd))
    assert roles["quiet"] == "2a"


# ─────────────────────── arrivals ───────────────────────

@pytest.mark.parametrize("n,window", [(0, 15), (1, 15), (7, 5), (250, 15), (600, 30)])
def test_arrival_profile_sums_to_n(n, window):
    per_min = arrival_profile(n, window, random.Random(0))
    assert sum(per_min) == n
    assert len(per_min) == 2 * window + 1


def test_arrival_profile_front_loaded():
    per_min = arrival_profile(300, 15, random.Random(1))
    assert sum(per_min[:5]) >= 0.55 * 300


# ─────────────────────── simulate_session ───────────────────────

def test_overflow_at_threshold(desks):
    # 60 students fit within 75 % of the two best zones, so no zone should exceed the threshold
    res = simulate_session(desks, readings(desks), attendance=60, duration_min=120, threshold=0.75, seed=3)
    caps = res.zone_table.set_index("zone")["capacity"]
    for z in res.session_counts.columns:
        assert res.session_counts[z].max() <= math.ceil(0.75 * caps[z]), z
    t = res.roles["teaching"]
    assert res.session_counts[t].max() == math.floor(0.75 * caps[t]) or res.session_counts[t].max() == math.ceil(0.75 * caps[t]) - 1
    assert any("reached" in e["message"] and "opened" in e["message"] for e in res.events)
    assert not any("quiet-study" in e["message"] for e in res.events)


def test_beyond_threshold_everywhere_seats_in_roomiest_zone(desks):
    # 100 students exceed 75 % of all four zones combined (90), so the last 10 go
    # to the zone with the most physical space rather than being turned away.
    res = simulate_session(desks, readings(desks), attendance=100, duration_min=120, threshold=0.75, seed=3)
    assert res.unseated == 0 and res.session_counts.sum(axis=1).max() == 100


def test_overflow_follows_air_score_order(desks):
    rd = readings(desks, **{"1b": 10})                         # make 1b noisier than 1a
    res = simulate_session(desks, rd, attendance=110, seed=1)
    opened = [e["message"].split("opened Zone ")[1].split(".")[0].rstrip(" ⚠️ quiet-study zone opened")
              for e in res.events if "opened Zone" in e["message"]]
    order = res.roles["overflow_order"]
    assert opened == order[:len(opened)]


def test_high_co2_triggers_early_overflow(desks):
    rd = {z: {"co2": CO2_LIMIT + 100, "noise": 40, "temperature": 21} for z in desks.zone.unique()}
    res = simulate_session(desks, rd, attendance=50, seed=2)
    first = [e for e in res.events if "opened Zone" in e["message"]][0]
    assert "CO₂" in first["message"] and "reached" not in first["message"]


def test_quiet_zone_last_resort_and_unseated(desks):
    res = simulate_session(desks, readings(desks), attendance=200, seed=3)
    assert any("quiet-study" in e["message"] and e["level"] == "warning" for e in res.events)
    assert res.unseated == 200 - len(desks)


def test_session_ends_and_late_departures(desks):
    res = simulate_session(desks, readings(desks), attendance=80, duration_min=120, seed=4)
    assert res.session_counts.loc[120].sum() == 0
    assert res.session_counts.loc[110].sum() < res.session_counts.loc[60].sum()
    assert res.events[-1]["message"].startswith("Session ends")


def test_timeline_shape_and_recommendations(desks):
    res = simulate_session(desks, readings(desks), attendance=60, duration_min=90, seed=5)
    assert list(res.timeline.index) == list(range(91))
    assert list(res.timeline.columns) == [res.roles["teaching"]] + res.roles["overflow_order"]
    assert ((res.timeline >= 0) & (res.timeline <= 1)).all().all()
    text = " ".join(res.recommendations)
    assert f"Zone {res.roles['teaching']}" in text and f"Zone {res.roles['quiet']}" in text
    assert set(res.zone_table["role"]) <= {"Teaching lab", "Overflow", "Quiet study"}


def test_existing_occupancy_reduces_room(desks):
    desks.loc[desks.zone == "2b", "in_use_now"] = True             # 2b full already
    res = simulate_session(desks, readings(desks), attendance=30, seed=6)
    assert res.session_counts["2b"].max() == 0


# ─────────────────────── desk_states ───────────────────────

def test_desk_states_labels_and_counts(desks):
    desks.loc[desks.zone == "1a", "in_use_now"] = [i < 5 for i in range(30)]
    res = simulate_session(desks, readings(desks), attendance=80, seed=7)
    for minute in (0, 20, 60):
        labels = desk_states(res, minute)
        assert set(labels.unique()) <= set(LABELS)
        joined = desks.set_index("pc_id").join(labels)
        assert (joined.loc[joined.in_use_now, "label"] == "Already in use").all()
        for z in desks.zone.unique():
            n = int(res.session_counts.loc[minute, z])
            seated = joined[(joined.zone == z) & joined.label.isin(["Teaching lab", "Overflow", "Quiet study"])]
            assert len(seated) == n


def test_desk_states_clamps_minute(desks):
    res = simulate_session(desks, readings(desks), attendance=40, duration_min=60, seed=8)
    assert desk_states(res, -5).equals(desk_states(res, 0))
    assert desk_states(res, 999).equals(desk_states(res, 60))
