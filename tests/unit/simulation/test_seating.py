import copy

import pandas as pd

from campus_occupancy.simulation.seating import FLOOR_PLAN_H, FLOOR_PLAN_W, ZONES, assess_zones, zone_shapes


def _seats():
    return pd.DataFrame([
        ("S001", 100, 1000), ("S002", 200, 1100), ("S003", 300, 1200),          # Front
        ("S004", 100, 500), ("S005", 200, 500), ("S006", 300, 500), ("S007", 100, 700), ("S008", 200, 700),  # Middle
        ("S009", 100, 100), ("S010", 200, 100), ("S011", 300, 200), ("S012", 400, 200),  # Back
    ], columns=["seat_id", "x_px", "y_px"])


def test_counts_capacity_free_and_recommendation():
    zones, rec = assess_zones(_seats(), {"S001", "S002", "S009"})
    by = {z["name"]: z for z in zones}
    assert (by["Front"]["capacity"], by["Front"]["count"], by["Front"]["free"]) == (3, 2, 1)
    assert (by["Middle"]["capacity"], by["Middle"]["count"], by["Middle"]["free"]) == (5, 0, 5)
    assert (by["Back"]["capacity"], by["Back"]["count"], by["Back"]["free"]) == (4, 1, 3)
    assert rec["name"] == "Front"


def test_front_full_recommends_middle():
    zones, rec = assess_zones(_seats(), {"S001", "S002", "S003"})
    assert rec["name"] == "Middle"


def test_all_full_falls_back_to_last_zone():
    seats = _seats()
    zones, rec = assess_zones(seats, set(seats.seat_id))
    assert all(z["free"] == 0 for z in zones) and rec["name"] == "Back"


def test_zones_ordered_front_to_back_and_bands_tile_plan():
    assert [z["name"] for z in ZONES] == ["Front", "Middle", "Back"]
    assert ZONES[0]["y_max"] == FLOOR_PLAN_H and ZONES[-1]["y_min"] == 0
    assert ZONES[0]["y_min"] == ZONES[1]["y_max"] and ZONES[1]["y_min"] == ZONES[2]["y_max"]


def test_empty_seats():
    zones, rec = assess_zones(pd.DataFrame(columns=["seat_id", "x_px", "y_px"]), set())
    assert all(z["capacity"] == 0 for z in zones) and rec["name"] == "Back"


def test_module_zones_not_mutated():
    before = copy.deepcopy(ZONES)
    assess_zones(_seats(), {"S001"})
    assert ZONES == before


def test_custom_zones_used_and_seats_outside_bands_ignored():
    custom = [{"name": "Only", "y_min": 0, "y_max": 600}]
    zones, rec = assess_zones(_seats(), {"S004"}, zones=custom)
    assert zones[0]["capacity"] == 7 and zones[0]["count"] == 1 and rec["name"] == "Only"


def test_zone_shapes_geometry():
    shape = zone_shapes({"name": "Front", "y_min": 887, "y_max": 1330})[0]
    assert (shape["x0"], shape["x1"], shape["y0"], shape["y1"]) == (0, FLOOR_PLAN_W, 887, 1330)
    assert shape["type"] == "rect" and shape["layer"] == "above"
