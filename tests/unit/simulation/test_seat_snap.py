import pandas as pd
import pytest
from scipy.spatial import cKDTree

from campus_occupancy.simulation.seat_snap import build_seat_tree, load_seats, snap_detections


@pytest.fixture
def seats():
    return pd.DataFrame([(f"S{r * 3 + c + 1:03d}", 100 + c * 100, 100 + r * 100)
                         for r in range(3) for c in range(3)], columns=["seat_id", "x_px", "y_px"])


@pytest.fixture
def tree(seats):
    return cKDTree(seats[["x_px", "y_px"]].to_numpy())


def test_load_seats_casts_to_float(tmp_path):
    p = tmp_path / "seats.csv"
    p.write_text("seat_id,x_px,y_px\nS001,10,20\n")
    df = load_seats.__wrapped__(str(p)) if hasattr(load_seats, "__wrapped__") else load_seats(str(p))
    assert df.x_px.dtype.kind == "f" and df.y_px.dtype.kind == "f"


def test_build_seat_tree_type(seats):
    t = build_seat_tree.__wrapped__(seats) if hasattr(build_seat_tree, "__wrapped__") else build_seat_tree(seats)
    assert isinstance(t, cKDTree) and t.n == 9


def test_empty_input(seats, tree):
    assert snap_detections([], seats, tree) == (set(), [])


def test_exact_hit(seats, tree):
    occ, dbg = snap_detections([(200.0, 200.0)], seats, tree)
    assert occ == {"S005"}
    assert dbg == [{"detection_index": 0, "x": 200.0, "y": 200.0, "seat_id": "S005",
                    "distance_px": 0.0, "reason": "ok"}]


def test_threshold_drops_far_detections(seats, tree):
    occ, dbg = snap_detections([(200.0, 200.0), (500.0, 500.0)], seats, tree, max_dist=80)
    assert occ == {"S005"}
    assert dbg[1]["seat_id"] is None and "S009" in dbg[1]["reason"] and "> 80px" in dbg[1]["reason"]


def test_greedy_fallback_to_next_seat(seats, tree):
    occ, dbg = snap_detections([(200.0, 200.0), (210.0, 210.0)], seats, tree, max_dist=120)
    assert occ == {"S005", "S008"}
    assert dbg[0]["seat_id"] == "S005" and dbg[1]["seat_id"] == "S008"


def test_no_fallback_within_tight_threshold(seats, tree):
    occ, dbg = snap_detections([(200.0, 200.0), (210.0, 210.0)], seats, tree, max_dist=80)
    assert occ == {"S005"}
    assert "claimed" in dbg[1]["reason"]


def test_identical_detections_get_distinct_seats(seats, tree):
    occ, _ = snap_detections([(200.0, 200.0), (200.0, 200.0)], seats, tree, max_dist=150)
    assert len(occ) == 2 and "S005" in occ


def test_closest_detection_wins(seats, tree):
    # second detection is closer to S005 → it should get S005, first one falls back
    occ, dbg = snap_detections([(230.0, 230.0), (201.0, 201.0)], seats, tree, max_dist=150)
    assert dbg[1]["seat_id"] == "S005" and dbg[0]["seat_id"] != "S005"


def test_k_larger_than_seat_count(seats, tree):
    occ, dbg = snap_detections([(100.0, 100.0)], seats, tree, k_candidates=50)
    assert occ == {"S001"} and dbg[0]["reason"] == "ok"


def test_debug_rows_keep_input_order(seats, tree):
    pts = [(300.0, 300.0), (100.0, 100.0), (200.0, 100.0)]
    _, dbg = snap_detections(pts, seats, tree)
    assert [d["detection_index"] for d in dbg] == [0, 1, 2]
    assert [(d["x"], d["y"]) for d in dbg] == pts


def test_single_seat_tree():
    seats = pd.DataFrame([("S001", 10.0, 10.0)], columns=["seat_id", "x_px", "y_px"])
    tree = cKDTree(seats[["x_px", "y_px"]].to_numpy())
    occ, dbg = snap_detections([(12.0, 12.0), (13.0, 13.0)], seats, tree, max_dist=50)
    assert occ == {"S001"} and dbg[1]["seat_id"] is None
