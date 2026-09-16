"""Smoke tests for the calibration tools through AppTest, in a temp working dir."""
import os
import shutil

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest



def _workdir(tmp_path, repo_root, assets):
    (tmp_path / "assets").mkdir()
    (tmp_path / "data").mkdir()
    for a in assets:
        src = repo_root / "assets" / a
        if not src.exists():
            pytest.skip(f"asset {a} not present")
        shutil.copy(src, tmp_path / "assets" / a)
    return tmp_path


def test_calibrate_coordinates_numbering_and_modes(repo_root, tmp_path, monkeypatch):
    wd = _workdir(tmp_path, repo_root, ["huxley_lab_floor_plan.jpg"])
    pd.DataFrame({"pc_id": ["Lab_202_PC001", "Lab_202_PC002", "Lab_206_PC003", "Lab_219_PC004", "Lab_219_PC005"],
                  "x_px": [10, 20, 30, 40, 50], "y_px": [10, 20, 30, 40, 50]}).to_csv(wd / "data/workstation_coordinates.csv", index=False)
    monkeypatch.chdir(wd)
    at = AppTest.from_file(str(repo_root / "src/campus_occupancy/tools/calibrate_coordinates.py"), default_timeout=60).run()
    assert not at.exception
    assert at.radio[0].value == "Add new workstation"
    assert at.selectbox[0].options == ["202", "206", "219", "Custom…"]
    assert at.metric[0].value == "Lab_202_PC006"
    at = at.selectbox[0].set_value("219").run()
    assert at.metric[0].value == "Lab_219_PC006"
    at = at.radio[0].set_value("Re-place existing").run()
    assert not at.exception and at.metric[0].value == "5 / 5"
    assert any("Placing" in m.value for m in at.markdown)


def test_calibrate_coordinates_without_any_csv(repo_root, tmp_path, monkeypatch):
    wd = _workdir(tmp_path, repo_root, ["huxley_lab_floor_plan.jpg"])
    monkeypatch.chdir(wd)
    at = AppTest.from_file(str(repo_root / "src/campus_occupancy/tools/calibrate_coordinates.py"), default_timeout=60).run()
    assert not at.exception and at.metric[0].value == "Lab_202_PC001"


def test_calibrate_wc_lab_modes(repo_root, tmp_path, monkeypatch):
    wd = _workdir(tmp_path, repo_root, ["wc_lab.png"])
    monkeypatch.chdir(wd)
    at = AppTest.from_file(str(repo_root / "src/campus_occupancy/tools/calibrate_wc_lab.py"), default_timeout=60).run()
    assert not at.exception
    assert any("Zone **1a**" in m.value for m in at.markdown)
    at = at.radio[0].set_value("Sensors").run()
    assert not at.exception and any("WC-S01" in m.value for m in at.markdown)


def test_calibrate_lecture_seats_fresh_and_resume(repo_root, tmp_path, monkeypatch):
    wd = _workdir(tmp_path, repo_root, ["lecture_hall_144.png"])
    monkeypatch.chdir(wd)
    at = AppTest.from_file(str(repo_root / "src/campus_occupancy/tools/calibrate_lecture_seats.py"), default_timeout=60).run()
    assert not at.exception and any("S001" in m.value for m in at.markdown)
    pd.DataFrame({"seat_id": ["S001", "S002"], "x_px": [10.0, 20.0], "y_px": [10.0, 20.0]}).to_csv(
        wd / "data/lecture_144_seats.csv", index=False)
    at = AppTest.from_file(str(repo_root / "src/campus_occupancy/tools/calibrate_lecture_seats.py"), default_timeout=60).run()
    assert not at.exception and any("S003" in m.value for m in at.markdown)
    assert {m.label: m.value for m in at.metric}["Seats placed"] == "2"


@pytest.mark.yolo
@pytest.mark.skipif(not (os.path.exists("data/lecture_video.mp4") and os.path.exists("data/lecture_144_seats.csv")),
                    reason="lecture video / seat map not present")
def test_label_lecture_frames_renders(repo_root):
    at = AppTest.from_file(str(repo_root / "src/campus_occupancy/tools/label_lecture_frames.py"), default_timeout=180).run()
    assert not at.exception
    assert at.metric[0].label == "Frames labelled" and at.metric[0].value.endswith("/ 30")
    assert any("Sample 00" in m.value for m in at.markdown)
    assert [b.label for b in at.button] == ["◀ Prev", "Next ▶", "Clear marks", "💾 Save this frame"]


@pytest.mark.yolo
@pytest.mark.skipif(not os.path.exists("data/lecture_video.mp4"), reason="lecture video not present")
def test_calibrate_homography_renders(repo_root):
    at = AppTest.from_file(str(repo_root / "src/campus_occupancy/tools/calibrate_homography.py"), default_timeout=120).run()
    assert not at.exception and any("Click point" in m.value for m in at.markdown)
