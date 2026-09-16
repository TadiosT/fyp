"""Page-level tests through Streamlit's AppTest, on the hermetic synthetic lab."""
import os
import time

import pytest
from streamlit.testing.v1 import AppTest

from campus_occupancy.data_io.preload import preload_status



def page(repo_root, rel, timeout=120):
    return AppTest.from_file(str(repo_root / rel), default_timeout=timeout)


def captions(at):
    return [c.value for c in at.caption]


def wait_preloaded(data_file, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = preload_status(data_file)
        if s and s["ready"]:
            return
        time.sleep(0.1)
    raise AssertionError("preload did not finish")


# ─────────────────────── home ───────────────────────

def test_home_cards_and_live_preload_status(repo_root, patched_lab_configs):
    at = page(repo_root, "app.py").run()
    assert not at.exception
    assert [s.value for s in at.subheader] == ["💻 Huxley Labs", "🎓 Lecture Hall 144", "🏙️ White City Labs"]
    assert [b.label for b in at.button] == ["Open Huxley dashboard →", "Open Lecture Hall view →", "Open White City view →"]
    wait_preloaded(patched_lab_configs.page.data_file)
    at = at.run()
    ready = [c for c in captions(at) if "⚡ Data preloaded" in c]
    import pandas as pd
    n_rows = len(pd.read_csv(patched_lab_configs.page.data_file, usecols=["pc_id"]))
    assert len(ready) == 2 and all(f"{n_rows:,} rows" in c for c in ready)


# ─────────────────────── Huxley (sensorless config) ───────────────────────

def test_huxley_page_renders_and_trend_ranges(repo_root, patched_lab_configs):
    at = page(repo_root, "src/campus_occupancy/pages/1_Huxley_Labs.py").run()
    assert not at.exception
    assert at.title[0].value.startswith("Huxley")
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Total Monitored PCs"] == str(patched_lab_configs.n_desks)
    assert "Loaded on demand" in captions(at)[0] or "Preloaded" in captions(at)[0]
    assert "Virtual Air-Quality Sensors" not in [s.value for s in at.subheader]
    for rng in ["This Week", "This Month", "Last 3 Months", "Today"]:
        at = at.radio[0].set_value(rng).run()
        assert not at.exception, rng
        assert not at.info                                        # history present → no "not available"
    at = at.checkbox[0].check().run()
    assert not at.exception and at.dataframe


def test_huxley_refresh_button(repo_root, patched_lab_configs):
    at = page(repo_root, "src/campus_occupancy/pages/1_Huxley_Labs.py").run()
    at = at.button[0].click().run()
    assert not at.exception


# ─────────────────────── White City (sensors + session simulator) ───────────────────────

def test_white_city_sensors_and_session_simulator(repo_root, patched_lab_configs):
    at = page(repo_root, "src/campus_occupancy/pages/3_White_City_Labs.py").run()
    assert not at.exception
    subs = [s.value for s in at.subheader]
    assert "🌿 Virtual Air-Quality Sensors" in subs and "🧪 Lab-Session Simulator" in subs
    live_rows = [m.value for m in at.markdown if "🟢 live" in m.value]
    assert len(live_rows) == patched_lab_configs.n_sensors
    assert any("Set the session parameters" in i.value for i in at.info)

    run_btn = [b for b in at.button if "Run session" in b.label]
    assert run_btn, [b.label for b in at.button]
    at = run_btn[0].click().run()
    assert not at.exception
    recs = [m.value for m in at.markdown if "Busy lab" in m.value]
    assert recs and any("Quiet study" in m.value for m in at.markdown)
    assert any(s.label == "Session minute" for s in at.slider)
    minute = [s for s in at.slider if s.label == "Session minute"][0]
    at = minute.set_value(5).run()
    assert not at.exception and any("session students seated" in c for c in captions(at))


def test_white_city_missing_data_hint(repo_root, patched_lab_configs, monkeypatch, tmp_path):
    import campus_occupancy.config as lc
    monkeypatch.setattr(lc.WC_CFG, "data_file", str(tmp_path / "missing.csv"))
    at = page(repo_root, "src/campus_occupancy/pages/3_White_City_Labs.py").run()
    assert not at.exception and at.error
    assert "missing.csv" in at.error[0].value and lc.WC_CFG.missing_data_hint in at.error[0].value
    assert not at.metric                                          # page stopped after the error


# ─────────────────────── Lecture Hall ───────────────────────

def test_lecture_hall_preconditions(repo_root, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = page(repo_root, "src/campus_occupancy/pages/2_Lecture_Hall.py").run()
    assert not at.exception and at.error and "Video not found" in at.error[0].value


ASSETS = ["data/lecture_video.mp4", "models/yolov8s.pt", "data/homography_lecture.npz", "data/lecture_144_seats.csv",
          "assets/lecture_hall_144.png"]


@pytest.mark.yolo
@pytest.mark.skipif(not all(os.path.exists(p) for p in ASSETS), reason="lecture-hall assets not present")
def test_lecture_hall_full_render_and_play(repo_root):
    at = page(repo_root, "src/campus_occupancy/pages/2_Lecture_Hall.py", timeout=300).run()
    assert not at.exception
    assert any(s.value.startswith("Seats occupied") for s in at.subheader)
    play = [b for b in at.button if "Play" in b.label][0]
    at = play.click().run()
    assert not at.exception
