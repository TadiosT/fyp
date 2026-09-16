import os

import numpy as np
import pytest

from campus_occupancy.cv import lecture_cv
from campus_occupancy.cv.lecture_cv import (
    CROP_X_PCT, CROP_Y_PCT, INTRO_OFFSET_SECONDS, detect_people, draw_boxes, project_to_floor_plan,
    read_and_crop,
)
from tests.support.helpers import synthetic_frame, tiny_video


def unwrap(fn):
    return getattr(fn, "__wrapped__", fn)


# ─────────────────────── projection ───────────────────────

def test_project_identity_and_empty():
    H = np.eye(3)
    assert project_to_floor_plan([(10, 20), (-5, 5)], H) == [(10.0, 20.0), (-5.0, 5.0)]
    assert project_to_floor_plan([], H) == []


def test_project_clips_to_dst_size():
    H = np.eye(3)
    out = project_to_floor_plan([(10, 20), (-5, 5), (100, 100), (49.9, 49.9)], H, dst_size=(50, 50))
    assert out == [(10.0, 20.0), (49.9, 49.9)] or [tuple(round(v, 1) for v in p) for p in out] == [(10.0, 20.0), (49.9, 49.9)]


def test_project_applies_homography():
    H = np.array([[2, 0, 5], [0, 3, 7], [0, 0, 1]], dtype=float)   # scale + translate
    (x, y), = project_to_floor_plan([(1, 1)], H)
    assert (x, y) == pytest.approx((7.0, 10.0))


def test_load_homography_from_npz(tmp_path):
    H = np.array([[1.5, 0, 3], [0, 2.0, 4], [0, 0, 1]])
    p = tmp_path / "h.npz"
    np.savez(p, H=H, dst_size=np.array([1110, 1330]), src_points=np.zeros((4, 2)), dst_points=np.zeros((4, 2)))
    H2, size = unwrap(lecture_cv.load_homography)(str(p))
    assert np.allclose(H2, H) and size == (1110, 1330)


# ─────────────────────── video ───────────────────────

@pytest.fixture
def video(tmp_path):
    p = tmp_path / "v.mp4"
    tiny_video(p, w=64, h=48, n_frames=10, fps=10.0)
    return str(p)


def test_video_info_fields(video):
    info = unwrap(lecture_cv.get_video_info)(video)
    assert (info["width"], info["height"], info["frame_count"]) == (64, 48, 10)
    assert info["fps"] == pytest.approx(10.0)
    assert info["crop_w"] == 64 - int(64 * CROP_X_PCT) and info["crop_h"] == 48 - int(48 * CROP_Y_PCT)
    assert info["intro_frame"] == int(10.0 * INTRO_OFFSET_SECONDS)
    assert info["duration_s"] == pytest.approx(1.0)


def test_read_and_crop_geometry_and_eof(video):
    crop = read_and_crop(video, 2)
    assert crop.shape == (48 - int(48 * CROP_Y_PCT), 64 - int(64 * CROP_X_PCT), 3)
    assert read_and_crop(video, 500) is None


def test_video_info_missing_file(tmp_path):
    info = unwrap(lecture_cv.get_video_info)(str(tmp_path / "missing.mp4"))
    assert info["frame_count"] == 0 and info["intro_frame"] == 0 and info["duration_s"] == 0.0


# ─────────────────────── detection with a fake model ───────────────────────

class _T:
    def __init__(self, a):
        self.a = np.array(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Box:
    def __init__(self, cls, xyxy, conf):
        self.cls = [cls]
        self.xyxy = [_T(xyxy)]
        self.conf = [_T(conf)]


class _Res:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    def __init__(self, boxes):
        self._boxes = boxes
        self.calls = []

    def __call__(self, frame, verbose=False):
        self.calls.append((frame.shape, verbose))
        return [_Res(self._boxes)]


def test_detect_people_keeps_only_persons():
    model = FakeModel([_Box(0, [0, 0, 10, 20], 0.9), _Box(2, [5, 5, 8, 8], 0.8), _Box(0, [10, 10, 30, 50], 0.5)])
    det = detect_people(model, synthetic_frame(64, 48))
    assert det == [(5.0, 10.0, pytest.approx(0.9), (0.0, 0.0, 10.0, 20.0)),
                   (20.0, 30.0, pytest.approx(0.5), (10.0, 10.0, 30.0, 50.0))]
    assert model.calls[0][1] is False


def test_draw_boxes_copies_and_marks():
    frame = synthetic_frame(64, 48)
    before = frame.copy()
    out = draw_boxes(frame, [(5.0, 10.0, 0.9, (0.0, 0.0, 10.0, 20.0))])
    assert out.shape == frame.shape and np.array_equal(frame, before)
    assert not np.array_equal(out, frame)


# ─────────────────────── real YOLO (opt-in) ───────────────────────

MODEL, VIDEO = "models/yolov8s.pt", "data/lecture_video.mp4"


@pytest.mark.yolo
@pytest.mark.skipif(not (os.path.exists(MODEL) and os.path.exists(VIDEO)), reason="YOLO weights / video not present")
def test_real_detection_on_real_frame():
    info = unwrap(lecture_cv.get_video_info)(VIDEO)
    crop = read_and_crop(VIDEO, info["intro_frame"] + int(info["fps"] * 30))
    assert crop is not None
    model = unwrap(lecture_cv.load_model)(MODEL)
    det = detect_people(model, crop)
    assert isinstance(det, list)
    for cx, cy, conf, (x1, y1, x2, y2) in det:
        assert 0 < conf <= 1 and x1 <= cx <= x2 and y1 <= cy <= y2
