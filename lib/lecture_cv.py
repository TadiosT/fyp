"""YOLO + video helpers for the Lecture Hall 144 live page.

Pipeline:
- Crop the bottom-right Picture-in-Picture region of the lecture feed.
- Run YOLOv8s person detection on the cropped frame.
- Project each detection's bbox-bottom-centre (the student's feet) into
  top-down floor-plan pixel space via a calibrated homography. The matrix is
  produced by `tools/calibrate_homography.py` and persisted to
  `data/homography_lecture.npz`.

Note on intro: for the first ~60 seconds the lecture footage takes the whole
frame (the student-room PiP only appears after that). Sampling earlier than
INTRO_OFFSET_SECONDS would feed YOLO the wrong region, so we skip past it.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# Pipeline constants — mirror the Colab notebook's values
CROP_X_PCT = 0.75
CROP_Y_PCT = 0.70
INTRO_OFFSET_SECONDS = 60

PERSON_CLASS_ID = 0  # COCO


@st.cache_resource
def load_model(weights_path: str = "models/yolov8s.pt") -> YOLO:
    return YOLO(weights_path)


@st.cache_data
def get_video_info(path: str) -> dict:
    """Probe video metadata once per session."""
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    crop_w = width - int(width * CROP_X_PCT)
    crop_h = height - int(height * CROP_Y_PCT)
    intro_frame = int(fps * INTRO_OFFSET_SECONDS) if fps > 0 else 0

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "crop_w": crop_w,
        "crop_h": crop_h,
        "intro_frame": intro_frame,
        "duration_s": frame_count / fps if fps > 0 else 0.0,
    }


def read_and_crop(video_path: str, frame_idx: int) -> np.ndarray | None:
    """Open the video, seek to frame_idx, return the bottom-right crop (BGR).

    Returns None if the seek/read fails. Caller is responsible for handling
    EOF wraparound.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return None

    h, w = frame.shape[:2]
    sx, sy = int(w * CROP_X_PCT), int(h * CROP_Y_PCT)
    return frame[sy:h, sx:w]


def detect_people(model: YOLO, frame_bgr: np.ndarray) -> list[tuple[float, float, float, tuple[float, float, float, float]]]:
    """Run YOLO on a cropped frame; return [(cx, cy, conf, (x1,y1,x2,y2)), ...]
    in *cropped* pixel coordinates. Only the person class (0) is kept.
    """
    results = model(frame_bgr, verbose=False)
    detections: list[tuple[float, float, float, tuple[float, float, float, float]]] = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != PERSON_CLASS_ID:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            detections.append((cx, cy, conf, (x1, y1, x2, y2)))
    return detections


@st.cache_data
def load_homography(
    path: str = "data/homography_lecture.npz",
) -> tuple[np.ndarray, tuple[int, int]]:
    """Load the calibrated homography matrix and the floor-plan size it was fit against.

    Returns ``(H, (dst_w, dst_h))``. Cached because the .npz is static between
    calibration runs and we don't want a disk read every tick.
    """
    data = np.load(path)
    H = data["H"]
    dst_w, dst_h = (int(v) for v in data["dst_size"])
    return H, (dst_w, dst_h)


def project_to_floor_plan(
    points: Sequence[tuple[float, float]],
    H: np.ndarray,
    dst_size: tuple[int, int] | None = None,
) -> list[tuple[float, float]]:
    """Map ``(x, y)`` points from cropped-PiP px space into floor-plan px space via H.

    If ``dst_size`` is provided, projections that land outside ``[0, W) × [0, H)``
    are dropped — those are usually detections outside the calibrated seating
    quadrilateral (e.g. the lecturer at the front of the room).
    """
    if not points:
        return []
    arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(arr, H).reshape(-1, 2)
    if dst_size is None:
        return [(float(x), float(y)) for x, y in out]
    w, h = dst_size
    return [(float(x), float(y)) for x, y in out if 0.0 <= x < w and 0.0 <= y < h]


def draw_boxes(frame_bgr: np.ndarray, detections, color=(0, 255, 0)) -> np.ndarray:
    """Return a copy of frame_bgr with bboxes, centre dots, and foot dots drawn.

    The foot point (bbox bottom-centre, blue) is the point fed into the
    homography, so showing both makes it easy to spot off-target detections.
    """
    img = frame_bgr.copy()
    for cx, cy, conf, (x1, y1, x2, y2) in detections:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
        cv2.circle(img, (int(cx), int(cy)), 3, (0, 0, 255), -1)  # centre (red)
        foot_x = int((x1 + x2) / 2)
        foot_y = int(y2)
        cv2.circle(img, (foot_x, foot_y), 4, (255, 0, 0), -1)    # feet (blue)
        label = f"{conf:.2f}"
        cv2.putText(img, label, (int(x1), max(10, int(y1) - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img
