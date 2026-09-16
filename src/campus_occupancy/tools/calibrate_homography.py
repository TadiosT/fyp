"""One-shot calibration tool: capture 4+ corresponding point pairs between the
cropped lecture-feed PiP and the top-down floor plan, fit a homography, save it
to data/homography_lecture.npz.

Run with:
    venv/bin/python -m streamlit run src/campus_occupancy/tools/calibrate_homography.py

The saved .npz contains:
    H:           3x3 float64 homography (cropped px -> floor-plan px)
    src_points:  clicked points in cropped px space (N x 2)
    dst_points:  clicked points in floor-plan px space (N x 2)
    frame_idx:   which video frame was calibrated against
    src_size:    (width, height) of cropped PiP
    dst_size:    (width, height) of floor-plan image
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from campus_occupancy.cv.lecture_cv import get_video_info, read_and_crop  # noqa: E402

VIDEO_PATH = "data/lecture_video.mp4"
FLOOR_PLAN_PATH = "assets/lecture_hall_144.png"
OUTPUT_NPZ = "data/homography_lecture.npz"

# Display widths for the two click panels. Picked so the small PiP becomes
# clickable and the tall floor plan fits next to it without scrolling.
SRC_DISPLAY_W = 640   # cropped PiP is 320x217 -> 2x upscale
DST_DISPLAY_W = 600   # floor plan is 1110x1330 -> ~0.54x downscale

SRC_DOT_COLOUR = "#FF6F00"   # orange
DST_DOT_COLOUR = "#1E88E5"   # blue
PROJECTED_COLOUR = "#00C853"  # green (after-save sanity overlay)


st.set_page_config(page_title="Lecture Hall Homography Calibrator", layout="wide")


# ─────────────────────── helpers ───────────────────────

@st.cache_data
def load_floor_plan() -> Image.Image:
    return Image.open(FLOOR_PLAN_PATH).convert("RGB")


def get_pip(frame_idx: int) -> Image.Image | None:
    """Materialise the cropped PiP for a given frame index as a PIL RGB image."""
    crop_bgr = read_and_crop(VIDEO_PATH, frame_idx)
    if crop_bgr is None:
        return None
    return Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))


def draw_dots(
    base: Image.Image,
    points: list[tuple[float, float]],
    colour: str,
    radius: int | None = None,
    labels: bool = True,
    extra_points: list[tuple[float, float]] | None = None,
    extra_colour: str | None = None,
) -> Image.Image:
    """Render numbered dots on top of base (does not modify the original)."""
    overlay = base.copy()
    draw = ImageDraw.Draw(overlay)
    r = radius or max(4, base.width // 120)
    for i, (x, y) in enumerate(points, start=1):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=colour, outline="black")
        if labels:
            draw.text((x + r + 2, y - r), str(i), fill="black")
    if extra_points and extra_colour:
        for i, (x, y) in enumerate(extra_points, start=1):
            draw.ellipse((x - r, y - r, x + r, y + r), outline=extra_colour, width=2)
    return overlay


def compute_homography(
    src: list[tuple[float, float]],
    dst: list[tuple[float, float]],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if len(src) < 4 or len(src) != len(dst):
        return None, None
    src_arr = np.array(src, dtype=np.float32)
    dst_arr = np.array(dst, dtype=np.float32)
    H, mask = cv2.findHomography(src_arr, dst_arr)
    return H, mask


def project(points: list[tuple[float, float]], H: np.ndarray) -> np.ndarray:
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, H).reshape(-1, 2)


# ─────────────────────── boot checks ───────────────────────

for label, path in [("Video", VIDEO_PATH), ("Floor plan", FLOOR_PLAN_PATH)]:
    if not os.path.exists(path):
        st.error(f"🚨 {label} not found at `{path}`.")
        st.stop()

video = get_video_info(VIDEO_PATH)
fps = video["fps"] or 30.0

# ─────────────────────── session state ───────────────────────

ss = st.session_state
ss.setdefault("hg_src_points", [])
ss.setdefault("hg_dst_points", [])
ss.setdefault("hg_next_side", "src")  # alternates "src" / "dst"
ss.setdefault("hg_frame_idx", min(video["intro_frame"] + int(fps * 30), video["frame_count"] - 1))
ss.setdefault("hg_last_save", None)


# ─────────────────────── UI ───────────────────────

st.title("📐 Lecture Hall Homography Calibrator")
st.caption(
    "Click matching reference points on the cropped video feed (left) and the "
    "top-down floor plan (right). Alternate sides. Four well-spread pairs are "
    "enough; more pairs ⇒ least-squares fit. Tip: aim for the **floor at the "
    "students' feet** (e.g. corners of the seating block, aisle markers)."
)

# Sidebar: frame picker + saved-pair list + controls
with st.sidebar:
    st.subheader("Frame")
    ss.hg_frame_idx = st.slider(
        "Video frame",
        min_value=video["intro_frame"],
        max_value=video["frame_count"] - 1,
        value=ss.hg_frame_idx,
        step=max(1, int(fps)),
        help="Pick a frame where students are visible and reference landmarks are sharp.",
    )
    t = ss.hg_frame_idx / fps if fps else 0.0
    st.caption(f"≈ video time **{t:0.1f}s** / {video['duration_s']:0.0f}s")

    st.divider()
    st.subheader("Pairs")
    pairs_for_table = []
    n_pairs = max(len(ss.hg_src_points), len(ss.hg_dst_points))
    for i in range(n_pairs):
        src = ss.hg_src_points[i] if i < len(ss.hg_src_points) else (None, None)
        dst = ss.hg_dst_points[i] if i < len(ss.hg_dst_points) else (None, None)
        pairs_for_table.append({
            "#": i + 1,
            "src_x": round(src[0], 1) if src[0] is not None else "—",
            "src_y": round(src[1], 1) if src[1] is not None else "—",
            "dst_x": round(dst[0], 1) if dst[0] is not None else "—",
            "dst_y": round(dst[1], 1) if dst[1] is not None else "—",
        })
    if pairs_for_table:
        st.dataframe(pd.DataFrame(pairs_for_table), use_container_width=True, hide_index=True)
    else:
        st.caption("_No clicks yet._")

    col_undo, col_reset = st.columns(2)
    if col_undo.button("↶ Undo", use_container_width=True,
                       disabled=not (ss.hg_src_points or ss.hg_dst_points)):
        # Roll back whichever side was clicked most recently.
        if len(ss.hg_src_points) > len(ss.hg_dst_points):
            ss.hg_src_points.pop()
            ss.hg_next_side = "src"
        elif ss.hg_dst_points:
            ss.hg_dst_points.pop()
            ss.hg_next_side = "dst"
        st.rerun()
    if col_reset.button("Reset", use_container_width=True,
                        disabled=not (ss.hg_src_points or ss.hg_dst_points)):
        ss.hg_src_points.clear()
        ss.hg_dst_points.clear()
        ss.hg_next_side = "src"
        ss.hg_last_save = None
        st.rerun()

    st.divider()
    n_complete_pairs = min(len(ss.hg_src_points), len(ss.hg_dst_points))
    can_save = n_complete_pairs >= 4
    if st.button(f"💾 Compute & Save ({n_complete_pairs} pairs)",
                 use_container_width=True, type="primary", disabled=not can_save):
        H, mask = compute_homography(
            ss.hg_src_points[:n_complete_pairs],
            ss.hg_dst_points[:n_complete_pairs],
        )
        if H is None:
            st.error("Homography fit failed. Check that your points aren't collinear.")
        else:
            floor = load_floor_plan()
            os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
            np.savez(
                OUTPUT_NPZ,
                H=H,
                src_points=np.array(ss.hg_src_points[:n_complete_pairs]),
                dst_points=np.array(ss.hg_dst_points[:n_complete_pairs]),
                frame_idx=ss.hg_frame_idx,
                src_size=np.array([video["crop_w"], video["crop_h"]]),
                dst_size=np.array([floor.width, floor.height]),
            )
            ss.hg_last_save = OUTPUT_NPZ
            st.success(f"Saved {OUTPUT_NPZ}")


# ─────────────────────── main click area ───────────────────────

pip_img = get_pip(ss.hg_frame_idx)
if pip_img is None:
    st.error(f"Could not read frame {ss.hg_frame_idx} from {VIDEO_PATH}.")
    st.stop()

floor_img = load_floor_plan()

# Pre-compute display scales so the click coords can be converted back to native px.
src_native_w, src_native_h = pip_img.size
src_scale = src_native_w / SRC_DISPLAY_W

dst_native_w, dst_native_h = floor_img.size
dst_scale = dst_native_w / DST_DISPLAY_W

# Status line: which side does the tool expect next?
n_src, n_dst = len(ss.hg_src_points), len(ss.hg_dst_points)
expecting = "src" if n_src <= n_dst else "dst"
ss.hg_next_side = expecting
side_word = "video feed (left)" if expecting == "src" else "floor plan (right)"
pair_n = min(n_src, n_dst) + (1 if expecting == "src" else 0) + 1
st.markdown(f"### 👉 Click point **{pair_n}** on the **{side_word}**")

col_l, col_r = st.columns(2)

with col_l:
    st.caption(f"Cropped PiP, frame {ss.hg_frame_idx:,} · native {src_native_w}×{src_native_h}")
    src_overlay = draw_dots(pip_img, ss.hg_src_points, SRC_DOT_COLOUR)
    src_click = streamlit_image_coordinates(
        src_overlay, width=SRC_DISPLAY_W, key=f"src_{ss.hg_frame_idx}_{n_src}_{n_dst}"
    )

with col_r:
    st.caption(f"Top-down floor plan, native {dst_native_w}×{dst_native_h}")
    # Show projected-src markers in green once a fit exists, so you can see how
    # good the current calibration is even before pressing Save.
    extra = None
    H_preview, _ = compute_homography(ss.hg_src_points[:min(n_src, n_dst)],
                                      ss.hg_dst_points[:min(n_src, n_dst)])
    if H_preview is not None and ss.hg_src_points:
        proj = project(ss.hg_src_points[:min(n_src, n_dst)], H_preview)
        extra = [(float(x), float(y)) for x, y in proj]
    dst_overlay = draw_dots(
        floor_img,
        ss.hg_dst_points,
        DST_DOT_COLOUR,
        extra_points=extra,
        extra_colour=PROJECTED_COLOUR,
    )
    dst_click = streamlit_image_coordinates(
        dst_overlay, width=DST_DISPLAY_W, key=f"dst_{ss.hg_frame_idx}_{n_src}_{n_dst}"
    )

# ─────────────────────── click handling ───────────────────────

def _record_click(click: dict, scale: float, native_w: int, native_h: int) -> tuple[float, float]:
    x = max(0.0, min(native_w - 1, click["x"] * scale))
    y = max(0.0, min(native_h - 1, click["y"] * scale))
    return float(x), float(y)


if expecting == "src" and src_click is not None and n_src == len(ss.hg_src_points):
    # Only accept a click when we're expecting the left side AND the click is
    # actually new (streamlit_image_coordinates re-yields the last click on rerun).
    new_pt = _record_click(src_click, src_scale, src_native_w, src_native_h)
    if not ss.hg_src_points or ss.hg_src_points[-1] != new_pt:
        ss.hg_src_points.append(new_pt)
        st.rerun()

if expecting == "dst" and dst_click is not None and n_dst == len(ss.hg_dst_points):
    new_pt = _record_click(dst_click, dst_scale, dst_native_w, dst_native_h)
    if not ss.hg_dst_points or ss.hg_dst_points[-1] != new_pt:
        ss.hg_dst_points.append(new_pt)
        st.rerun()


# ─────────────────────── after-save sanity readout ───────────────────────

if ss.hg_last_save and os.path.exists(ss.hg_last_save):
    st.divider()
    st.subheader("Saved calibration: residual check")
    data = np.load(ss.hg_last_save)
    H = data["H"]
    src = data["src_points"].astype(np.float32)
    dst = data["dst_points"].astype(np.float32)
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    residuals = np.linalg.norm(proj - dst, axis=1)
    table = pd.DataFrame({
        "#": range(1, len(src) + 1),
        "src_x": src[:, 0].round(1),
        "src_y": src[:, 1].round(1),
        "dst_x": dst[:, 0].round(1),
        "dst_y": dst[:, 1].round(1),
        "projected_x": proj[:, 0].round(1),
        "projected_y": proj[:, 1].round(1),
        "residual_px": residuals.round(2),
    })
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(
        f"Mean residual: **{residuals.mean():.2f} px** · "
        f"Max: **{residuals.max():.2f} px**. "
        "Exact (≈0) for 4-point fits; small but non-zero for ≥5-point least-squares fits. "
        "Large residuals → re-click that pair."
    )
