import os
import sys
import time

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.floor_map import build_floor_map  # noqa: E402
from lib.lecture_cv import (  # noqa: E402
    detect_people,
    draw_boxes,
    get_video_info,
    load_homography,
    load_model,
    project_to_floor_plan,
    read_and_crop,
)
from lib.seating import assess_zones, zone_shapes  # noqa: E402

VIDEO_PATH = "data/lecture_video.mp4"
MODEL_PATH = "models/yolov8s.pt"
FLOOR_PLAN_PATH = "assets/lecture_hall_144.png"
HOMOGRAPHY_PATH = "data/homography_lecture.npz"

DETECTION_COLOR = "#1E88E5"

st.set_page_config(
    page_title="Lecture Hall 144 — Live Occupancy",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Lecture Hall 144 — Live Occupancy")

# Preconditions
for label, path in [
    ("Video", VIDEO_PATH),
    ("Model weights", MODEL_PATH),
    ("Floor plan", FLOOR_PLAN_PATH),
]:
    if not os.path.exists(path):
        st.error(f"🚨 {label} not found at `{path}`.")
        st.stop()

if not os.path.exists(HOMOGRAPHY_PATH):
    st.error(
        f"🚨 Homography matrix not found at `{HOMOGRAPHY_PATH}`. "
        "Run the calibration tool first:\n\n"
        "```\nvenv/bin/python -m streamlit run tools/calibrate_homography.py\n```"
    )
    st.stop()


@st.cache_data
def load_floor_plan():
    return Image.open(FLOOR_PLAN_PATH)


video = get_video_info(VIDEO_PATH)

# Session state
ss = st.session_state
ss.setdefault("lh_playing", False)
ss.setdefault("lh_frame_idx", video["intro_frame"])
ss.setdefault("lh_latest_points", [])
ss.setdefault("lh_latest_detections", [])
ss.setdefault("lh_last_inference_ms", 0.0)
ss.setdefault("lh_tick_count", 0)
ss.setdefault("lh_interval_s", 2)
ss.setdefault("lh_zones", None)
ss.setdefault("lh_recommended", None)

# Controls row
ctrl_l, ctrl_m, ctrl_r = st.columns([1, 1, 3])
with ctrl_l:
    label = "⏸ Pause" if ss.lh_playing else "▶ Play"
    if st.button(label, use_container_width=True, type="primary"):
        ss.lh_playing = not ss.lh_playing
with ctrl_m:
    if st.button("↻ Restart", use_container_width=True):
        ss.lh_frame_idx = video["intro_frame"]
        ss.lh_latest_points = []
        ss.lh_latest_detections = []
        ss.lh_tick_count = 0
with ctrl_r:
    ss.lh_interval_s = st.slider(
        "Sample interval (seconds of wall time per tick)",
        min_value=1, max_value=5, value=ss.lh_interval_s, step=1,
    )

# Stats row
fps = video["fps"] or 1.0
current_video_time = ss.lh_frame_idx / fps
st.caption(
    f"Frame **{ss.lh_frame_idx:,}** / {video['frame_count']:,} · "
    f"Video time **{current_video_time:0.1f}s** / {video['duration_s']:0.0f}s · "
    f"Crop **{video['crop_w']}×{video['crop_h']}** · "
    f"Last inference **{ss.lh_last_inference_ms:0.0f}ms** · "
    f"Ticks **{ss.lh_tick_count}**"
)


def _render_recommendation():
    """Banner above the map showing the recommended seating zone."""
    recommended = ss.lh_recommended
    zones = ss.lh_zones
    if recommended is None or zones is None:
        st.info("Hit **▶ Play** to start detection and get a seating recommendation.")
        return

    total_free = sum(z["free"] for z in zones)
    if total_free == 0:
        st.error("🚫 Hall is at capacity — no seats remaining.")
        return

    name = recommended["name"]
    free = recommended["free"]
    cap = recommended["capacity"]

    is_first = (recommended == zones[0])
    if is_first:
        msg = f"👉 **Sit in the {name}** rows — {free} of {cap} seats free."
        st.success(msg)
    else:
        prev = zones[zones.index(recommended) - 1]["name"]
        msg = f"⚠️ {prev} is full. Try the **{name}** — {free} of {cap} seats free."
        st.warning(msg)

    cols = st.columns(len(zones))
    for col, z in zip(cols, zones):
        pct = z["count"] / z["capacity"] * 100 if z["capacity"] else 0
        label = f"{z['name']} ◀" if z is recommended else z["name"]
        col.metric(
            label=label,
            value=f"{z['count']} / {z['capacity']}",
            delta=f"{z['free']} free · {pct:.0f}% full",
            delta_color="off",
        )


def _render_map():
    """Plotly floor plan with current detections overlaid."""
    img = load_floor_plan()
    points = ss.lh_latest_points

    points_df = pd.DataFrame(points, columns=["x_px", "y_px"]) if points else \
                pd.DataFrame(columns=["x_px", "y_px"])
    points_df["category"] = "Detected person"
    points_df["person_id"] = [f"#{i+1}" for i in range(len(points_df))]

    shapes = zone_shapes(ss.lh_recommended) if ss.lh_recommended is not None else None

    fig = build_floor_map(
        image=img,
        points_df=points_df,
        color_col="category",
        color_map={"Detected person": DETECTION_COLOR},
        custom_data_cols=["person_id", "x_px", "y_px"],
        hover_template=(
            "<b>%{customdata[0]}</b><br>"
            "x: %{customdata[1]:.0f}<br>"
            "y: %{customdata[2]:.0f}"
            "<extra></extra>"
        ),
        marker={"size": 10, "opacity": 0.75, "line": {"width": 1, "color": "white"}},
        legend_title="Live detections",
        height=700,
        extra_shapes=shapes,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_debug_expander():
    """Cropped feed with YOLO overlays — useful while tuning, collapsed by default."""
    with st.expander("🔍 Debug: cropped feed with YOLO overlays", expanded=False):
        crop = read_and_crop(VIDEO_PATH, ss.lh_frame_idx)
        if crop is None:
            st.info("No frame available yet.")
            return
        annotated = draw_boxes(crop, ss.lh_latest_detections)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(rgb, caption=f"Frame {ss.lh_frame_idx:,} — cropped PiP", use_container_width=True)
        if ss.lh_latest_points:
            pts_df = pd.DataFrame(ss.lh_latest_points, columns=["x_px", "y_px"]).round(1)
            pts_df.insert(0, "#", range(1, len(pts_df) + 1))
            st.dataframe(pts_df, use_container_width=True, hide_index=True)


@st.fragment(run_every=f"{ss.lh_interval_s}s")
def cv_tick():
    """Advance video + run YOLO + render map. Re-runs only this fragment."""
    if ss.lh_playing:
        step = max(1, int(fps * ss.lh_interval_s))
        next_idx = ss.lh_frame_idx + step
        if next_idx >= video["frame_count"]:
            next_idx = video["intro_frame"]
        ss.lh_frame_idx = next_idx

        crop = read_and_crop(VIDEO_PATH, ss.lh_frame_idx)
        if crop is not None:
            model = load_model(MODEL_PATH)
            t0 = time.perf_counter()
            detections = detect_people(model, crop)
            ss.lh_last_inference_ms = (time.perf_counter() - t0) * 1000.0

            # Foot point (bbox bottom-centre) is the one lying on the floor
            # plane the homography models, not the bbox centre.
            feet = [((x1 + x2) / 2.0, y2)
                    for (_, _, _, (x1, y1, x2, y2)) in detections]
            H, dst_size = load_homography(HOMOGRAPHY_PATH)
            ss.lh_latest_points = project_to_floor_plan(feet, H, dst_size=dst_size)
            ss.lh_latest_detections = detections
            ss.lh_tick_count += 1

        zones, recommended = assess_zones(ss.lh_latest_points)
        ss.lh_zones = zones
        ss.lh_recommended = recommended

    st.subheader(f"Detected people: {len(ss.lh_latest_points)}")
    _render_recommendation()
    _render_map()
    _render_debug_expander()


cv_tick()
