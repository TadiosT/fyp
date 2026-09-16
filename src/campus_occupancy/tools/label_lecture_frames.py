"""Ground-truth labelling tool for the lecture-hall evaluation (experiment E1).

Shows 30 frames sampled from the lecture recording (every 90 s after the
intro) next to the seat map. For each frame you: enter the head count you
can see in the cropped region, click the seats that are occupied on the plan
(click again to un-mark), and press Save. Labels go to
``data/eval/lecture_labels.csv`` and the tool resumes from it.

The tool is *blind*: it never runs or shows the detector, so labels are not
biased by the system under test.

Run:  venv/bin/python -m streamlit run src/campus_occupancy/tools/label_lecture_frames.py
"""
from __future__ import annotations

import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from campus_occupancy.cv.lecture_cv import get_video_info, read_and_crop

VIDEO_PATH = "data/lecture_video.mp4"
SEATS_CSV = "data/lecture_144_seats.csv"
FLOOR_PLAN = "assets/lecture_hall_144.png"
LABELS_CSV = "data/eval/lecture_labels.csv"

N_FRAMES = 30
FIRST_OFFSET_S = 30          # seconds after the intro for sample 0
STEP_S = 90                  # seconds between samples
CROP_DISPLAY_W = 900         # cropped PiP is 320x217 → ~2.8x upscale
PLAN_DISPLAY_W = 520
CLICK_RADIUS = 28            # floor-plan px: nearest seat within this toggles

OCCUPIED = "#32CD32"
EMPTY = "#FF0000"


st.set_page_config(page_title="Lecture frame labeller", layout="wide")


@st.cache_data
def load_seats() -> pd.DataFrame:
    df = pd.read_csv(SEATS_CSV)
    df["x_px"] = df["x_px"].astype(float)
    df["y_px"] = df["y_px"].astype(float)
    return df


@st.cache_data
def load_plan() -> Image.Image:
    return Image.open(FLOOR_PLAN).convert("RGB")


@st.cache_data
def sample_frames() -> list[int]:
    info = get_video_info(VIDEO_PATH)
    fps = info["fps"] or 30.0
    return [min(info["frame_count"] - 1, info["intro_frame"] + int(fps * (FIRST_OFFSET_S + STEP_S * k)))
            for k in range(N_FRAMES)]


def load_labels() -> dict[int, dict]:
    if not os.path.exists(LABELS_CSV):
        return {}
    df = pd.read_csv(LABELS_CSV, keep_default_na=False)
    out = {}
    for r in df.itertuples(index=False):
        seats = [s for s in str(r.occupied_seats).split(";") if s]
        out[int(r.frame_idx)] = {"sample_k": int(r.sample_k), "head_count": int(r.head_count),
                                 "seats": seats, "labelled_at": r.labelled_at}
    return out


def save_labels(labels: dict[int, dict]) -> None:
    os.makedirs(os.path.dirname(LABELS_CSV), exist_ok=True)
    rows = [{"sample_k": v["sample_k"], "frame_idx": k, "head_count": v["head_count"],
             "occupied_seats": ";".join(v["seats"]), "n_seats": len(v["seats"]),
             "labelled_at": v["labelled_at"]} for k, v in sorted(labels.items())]
    pd.DataFrame(rows, columns=["sample_k", "frame_idx", "head_count", "occupied_seats", "n_seats",
                                "labelled_at"]).to_csv(LABELS_CSV, index=False)


def draw_plan(plan: Image.Image, seats: pd.DataFrame, marked: set[str]) -> Image.Image:
    img = plan.copy()
    d = ImageDraw.Draw(img)
    r = max(5, plan.width // 150)
    for s in seats.itertuples(index=False):
        colour = OCCUPIED if s.seat_id in marked else EMPTY
        d.ellipse((s.x_px - r, s.y_px - r, s.x_px + r, s.y_px + r), fill=colour, outline="black")
    return img


# ─────────────────────── state ───────────────────────

for label, path in [("Video", VIDEO_PATH), ("Seat map", SEATS_CSV), ("Floor plan", FLOOR_PLAN)]:
    if not os.path.exists(path):
        st.error(f"{label} not found at `{path}`.")
        st.stop()

frames = sample_frames()
seats = load_seats()
plan = load_plan()
ss = st.session_state
ss.setdefault("lbl_labels", load_labels())
ss.setdefault("lbl_k", next((k for k, f in enumerate(frames) if f not in ss.lbl_labels), 0))
ss.setdefault("lbl_marked", None)   # set[str] for the current frame, or None until initialised

k = ss.lbl_k
frame_idx = frames[k]
existing = ss.lbl_labels.get(frame_idx)
if ss.lbl_marked is None:
    ss.lbl_marked = set(existing["seats"]) if existing else set()

# ─────────────────────── sidebar ───────────────────────

with st.sidebar:
    st.subheader("Progress")
    done = sum(1 for f in frames if f in ss.lbl_labels)
    st.metric("Frames labelled", f"{done} / {N_FRAMES}")
    st.progress(done / N_FRAMES)
    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("◀ Prev", use_container_width=True, disabled=k == 0):
        ss.lbl_k, ss.lbl_marked = k - 1, None
        st.rerun()
    if c2.button("Next ▶", use_container_width=True, disabled=k == N_FRAMES - 1):
        ss.lbl_k, ss.lbl_marked = k + 1, None
        st.rerun()
    jump = st.selectbox("Jump to sample", list(range(N_FRAMES)), index=k,
                        format_func=lambda i: f"{i:02d}: frame {frames[i]:,}" + (" ✓" if frames[i] in ss.lbl_labels else ""))
    if jump != k:
        ss.lbl_k, ss.lbl_marked = jump, None
        st.rerun()
    st.divider()
    head = st.number_input("Head count in the crop", min_value=0, max_value=200,
                           value=int(existing["head_count"]) if existing else len(ss.lbl_marked), step=1)
    st.caption(f"Seats marked: **{len(ss.lbl_marked)}**")
    if st.button("Clear marks", use_container_width=True, disabled=not ss.lbl_marked):
        ss.lbl_marked = set()
        st.rerun()
    if st.button("💾 Save this frame", type="primary", use_container_width=True):
        ss.lbl_labels[frame_idx] = {"sample_k": k, "head_count": int(head), "seats": sorted(ss.lbl_marked),
                                    "labelled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        save_labels(ss.lbl_labels)
        st.success(f"Saved frame {frame_idx:,} ({int(head)} people, {len(ss.lbl_marked)} seats).")
        if k < N_FRAMES - 1:
            ss.lbl_k, ss.lbl_marked = k + 1, None
            st.rerun()
    st.divider()
    st.caption("Protocol: count every person visible in the crop (including standing). Mark a seat only "
               "if someone is sitting in it. The tool never shows detector output.")

# ─────────────────────── main ───────────────────────

info = get_video_info(VIDEO_PATH)
t_s = frame_idx / (info["fps"] or 30.0)
st.markdown(f"### Sample {k:02d} / {N_FRAMES - 1}: frame {frame_idx:,} (video time {t_s / 60:.1f} min)"
            + ("  ·  ✓ labelled" if existing else ""))

crop = read_and_crop(VIDEO_PATH, frame_idx)
if crop is None:
    st.error("Could not read this frame.")
    st.stop()

left, right = st.columns([1.4, 1])
with left:
    st.caption("Cropped picture-in-picture region (enlarged)")
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    big = cv2.resize(rgb, (CROP_DISPLAY_W, int(CROP_DISPLAY_W * rgb.shape[0] / rgb.shape[1])),
                     interpolation=cv2.INTER_CUBIC)
    st.image(big, use_container_width=True)
with right:
    st.caption("Click occupied seats (click again to un-mark)")
    overlay = draw_plan(plan, seats, ss.lbl_marked)
    scale = plan.width / PLAN_DISPLAY_W
    click = streamlit_image_coordinates(overlay, width=PLAN_DISPLAY_W,
                                        key=f"plan_{frame_idx}_{len(ss.lbl_marked)}")
    if click is not None:
        x, y = click["x"] * scale, click["y"] * scale
        d = np.hypot(seats.x_px - x, seats.y_px - y)
        i = int(d.idxmin())
        if d.iloc[i] <= CLICK_RADIUS:
            sid = seats.seat_id.iloc[i]
            if sid in ss.lbl_marked:
                ss.lbl_marked.discard(sid)
            else:
                ss.lbl_marked.add(sid)
            st.rerun()
