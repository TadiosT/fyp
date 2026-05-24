"""One-shot calibration tool: click each workstation on the floor plan to
produce data/workstation_coordinates.csv (pc_id, x_px, y_px).

Run with:
    streamlit run tools/calibrate_coordinates.py

Dependencies:
    pip install streamlit-image-coordinates
"""
from __future__ import annotations

import os
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

DATA_FILE = "data/huxley_mock_access_logs_with_users.csv"
FLOOR_PLAN = "assets/huxley_lab_floor_plan.jpg"
OUTPUT_CSV = "data/workstation_coordinates.csv"
DISPLAY_WIDTH = 1100  # rendered image width in the browser

STATE_COLOURS = {"placed": "#32CD32", "current": "#FFA500"}


st.set_page_config(page_title="Workstation Coordinate Calibrator", layout="wide")


@st.cache_data
def load_pc_ids() -> list[str]:
    df = pd.read_csv(DATA_FILE, usecols=["pc_id"])
    return sorted(df["pc_id"].unique().tolist())


@st.cache_data
def load_image() -> tuple[Image.Image, int, int]:
    img = Image.open(FLOOR_PLAN).convert("RGB")
    return img, img.width, img.height


def load_existing_coords() -> dict[str, tuple[int, int]]:
    """Resume from a previously-saved CSV if present."""
    if not os.path.exists(OUTPUT_CSV):
        return {}
    df = pd.read_csv(OUTPUT_CSV)
    return {row.pc_id: (int(row.x_px), int(row.y_px)) for row in df.itertuples()}


def render_overlay(
    base: Image.Image,
    coords: dict[str, tuple[int, int]],
    current_pc: str | None,
) -> Image.Image:
    """Draw already-placed dots + a highlight ring for the current PC."""
    overlay = base.copy()
    draw = ImageDraw.Draw(overlay)
    r = max(4, base.width // 250)
    for pc_id, (x, y) in coords.items():
        colour = STATE_COLOURS["current"] if pc_id == current_pc else STATE_COLOURS["placed"]
        draw.ellipse((x - r, y - r, x + r, y + r), fill=colour, outline="black")
    return overlay


def export_csv(coords: dict[str, tuple[int, int]]) -> bytes:
    df = pd.DataFrame(
        [(pc, x, y) for pc, (x, y) in sorted(coords.items())],
        columns=["pc_id", "x_px", "y_px"],
    )
    buf = BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# Session state initialisation
pc_ids = load_pc_ids()
if "coords" not in st.session_state:
    st.session_state.coords = load_existing_coords()
if "cursor" not in st.session_state:
    placed = st.session_state.coords
    first_unplaced = next((i for i, p in enumerate(pc_ids) if p not in placed), 0)
    st.session_state.cursor = first_unplaced
if "history" not in st.session_state:
    st.session_state.history = []  # stack of pc_ids whose placement can be undone

img, native_w, native_h = load_image()

st.title("🖱️ Workstation Coordinate Calibrator")
st.caption(
    f"Click the floor plan to record the location of the highlighted PC. "
    f"Image native size: {native_w} × {native_h} px."
)

# Sidebar — progress + controls
with st.sidebar:
    placed_count = len(st.session_state.coords)
    total = len(pc_ids)
    st.metric("Placed", f"{placed_count} / {total}")
    st.progress(placed_count / total if total else 0.0)

    st.divider()
    jump = st.selectbox(
        "Jump to PC",
        options=pc_ids,
        index=st.session_state.cursor,
        key="jump_select",
    )
    if jump != pc_ids[st.session_state.cursor]:
        st.session_state.cursor = pc_ids.index(jump)
        st.rerun()

    col_skip, col_undo = st.columns(2)
    if col_skip.button("Skip ▶", use_container_width=True):
        st.session_state.cursor = (st.session_state.cursor + 1) % len(pc_ids)
        st.rerun()
    if col_undo.button("Undo ↶", use_container_width=True, disabled=not st.session_state.history):
        last = st.session_state.history.pop()
        st.session_state.coords.pop(last, None)
        st.session_state.cursor = pc_ids.index(last)
        st.rerun()

    st.divider()
    st.download_button(
        "💾 Export CSV",
        data=export_csv(st.session_state.coords),
        file_name="workstation_coordinates.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=not st.session_state.coords,
    )
    if st.button("Save to data/workstation_coordinates.csv", use_container_width=True,
                 disabled=not st.session_state.coords):
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        with open(OUTPUT_CSV, "wb") as f:
            f.write(export_csv(st.session_state.coords))
        st.success(f"Wrote {len(st.session_state.coords)} rows to {OUTPUT_CSV}")

    st.divider()
    unplaced = [p for p in pc_ids if p not in st.session_state.coords]
    with st.expander(f"Unplaced ({len(unplaced)})", expanded=False):
        st.write(unplaced)

# Main area
if st.session_state.cursor >= len(pc_ids):
    st.session_state.cursor = 0
current_pc = pc_ids[st.session_state.cursor]

already = current_pc in st.session_state.coords
banner = f"### Placing: `{current_pc}` ({st.session_state.cursor + 1} / {len(pc_ids)})"
if already:
    x_old, y_old = st.session_state.coords[current_pc]
    banner += f" — already placed at ({x_old}, {y_old}); click to overwrite"
st.markdown(banner)

overlay = render_overlay(img, st.session_state.coords, current_pc)

# streamlit-image-coordinates returns click coords in the *displayed* image space.
# Scale them back to native pixels so the CSV is resolution-stable.
display_w = min(DISPLAY_WIDTH, native_w)
scale = native_w / display_w
click = streamlit_image_coordinates(overlay, width=display_w, key="floor_plan")

if click is not None:
    x_native = int(round(click["x"] * scale))
    y_native = int(round(click["y"] * scale))
    # Clamp to image bounds defensively
    x_native = max(0, min(native_w - 1, x_native))
    y_native = max(0, min(native_h - 1, y_native))

    prev = st.session_state.coords.get(current_pc)
    if prev != (x_native, y_native):
        st.session_state.coords[current_pc] = (x_native, y_native)
        st.session_state.history.append(current_pc)
        # advance to next unplaced PC, wrapping around
        n = len(pc_ids)
        for offset in range(1, n + 1):
            idx = (st.session_state.cursor + offset) % n
            if pc_ids[idx] not in st.session_state.coords:
                st.session_state.cursor = idx
                break
        else:
            st.session_state.cursor = (st.session_state.cursor + 1) % n
        st.rerun()
