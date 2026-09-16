"""One-shot calibration tool: click every seat in Lecture Hall 144 to produce
data/lecture_144_seats.csv (seat_id, x_px, y_px).

Seat IDs auto-increment as you click (S001, S002, …), so no pre-existing
seat list is needed. The CSV is what the Lecture Hall page later uses to
snap YOLO detections to discrete seats.

Run with:
    venv/bin/python -m streamlit run src/campus_occupancy/tools/calibrate_lecture_seats.py

Dependencies (already in the venv):
    streamlit-image-coordinates, pandas, pillow
"""
from __future__ import annotations

import os
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates

FLOOR_PLAN = "assets/lecture_hall_144.png"
OUTPUT_CSV = "data/lecture_144_seats.csv"
DISPLAY_WIDTH = 900  # rendered image width in the browser

PLACED_COLOUR = "#32CD32"   # green, same as Huxley "In Use"
CURRENT_COLOUR = "#FFA500"  # orange, the most-recently placed seat


st.set_page_config(page_title="Lecture Seat Calibrator", layout="wide")


@st.cache_data
def load_image() -> tuple[Image.Image, int, int]:
    img = Image.open(FLOOR_PLAN).convert("RGB")
    return img, img.width, img.height


def load_existing_seats() -> list[tuple[float, float]]:
    """Hydrate the seat list from a previously-saved CSV so the user can resume."""
    if not os.path.exists(OUTPUT_CSV):
        return []
    df = pd.read_csv(OUTPUT_CSV)
    return [(float(row.x_px), float(row.y_px)) for row in df.itertuples()]


def seat_id_for(idx: int) -> str:
    """Position-driven ID; index 0 → 'S001'."""
    return f"S{idx + 1:03d}"


def render_overlay(base: Image.Image, seats: list[tuple[float, float]]) -> Image.Image:
    """Draw all placed seats; the most recent one gets the highlight colour."""
    overlay = base.copy()
    draw = ImageDraw.Draw(overlay)
    r = max(5, base.width // 220)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    last_idx = len(seats) - 1
    for i, (x, y) in enumerate(seats):
        colour = CURRENT_COLOUR if i == last_idx else PLACED_COLOUR
        draw.ellipse((x - r, y - r, x + r, y + r), fill=colour, outline="black")
        if font is not None:
            draw.text((x + r + 2, y - r - 2), seat_id_for(i), fill="black", font=font)
    return overlay


def to_dataframe(seats: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [(seat_id_for(i), x, y) for i, (x, y) in enumerate(seats)],
        columns=["seat_id", "x_px", "y_px"],
    )


def export_csv_bytes(seats: list[tuple[float, float]]) -> bytes:
    buf = BytesIO()
    to_dataframe(seats).to_csv(buf, index=False)
    return buf.getvalue()


# ─────────────────────── session state ───────────────────────

if "seats" not in st.session_state:
    st.session_state.seats = load_existing_seats()
if "saved_count" not in st.session_state:
    st.session_state.saved_count = len(st.session_state.seats)


img, native_w, native_h = load_image()

st.title("🪑 Lecture Hall Seat Calibrator")
st.caption(
    f"Click each seat on the top-down floor plan. Seat IDs are assigned in "
    f"click order (S001, S002, …) and saved to `{OUTPUT_CSV}`. "
    f"Image native size: {native_w} × {native_h} px."
)

# ─────────────────────── sidebar: progress + controls ───────────────────────

with st.sidebar:
    placed = len(st.session_state.seats)
    unsaved = placed != st.session_state.saved_count
    st.metric("Seats placed", placed)
    if unsaved:
        st.warning(f"⚠️ {placed - st.session_state.saved_count} unsaved change(s)"
                   if placed > st.session_state.saved_count
                   else "⚠️ Unsaved changes")

    st.divider()
    col_undo, col_reset = st.columns(2)
    if col_undo.button("↶ Undo last", use_container_width=True,
                       disabled=not st.session_state.seats):
        st.session_state.seats.pop()
        st.rerun()
    if col_reset.button("Reset", use_container_width=True,
                        disabled=not st.session_state.seats):
        st.session_state.seats.clear()
        st.rerun()

    st.divider()
    st.download_button(
        "💾 Download CSV",
        data=export_csv_bytes(st.session_state.seats),
        file_name="lecture_144_seats.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=not st.session_state.seats,
    )
    if st.button(f"Save to {OUTPUT_CSV}", use_container_width=True,
                 type="primary", disabled=not st.session_state.seats):
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        with open(OUTPUT_CSV, "wb") as f:
            f.write(export_csv_bytes(st.session_state.seats))
        st.session_state.saved_count = len(st.session_state.seats)
        st.success(f"Wrote {len(st.session_state.seats)} rows to {OUTPUT_CSV}")

    if st.session_state.seats:
        st.divider()
        with st.expander("Seat table", expanded=False):
            st.dataframe(
                to_dataframe(st.session_state.seats).round(1),
                use_container_width=True, hide_index=True,
            )

# ─────────────────────── main click area ───────────────────────

next_id = seat_id_for(len(st.session_state.seats))
st.markdown(f"### Placing **{next_id}**: click the next seat")

overlay = render_overlay(img, st.session_state.seats)

# streamlit-image-coordinates returns click coords in the *displayed* image
# space. Scale them back to native px so the saved CSV is resolution-stable.
display_w = min(DISPLAY_WIDTH, native_w)
scale = native_w / display_w
click = streamlit_image_coordinates(
    overlay, width=display_w, key=f"floor_{len(st.session_state.seats)}"
)

if click is not None:
    x_native = max(0.0, min(native_w - 1, click["x"] * scale))
    y_native = max(0.0, min(native_h - 1, click["y"] * scale))
    pt = (float(x_native), float(y_native))
    # streamlit-image-coordinates re-yields the last click on rerun; guard
    # against double-appending the same point.
    if not st.session_state.seats or st.session_state.seats[-1] != pt:
        st.session_state.seats.append(pt)
        st.rerun()
