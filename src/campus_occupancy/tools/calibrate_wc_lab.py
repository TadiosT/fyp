"""Calibration tool for the White City computer labs floor plan.

Two modes (sidebar radio):

* **Desk rows**: click the two ends of a straight desk row and enter how many
  desks it holds; the tool interpolates evenly spaced desk positions. IDs are
  assigned per zone (``WC1A-001`` …). Undo removes a whole row. Saves
  ``data/wc_workstation_coordinates.csv`` (``pc_id,x_px,y_px,zone``) plus a
  sidecar ``data/wc_desk_rows.json`` so the tool can resume with row structure.

* **Sensors**: each click places a virtual Netatmo sensor (``WC-S01`` …).
  Saves ``data/wc_sensor_positions.csv`` (``sensor_id,x_px,y_px``).

Run with:
    venv/bin/python -m streamlit run src/campus_occupancy/tools/calibrate_wc_lab.py
"""
from __future__ import annotations

import json
import os
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates

FLOOR_PLAN = "assets/wc_lab.png"
DESKS_CSV = "data/wc_workstation_coordinates.csv"
ROWS_JSON = "data/wc_desk_rows.json"
SENSORS_CSV = "data/wc_sensor_positions.csv"
DISPLAY_WIDTH = 900

ZONES = ["1a", "1b", "2a", "2b", "other"]

DESK_COLOUR = "#32CD32"      # green
PENDING_COLOUR = "#FFA500"   # orange, first endpoint of the row in progress
SENSOR_COLOUR = "#1E88E5"    # blue squares


st.set_page_config(page_title="White City Lab Calibrator", layout="wide")


# ─────────────────────── helpers ───────────────────────

@st.cache_data
def load_image() -> tuple[Image.Image, int, int]:
    img = Image.open(FLOOR_PLAN).convert("RGB")
    return img, img.width, img.height


def zone_prefix(zone: str) -> str:
    return f"WC{zone.upper()}"


def derive_desks(rows: list[dict]) -> list[dict]:
    """Expand row definitions into individual desks with per-zone sequential IDs."""
    counters: dict[str, int] = {}
    desks: list[dict] = []
    for row in rows:
        zone = row["zone"]
        n = max(1, int(row["n"]))
        (x0, y0), (x1, y1) = row["start"], row["end"]
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        for x, y in zip(xs, ys):
            counters[zone] = counters.get(zone, 0) + 1
            desks.append({
                "pc_id": f"{zone_prefix(zone)}-{counters[zone]:03d}",
                "x_px": round(float(x), 1),
                "y_px": round(float(y), 1),
                "zone": zone,
            })
    return desks


def load_rows() -> list[dict]:
    if not os.path.exists(ROWS_JSON):
        return []
    with open(ROWS_JSON) as f:
        data = json.load(f)
    return [
        {"zone": r["zone"], "start": tuple(r["start"]), "end": tuple(r["end"]), "n": int(r["n"])}
        for r in data
    ]


def load_sensors() -> list[tuple[float, float]]:
    if not os.path.exists(SENSORS_CSV):
        return []
    df = pd.read_csv(SENSORS_CSV)
    return [(float(r.x_px), float(r.y_px)) for r in df.itertuples()]


def sensor_id_for(idx: int) -> str:
    return f"WC-S{idx + 1:02d}"


def render_overlay(base, desks, pending, sensors) -> Image.Image:
    overlay = base.copy()
    draw = ImageDraw.Draw(overlay)
    r = max(4, base.width // 300)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for d in desks:
        x, y = d["x_px"], d["y_px"]
        draw.ellipse((x - r, y - r, x + r, y + r), fill=DESK_COLOUR, outline="black")

    if pending is not None:
        x, y = pending
        rr = r + 3
        draw.ellipse((x - rr, y - rr, x + rr, y + rr), fill=PENDING_COLOUR, outline="black")

    sr = r + 4
    for i, (x, y) in enumerate(sensors):
        draw.rectangle((x - sr, y - sr, x + sr, y + sr), fill=SENSOR_COLOUR, outline="black")
        if font is not None:
            draw.text((x + sr + 2, y - sr), sensor_id_for(i), fill="black", font=font)
    return overlay


def desks_csv_bytes(desks: list[dict]) -> bytes:
    buf = BytesIO()
    pd.DataFrame(desks, columns=["pc_id", "x_px", "y_px", "zone"]).to_csv(buf, index=False)
    return buf.getvalue()


def sensors_csv_bytes(sensors) -> bytes:
    buf = BytesIO()
    pd.DataFrame(
        [(sensor_id_for(i), x, y) for i, (x, y) in enumerate(sensors)],
        columns=["sensor_id", "x_px", "y_px"],
    ).to_csv(buf, index=False)
    return buf.getvalue()


# ─────────────────────── session state ───────────────────────

ss = st.session_state
ss.setdefault("wc_rows", load_rows())
ss.setdefault("wc_pending", None)
ss.setdefault("wc_sensors", load_sensors())
ss.setdefault("wc_saved_rows", len(ss.wc_rows))
ss.setdefault("wc_saved_sensors", len(ss.wc_sensors))
ss.setdefault("wc_desks_per_row", 10)

img, native_w, native_h = load_image()
desks = derive_desks(ss.wc_rows)

st.title("🏙️ White City Lab Calibrator")
st.caption(
    f"Floor plan native size {native_w} × {native_h} px. "
    "Desk rows: click row start, then row end. Sensors: click to place."
)

# ─────────────────────── sidebar ───────────────────────

with st.sidebar:
    mode = st.radio("Mode", ["Desk rows", "Sensors"], key="wc_mode")
    st.divider()

    if mode == "Desk rows":
        zone = st.selectbox("Zone", ZONES, key="wc_zone")
        ss.wc_desks_per_row = st.number_input(
            "Desks in this row", min_value=1, max_value=60,
            value=int(ss.wc_desks_per_row), step=1,
        )
        st.metric("Rows placed", len(ss.wc_rows))
        st.metric("Desks total", len(desks))
        per_zone = pd.Series([d["zone"] for d in desks]).value_counts() if desks else pd.Series(dtype=int)
        if not per_zone.empty:
            st.caption(" · ".join(f"{z}: {c}" for z, c in per_zone.items()))
        if len(ss.wc_rows) != ss.wc_saved_rows:
            st.warning("Unsaved row changes")

        c1, c2 = st.columns(2)
        if c1.button("↶ Undo row", use_container_width=True, disabled=not ss.wc_rows):
            ss.wc_rows.pop()
            st.rerun()
        if c2.button("Cancel point", use_container_width=True, disabled=ss.wc_pending is None):
            ss.wc_pending = None
            st.rerun()
        if st.button("Reset all rows", use_container_width=True, disabled=not ss.wc_rows):
            ss.wc_rows.clear()
            ss.wc_pending = None
            st.rerun()

        st.divider()
        st.download_button(
            "💾 Download desks CSV", data=desks_csv_bytes(desks),
            file_name="wc_workstation_coordinates.csv", mime="text/csv",
            use_container_width=True, disabled=not desks,
        )
        if st.button(f"Save to {DESKS_CSV}", type="primary",
                     use_container_width=True, disabled=not desks):
            os.makedirs(os.path.dirname(DESKS_CSV), exist_ok=True)
            with open(DESKS_CSV, "wb") as f:
                f.write(desks_csv_bytes(desks))
            with open(ROWS_JSON, "w") as f:
                json.dump(ss.wc_rows, f, indent=1)
            ss.wc_saved_rows = len(ss.wc_rows)
            st.success(f"Wrote {len(desks)} desks to {DESKS_CSV} (+ {ROWS_JSON})")

        if desks:
            with st.expander("Desk table", expanded=False):
                st.dataframe(pd.DataFrame(desks), use_container_width=True, hide_index=True)

    else:  # Sensors
        st.metric("Sensors placed", len(ss.wc_sensors))
        if len(ss.wc_sensors) != ss.wc_saved_sensors:
            st.warning("Unsaved sensor changes")
        c1, c2 = st.columns(2)
        if c1.button("↶ Undo", use_container_width=True, disabled=not ss.wc_sensors):
            ss.wc_sensors.pop()
            st.rerun()
        if c2.button("Reset", use_container_width=True, disabled=not ss.wc_sensors):
            ss.wc_sensors.clear()
            st.rerun()

        st.divider()
        st.download_button(
            "💾 Download sensors CSV", data=sensors_csv_bytes(ss.wc_sensors),
            file_name="wc_sensor_positions.csv", mime="text/csv",
            use_container_width=True, disabled=not ss.wc_sensors,
        )
        if st.button(f"Save to {SENSORS_CSV}", type="primary",
                     use_container_width=True, disabled=not ss.wc_sensors):
            os.makedirs(os.path.dirname(SENSORS_CSV), exist_ok=True)
            with open(SENSORS_CSV, "wb") as f:
                f.write(sensors_csv_bytes(ss.wc_sensors))
            ss.wc_saved_sensors = len(ss.wc_sensors)
            st.success(f"Wrote {len(ss.wc_sensors)} sensors to {SENSORS_CSV}")

        if ss.wc_sensors:
            with st.expander("Sensor table", expanded=False):
                st.dataframe(
                    pd.DataFrame(
                        [(sensor_id_for(i), round(x, 1), round(y, 1))
                         for i, (x, y) in enumerate(ss.wc_sensors)],
                        columns=["sensor_id", "x_px", "y_px"],
                    ),
                    use_container_width=True, hide_index=True,
                )

# ─────────────────────── main click area ───────────────────────

if mode == "Desk rows":
    if ss.wc_pending is None:
        st.markdown(f"### Zone **{ss.wc_zone}** · click the **start** of the next row "
                    f"({int(ss.wc_desks_per_row)} desks)")
    else:
        st.markdown(f"### Zone **{ss.wc_zone}** · now click the **end** of the row")
else:
    st.markdown(f"### Click to place **{sensor_id_for(len(ss.wc_sensors))}**")

overlay = render_overlay(img, desks, ss.wc_pending, ss.wc_sensors)

display_w = min(DISPLAY_WIDTH, native_w)
scale = native_w / display_w
click_key = f"wc_{mode}_{len(ss.wc_rows)}_{ss.wc_pending is not None}_{len(ss.wc_sensors)}"
click = streamlit_image_coordinates(overlay, width=display_w, key=click_key)

if click is not None:
    x = float(max(0.0, min(native_w - 1, click["x"] * scale)))
    y = float(max(0.0, min(native_h - 1, click["y"] * scale)))
    pt = (x, y)

    if mode == "Desk rows":
        if ss.wc_pending is None:
            ss.wc_pending = pt
            st.rerun()
        elif ss.wc_pending != pt:
            ss.wc_rows.append({
                "zone": ss.wc_zone,
                "start": ss.wc_pending,
                "end": pt,
                "n": int(ss.wc_desks_per_row),
            })
            ss.wc_pending = None
            st.rerun()
    else:
        if not ss.wc_sensors or ss.wc_sensors[-1] != pt:
            ss.wc_sensors.append(pt)
            st.rerun()
