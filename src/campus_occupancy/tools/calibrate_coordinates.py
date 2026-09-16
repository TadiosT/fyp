"""Huxley workstation calibrator: click the floor plan to produce / extend
data/workstation_coordinates.csv (pc_id, x_px, y_px).

Two modes (sidebar radio):

* **Re-place existing**: walk through known PC ids and click each one
  (the original workflow; click overwrites a placed PC).
* **Add new workstation**: pick a lab (202 / 206 / 210 / 219 / custom); each
  click appends a new PC with the next free number in that lab, e.g.
  ``Lab_219_PC099``. Use this for seats visible on the plan that aren't in
  the CSV yet.

Known ids come from the coordinates CSV plus any ids in the Huxley log. After
saving, restart the launcher: the simulator reconciles its desk list with the
CSV on boot, so new seats go live without re-seeding.

Run with:
    venv/bin/python -m streamlit run src/campus_occupancy/tools/calibrate_coordinates.py
"""
from __future__ import annotations

import os
import re
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

LOG_FILE = "data/huxley_mock_access_logs_with_users.csv"
FLOOR_PLAN = "assets/huxley_lab_floor_plan.jpg"
OUTPUT_CSV = "data/workstation_coordinates.csv"
DISPLAY_WIDTH = 1100

STATE_COLOURS = {"placed": "#32CD32", "current": "#FFA500"}
ID_RE = re.compile(r"^Lab_(\w+?)_PC(\d+)$")


st.set_page_config(page_title="Workstation Coordinate Calibrator", layout="wide")


@st.cache_data
def load_image() -> tuple[Image.Image, int, int]:
    img = Image.open(FLOOR_PLAN).convert("RGB")
    return img, img.width, img.height


def load_existing_coords() -> dict[str, tuple[int, int]]:
    if not os.path.exists(OUTPUT_CSV):
        return {}
    df = pd.read_csv(OUTPUT_CSV)
    return {row.pc_id: (int(row.x_px), int(row.y_px)) for row in df.itertuples()}


def load_log_ids() -> set[str]:
    if not os.path.exists(LOG_FILE):
        return set()
    try:
        return set(pd.read_csv(LOG_FILE, usecols=["pc_id"])["pc_id"].unique())
    except Exception:
        return set()


def parse_id(pc_id: str) -> tuple[str, int] | None:
    m = ID_RE.match(pc_id)
    return (m.group(1), int(m.group(2))) if m else None


def labs_in(ids) -> list[str]:
    return sorted({p[0] for p in map(parse_id, ids) if p}, key=lambda s: (len(s), s))


def next_id(lab: str, ids) -> str:
    """PC numbers are global across labs (202→001-030, 206→031-060, …), so the
    next id continues from the highest number anywhere, in the chosen lab."""
    nums = [p[1] for p in map(parse_id, ids) if p]
    return f"Lab_{lab}_PC{(max(nums) + 1) if nums else 1:03d}"


def render_overlay(base, coords, highlight: str | None) -> Image.Image:
    overlay = base.copy()
    draw = ImageDraw.Draw(overlay)
    r = max(4, base.width // 250)
    for pc_id, (x, y) in coords.items():
        colour = STATE_COLOURS["current"] if pc_id == highlight else STATE_COLOURS["placed"]
        draw.ellipse((x - r, y - r, x + r, y + r), fill=colour, outline="black")
    return overlay


def export_csv(coords) -> bytes:
    df = pd.DataFrame([(pc, x, y) for pc, (x, y) in sorted(coords.items())],
                      columns=["pc_id", "x_px", "y_px"])
    buf = BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# ─────────────────────── session state ───────────────────────

ss = st.session_state
if "coords" not in ss:
    ss.coords = load_existing_coords()
if "known_ids" not in ss:
    ss.known_ids = sorted(set(ss.coords) | load_log_ids())
if "history" not in ss:
    ss.history = []          # (pc_id, previous_coords_or_None)
if "cursor" not in ss:
    ss.cursor = next((i for i, p in enumerate(ss.known_ids) if p not in ss.coords), 0)
if "saved_count" not in ss:
    ss.saved_count = len(ss.coords)
if "last_added" not in ss:
    ss.last_added = None

img, native_w, native_h = load_image()

st.title("🖱️ Workstation Coordinate Calibrator")
st.caption(f"Image native size: {native_w} × {native_h} px. "
           f"{len(ss.coords)} placed · {len(ss.known_ids)} known ids.")

# ─────────────────────── sidebar ───────────────────────

with st.sidebar:
    mode = st.radio("Mode", ["Add new workstation", "Re-place existing"], key="mode")
    st.divider()

    if mode == "Add new workstation":
        labs = labs_in(ss.known_ids) or ["202"]
        lab_choice = st.selectbox("Lab", labs + ["Custom…"], key="lab_choice")
        lab = st.text_input("Custom lab id", "", key="lab_custom").strip() if lab_choice == "Custom…" else lab_choice
        pending_id = next_id(lab, set(ss.known_ids) | set(ss.coords)) if lab else None
        st.metric("Next id", pending_id or "—")
        added = [pc for pc, prev in ss.history if prev is None]
        st.caption(f"Added this session: {len(added)}")
    else:
        pending_id = None
        placed = len([p for p in ss.known_ids if p in ss.coords])
        st.metric("Placed", f"{placed} / {len(ss.known_ids)}")
        st.progress(placed / len(ss.known_ids) if ss.known_ids else 0.0)
        jump = st.selectbox("Jump to PC", options=ss.known_ids, index=min(ss.cursor, len(ss.known_ids) - 1),
                            key="jump_select")
        if jump != ss.known_ids[ss.cursor]:
            ss.cursor = ss.known_ids.index(jump)
            st.rerun()
        if st.button("Skip ▶", use_container_width=True):
            ss.cursor = (ss.cursor + 1) % len(ss.known_ids)
            st.rerun()

    if st.button("Undo ↶", use_container_width=True, disabled=not ss.history):
        pc, prev = ss.history.pop()
        if prev is None:
            ss.coords.pop(pc, None)
            if pc in ss.known_ids:
                ss.known_ids.remove(pc)
        else:
            ss.coords[pc] = prev
        ss.last_added = None
        st.rerun()

    if len(ss.coords) != ss.saved_count:
        st.warning("Unsaved changes")

    st.divider()
    st.download_button("💾 Export CSV", data=export_csv(ss.coords), file_name="workstation_coordinates.csv",
                       mime="text/csv", use_container_width=True, disabled=not ss.coords)
    if st.button(f"Save to {OUTPUT_CSV}", type="primary", use_container_width=True, disabled=not ss.coords):
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        with open(OUTPUT_CSV, "wb") as f:
            f.write(export_csv(ss.coords))
        ss.saved_count = len(ss.coords)
        st.success(f"Wrote {len(ss.coords)} rows to {OUTPUT_CSV}. Restart the launcher to simulate new seats.")

    st.divider()
    unplaced = [p for p in ss.known_ids if p not in ss.coords]
    with st.expander(f"Unplaced known ids ({len(unplaced)})", expanded=False):
        st.write(unplaced)
    with st.expander("Placed table", expanded=False):
        st.dataframe(pd.DataFrame([(pc, x, y) for pc, (x, y) in sorted(ss.coords.items())],
                                  columns=["pc_id", "x_px", "y_px"]),
                     use_container_width=True, hide_index=True)

# ─────────────────────── main click area ───────────────────────

if mode == "Add new workstation":
    if not pending_id:
        st.info("Enter a custom lab id in the sidebar.")
        st.stop()
    st.markdown(f"### Adding `{pending_id}`: click its seat on the plan")
    highlight = ss.last_added
else:
    if ss.cursor >= len(ss.known_ids):
        ss.cursor = 0
    current_pc = ss.known_ids[ss.cursor]
    banner = f"### Placing: `{current_pc}` ({ss.cursor + 1} / {len(ss.known_ids)})"
    if current_pc in ss.coords:
        x_old, y_old = ss.coords[current_pc]
        banner += f". Already placed at ({x_old}, {y_old}); click to overwrite"
    st.markdown(banner)
    highlight = current_pc

overlay = render_overlay(img, ss.coords, highlight)

display_w = min(DISPLAY_WIDTH, native_w)
scale = native_w / display_w
click_key = f"plan_{mode}_{len(ss.coords)}_{len(ss.history)}_{ss.cursor}"
click = streamlit_image_coordinates(overlay, width=display_w, key=click_key)

if click is not None:
    x_native = max(0, min(native_w - 1, int(round(click["x"] * scale))))
    y_native = max(0, min(native_h - 1, int(round(click["y"] * scale))))
    pt = (x_native, y_native)

    if mode == "Add new workstation":
        if pt not in ss.coords.values():
            ss.coords[pending_id] = pt
            ss.known_ids = sorted(set(ss.known_ids) | {pending_id})
            ss.history.append((pending_id, None))
            ss.last_added = pending_id
            st.rerun()
    else:
        prev = ss.coords.get(current_pc)
        if prev != pt:
            ss.coords[current_pc] = pt
            ss.history.append((current_pc, prev))
            n = len(ss.known_ids)
            for offset in range(1, n + 1):
                idx = (ss.cursor + offset) % n
                if ss.known_ids[idx] not in ss.coords:
                    ss.cursor = idx
                    break
            else:
                ss.cursor = (ss.cursor + 1) % n
            st.rerun()
