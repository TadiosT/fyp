"""Shared lab-occupancy dashboard renderer.

Used by ``pages/1_Huxley_Labs.py`` and ``pages/3_White_City_Labs.py``. Each page
sets its own ``st.set_page_config`` and then calls ``render_lab_dashboard`` with
a ``LabConfig``.

Data comes as a :class:`campus_occupancy.data_io.preload.Bundle`, preloaded at boot by the
background preloader when the lab is registered in ``app.py`` (White City),
otherwise built on demand and cached for 60 s (Huxley).

Optional extras (White City):
* ``history_file``: hourly state counts enabling Week / Month / 3-Month trends.
* ``sensor_positions``: virtual sensors whose readings are computed **live**
  from the current desk state on every render (always on), with the stored
  history from ``campus_occupancy.data_io.database`` shown alongside.
"""
from __future__ import annotations

import os
import random
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from campus_occupancy.config import LabConfig  # noqa: F401  (re-exported for callers)
from campus_occupancy.dashboard.floor_map import build_floor_map
from campus_occupancy.data_io.preload import Bundle, build_bundle, get_preloaded

STATE_COLOURS = {"In Use": "#32CD32", "Idle": "#FFA500", "Offline": "#FF0000"}
STATE_ORDER = ["In Use", "Idle", "Offline"]
SENSOR_COLOUR = "#1E88E5"


# ─────────────────────── data access ───────────────────────

@st.cache_resource(ttl=60, show_spinner="Loading lab data…")
def _bundle_on_demand(data_file: str, coords_file: str, floor_plan: str,
                      sensor_positions: str | None, history_file: str | None) -> Bundle:
    return build_bundle(data_file, coords_file, floor_plan, sensor_positions, history_file)


def get_bundle(cfg: LabConfig) -> Bundle:
    """Preloaded bundle if the boot-time preloader has one, else on-demand (cached)."""
    b = get_preloaded(cfg.data_file)
    if b is not None:
        return b
    return _bundle_on_demand(cfg.data_file, cfg.coords_file, cfg.floor_plan,
                             cfg.sensor_positions, cfg.history_file)


def co2_band(co2) -> str | None:
    if co2 is None:
        return None
    return "Good" if co2 < 800 else "Fair" if co2 < 1200 else "Poor"


def fmt(value, unit: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{value:.1f} {unit}" if isinstance(value, float) else f"{value} {unit}"


def age_str(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    m = seconds // 60
    return f"{m}m" if m < 60 else f"{m // 60}h {m % 60}m"


def current_desk_state(latest: pd.DataFrame) -> dict:
    """{pc_id: {'ttl': 1|0}} from the latest row per desk, enough for sensor_snapshots."""
    return {pc: {"ttl": 1 if pd.notna(ttl) and ttl > 0 else 0}
            for pc, ttl in zip(latest["pc_id"], latest["session_ttl_remaining"])}


def live_sensor_readings(b: Bundle) -> dict[str, dict]:
    """sensor_id → Netatmo-shaped reading computed from the current desk state.
    Seeded by the current minute so reruns within a minute are stable."""
    if not b.nearby:
        return {}
    from campus_occupancy.simulation.wc_sim import sensor_snapshots
    now = datetime.now().replace(second=0, microsecond=0)
    rng = random.Random(int(now.timestamp() // 60))
    return {s["device_name"]: s for s in sensor_snapshots(b.nearby, current_desk_state(b.latest), now, rng)}


# ─────────────────────── sections ───────────────────────

def render_header(cfg: LabConfig, b: Bundle | None = None):
    col_a, col_b = st.columns([5, 1])
    with col_a:
        st.title(cfg.title)
        st.markdown("Welcome to the live occupancy tracker. Data is currently simulated.")
        parts = [f"🕒 Now **{datetime.now():%H:%M}** (local)"]
        if b is not None:
            age = age_str((datetime.now() - b.loaded_at).total_seconds())
            if b.source == "preloaded":
                parts.append(f"⚡ Preloaded {age} ago (load took {b.load_seconds:.1f}s in the background)")
            else:
                parts.append(f"Loaded on demand {age} ago ({b.load_seconds:.1f}s)")
        st.caption(" · ".join(parts))
    with col_b:
        if st.button("🔄 Refresh Live Data", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()


def render_metrics(df: pd.DataFrame):
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Monitored PCs", df["pc_id"].nunique())
    c2.metric("Unique Users Today", df[df["user_id"] != "N/A"]["user_id"].nunique())
    c3.metric("Last Data Refresh", df["timestamp"].max().strftime("%H:%M"))


def render_trend(cfg: LabConfig, b: Bundle):
    st.subheader("Utilisation Trend")
    time_filter = st.radio("Select Time Range:",
                           ["Today", "This Week", "This Month", "Last 3 Months"], horizontal=True)
    now = pd.Timestamp.now()

    if time_filter == "Today":
        today_df = b.df[b.df["timestamp"] >= now.normalize()]
        pivot = today_df.pivot_table(index="timestamp", columns="state", aggfunc="size", fill_value=0)
        chart = pivot.reindex(columns=STATE_ORDER, fill_value=0)
        st.caption("Desks per state, per minute, since midnight.")
    elif b.history is not None:
        hist = b.history.rename(columns={"in_use": "In Use", "idle": "Idle", "offline": "Offline"})[STATE_ORDER]
        spec = {"This Week": (7, "D", "day"), "This Month": (30, "W", "week"),
                "Last 3 Months": (90, "MS", "month")}[time_filter]
        window = hist[hist.index >= now - pd.Timedelta(days=spec[0])]
        chart = window.resample(spec[1]).mean().round(1)
        st.caption(f"Mean desks per state, per {spec[2]}, over the last {spec[0]} days.")
    else:
        st.info(f"Historical data for '{time_filter}' is not yet available.")
        chart = pd.DataFrame(columns=STATE_ORDER)

    st.line_chart(chart, color=[STATE_COLOURS[s] for s in STATE_ORDER], use_container_width=True)


def render_sensors(b: Bundle) -> dict[str, dict]:
    """Always-on virtual sensors: live values + stored history. Returns live readings."""
    live = live_sensor_readings(b)
    if not live:
        return {}
    from campus_occupancy.data_io.database import latest_reading, readings_between, utc

    st.subheader("🌿 Virtual Air-Quality Sensors")
    st.caption("🟢 Live values are derived from the current desk occupancy around each sensor; "
               "stored readings are written every 5 minutes.")
    now = datetime.now(timezone.utc)
    for sid, r in live.items():
        c0, c1, c2, c3, c4, c5 = st.columns([1.3, 1, 1, 1, 1, 1.6])
        stored_txt, hist = "no stored readings yet", pd.DataFrame()
        try:
            row = latest_reading(sid)
            if row is not None:
                stored_txt = f"stored {age_str((now - utc(row.timestamp)).total_seconds())} ago"
            hist = readings_between(sid, now - timedelta(hours=24), now)
        except Exception as e:  # DB trouble must not hide the live values
            stored_txt = f"stored history unavailable ({e.__class__.__name__})"
        c0.markdown(f"**{sid}** 🟢 live  \n<small>{stored_txt}</small>", unsafe_allow_html=True)
        c1.metric("Temperature", fmt(r["temperature"], "°C"))
        c2.metric("Humidity", fmt(r["humidity"], "%"))
        c3.metric("CO₂", fmt(r["co2"], "ppm"), delta=co2_band(r["co2"]), delta_color="off")
        c4.metric("Noise", fmt(r["noise"], "dB"))
        with c5:
            if not hist.empty:
                s = hist.set_index("timestamp")["co2"]
                s.index = s.index.tz_convert(None)
                st.caption("CO₂ · last 24 h")
                st.line_chart(s, height=90, use_container_width=True)
    return live


def render_map(cfg: LabConfig, b: Bundle, live: dict[str, dict]):
    st.subheader("Live Floor Plan")
    st.caption("Hover a workstation for user and TTL details.")
    if b.floor_plan is None:
        st.warning(f"⚠️ Could not find '{cfg.floor_plan}'.")
        return
    if b.coords is None:
        st.warning(f"⚠️ Coordinate file '{cfg.coords_file}' not found. Run `{cfg.calibrate_hint}`.")
        return

    merged = b.coords.merge(b.latest[["pc_id", "state", "user_id", "session_ttl_remaining"]],
                            on="pc_id", how="left")
    merged["state"] = merged["state"].fillna("Offline")
    merged["user_id"] = merged["user_id"].fillna("—").replace("N/A", "—")
    merged["ttl_display"] = merged["session_ttl_remaining"].apply(
        lambda v: f"{int(v)} min" if pd.notna(v) else "—")

    fig = build_floor_map(
        image=b.floor_plan, points_df=merged, color_col="state",
        color_map=STATE_COLOURS, category_order=STATE_ORDER,
        hover_template=("<b>%{customdata[0]}</b><br>User: %{customdata[1]}<br>"
                        "TTL: %{customdata[2]}<extra></extra>"),
        custom_data_cols=["pc_id", "user_id", "ttl_display"],
        legend_title="Workstation state", height=cfg.map_height,
    )
    if live and b.sensors is not None:
        hover = [(f"<b>{sid}</b><br>T {fmt(live[sid]['temperature'], '°C')} · "
                  f"H {fmt(live[sid]['humidity'], '%')}<br>CO₂ {fmt(live[sid]['co2'], 'ppm')} · "
                  f"Noise {fmt(live[sid]['noise'], 'dB')}") if sid in live else f"<b>{sid}</b>"
                 for sid in b.sensors["sensor_id"]]
        fig.add_trace(go.Scatter(
            x=b.sensors["x_px"], y=b.sensors["y_px"], mode="markers", name="Sensor",
            marker=dict(symbol="square", size=14, color=SENSOR_COLOUR, line=dict(width=1.5, color="white")),
            hovertext=hover, hoverinfo="text"))
    st.plotly_chart(fig, use_container_width=True)


def render_raw(df: pd.DataFrame):
    st.subheader("Raw Data Explorer")
    if st.checkbox("Show Raw Log Data"):
        st.dataframe(df, use_container_width=True)


# ─────────────────────── entry point ───────────────────────

def render_lab_dashboard(cfg: LabConfig) -> Bundle | None:
    """Render the dashboard; returns the data bundle so page-specific sections can reuse it."""
    try:
        b = get_bundle(cfg)
    except FileNotFoundError:
        render_header(cfg)
        st.error(f"🚨 Could not find the data file at '{cfg.data_file}'. {cfg.missing_data_hint}")
        st.stop()
        return None

    render_header(cfg, b)
    st.divider()
    render_metrics(b.df)
    st.divider()
    live = render_sensors(b)
    if live:
        st.divider()
    render_trend(cfg, b)
    st.divider()
    render_map(cfg, b, live)
    st.divider()
    render_raw(b.df)
    return b
