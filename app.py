from __future__ import annotations

import time

import streamlit as st

from lib.netatmo import fetch_homecoach_snapshot

st.set_page_config(
    page_title="Imperial Campus Occupancy",
    page_icon="🏛️",
    layout="wide",
)


def _fmt(value, unit: str) -> str:
    """Format a metric value with unit, or '—' when missing."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.1f} {unit}"
    return f"{value} {unit}"


def _co2_band(co2) -> str | None:
    """Coarse air-quality label for the CO₂ delta slot."""
    if co2 is None:
        return None
    if co2 < 800:
        return "Good"
    if co2 < 1200:
        return "Fair"
    return "Poor"


def _humanize_age(seconds: int) -> str:
    """'45s', '7m', '1h 12m' — single-glance freshness indicator."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m"


def _render_air_quality():
    """Live Netatmo Home Coach reading — degrades to a warning row on any failure."""
    snap = fetch_homecoach_snapshot()
    st.subheader("🌿 Indoor Air Quality")

    if not snap["ok"]:
        st.warning(f"Air-quality sensor unavailable: {snap['error']}")
        return

    c_temp, c_hum, c_co2, c_noise = st.columns(4)
    c_temp.metric("Temperature", _fmt(snap["temperature"], "°C"))
    c_hum.metric("Humidity", _fmt(snap["humidity"], "%"))
    c_co2.metric(
        "CO₂",
        _fmt(snap["co2"], "ppm"),
        delta=_co2_band(snap["co2"]),
        delta_color="off",
    )
    c_noise.metric("Noise", _fmt(snap["noise"], "dB"))

    if snap["time_utc"]:
        age_s = int(time.time() - snap["time_utc"])
        st.caption(
            f"Last reading {_humanize_age(age_s)} ago · "
            f"sensor: {snap['device_name'] or 'Home Coach'}"
        )

huxley_page = st.Page(
    "pages/1_Huxley_Labs.py",
    title="Huxley Labs",
    icon="💻",
    url_path="Huxley_Labs",
)
lecture_page = st.Page(
    "pages/2_Lecture_Hall.py",
    title="Lecture Hall 144",
    icon="🎓",
    url_path="Lecture_Hall",
)


def render_home():
    st.title("🏛️ Imperial Campus — Live Occupancy")
    st.markdown("Pick a view from the sidebar, or jump in with one of the shortcuts below.")

    _render_air_quality()

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💻 Huxley Labs")
        st.markdown(
            "Live workstation status across all Huxley computing labs. "
            "See per-PC state (In Use / Idle / Offline), session TTLs, "
            "and the 24-hour utilisation trend."
        )
        if st.button("Open Huxley dashboard →", key="goto_huxley", use_container_width=True):
            st.switch_page(huxley_page)

    with col2:
        st.subheader("🎓 Lecture Hall 144")
        st.markdown(
            "Live lecture-hall occupancy from a simulated camera feed. "
            "Play to detect students in real time and get a "
            "front-to-back seating recommendation."
        )
        if st.button("Open Lecture Hall view →", key="goto_lecture", use_container_width=True):
            st.switch_page(lecture_page)

    st.divider()
    st.caption(
        "Background simulator writes to `data/huxley_mock_access_logs_with_users.csv` every 60 s. "
        "Lecture-hall stream is sampled from `data/lecture_video.mp4`."
    )


home_page = st.Page(render_home, title="Home", icon="🏛️", default=True, url_path="home")

navigation = st.navigation([home_page, huxley_page, lecture_page], position="sidebar")
navigation.run()
