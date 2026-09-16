"""Home page + multipage navigation.

The Streamlit entry point is the thin `app.py` at the project root, which just
calls :func:`main`. Page scripts live next to this module in ``pages/``.
"""
from __future__ import annotations

import streamlit as st

from campus_occupancy.config import HUXLEY_CFG, WC_CFG
from campus_occupancy.data_io.preload import preload_status, start_preloader
from campus_occupancy.paths import PAGES_DIR

PRELOADED_LABS = [HUXLEY_CFG, WC_CFG]


def _humanize_age(seconds: int) -> str:
    """'45s', '7m', '1h 12m': a single-glance freshness indicator."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, _ = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m"


def _preload_caption(cfg) -> None:
    status = preload_status(cfg.data_file)
    if status and status["ready"]:
        st.caption(f"⚡ Data preloaded · {status['rows']:,} rows · "
                   f"refreshed {_humanize_age(status['age_s'])} ago")
    elif status and status["error"]:
        st.caption(f"⚠️ Preload: {status['error']}")
    else:
        st.caption("⏳ Preloading data in the background…")


def _all_preloaded() -> bool:
    return all((preload_status(c.data_file) or {}).get("ready") for c in PRELOADED_LABS)


def _render_preload_status_row() -> None:
    """Live preload status under the lab cards; no manual refresh needed.

    Polls every second while any lab is still loading; once everything is
    ready it triggers one app rerun so the fragment is re-created with a slow
    30 s cadence that just keeps the "refreshed Xs ago" ages current.
    """
    ready_at_page_run = _all_preloaded()
    interval = "30s" if ready_at_page_run else "1s"

    @st.fragment(run_every=interval)
    def _row():
        c_hux, _, c_wc = st.columns(3)
        with c_hux:
            _preload_caption(HUXLEY_CFG)
        with c_wc:
            _preload_caption(WC_CFG)
        if not ready_at_page_run and _all_preloaded():
            st.rerun(scope="app")

    _row()


def main() -> None:
    st.set_page_config(page_title="Imperial Campus Occupancy", page_icon="🏛️", layout="wide")

    # Optimistic preloading: warm both labs' data in a background thread as soon
    # as the server boots (idempotent: one thread per server process).
    start_preloader(PRELOADED_LABS, refresh_s=60)

    huxley_page = st.Page(str(PAGES_DIR / "1_Huxley_Labs.py"), title="Huxley Labs", icon="💻",
                          url_path="Huxley_Labs")
    lecture_page = st.Page(str(PAGES_DIR / "2_Lecture_Hall.py"), title="Lecture Hall 144", icon="🎓",
                           url_path="Lecture_Hall")
    white_city_page = st.Page(str(PAGES_DIR / "3_White_City_Labs.py"), title="White City Labs", icon="🏙️",
                              url_path="White_City_Labs")

    def render_home():
        st.title("🏛️ Imperial Campus: Live Occupancy")
        st.markdown("Pick a view from the sidebar, or jump in with one of the shortcuts below.")
        st.divider()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("💻 Huxley Labs")
            st.markdown("Live workstation status across all Huxley computing labs. "
                        "See per-PC state (In Use / Idle / Offline), session TTLs, "
                        "and the 24-hour utilisation trend.")
            if st.button("Open Huxley dashboard →", key="goto_huxley", use_container_width=True):
                st.switch_page(huxley_page)
        with col2:
            st.subheader("🎓 Lecture Hall 144")
            st.markdown("Live lecture-hall occupancy from a simulated camera feed. "
                        "Play to detect students in real time and get a "
                        "front-to-back seating recommendation.")
            if st.button("Open Lecture Hall view →", key="goto_lecture", use_container_width=True):
                st.switch_page(lecture_page)
        with col3:
            st.subheader("🏙️ White City Labs")
            st.markdown("Simulated occupancy across the White City computing zones, "
                        "with virtual air-quality sensors whose readings track "
                        "how busy each zone is.")
            if st.button("Open White City view →", key="goto_white_city", use_container_width=True):
                st.switch_page(white_city_page)

        _render_preload_status_row()

        st.divider()
        st.caption("Background simulators write to `data/huxley_mock_access_logs_with_users.csv` and "
                   "`data/wc_mock_access_logs.csv` every 60 s. "
                   "Lecture-hall stream is sampled from `data/lecture_video.mp4`.")

    home_page = st.Page(render_home, title="Home", icon="🏛️", default=True, url_path="home")
    st.navigation([home_page, huxley_page, lecture_page, white_city_page], position="sidebar").run()
