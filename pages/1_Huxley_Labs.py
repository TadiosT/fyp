import os
import sys

import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.floor_map import build_floor_map  # noqa: E402

DATA_FILE = "data/huxley_mock_access_logs_with_users.csv"
COORDS_FILE = "data/workstation_coordinates.csv"
FLOOR_PLAN = "assets/huxley_lab_floor_plan.jpg"

STATE_COLOURS = {"In Use": "#32CD32", "Idle": "#FFA500", "Offline": "#FF0000"}

st.set_page_config(
    page_title="Huxley Labs Occupancy",
    page_icon="💻",
    layout="wide",
)


@st.cache_data(ttl=60)
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["session_ttl_remaining"] = pd.to_numeric(
        df["session_ttl_remaining"], errors="coerce"
    ).astype("Int64")
    return df


@st.cache_data
def load_coordinates():
    return pd.read_csv(COORDS_FILE)


@st.cache_data
def load_floor_plan():
    return Image.open(FLOOR_PLAN)


def render_header():
    colA, colB = st.columns([5, 1])
    with colA:
        st.title("Huxley Computing Labs - Occupancy Dashboard")
        st.markdown("Welcome to the live occupancy tracker. Data is currently simulated.")
    with colB:
        if st.button("🔄 Refresh Live Data", use_container_width=True):
            st.cache_data.clear()


def render_metrics(df):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Monitored PCs", df["pc_id"].nunique())
    with col2:
        total_users = df[df["user_id"] != "N/A"]["user_id"].nunique()
        st.metric("Unique Users Today", total_users)
    with col3:
        last_update = df["timestamp"].max().strftime("%H:%M")
        st.metric("Last Data Refresh", last_update)


def render_trend(df):
    st.subheader("24-Hour Utilisation Trend")
    time_filter = st.radio(
        "Select Time Range:",
        ["Today", "This Week", "This Month", "Last 3 Months"],
        horizontal=True,
    )

    if time_filter == "Today":
        today = df["timestamp"].max().normalize()
        today_df = df[df["timestamp"] >= today]
        time_series_df = today_df.pivot_table(
            index="timestamp", columns="state", aggfunc="size", fill_value=0
        )
        chart_data = time_series_df.reindex(
            columns=["In Use", "Idle", "Offline"], fill_value=0
        )
    else:
        st.info(f"Historical data for '{time_filter}' is not yet available.")
        chart_data = pd.DataFrame(columns=["In Use", "Idle", "Offline"])

    st.line_chart(
        chart_data,
        color=["#32CD32", "#FFA500", "#FF0000"],
        use_container_width=True,
    )


def render_map(df):
    st.subheader("Live Floor Plan")
    st.caption("Hover a workstation for user and TTL details.")

    if not os.path.exists(FLOOR_PLAN):
        st.warning(f"⚠️ Could not find '{FLOOR_PLAN}'. Check the folder structure!")
        return
    if not os.path.exists(COORDS_FILE):
        st.warning(
            f"⚠️ Coordinate file '{COORDS_FILE}' not found. "
            "Run `streamlit run tools/calibrate_coordinates.py` to generate it."
        )
        return

    coords = load_coordinates()
    img = load_floor_plan()

    latest = df.sort_values("timestamp").groupby("pc_id").tail(1)
    merged = coords.merge(
        latest[["pc_id", "state", "user_id", "session_ttl_remaining"]],
        on="pc_id",
        how="left",
    )
    merged["state"] = merged["state"].fillna("Offline")
    merged["user_id"] = merged["user_id"].fillna("—").replace("N/A", "—")
    merged["ttl_display"] = merged["session_ttl_remaining"].apply(
        lambda v: f"{int(v)} min" if pd.notna(v) else "—"
    )

    hover_template = (
        "<b>%{customdata[0]}</b><br>"
        "User: %{customdata[1]}<br>"
        "TTL: %{customdata[2]}"
        "<extra></extra>"
    )

    fig = build_floor_map(
        image=img,
        points_df=merged,
        color_col="state",
        color_map=STATE_COLOURS,
        category_order=["In Use", "Idle", "Offline"],
        hover_template=hover_template,
        custom_data_cols=["pc_id", "user_id", "ttl_display"],
        legend_title="Workstation state",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_raw(df):
    st.subheader("Raw Data Explorer")
    if st.checkbox("Show Raw Log Data"):
        st.dataframe(df, use_container_width=True)


render_header()
st.divider()

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        f"🚨 Could not find the data file at '{DATA_FILE}'. "
        "Ensure the simulator has been started via launcher.py."
    )
    st.stop()

render_metrics(df)
st.divider()
render_trend(df)
st.divider()
render_map(df)
st.divider()
render_raw(df)
