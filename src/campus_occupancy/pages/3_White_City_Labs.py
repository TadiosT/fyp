import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from campus_occupancy.dashboard.floor_map import build_floor_map  # noqa: E402
from campus_occupancy.dashboard.lab_dashboard import current_desk_state, live_sensor_readings, render_lab_dashboard  # noqa: E402
from campus_occupancy.data_io.preload import Bundle  # noqa: E402
from campus_occupancy.config import WC_CFG as CFG  # noqa: E402
from campus_occupancy.simulation.wc_session import LABELS, desk_states, simulate_session  # noqa: E402

st.set_page_config(page_title="White City Labs Occupancy", page_icon="🏙️", layout="wide")

SESSION_COLOURS = {"Teaching lab": "#32CD32", "Overflow": "#FFA500", "Quiet study": "#1E88E5",
                   "Already in use": "#9E9E9E", "Empty": "#FF0000"}


def render_session_simulator(b: Bundle) -> None:
    st.divider()
    st.subheader("🧪 Lab-Session Simulator")
    st.caption("Plan a timetabled session: students are directed to one lab and only move to an "
               "additional lab once it reaches the capacity threshold. Zone roles, overflow order and "
               "early-overflow decisions use the live sensor readings above.")

    if b.coords is None:
        st.info("Calibrate desks first.")
        return
    desks = b.coords.copy()
    if "zone" not in desks.columns:
        st.info("The desk CSV has no zone column. Re-run the calibrator.")
        return
    state = current_desk_state(b.latest)
    desks["in_use_now"] = desks["pc_id"].map(lambda pc: state.get(pc, {}).get("ttl", 0) > 0)

    live = live_sensor_readings(b)
    zone_readings = {b.zone_map.get(sid, "?"): r for sid, r in live.items()}

    with st.form("wc_session_form"):
        c1, c2, c3, c4, c5 = st.columns(5)
        attendance = c1.number_input("Students attending", 10, 600, 250, 10)
        duration = c2.number_input("Duration (min)", 30, 240, 120, 15)
        threshold = c3.slider("Overflow threshold", 0.5, 1.0, 0.75, 0.05, format="%.2f")
        window = c4.number_input("Arrival window (min)", 5, 60, 15, 5)
        seed = c5.number_input("Random seed", 0, 9999, 1)
        run = st.form_submit_button("▶ Run session", type="primary")
    if run:
        st.session_state["wc_session"] = simulate_session(
            desks, zone_readings, attendance=int(attendance), duration_min=int(duration),
            threshold=float(threshold), arrival_window_min=int(window), seed=int(seed))
    res = st.session_state.get("wc_session")
    if res is None:
        st.info("Set the session parameters and press **Run session**.")
        return

    for rec in res.recommendations:
        st.markdown(f"- {rec}")
    st.dataframe(res.zone_table.rename(columns={
        "in_use_now": "in use now", "free_now": "free now", "air_score": "air score"}),
        use_container_width=True, hide_index=True)

    max_min = int(res.timeline.index.max())
    minute = st.slider("Session minute", 0, max_min, min(20, max_min))

    fig = go.Figure()
    roles = res.zone_table.set_index("zone")["role"]
    for z in res.timeline.columns:
        fig.add_trace(go.Scatter(x=res.timeline.index, y=res.timeline[z] * 100, mode="lines",
                                 name=f"Zone {z} ({roles[z]})", line=dict(color=SESSION_COLOURS[roles[z]])))
    fig.add_hline(y=threshold * 100, line_dash="dash", line_color="gray",
                  annotation_text=f"{threshold:.0%} threshold")
    fig.add_vline(x=minute, line_color="black", line_dash="dot")
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis_title="Occupancy (%)", xaxis_title="Minute", legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    ev = pd.DataFrame([e for e in res.events if e["minute"] <= minute])
    if not ev.empty:
        st.dataframe(ev[["minute", "message"]], use_container_width=True, hide_index=True)

    pts = desks.copy()
    pts["label"] = pts["pc_id"].map(desk_states(res, minute)).fillna("Empty")
    seated = int(pts["label"].isin(["Teaching lab", "Overflow", "Quiet study"]).sum())
    st.caption(f"Minute {minute}: {seated} session students seated.")
    if b.floor_plan is not None:
        st.plotly_chart(build_floor_map(
            image=b.floor_plan, points_df=pts, color_col="label",
            color_map=SESSION_COLOURS, category_order=LABELS,
            hover_template="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
            custom_data_cols=["pc_id", "label"], legend_title="Session allocation", height=1100,
        ), use_container_width=True)


bundle = render_lab_dashboard(CFG)
if bundle is not None:
    render_session_simulator(bundle)
