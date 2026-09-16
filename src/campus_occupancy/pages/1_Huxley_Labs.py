import streamlit as st

from campus_occupancy.config import HUXLEY_CFG
from campus_occupancy.dashboard.lab_dashboard import render_lab_dashboard

st.set_page_config(page_title="Huxley Labs Occupancy", page_icon="💻", layout="wide")

render_lab_dashboard(HUXLEY_CFG)
