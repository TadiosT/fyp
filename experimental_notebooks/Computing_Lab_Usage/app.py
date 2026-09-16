import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Huxley Labs Dashboard", layout="wide")

st.title("Huxley Computing Labs - Live Dashboard")
st.write("This app is running directly from Google Colab!")

# Generate some dummy data just to prove it works
chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['Lab 202', 'Lab 210', 'Lab 219'])

st.line_chart(chart_data)
