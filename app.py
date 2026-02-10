import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data.load_data import load_energy_data
from src.data.clean_data import clean_energy_data

st.set_page_config(page_title="Energy Forecasting", layout="wide")

st.title("⚡ Smart City Energy Demand Analysis")

df = load_energy_data("data/raw/energy.csv")
df = clean_energy_data(df)

st.subheader("Metropolitan Energy Demand")

st.line_chart(df["metropolitan_demand"])

st.markdown("""
### 📌 Project Highlights
- Government open energy data
- 5-minute resolution
- Time-series forecasting
- Baseline, SARIMAX, XGBoost, LSTM
""")
