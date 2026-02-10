# ⚡ Smart City Energy Demand Forecasting

## Overview
This project focuses on forecasting high-frequency electricity demand using
real-world government open energy data from Thailand.

The dataset contains 5-minute interval electricity generation and demand
across multiple regions, with a focus on metropolitan demand forecasting.

## Dataset
- Source: Thailand Government Open Data
- Frequency: 5-minute
- Target: Metropolitan electricity demand
- Period: 2023–

## Models Implemented
- Naive Baseline
- SARIMAX (seasonal time-series model)
- XGBoost (machine learning)
- LSTM (deep learning)

## Features
- Time-based features (hour, weekday)
- Lag features (1 step, 1 hour, 1 day)
- Rolling statistics

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Statsmodels (SARIMAX)
- TensorFlow / Keras (LSTM)
- Streamlit (Visualization)

## How to Run
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py

