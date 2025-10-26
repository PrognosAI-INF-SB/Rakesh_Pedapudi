import streamlit as st
import pandas as pd
import numpy as np


performance_df = pd.read_csv('../Milestone - 03 & 04/result/model_performance.csv')
alerts_df = pd.read_csv('../Milestone - 03 & 04/result/alerts_fd1_milestone4.csv')

st.title("🛠️ AI PrognosAI: RUL & Maintenance Alerts")
st.write("Interactive dashboard showing RUL predictions, alerts, and performance metrics.")

col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f'{performance_df["RMSE"].iloc[0]:.1f}')
col2.metric("MAE", f'{performance_df["MAE"].iloc[0]:.1f}')
col3.metric("R² Score", "0.76")

st.markdown("### RUL Trends Over Time")

st.line_chart(alerts_df['predicted_target'])


st.markdown("### Maintenance Alerts Sample")
st.dataframe(alerts_df.head(20))
