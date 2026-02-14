import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Energy Anomaly Detector", layout="wide")
st.title("AI Energy Usage Anomaly Detector")
st.caption("Detect and explain anomalies in energy consumption data using AI")
# df = pd.read_csv("../data/electricity_anomalies_explained.csv", parse_dates=["timestamp"])

uploaded = st.file_uploader("Upload your processed CSV (must include timestamp, energy_kw, is_anomaly, explanation, excess_kwh_anom, cost_impact_$)", type=["csv"])
if uploaded is None:
    st.info("Upload your processed CSV to view the dashboard.")
    st.stop()

df = pd.read_csv(uploaded, parse_dates=["timestamp"])

# sidebar
st.sidebar.header("Filters")
min_date = df["timestamp"].min()
max_date = df["timestamp"].max()
date_range = st.sidebar.date_input("Date range", value=(min_date.date(), max_date.date()))

show_only_anomalies = st.sidebar.checkbox("Show only anomalies", value=False)
reasons = df["explanation"].unique()
selected_reasons = st.sidebar.multiselect("Anomaly reasons", options=reasons, default=reasons)
rate = st.sidebar.number_input("Electricity rate ($/kWh)", value=0.15, step=0.01)
# recompute cost impact based on user input rate
if "excess_kwh_anomaly" in df.columns:
    df["cost_impact"] = df["excess_kwh_anomaly"] * rate

# filters

start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
f = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].copy()

if show_only_anomalies :
    f = f[f["is_anomaly"] == True]

if "z_score" in f.columns and show_only_anomalies:
    f = f[(f["is_anomaly"] == False) | (f["z_score"] >= 3)]
if "explanation" in f.columns and selected_reasons:
    f = f[(f["is_anomaly"] == False) | (f["explanation"].isin(selected_reasons))]

# kpis
total_anomalies = f["is_anomaly"].sum()
total_excess_kwh = f["excess_kwh_anomaly"].sum() if "excess_kwh_anomaly" in f.columns else 0
total_cost_impact = f["cost_impact"].sum() if "cost_impact" in f.columns else 0     

c1, c2, c3, c4 = st.columns(4)
c1.metric("Anomalous hours", f"{total_anomalies}")
c2.metric("Excess energy (kWh)", f"{total_excess_kwh:.2f}")
c3.metric("Estimated cost ($)", f"{total_cost_impact:.2f}")
c4.metric("Avg $ / anomaly hour", f"{(total_cost_impact / total_anomalies) if total_anomalies > 0 else 0:.2f}")

# plot
st.subheader("Energy Consumption Over Time")
plt.figure(figsize=(14,4))
plt.plot(f["timestamp"], f["energy_kw"], label="Energy (kW)")
if "is_anomaly" in f.columns:
    important = f["is_anomaly"] & (f["z_score"] >= 3)
    plt.scatter(f.loc[important, "timestamp"],
                f.loc[important, "energy_kw"], color="red",
                marker="x", s=25, label="High-severity anomalies (z_score>=3)")
plt.legend()
plt.title("Energy with High-Severity Anomalies")
plt.xlabel("Timestamp")
plt.ylabel("kW")
st.pyplot(plt)

# top anomalies table


st.subheader("Anomaly log (actionable)")

cols = ["timestamp",
    "energy_kw",
    "baseline_kw",
    "z_score",
    "excess_kwh_anomaly",
    "cost_impact",
    "priority",
    "explanation",
    "recommended_action",]
cols = [c for c in cols if c in f.columns]

anom_table = (
    f[f["is_anomaly"] == True][cols]
    .sort_values("cost_impact", ascending=False)
)

display_table = anom_table.rename(columns={
    "timestamp": "Timestamp",
    "energy_kw": "Energy (kW)",
    "baseline_kw": "Baseline (kW)",
    "z_score": "Severity (z-score)",
    "excess_kwh_anomaly": "Excess Energy (kWh)",
    "cost_impact": "Estimated Cost ($)",
    "priority": "Priority",
    "explanation": "Likely Cause",
    "recommended_action": "Recommended Action",
})


st.dataframe(display_table, use_container_width=True)