import pandas

# Loads the cleaned electricity dataset
full_dataset = pandas.read_csv("data/electricity_cleaned.csv")

# get timestamp and electricity consumption for specific office
df = full_dataset[["timestamp", "Panther_office_Hannah"]]
df = df.rename(columns={'Panther_office_Hannah': 'energy_kw'})

# drop nan values
df = df.dropna()

df["timestamp"] = pandas.to_datetime(df["timestamp"])

# add time feature
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = df["timestamp"].dt.weekday >= 5
df["is_overnight"] = df["hour"].between(0, 6)

# add baseline
df["roll_24hr_mean"] = df["energy_kw"].rolling(window=24).mean()
df["roll_24hr_std"] = df["energy_kw"].rolling(window=24).std()

# drop nan values from rolling features
df = df.dropna(subset=["roll_24hr_mean", "roll_24hr_std"])

df["timestamp"].diff().value_counts().head()

# select features for modeling
features = ["energy_kw", "hour", "day_of_week", "is_weekend", "is_overnight", "roll_24hr_mean", "roll_24hr_std"]

X = df[features]

# fit an isolation model to detect anomalies
from sklearn.ensemble import IsolationForest

# isolation forest with 100 trees and 2% contamination (anomalies)
iso = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)

# predict anomalies given the features
df["anomaly"] = iso.fit_predict(X)

# anomaly = -1 if an anomaly, 1 if normal
df["is_anomaly"] = df["anomaly"] == -1


import matplotlib.pyplot as plt

# plot energy consumption with anomalies highlighted
# plt.figure(figsize=(15, 6))
# plt.plot(df["timestamp"], df["energy_kw"], label="Energy Consumption (kW)")
# plt.scatter(df[df["is_anomaly"]]["timestamp"], df[df["is_anomaly"]]["energy_kw"], color="red", label="Anomalies", marker="x")
# plt.xlabel("Timestamp")
# plt.ylabel("Energy Consumption (kW)")
# plt.title("Energy Consumption with Anomalies Highlighted")
# plt.legend()
# plt.show()

# # save plot
# plt.savefig("energy_anomalies.png")

# create explanation for anomalties

# anomaly score, z = 0 is normal, z > 2 means suspicous, z > 3 means very suspicous
df["z_score"] = (df["energy_kw"] - df["roll_24hr_mean"]) / df["roll_24hr_std"]

# create flags
df["overnight_spike"] = (df["is_anomaly"] & df["is_overnight"]) & (df["z_score"] >= 1.5)
df["weekend_spike"] = (df["is_anomaly"] & df["is_weekend"]) & (df["z_score"] >= 1.5)
df["big_spike"] = (df["is_anomaly"]) & (df["z_score"] >= 2.5)
df["drop"] = (df["is_anomaly"]) & (df["z_score"] <= -1.5)

# explain anomalies
def explain_anomaly(row):
    if row["overnight_spike"]:
        return "After hours HVAC running"
    elif row["weekend_spike"]:
        return "Weekend schedule override or event"
    elif row["big_spike"]:
        return "Sudden load spike (possible equipment fault or override)"
    elif row["drop"]:
        return "Unexpected load drop (shutdown or schedule change)"
    else:
        return "Atypical energy behavior (requires review)"


    
df["explanation"] = ""
df.loc[df["is_anomaly"], "explanation"] = df.loc[df["is_anomaly"]].apply(explain_anomaly, axis=1)

# top 20 anomalies with explanations
top = df[df["is_anomaly"]].copy().sort_values(by="z_score", ascending=False)[["timestamp", "energy_kw", "z_score", "explanation"]].head(20)

# plot top 20 anomalies with explanations
important = df["is_anomaly"] & (df["z_score"] >= 3)

# plt.figure(figsize=(14,4))
# plt.plot(df["timestamp"], df["energy_kw"], label="Energy (kW)")
# plt.scatter(df.loc[important, "timestamp"],
#             df.loc[important, "energy_kw"], color="red",
#             marker="x", s=25, label="High-severity anomalies (z_score>=3)")
# plt.legend()
# plt.title("Energy with High-Severity Anomalies")
# plt.xlabel("Timestamp")
# plt.ylabel("kW")
# plt.show()
# plt.savefig("high_severity_anomalies.png")

# baseline vs excess money
rate = 0.15

df["baseline_kw"] = df["energy_kw"].rolling(window=24*7, min_periods=24).mean()
df["excess_kw"] = df["energy_kw"] - df["baseline_kw"]
df["excess_kw_anomaly"] = df["excess_kw"].where(df["is_anomaly"], 0)

# conert excess kW to dollars
df["excess_kwh_anomaly"] = df["excess_kw_anomaly"]
df["cost_impact"] = df["excess_kwh_anomaly"] * rate

total_kwh = df["excess_kwh_anomaly"].sum()
total_cost = df["cost_impact"].sum()

# create kpis
kpis = {
    "total_anomalies": df["is_anomaly"].sum(),
    "total_excess_kwh": total_kwh.round(2),
    "total_cost_impact": total_cost.round(2),
}

print("KPIs:")
for k, v in kpis.items():
    print(f"{k}: {v}")

impact_by_reason = (
    df[df["is_anomaly"]]
    .groupby("explanation")[["excess_kwh_anomaly", "cost_impact"]]
    .sum()
    .sort_values("cost_impact", ascending=False)
)

# actions
ACTION_MAP = {
    "After-hours HVAC operation":
        "Check BAS occupancy schedule & overrides. Verify AHU/VAV start-stop times; look for manual override left ON.",
    "Weekend schedule override":
        "Verify weekend/holiday schedule and after-hours requests. Check if equipment is scheduled to run unnecessarily.",
    "Sustained high load (possible HVAC inefficiency)":
        "Review setpoints and reset strategies (SAT/CHW). Check simultaneous heating/cooling, economizer operation, and stuck valves/dampers.",
    "Sudden load spike (possible equipment fault or override)":
        "Check alarms and trend logs around this time. Verify major equipment starts (AHU/chiller/boiler), demand events, or meter anomalies.",
    "Unexpected load drop (shutdown or schedule change)":
        "Confirm planned shutdown/holiday. If unplanned, check for equipment trips, power issues, or meter/data gaps.",
    "Atypical energy behavior (requires review)":
        "Inspect trend plots before/after. Compare to similar days (same day-of-week/hour). Check recent control changes or maintenance work."
}

df["recommended_action"] = df["explanation"].map(ACTION_MAP).fillna("Review energy trends and investigate potential causes.")

def priority(row):
    if row["z_score"] >= 3.5:
        return "High"
    elif row["z_score"] >= 2.5:
        return "Medium"
    else:
        return "Low"
    
df["priority"] = df.apply(priority, axis=1)

#save final csv with anomalies and explanations
df.to_csv("data/electricity_anomalies_explained.csv", index=False)
