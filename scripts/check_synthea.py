import pandas as pd

path = r"E:\healthcare\healthcare_capstone\synthea\output\csv"
enc = pd.read_csv(f"{path}\\encounters.csv", parse_dates=["START", "STOP"])

print("Encounter classes:")
print(enc["ENCOUNTERCLASS"].value_counts(), "\n")

inp = enc[enc["ENCOUNTERCLASS"] == "inpatient"].copy()
inp = inp.sort_values(["PATIENT", "START"])
inp["los_days"] = (inp["STOP"] - inp["START"]).dt.total_seconds() / 86400

print("Inpatient stays:", len(inp))
print("Patients with at least one stay:", inp["PATIENT"].nunique())
print("Patients with 2+ stays:", (inp.groupby("PATIENT").size() >= 2).sum(), "\n")

print("Length of stay (days):")
print(inp["los_days"].describe(), "\n")

# 30-day readmission: next inpatient start within 30 days of this stop
inp["next_start"] = inp.groupby("PATIENT")["START"].shift(-1)
gap = (inp["next_start"] - inp["STOP"]).dt.total_seconds() / 86400
inp["readmit_30d"] = (gap >= 0) & (gap <= 30)

print("30-day readmissions:", inp["readmit_30d"].sum())
print("Readmission rate:", round(inp["readmit_30d"].mean() * 100, 2), "%")
print("Date range:", inp["START"].min(), "to", inp["START"].max())