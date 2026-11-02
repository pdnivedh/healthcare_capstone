import pandas as pd

path = r"E:\healthcare\healthcare_capstone\synthea\output\csv"
enc = pd.read_csv(f"{path}\\encounters.csv", parse_dates=["START", "STOP"])
pat = pd.read_csv(f"{path}\\patients.csv", parse_dates=["BIRTHDATE", "DEATHDATE"])

inp = enc[enc["ENCOUNTERCLASS"] == "inpatient"].sort_values(["PATIENT", "START"]).copy()
inp["year"] = inp["START"].dt.year
print("Stays per year (last 15 years):")
print(inp["year"].value_counts().sort_index().tail(15), "\n")

inp["next_start"] = inp.groupby("PATIENT")["START"].shift(-1)
inp["gap_days"] = (inp["next_start"] - inp["STOP"]).dt.total_seconds() / 86400
near = inp[(inp["gap_days"] >= 0) & (inp["gap_days"] <= 30)]
print("Gap distribution for 'readmissions':")
print(near["gap_days"].describe())
print("Gap under 1 day:", (near["gap_days"] < 1).sum())
print("Gap 1-30 days:", (near["gap_days"] >= 1).sum(), "\n")

died = pat[pat["DEATHDATE"].notna()]["Id"]
print("Patients who died:", len(died))
print("Stays by patients who died:", inp["PATIENT"].isin(died).sum())