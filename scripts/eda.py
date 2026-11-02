import duckdb, pandas as pd, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("reports/tables", exist_ok=True)
con = duckdb.connect("hospital.duckdb")

# ================= SYNTHEA =================
enc = con.execute("SELECT * FROM syn_encounters").df()
pat = con.execute("SELECT * FROM syn_patients").df()
for c in ["START", "STOP"]:
    enc[c] = pd.to_datetime(enc[c], utc=True)
pat["BIRTHDATE"] = pd.to_datetime(pat["BIRTHDATE"], utc=True)
pat["DEATHDATE"] = pd.to_datetime(pat["DEATHDATE"], utc=True)

# missingness of the patient table
(pat.isna().mean() * 100).round(1).sort_values(ascending=False).to_csv("reports/tables/synthea_patients_missing_pct.csv")

inp = enc[enc["ENCOUNTERCLASS"] == "inpatient"].sort_values(["PATIENT", "START"]).copy()
n_raw = len(inp)

# merge continuation stays (gap < 1 day)
prev_stop = inp.groupby("PATIENT")["STOP"].shift(1)
gap_prev = (inp["START"] - prev_stop).dt.total_seconds() / 86400
inp["adm_id"] = (~(gap_prev < 1)).groupby(inp["PATIENT"]).cumsum()
stays = (inp.groupby(["PATIENT", "adm_id"])
            .agg(START=("START", "min"), STOP=("STOP", "max"), reason=("REASONDESCRIPTION", "first"))
            .reset_index())
n_merged = len(stays)
stays["los_days"] = (stays["STOP"] - stays["START"]).dt.total_seconds() / 86400

# 30-day readmission label, computed on ALL stays
stays = stays.sort_values(["PATIENT", "START"])
nxt = stays.groupby("PATIENT")["START"].shift(-1)
gap = (nxt - stays["STOP"]).dt.total_seconds() / 86400
stays["readmit_30d"] = gap.between(0, 30)

stays = stays.merge(pat[["Id", "BIRTHDATE", "DEATHDATE", "GENDER", "RACE"]], left_on="PATIENT", right_on="Id", how="left")
stays["age"] = (stays["START"] - stays["BIRTHDATE"]).dt.days / 365.25

died = stays["DEATHDATE"].notna() & (stays["DEATHDATE"] <= stays["STOP"] + pd.Timedelta(days=1))
yr = stays["START"].dt.year
cohort = stays[~died & yr.between(2017, 2025)].copy()

summary = {
    "inpatient_encounters_raw": n_raw,
    "admissions_after_merging_continuations": n_merged,
    "stays_ending_in_death_excluded": int(died.sum()),
    "final_cohort_2017_2025": len(cohort),
    "patients_in_cohort": cohort["PATIENT"].nunique(),
    "readmissions_30d": int(cohort["readmit_30d"].sum()),
    "readmission_rate_pct": round(cohort["readmit_30d"].mean() * 100, 2),
    "median_los_days": round(cohort["los_days"].median(), 2),
    "median_age": round(cohort["age"].median(), 1),
}
pd.Series(summary).to_csv("reports/tables/synthea_cohort_summary.csv", header=["value"])
print("\n=== SYNTHEA COHORT ===")
for k, v in summary.items():
    print(k, ":", v)

# does the label depend on the reason? (rule-driven check)
rb = cohort.groupby("reason")["readmit_30d"].agg(["count", "mean"]).sort_values("count", ascending=False).head(10)
rb.to_csv("reports/tables/synthea_readmit_by_reason.csv")
print("\nReadmission rate by top reasons:\n", rb)

# figures
stays.groupby(stays["START"].dt.year).size().loc[2005:2026].plot(kind="bar", figsize=(8, 4))
plt.title("Synthea inpatient admissions per year"); plt.xlabel("Year"); plt.ylabel("Admissions")
plt.tight_layout(); plt.savefig("reports/figures/synthea_admissions_per_year.png", dpi=150); plt.close()

cohort["los_days"].clip(upper=30).plot(kind="hist", bins=30, figsize=(7, 4))
plt.title("Length of stay (cohort 2017-2025, capped at 30 days)"); plt.xlabel("Days")
plt.tight_layout(); plt.savefig("reports/figures/synthea_los_hist.png", dpi=150); plt.close()

cohort["age_band"] = pd.cut(cohort["age"], [0, 18, 40, 60, 75, 120])
(cohort.groupby("age_band", observed=True)["readmit_30d"].mean() * 100).plot(kind="bar", figsize=(6, 4))
plt.title("Synthea 30-day readmission rate by age band"); plt.ylabel("%")
plt.tight_layout(); plt.savefig("reports/figures/synthea_readmit_by_age.png", dpi=150); plt.close()

# save cleaned cohort
out = cohort.drop(columns=["BIRTHDATE", "DEATHDATE", "Id", "age_band"]).copy()
out["age_band"] = None
con.register("cohort_df", out.drop(columns=["age_band"]))
con.execute("CREATE OR REPLACE TABLE syn_stays AS SELECT * FROM cohort_df")

# ================= UCI =================
u = con.execute("SELECT * FROM uci_diabetes").df()
miss = (u.isna().mean() * 100).round(1).sort_values(ascending=False)
miss[miss > 0].to_csv("reports/tables/uci_missing_pct.csv", header=["missing_pct"])

u_info = {
    "rows": len(u),
    "unique_patients": u["patient_nbr"].nunique(),
    "gender_unknown_rows": int((u["gender"] == "Unknown/Invalid").sum()),
}
print("\n=== UCI ===")
print(u_info)
print("\nMissing % (columns with any missing):\n", miss[miss > 0])
print("\nReadmitted classes (%):\n", (u["readmitted"].value_counts(normalize=True) * 100).round(1))

# preprocessing: remove discharges to death/hospice (VERIFY these ids in IDS_mapping.csv)
excl_ids = [11, 13, 14, 19, 20, 21]
n_excl = int(u["discharge_disposition_id"].isin(excl_ids).sum())
uc = u[~u["discharge_disposition_id"].isin(excl_ids)].copy()
uc["readmit_30d"] = (uc["readmitted"] == "<30")
print("\nRows removed (death/hospice discharge):", n_excl)
print("Clean rows:", len(uc), "| 30-day readmission rate %:", round(uc["readmit_30d"].mean() * 100, 2))
con.register("uci_df", uc)
con.execute("CREATE OR REPLACE TABLE uci_clean AS SELECT * FROM uci_df")

# figures
u["readmitted"].value_counts().plot(kind="bar", figsize=(5, 4))
plt.title("UCI readmission classes"); plt.tight_layout()
plt.savefig("reports/figures/uci_readmitted_classes.png", dpi=150); plt.close()

miss[miss > 0].head(10).plot(kind="barh", figsize=(7, 4))
plt.title("UCI: columns with most missing values (%)"); plt.gca().invert_yaxis(); plt.tight_layout()
plt.savefig("reports/figures/uci_missing.png", dpi=150); plt.close()

print("\nDone. Tables in reports/tables, figures in reports/figures.")
con.close()
