import duckdb, pandas as pd, numpy as np
con = duckdb.connect("hospital.duckdb")
u = con.execute("SELECT patient_nbr, readmit_30d FROM uci_clean").df().reset_index(drop=True)
r = pd.read_csv("reports/tables/uci_risk_scores.csv")
assert (u["patient_nbr"].values == r["patient_nbr"].values).all(), "row order mismatch"
t = r[r["split_test"]].copy()
t["y"] = u.loc[r["split_test"].values, "readmit_30d"].astype(int).values
base = t["y"].mean()
print("Test rows:", len(t), "| base rate:", round(base, 3))
rows = []
for k in (5, 10, 20, 30):
    top = t.sort_values("p_raw", ascending=False).head(int(len(t) * k / 100))
    rows.append({"top_pct": k, "captured_pct": round(100 * top["y"].sum() / t["y"].sum(), 1),
                 "precision_pct": round(100 * top["y"].mean(), 1), "lift": round(top["y"].mean() / base, 2)})
res = pd.DataFrame(rows)
print(res.to_string(index=False))
res.to_csv("reports/tables/uci_topk_capture.csv", index=False)
