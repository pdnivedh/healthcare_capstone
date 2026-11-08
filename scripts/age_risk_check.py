import duckdb, pandas as pd
con = duckdb.connect("hospital.duckdb")
u = con.execute("SELECT age FROM uci_clean").df().reset_index(drop=True)
r = pd.read_csv("reports/tables/uci_risk_scores.csv")
t = pd.DataFrame({"age": u["age"], "p": r["p_raw"], "test": r["split_test"]})
t["mid"] = t["age"].map(lambda a: int(str(a).strip("[)").split("-")[0]) + 5)
t["band"] = pd.cut(t["mid"], [0, 40, 60, 80, 120], labels=["<40", "40-59", "60-79", "80+"])
print(t[t["test"]].groupby("band", observed=True)["p"].agg(["count", "mean", "median"]).round(3))
