import duckdb, pandas as pd, numpy as np, os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
try:
    from xgboost import XGBClassifier
    GB_NAME = "XGBoost"
    def make_gb():
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, eval_metric="logloss", random_state=42)
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    GB_NAME = "HistGradientBoosting (xgboost unavailable)"
    def make_gb():
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, random_state=42)

os.makedirs("reports/tables", exist_ok=True)
con = duckdb.connect("hospital.duckdb")
stays = con.execute("SELECT * FROM syn_stays").df()
enc = con.execute("SELECT PATIENT, START, STOP, ENCOUNTERCLASS FROM syn_encounters").df()
cond = con.execute("SELECT PATIENT, START, STOP, DESCRIPTION FROM syn_conditions").df()
for df in (stays, enc, cond):
    for c in ("START", "STOP"):
        df[c] = pd.to_datetime(df[c], utc=True)
stays = stays.reset_index(drop=True)
stays["sid"] = stays.index
base = stays[["sid", "PATIENT", "START"]]
DAY = 86400.0

def count_events(events, lo, hi):
    m = base.merge(events[["PATIENT", "START"]].rename(columns={"START": "EV"}), on="PATIENT")
    d = (m["START"] - m["EV"]).dt.total_seconds() / DAY
    m = m[(d > lo) & (d <= hi)]
    return m.groupby("sid").size().reindex(stays["sid"], fill_value=0).values

inpt = enc[enc["ENCOUNTERCLASS"] == "inpatient"]
ed = enc[enc["ENCOUNTERCLASS"] == "emergency"]
stays["prior_adm_365"] = count_events(inpt, 0, 365)
stays["prior_ed_180"] = count_events(ed, 1, 180)          # excludes the ED visit that led to this admission
stays["acute"] = (count_events(ed, -0.001, 1) > 0).astype(int)  # ED visit within 1 day before admission (proxy for emergency admission)

cd = cond[cond["DESCRIPTION"].str.contains(r"\(disorder\)", na=False)]
m = base.merge(cd.rename(columns={"START": "CSTART", "STOP": "CSTOP"}), on="PATIENT")
active = (m["CSTART"] < m["START"]) & (m["CSTOP"].isna() | (m["CSTOP"] > m["START"]))
comorb = m[active].groupby("sid")["DESCRIPTION"].nunique()
stays["comorb"] = comorb.reindex(stays["sid"], fill_value=0).values

# LACE index (approximate inputs, see report)
los = stays["los_days"]
L = np.select([los < 1, los < 2, los < 3, los < 4, los < 7, los < 14], [0, 1, 2, 3, 4, 5], 7)
A = 3 * stays["acute"]
c = stays["comorb"]
C = np.select([c == 0, c == 1, c == 2, c == 3], [0, 1, 2, 3], 5)
E = np.minimum(stays["prior_ed_180"], 4)
stays["lace"] = L + A + C + E

# features
X = pd.DataFrame({
    "age": stays["age"], "male": (stays["GENDER"] == "M").astype(int), "los_days": los,
    "prior_adm_365": stays["prior_adm_365"], "prior_ed_180": stays["prior_ed_180"],
    "acute": stays["acute"], "comorb": stays["comorb"]})
X = pd.concat([X, pd.get_dummies(stays["RACE"], prefix="race", dtype=int)], axis=1)
top = stays["reason"].fillna("none").value_counts().head(15).index
reason = stays["reason"].fillna("none").where(stays["reason"].fillna("none").isin(top), "other")
R = pd.get_dummies(reason, prefix="reason", dtype=int)
y = stays["readmit_30d"].astype(int).values

# patient-level split (no patient in both train and test)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, te = next(gss.split(X, y, groups=stays["PATIENT"]))
print("Train stays:", len(tr), "| Test stays:", len(te))
print("Positives train/test:", y[tr].sum(), "/", y[te].sum())
print("Patients overlapping train/test:", len(set(stays["PATIENT"].iloc[tr]) & set(stays["PATIENT"].iloc[te])))

def metrics(name, fs, yt, p, probability=True):
    return {"model": name, "features": fs, "AUROC": round(roc_auc_score(yt, p), 3),
            "AUPRC": round(average_precision_score(yt, p), 3),
            "Brier": round(brier_score_loss(yt, p), 3) if probability else None}

rows = [metrics("LACE index", "LACE score", y[te], stays["lace"].values[te], probability=False)]
for fs_name, Xf in [("no reason", X), ("with reason", pd.concat([X, R], axis=1))]:
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    lr.fit(Xf.iloc[tr], y[tr])
    rows.append(metrics("Logistic regression", fs_name, y[te], lr.predict_proba(Xf.iloc[te])[:, 1]))
    gb = make_gb()
    gb.fit(Xf.iloc[tr], y[tr])
    rows.append(metrics(GB_NAME, fs_name, y[te], gb.predict_proba(Xf.iloc[te])[:, 1]))

res = pd.DataFrame(rows)
print("\n", res.to_string(index=False))
print("\nBase rate in test set:", round(y[te].mean(), 3), "(AUPRC of a random model equals this)")
res.to_csv("reports/tables/synthea_model_results_v1.csv", index=False)

stays.drop(columns=["sid"]).to_csv("data/processed_synthea_features.csv", index=False) if os.path.isdir("data") else None
con.register("feat", stays.drop(columns=["sid"]))
con.execute("CREATE OR REPLACE TABLE syn_features AS SELECT * FROM feat")
print("Saved table syn_features. Done.")
con.close()
