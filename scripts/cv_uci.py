import duckdb, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
try:
    from xgboost import XGBClassifier
    def make_gb():
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, eval_metric="logloss", random_state=42)
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    def make_gb():
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, random_state=42)

con = duckdb.connect("hospital.duckdb")
u = con.execute("SELECT * FROM uci_clean").df().reset_index(drop=True)
y = u["readmit_30d"].astype(int).values
groups = u["patient_nbr"].values

def age_mid(a):
    return int(str(a).strip("[)").split("-")[0]) + 5

def dx_group(code):
    if pd.isna(code): return "missing"
    c = str(code)
    if c.startswith(("V", "E")): return "other"
    try: v = float(c)
    except ValueError: return "other"
    if 250 <= v < 251: return "diabetes"
    if 390 <= v <= 459 or v == 785: return "circulatory"
    if 460 <= v <= 519 or v == 786: return "respiratory"
    if 520 <= v <= 579 or v == 787: return "digestive"
    if 800 <= v <= 999: return "injury"
    if 710 <= v <= 739: return "musculoskeletal"
    if 580 <= v <= 629 or v == 788: return "genitourinary"
    if 140 <= v <= 239: return "neoplasms"
    return "other"

num_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
            "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
X = u[num_cols].copy().astype(float)
X["age"] = u["age"].map(age_mid)
u["dx1"] = u["diag_1"].map(dx_group)
cat_cols = ["race", "gender", "admission_type_id", "admission_source_id", "discharge_disposition_id",
            "dx1", "A1Cresult", "max_glu_serum", "insulin", "change", "diabetesMed"]
cats = u[cat_cols].astype(str).fillna("missing")
X = pd.concat([X, pd.get_dummies(cats, dtype=int)], axis=1)

# LACE (approximate inputs, state this in the report)
los = u["time_in_hospital"]
L = np.select([los < 1, los < 2, los < 3, los < 4, los < 7, los < 14], [0, 1, 2, 3, 4, 5], 7)
A = 3 * (u["admission_type_id"] == 1).astype(int)           # type 1 = Emergency
c = np.clip(u["number_diagnoses"] - 1, 0, None)             # crude comorbidity proxy
C = np.select([c == 0, c == 1, c == 2, c == 3], [0, 1, 2, 3], 5)
E = np.minimum(u["number_emergency"], 4)                    # prior YEAR, not 6 months
lace = (L + A + C + E).values

print("Rows:", len(u), "| patients:", len(set(groups)), "| base rate:", round(y.mean(), 3))
print("\nSingle-feature AUROC:")
for col in ["number_inpatient", "number_emergency", "time_in_hospital", "number_diagnoses", "age", "num_medications"]:
    print(f"  {col:18s} {roc_auc_score(y, X[col]):.3f}")

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof = {"LACE": np.zeros(len(y)), "LogReg": np.zeros(len(y)), "GradBoost": np.zeros(len(y))}
fa = {k: [] for k in oof}
for i, (tr, te) in enumerate(cv.split(X, y, groups=groups), 1):
    oof["LACE"][te] = lace[te]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X.iloc[tr], y[tr])
    oof["LogReg"][te] = lr.predict_proba(X.iloc[te])[:, 1]
    gb = make_gb().fit(X.iloc[tr], y[tr])
    oof["GradBoost"][te] = gb.predict_proba(X.iloc[te])[:, 1]
    for k in oof:
        fa[k].append(roc_auc_score(y[te], oof[k][te]))
    print("fold", i, "done")

rows = []
for k in oof:
    a = np.array(fa[k])
    rows.append({"model": k, "AUROC_mean": round(a.mean(), 3), "AUROC_sd": round(a.std(), 3),
                 "AUPRC_pooled": round(average_precision_score(y, oof[k]), 3),
                 "Brier_pooled": None if k == "LACE" else round(brier_score_loss(y, oof[k]), 3)})
res = pd.DataFrame(rows)
print("\nUCI 5-fold patient-grouped CV:\n", res.to_string(index=False))
print("Brier of always-predict-base-rate:", round(y.mean() * (1 - y.mean()), 3))
res.to_csv("reports/tables/uci_cv_results.csv", index=False)

plt.figure(figsize=(5, 5))
for k in ["LogReg", "GradBoost"]:
    fp, mp = calibration_curve(y, oof[k], n_bins=10, strategy="quantile")
    plt.plot(mp, fp, marker="o", label=k)
plt.plot([0, 0.4], [0, 0.4], "k--", label="perfect")
plt.xlabel("Predicted risk"); plt.ylabel("Observed readmission rate")
plt.title("Calibration (out-of-fold, UCI)"); plt.legend(); plt.tight_layout()
plt.savefig("reports/figures/uci_calibration.png", dpi=150)
print("Saved reports/figures/uci_calibration.png")
con.close()
