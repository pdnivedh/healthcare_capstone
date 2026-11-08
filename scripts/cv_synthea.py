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
s = con.execute("SELECT * FROM syn_features").df()
y = s["readmit_30d"].astype(int).values
X = pd.DataFrame({"age": s["age"], "male": (s["GENDER"] == "M").astype(int), "los_days": s["los_days"],
                  "prior_adm_365": s["prior_adm_365"], "prior_ed_180": s["prior_ed_180"],
                  "acute": s["acute"], "comorb": s["comorb"]})
X = pd.concat([X, pd.get_dummies(s["RACE"], prefix="race", dtype=int)], axis=1)

# 1) which single features carry the signal? (raw AUROC, no model)
print("Single-feature AUROC:")
for c in ["prior_adm_365", "los_days", "comorb", "prior_ed_180", "age", "acute"]:
    print(f"  {c:15s} {roc_auc_score(y, X[c]):.3f}")

# 2) patient-grouped 5-fold CV (no reason feature)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof = {"LACE": np.zeros(len(y)), "LogReg": np.zeros(len(y)), "GradBoost": np.zeros(len(y))}
fold_auc = {k: [] for k in oof}
for tr, te in cv.split(X, y, groups=s["PATIENT"]):
    oof["LACE"][te] = s["lace"].values[te]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X.iloc[tr], y[tr])
    oof["LogReg"][te] = lr.predict_proba(X.iloc[te])[:, 1]
    gb = make_gb().fit(X.iloc[tr], y[tr])
    oof["GradBoost"][te] = gb.predict_proba(X.iloc[te])[:, 1]
    for k in oof:
        fold_auc[k].append(roc_auc_score(y[te], oof[k][te]))

rows = []
for k in oof:
    a = np.array(fold_auc[k])
    row = {"model": k, "AUROC_mean": round(a.mean(), 3), "AUROC_sd": round(a.std(), 3),
           "AUPRC_pooled": round(average_precision_score(y, oof[k]), 3)}
    row["Brier_pooled"] = None if k == "LACE" else round(brier_score_loss(y, oof[k]), 3)
    rows.append(row)
res = pd.DataFrame(rows)
print("\n5-fold patient-grouped CV (no reason feature):\n", res.to_string(index=False))
print("Base rate:", round(y.mean(), 3), "| Brier of always-predict-base-rate:", round(y.mean() * (1 - y.mean()), 3))
res.to_csv("reports/tables/synthea_cv_results.csv", index=False)

# 3) calibration curves
plt.figure(figsize=(5, 5))
for k in ["LogReg", "GradBoost"]:
    fp, mp = calibration_curve(y, oof[k], n_bins=8, strategy="quantile")
    plt.plot(mp, fp, marker="o", label=k)
plt.plot([0, 1], [0, 1], "k--", label="perfect")
plt.xlabel("Predicted risk"); plt.ylabel("Observed readmission rate")
plt.title("Calibration (out-of-fold, Synthea)"); plt.legend(); plt.tight_layout()
plt.savefig("reports/figures/synthea_calibration.png", dpi=150)
print("Saved reports/figures/synthea_calibration.png")
con.close()
