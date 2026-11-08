import duckdb, pandas as pd, numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
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

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
p_lr = np.zeros(len(y)); p_gb = np.zeros(len(y))
for tr, te in cv.split(X, y, groups=s["PATIENT"]):
    p_lr[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    p_gb[te] = make_gb().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]

print("Out-of-fold  AUROC / Brier (base-rate Brier = %.3f)" % (y.mean() * (1 - y.mean())))
for n, p in [("LogReg", p_lr), ("GradBoost", p_gb)]:
    print(f"  {n:10s} {roc_auc_score(y, p):.3f} / {brier_score_loss(y, p):.4f}")

# top-k capture: if support goes to the top k% of risk, how many readmissions are in that group?
rows = []
for name, score in [("LogReg", p_lr), ("GradBoost", p_gb), ("LACE", s["lace"].values.astype(float) + np.random.default_rng(0).random(len(y)) * 1e-6)]:
    order = np.argsort(-score)
    for k in (5, 10, 20, 30):
        top = order[: int(len(y) * k / 100)]
        rows.append({"score": name, "top_pct": k, "share_of_readmissions_captured_pct": round(100 * y[top].sum() / y.sum(), 1),
                     "precision_pct": round(100 * y[top].mean(), 1)})
tk = pd.DataFrame(rows)
print("\nTop-k capture (random targeting would capture k%):\n", tk.to_string(index=False))
tk.to_csv("reports/tables/synthea_topk_capture.csv", index=False)

risk = s[["PATIENT", "START", "STOP", "los_days", "age", "GENDER", "RACE", "readmit_30d"]].copy()
risk["risk_lr"] = p_lr
risk["risk_gb"] = p_gb
risk["lace"] = s["lace"].values
con.register("risk_df", risk)
con.execute("CREATE OR REPLACE TABLE syn_risk_scores AS SELECT * FROM risk_df")
risk.to_csv("reports/tables/synthea_risk_scores.csv", index=False)
print("\nSaved table syn_risk_scores with", len(risk), "rows")
con.close()
