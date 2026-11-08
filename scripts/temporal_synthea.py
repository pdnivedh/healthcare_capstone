import duckdb, pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

con = duckdb.connect("hospital.duckdb")
s = con.execute("SELECT * FROM syn_features").df()
s["START"] = pd.to_datetime(s["START"], utc=True)
y = s["readmit_30d"].astype(int).values
X = pd.DataFrame({"age": s["age"], "male": (s["GENDER"] == "M").astype(int), "los_days": s["los_days"],
                  "prior_adm_365": s["prior_adm_365"], "prior_ed_180": s["prior_ed_180"],
                  "acute": s["acute"], "comorb": s["comorb"]})
X = pd.concat([X, pd.get_dummies(s["RACE"], prefix="race", dtype=int)], axis=1)

yr = s["START"].dt.year
train_all = (yr <= 2022).values
test = (yr >= 2023).values
print("Train stays:", train_all.sum(), "| test stays:", test.sum())
print("Positives train/test:", y[train_all].sum(), "/", y[test].sum())
print("Base rate train/test:", round(y[train_all].mean(), 3), "/", round(y[test].mean(), 3))
test_pats = set(s.loc[test, "PATIENT"])
overlap = s["PATIENT"].isin(test_pats).values & train_all
print("Train stays belonging to patients also seen in test:", overlap.sum())

def boot_ci(yt, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(yt), len(yt))
        if yt[i].min() == yt[i].max():
            continue
        vals.append(roc_auc_score(yt[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])

def run(label, train):
    rows = []
    lace = s["lace"].values.astype(float)
    auc = roc_auc_score(y[test], lace[test]); lo, hi = boot_ci(y[test], lace[test])
    rows.append({"setting": label, "model": "LACE", "AUROC": round(auc, 3), "CI_low": round(lo, 3), "CI_high": round(hi, 3),
                 "AUPRC": round(average_precision_score(y[test], lace[test]), 3), "Brier": None})
    models = {"LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
              "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                       eval_metric="logloss", random_state=42)}
    for name, m in models.items():
        m.fit(X[train], y[train])
        p = m.predict_proba(X[test])[:, 1]
        lo, hi = boot_ci(y[test], p)
        rows.append({"setting": label, "model": name, "AUROC": round(roc_auc_score(y[test], p), 3),
                     "CI_low": round(lo, 3), "CI_high": round(hi, 3),
                     "AUPRC": round(average_precision_score(y[test], p), 3),
                     "Brier": round(brier_score_loss(y[test], p), 4)})
    return rows

rows = run("temporal (train 2017-2022)", train_all)
rows += run("temporal, test patients removed from train", train_all & ~overlap)
res = pd.DataFrame(rows)
print("\n", res.to_string(index=False))
print("\nBase rate in test:", round(y[test].mean(), 3), "| Brier of always-predict-base-rate:", round(y[test].mean() * (1 - y[test].mean()), 3))
res.to_csv("reports/tables/synthea_temporal_results.csv", index=False)
con.close()
