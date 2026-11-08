import duckdb, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
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
u = con.execute("SELECT * FROM uci_clean").df().reset_index(drop=True)
y = u["readmit_30d"].astype(int).values

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
X = pd.concat([X, pd.get_dummies(u[cat_cols].astype(str), dtype=int)], axis=1)

# patient-level split: 50% train, 15% recalibration, 15% conformal calibration, 20% test
rng = np.random.default_rng(42)
pats = u["patient_nbr"].unique()
r = pd.Series(rng.random(len(pats)), index=pats)[u["patient_nbr"]].values
tr, rc, cc, te = r < 0.50, (r >= 0.50) & (r < 0.65), (r >= 0.65) & (r < 0.80), r >= 0.80
print("Sizes train/recal/conformal/test:", tr.sum(), rc.sum(), cc.sum(), te.sum())

gb = make_gb().fit(X[tr], y[tr])
p_raw = gb.predict_proba(X)[:, 1]
iso = IsotonicRegression(out_of_bounds="clip").fit(p_raw[rc], y[rc])
p = iso.predict(p_raw)

def ece(yt, pt, bins=10):
    q = pd.qcut(pt, bins, duplicates="drop")
    g = pd.DataFrame({"y": yt, "p": pt, "q": q}).groupby("q", observed=True)
    return float(np.sum(g.size() / len(yt) * (g["p"].mean() - g["y"].mean()).abs()))

print("\n=== Calibration on test set ===")
for name, pp in [("raw", p_raw), ("isotonic", p)]:
    print(f"{name:9s} AUROC {roc_auc_score(y[te], pp[te]):.3f} | Brier {brier_score_loss(y[te], pp[te]):.4f} | ECE {ece(y[te], pp[te]):.4f}")

plt.figure(figsize=(5, 5))
for name, pp in [("raw", p_raw), ("isotonic", p)]:
    fp, mp = calibration_curve(y[te], pp[te], n_bins=10, strategy="quantile")
    plt.plot(mp, fp, marker="o", label=name)
plt.plot([0, 0.4], [0, 0.4], "k--")
plt.xlabel("Predicted risk"); plt.ylabel("Observed rate"); plt.legend()
plt.title("UCI calibration before/after isotonic"); plt.tight_layout()
plt.savefig("reports/figures/uci_recalibration.png", dpi=150); plt.close()

# split conformal prediction (nonconformity = 1 - probability of the true class)
alpha = 0.10
s_cal = np.where(y[cc] == 1, 1 - p[cc], p[cc])
n = cc.sum()
qhat = np.quantile(s_cal, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
in1 = (1 - p) <= qhat      # class 1 in prediction set
in0 = p <= qhat            # class 0 in prediction set
covered = np.where(y == 1, in1, in0)
size = in1.astype(int) + in0.astype(int)
t = pd.DataFrame({"covered": covered, "size": size, "y": y,
                  "sex": u["gender"], "age_mid": X["age"], "race": u["race"].fillna("Unknown")})
t["age_band"] = pd.cut(t["age_mid"], [0, 40, 60, 80, 120], labels=["<40", "40-59", "60-79", "80+"])
tt = t[te].copy()

print(f"\n=== Conformal prediction (target coverage {1 - alpha:.0%}) ===")
print(f"Overall coverage: {tt['covered'].mean():.3f}")
print(f"Coverage when true outcome is readmitted (class 1): {tt.loc[tt['y']==1, 'covered'].mean():.3f}")
print("Prediction set sizes (share of patients):")
print(tt["size"].map({0: "empty", 1: "single label", 2: "both labels (uncertain)"}).value_counts(normalize=True).round(3).to_string())

def by(col):
    g = tt.groupby(col, observed=True).agg(n=("covered", "size"), coverage=("covered", "mean"),
                                           readmit_rate=("y", "mean"), uncertain_share=("size", lambda s: (s == 2).mean())).round(3)
    return g[g["n"] >= 50]
sub = pd.concat({"sex": by("sex"), "age_band": by("age_band"), "race": by("race")})
print("\nSubgroup coverage (groups with n >= 50):\n", sub.to_string())
sub.to_csv("reports/tables/uci_conformal_subgroups.csv")

out = pd.DataFrame({"patient_nbr": u["patient_nbr"], "p_raw": p_raw, "p_calibrated": p, "split_test": te})
out.to_csv("reports/tables/uci_risk_scores.csv", index=False)


# class-conditional (Mondrian) conformal, using the raw (already calibrated) probabilities
def qh(scores):
    m = len(scores)
    return np.quantile(scores, min(1.0, np.ceil((m + 1) * (1 - alpha)) / m), method="higher")
q1 = qh(1 - p_raw[cc & (y == 1)])
q0 = qh(p_raw[cc & (y == 0)])
in1 = (1 - p_raw) <= q1
in0 = p_raw <= q0
size2 = in1.astype(int) + in0.astype(int)
cov2 = np.where(y == 1, in1, in0)
t2 = pd.DataFrame({"covered": cov2, "size": size2, "y": y, "age_mid": X["age"]})
t2["age_band"] = pd.cut(t2["age_mid"], [0, 40, 60, 80, 120], labels=["<40", "40-59", "60-79", "80+"])
t2 = t2[te]
print("\n=== Class-conditional conformal ===")
print("Coverage, readmitted:", round(t2.loc[t2.y == 1, "covered"].mean(), 3),
      "| not readmitted:", round(t2.loc[t2.y == 0, "covered"].mean(), 3))
print(t2["size"].map({0: "empty", 1: "single label", 2: "both labels (uncertain)"}).value_counts(normalize=True).round(3).to_string())
for band, g in t2.groupby("age_band", observed=True):
    print(band, "| cov readmitted:", round(g.loc[g.y == 1, "covered"].mean(), 3),
          "| cov not:", round(g.loc[g.y == 0, "covered"].mean(), 3),
          "| uncertain:", round((g["size"] == 2).mean(), 3))



# SHAP
try:
    import shap
    samp = X[te].sample(min(2000, te.sum()), random_state=42)
    sv = shap.TreeExplainer(gb).shap_values(samp)
    shap.summary_plot(sv, samp, max_display=15, show=False)
    plt.tight_layout(); plt.savefig("reports/figures/uci_shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()
    print("\nSaved reports/figures/uci_shap_bar.png")
except Exception as e:
    print("\nSHAP skipped:", type(e).__name__, e)
con.close()
