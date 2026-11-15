"""Make the AUROC comparison chart for slide 4 from reports/tables/uci_cv_results.csv.

Run from the project root:  python scripts/make_slide_figures.py
Writes reports/figures/uci_auroc_comparison.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
res = pd.read_csv(ROOT / "reports" / "tables" / "uci_cv_results.csv")

names = {"LACE": "LACE\n(approximated)", "LogReg": "Logistic\nregression", "GradBoost": "XGBoost"}
order = ["LACE", "LogReg", "GradBoost"]
res = res.set_index("model").loc[order]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
colors = ["#9aa5b1", "#5b8def", "#1f4fd8"]
bars = ax.bar([names[m] for m in order], res["AUROC_mean"], yerr=res["AUROC_sd"],
              capsize=6, color=colors, width=0.55)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
ax.text(2.45, 0.505, "chance (0.5)", ha="right", va="bottom", fontsize=9)
for rect, v in zip(bars, res["AUROC_mean"]):
    ax.annotate(f"{v:.3f}", (rect.get_x() + rect.get_width() / 2, v), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=11, fontweight="bold")
ax.set_ylim(0.45, 0.75)
ax.set_ylabel("AUROC (5-fold, patient-grouped)")
ax.set_title("Readmission prediction on real data (UCI Diabetes 130-US Hospitals)", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = ROOT / "reports" / "figures" / "uci_auroc_comparison.png"
fig.savefig(out, dpi=200)
print("Saved", out)
