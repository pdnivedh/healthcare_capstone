# Hand-off to B: risk scores for the simulation

## File
reports/tables/synthea_risk_scores.csv (also built by scripts/risk_scores_synthea.py as table syn_risk_scores in hospital.duckdb)

## Contents
One row per Synthea inpatient stay in the cohort (2,023 stays, 946 patients, 2017-2025, deaths excluded).
Columns: PATIENT, START, STOP (UTC), los_days, age, GENDER, RACE, readmit_30d (observed), risk_lr, risk_gb, lace.

## Which score to use
Use risk_lr. It is the predicted probability that this discharge is followed by an inpatient return within 30 days. Scores are out-of-fold: every stay was scored by a model that never saw that patient, so they are not inflated by training on the same data. risk_lr is better calibrated than risk_gb. lace is the benchmark and is a weak predictor on Synthea.

## Using it for policies (top-k targeting)
Rank stays by risk_lr and give "discharge support" to the top X%. Observed capture on Synthea (logistic regression): top 5% captures 14.9% of readmissions, top 10% 28.2%, top 20% 54.1%, top 30% 76.9%. The realistic UCI figures are lower: 13.5%, 23.0%, 38.1%, 51.4% (reports/tables/uci_topk_capture.csv). Treat the UCI numbers as the cautious case in the sensitivity analysis.

## Cautions
- Synthea readmissions are partly rule-driven, so these scores overstate how predictable real readmissions are.
- The intervention effect (how many targeted readmissions are actually prevented) is NOT estimated from data. It is an assumption and must be varied in the sensitivity analysis (for example 10%, 20%, 30%) and stated in the report.
- The cohort is a small hospital (about 250 stays per year after exclusions). Use an arrival_scale setting and validate at scale 1.0 before scaling up.
- Arrival and length-of-stay fitting can use the table syn_stays (same 2,023 stays, built by scripts/eda.py).

## Questions
Ask A.
