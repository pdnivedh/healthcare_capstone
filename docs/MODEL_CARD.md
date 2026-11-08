# Model Card: 30-day readmission risk model (prototype)

## Intended use
Research and teaching prototype for a capstone project. It ranks inpatient stays by estimated 30-day readmission risk to feed a bed-capacity simulation. It is not validated for clinical use and must not be used to make decisions about real patients.

## Data
- UCI Diabetes 130-US Hospitals (1999-2008): 99,343 encounters after removing death/hospice discharges, 69,990 patients, 11.4% readmitted within 30 days. Diabetic patients only, stays of 1 to 14 days.
- Synthea synthetic cohort (2017-2025): 2,023 stays, 946 patients, 12.6% readmitted. Rule-generated; the readmission signal partly reflects generator rules.

## Models
Logistic regression and gradient boosting ([XGBoost / HistGradientBoosting: confirm]) on demographics, utilisation, diagnosis group, discharge and admission fields. Baseline: approximated LACE index.
Splits are by patient so no patient appears in both training and test data.

## Performance (UCI, 5-fold patient-grouped CV)
| Model | AUROC | AUPRC | Brier |
|---|---|---|---|
| LACE (approximated) | 0.561 | 0.135 | n/a |
| Logistic regression | 0.661 | 0.213 | 0.097 |
| Gradient boosting | 0.673 | 0.228 | 0.096 |
Base rate 0.114; Brier of always predicting the base rate 0.101. Gradient boosting was well calibrated without adjustment (ECE 0.005); isotonic recalibration did not help.
Targeting the top 10% of risk captured 23.0% of readmissions (lift 2.31).

## Uncertainty and subgroups
Class-conditional conformal prediction (90% target) gave 89.6% coverage for readmitted and 90.7% for non-readmitted patients, but 66.2% of patients received an uncertain label. Coverage for readmitted patients varied by age (0.81 under 40, 0.84 for 40-59, 0.91 for 60-79, 0.96 for 80+). Coverage by sex was similar. Race groups other than Caucasian and African American were too small for reliable conclusions.

## Limitations
- Modest discrimination (AUROC about 0.67); useful for ranking, not for confident individual prediction.
- UCI is old, covers diabetic patients only, and excludes stays over 14 days; it may not transfer to other populations or years.
- LACE inputs are approximated on both datasets, so its score is a lower bound.
- Synthea results demonstrate the pipeline and are not clinical findings.
- Explanations (SHAP) describe model reliance, not causal effects.
