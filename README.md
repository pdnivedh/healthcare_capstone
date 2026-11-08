# Hospital Capacity Digital Twin (Healthcare Analytics Capstone)

Readmission risk model, discrete-event bed-capacity simulation, and a local LLM text-to-SQL assistant, built on synthetic and open EHR data.

## Status
- Done (Part A): data loading, cohort definition, preprocessing, EDA, LACE baseline, logistic regression and XGBoost on both datasets, calibration, class-conditional conformal prediction with subgroup checks, SHAP, risk scores for the simulation, model card.
- In progress: digital twin (Part B), text-to-SQL assistant and dashboard (Part C).

## Data (not stored in this repo)
- Synthea synthetic patients (JDK 17), generated with:

      run_synthea.bat -p 5000 -s 42 Massachusetts --exporter.csv.export=true

  Copy output/csv/*.csv into data/raw/synthea/
- UCI Diabetes 130-US Hospitals (CC BY 4.0): https://archive.ics.uci.edu/dataset/296
  Put diabetic_data.csv and IDS_mapping.csv in data/raw/uci/

## Setup

    pip install -r requirements.txt

## Reproduce (run from the project root, in this order)

    python scripts/load_db.py
    python scripts/eda.py
    python scripts/model_synthea.py
    python scripts/cv_synthea.py
    python scripts/cv_uci.py
    python scripts/calibrate_conformal_uci.py
    python scripts/risk_scores_synthea.py
    python scripts/topk_uci.py
    python scripts/age_risk_check.py
    python scripts/temporal_synthea.py

load_db.py creates hospital.duckdb. eda.py creates the cleaned tables syn_stays and uci_clean. The later scripts write tables to reports/tables and figures to reports/figures.

## Structure
- scripts/ : loading, preprocessing, EDA, modelling
- reports/tables, reports/figures : outputs
- docs/ : model card and hand-off notes

## Key results (Part A)
- UCI, 5-fold patient-grouped CV: AUROC 0.673 (XGBoost), 0.661 (logistic regression), 0.561 (approximated LACE).
- Targeting the top 10% of risk captures 23.0% of readmissions on UCI (lift 2.31).
- Synthea readmissions are partly rule-driven, so Synthea results demonstrate the pipeline and are not clinical findings.

## Citations
- Synthea: Walonoski et al., 2018, JAMIA.
- Clore, Cios, DeShazo and Strack (2014). Diabetes 130-US Hospitals for Years 1999-2008. UCI ML Repository. doi:10.24432/C5230J

## Team
A: data and model | B: simulation | C: assistant and dashboard

    
