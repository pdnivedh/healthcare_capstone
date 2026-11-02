# Hospital Capacity Digital Twin (Healthcare Analytics Capstone)

Readmission risk model, discrete-event bed-capacity simulation, and a local LLM text-to-SQL assistant, built on synthetic and open EHR data.

## Status
- Done: data acquisition, loading into DuckDB, cohort definition, preprocessing, EDA.
- Next: readmission model and LACE benchmark, simulation, assistant, dashboard.

## Data (not stored in this repo)
- Synthea synthetic patients (JDK 17), generated with:
      run_synthea.bat -p 5000 -s 42 Massachusetts --exporter.csv.export=true
  Copy output/csv/*.csv into data/raw/synthea/
- UCI Diabetes 130-US Hospitals (CC BY 4.0): https://archive.ics.uci.edu/dataset/296
  Put diabetic_data.csv and IDS_mapping.csv in data/raw/uci/

## Reproduce
    pip install duckdb pandas matplotlib
    python scripts/load_db.py
    python scripts/eda.py

## Structure
- scripts/ : data loading, feasibility checks, preprocessing and EDA
- reports/tables, reports/figures : EDA outputs

## Citations
- Synthea: Walonoski et al., 2018, JAMIA.
- Clore, Cios, DeShazo and Strack (2014). Diabetes 130-US Hospitals for Years 1999-2008. UCI ML Repository. doi:10.24432/C5230J

## Team
A: data and model | B: simulation | C: assistant and dashboard
