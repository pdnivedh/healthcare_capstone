# Hospital Digital Twin – Simulation Assumptions

## 1. Patient Arrivals
- New patient arrivals are modelled using a Poisson process.
- The arrival rate is calculated from historical new admissions between 2017 and 2025 using the `syn_stays` table.
- The `arrival_scale` parameter adjusts the arrival rate for different simulation scenarios.

## 2. Patient Risk and Length of Stay
- Patient readmission risk (`risk_lr`) and length of stay (`los_days`) are obtained from the `syn_risk_scores` table.
- Risk and length-of-stay values are sampled from the available records using bootstrapping.
- A returning patient receives a fresh risk and length-of-stay sample.

## 3. Hospital Bed Capacity
- The hospital has a fixed bed capacity, controlled by the `beds` parameter.
- Patients are admitted only when a bed is available.
- If all beds are occupied, incoming patients are blocked.
- No waiting queue is modelled.

## 4. Readmission Modelling
- Readmission probability is based on the patient's risk score.
- Returning patients come back after an empirically observed gap between 0 and 30 days, obtained from historical patient stays.
- The same random probability draw is used to compare readmission outcomes with and without intervention.

## 5. Intervention Policy
- The intervention policy identifies patients in the highest-risk percentage, controlled by `top_pct`.
- For flagged patients, readmission probability is reduced according to the `effectiveness` parameter.
- The intervention effectiveness is a simulation assumption and is not a clinically established treatment effect.

## 6. Randomness and Reproducibility
- A fixed random seed is used to support reproducibility.
- Common random numbers are used so that simulation scenarios with the same seed can be compared under similar random conditions.
- Multiple replications are performed to assess variation in simulation outcomes.

## 7. Simulation Outputs
The simulation reports:
- Mean hospital occupancy
- Maximum hospital occupancy
- Total bed-days
- Blocked admissions
- Readmissions
- Readmissions prevented
- Bed-days avoided
- Number of flagged discharges

## 8. Model Limitations
- The simulation uses historical data and simplified assumptions to represent hospital operations.
- Patients are not placed in a waiting queue when beds are unavailable.
- Intervention effectiveness is an assumed parameter.
- The simulation represents a simplified hospital system and may not capture every factor affecting real-world admissions and readmissions.
