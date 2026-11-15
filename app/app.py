from pathlib import Path
import sys

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from scipy.stats import t

# Make imports work when launched from the project root.
ROOT = Path(__file__).resolve().parents[1]
# If you download this file, place it at D:\healthcare_capstone\app\app.py.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assistant.text_to_sql import ask
from src.sim.simulate import simulate

DB_PATH = ROOT / "hospital.duckdb"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"

st.set_page_config(
    page_title="Hospital Decision Support",
    page_icon="🏥",
    layout="wide",
)

st.title("Hospital Decision-Support Platform")
st.warning("Prototype using synthetic data. Not for clinical use.")

tab1, tab2, tab3 = st.tabs([
    "Cohort Overview",
    "Text-to-SQL Assistant",
    "Digital Twin Simulator",
])


# TAB 1 — COHORT OVERVIEW
with tab1:
    st.header("Hospital Cohort Overview")

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        total_stays = con.execute(
            "SELECT COUNT(*) FROM syn_stays"
        ).fetchone()[0]
        total_patients = con.execute(
            "SELECT COUNT(DISTINCT PATIENT) FROM syn_stays"
        ).fetchone()[0]
        readmissions = con.execute(
            "SELECT COUNT(*) FROM syn_stays WHERE readmit_30d = TRUE"
        ).fetchone()[0]
    finally:
        con.close()

    col1, col2, col3 = st.columns(3)
    col1.metric("Hospital Stays", total_stays)
    col2.metric("Unique Patients", total_patients)
    col3.metric("30-Day Readmissions", readmissions)
    st.metric("Readmission Rate", f"{readmissions / total_stays * 100:.1f}%")

    st.divider()
    st.subheader("Synthea Cohort Analysis")
    st.caption("Exploratory analysis of synthetic hospital admissions from 2017 to 2025.")

    fig1, fig2 = st.columns(2)
    with fig1:
        st.markdown("**Admissions per Year**")
        st.image(str(FIGURES / "synthea_admissions_per_year.png"), use_container_width=True)
    with fig2:
        st.markdown("**Length-of-Stay Distribution**")
        st.image(str(FIGURES / "synthea_los_hist.png"), use_container_width=True)

    st.markdown("**Readmission Analysis by Age**")
    st.image(str(FIGURES / "synthea_readmit_by_age.png"), use_container_width=True)

    st.divider()
    st.subheader("Readmission Prediction Performance")
    st.caption("Results produced by the predictive modelling component of the capstone.")

    with st.expander("Synthea Cross-Validation Results"):
        df = pd.read_csv(TABLES / "synthea_cv_results.csv")
        st.dataframe(df, use_container_width=True)
    with st.expander("Synthea Temporal Validation"):
        df = pd.read_csv(TABLES / "synthea_temporal_results.csv")
        st.dataframe(df, use_container_width=True)
    with st.expander("Initial Model Comparison"):
        df = pd.read_csv(TABLES / "synthea_model_results_v1.csv")
        st.dataframe(df, use_container_width=True)

    st.markdown("**Synthea Calibration**")
    st.image(str(FIGURES / "synthea_calibration.png"), use_container_width=True)

    st.divider()
    st.subheader("UCI Diabetes — External Dataset Check")
    st.info(
        "The UCI Diabetes dataset is used to evaluate the modelling pipeline "
        "on real, de-identified data. It is not used in the hospital simulator."
    )
    with st.expander("UCI Model Evaluation"):
        df = pd.read_csv(TABLES / "uci_cv_results.csv")
        st.dataframe(df, use_container_width=True)
    with st.expander("Top-K Risk Capture Comparison"):
        syn_topk = pd.read_csv(TABLES / "synthea_topk_capture.csv")
        uci_topk = pd.read_csv(TABLES / "uci_topk_capture.csv")
        st.markdown("**Synthea**")
        st.dataframe(syn_topk, use_container_width=True)
        st.markdown("**UCI Diabetes**")
        st.dataframe(uci_topk, use_container_width=True)

    fig3, fig4 = st.columns(2)
    with fig3:
        st.markdown("**UCI Calibration**")
        st.image(str(FIGURES / "uci_calibration.png"), use_container_width=True)
    with fig4:
        st.markdown("**UCI SHAP Feature Importance**")
        st.image(str(FIGURES / "uci_shap_bar.png"), use_container_width=True)


# TAB 2 — TEXT-TO-SQL ASSISTANT
with tab2:
    st.header("Ask Hospital Data")
    question = st.text_input("Ask a question about hospital admissions:")

    if st.button("Generate Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating SQL and retrieving results..."):
                result = ask(question)

            if result["sql"]:
                st.subheader("Generated SQL")
                st.code(result["sql"], language="sql")
            if result["ok"]:
                st.success("Query executed successfully.")
                st.dataframe(result["rows"], use_container_width=True)
            else:
                st.error(result["error"] or "Query failed.")


# TAB 3 — DIGITAL TWIN SIMULATOR
with tab3:
    st.header("Hospital Capacity Digital Twin")
    st.caption(
        "Test bed-capacity and discharge-support assumptions using "
        "Person B's SimPy simulation."
    )
    st.info(
        "Simulation uses synthetic data. Arrival scaling and intervention "
        "effectiveness are assumptions, not measured clinical effects."
    )

    st.subheader("Policy Configuration")
    with st.form("simulation_form"):
        c1, c2 = st.columns(2)
        with c1:
            beds = st.slider("Hospital beds", 10, 120, 80, 5)
            top_pct = st.slider("Top-risk discharges supported (%)", 0, 30, 10, 5)
        with c2:
            arrival_scale = st.slider("Arrival multiplier", 1, 30, 20)
            effectiveness_pct = st.slider("Assumed intervention effectiveness (%)", 0, 50, 20, 5)

        c3, c4 = st.columns(2)
        with c3:
            n_days = st.selectbox("Simulation days", [30, 90, 365], index=0)
        with c4:
            n_reps = st.selectbox("Replications", [2, 5, 10, 30], index=0)

        submitted = st.form_submit_button("Run Simulation", type="primary")

    if submitted:
        config = {
            "beds": beds,
            "arrival_scale": float(arrival_scale),
            "top_pct": top_pct,
            "effectiveness": effectiveness_pct / 100.0,
            "n_days": n_days,
            "n_reps": n_reps,
            "seed": 42,
            "warmup_days": 10 if n_days == 30 else 60,
            "db_path": str(DB_PATH),
        }
        try:
            with st.spinner("Running simulation replications..."):
                new_results = simulate(config)

            required = {
                "rep", "mean_occupancy", "max_occupancy", "bed_days",
                "blocked_admissions", "readmissions", "readmissions_prevented",
            }
            if new_results.empty or not required.issubset(new_results.columns):
                st.error("The simulation returned no data or is missing expected columns.")
            else:
                st.session_state["sim_results"] = new_results
                st.session_state["sim_config"] = config
                st.success("Simulation completed.")
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")

    if "sim_results" in st.session_state:
        df = st.session_state["sim_results"]
        used = st.session_state["sim_config"]

        st.divider()
        st.subheader("Simulation Results")
        st.caption(
            f"Last successful run: {used['n_reps']} replications, "
            f"{used['n_days']} simulated days, {used['beds']} beds, seed {used['seed']}."
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean Bed Occupancy", f"{df['mean_occupancy'].mean():.1f}")
        m2.metric("Mean Blocked Admissions", f"{df['blocked_admissions'].mean():.1f}")
        m3.metric("Mean Readmissions Prevented", f"{df['readmissions_prevented'].mean():.1f}")
        if "bed_days_avoided" in df.columns:
            st.metric("Mean Bed-Days Avoided", f"{df['bed_days_avoided'].mean():.1f}")

        st.subheader("Results by Replication")
        st.dataframe(df.round(2), use_container_width=True, hide_index=True)
        plot1, plot2 = st.columns(2)
        with plot1:
            st.markdown("**Bed Occupancy by Replication**")
            st.bar_chart(df.set_index("rep")["mean_occupancy"])
        with plot2:
            st.markdown("**Blocked Admissions by Replication**")
            st.bar_chart(df.set_index("rep")["blocked_admissions"])

        st.subheader("95% Confidence Intervals Across Replications")
        metrics = [
            "mean_occupancy", "blocked_admissions", "readmissions",
            "readmissions_prevented", "bed_days",
        ]
        if "bed_days_avoided" in df.columns:
            metrics.append("bed_days_avoided")

        summary = []
        for metric in metrics:
            values = pd.to_numeric(df[metric], errors="coerce").dropna()
            n = len(values)
            avg = values.mean()
            margin = (
                t.ppf(0.975, df=n - 1) * values.std(ddof=1) / (n ** 0.5)
                if n > 1 else float("nan")
            )
            summary.append({
                "Metric": metric,
                "Mean": avg,
                "CI Lower": avg - margin,
                "CI Upper": avg + margin,
            })
        ci_df = pd.DataFrame(summary)
        st.dataframe(ci_df.round(2), use_container_width=True, hide_index=True)

        # Mean and confidence interval plots for the two operational outcomes.
        graph1, graph2 = st.columns(2)
        for column, metric, label in [
            (graph1, "mean_occupancy", "Mean bed occupancy"),
            (graph2, "blocked_admissions", "Blocked admissions"),
        ]:
            row = ci_df.loc[ci_df["Metric"] == metric].iloc[0]
            avg = float(row["Mean"])
            margin = float(row["CI Upper"] - row["Mean"])
            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.errorbar([0], [avg], yerr=[margin], fmt="o", capsize=6)
            ax.set_xticks([0], [label])
            ax.set_xlim(-0.8, 0.8)
            ax.set_ylabel("Mean with 95% CI")
            ax.grid(axis="y", alpha=0.2)
            with column:
                st.pyplot(fig)
            plt.close(fig)

        if len(df) < 30:
            st.warning(
                "Development run: the guide requires at least 30 replications "
                "for final scenario reporting. CIs from only 2 replications "
                "are especially unstable."
            )

    st.divider()
    st.subheader("Saved Policy Experiments from Person B")
    saved_files = {
        "Main Policy Results": "sim_main_results.csv",
        "Sensitivity Analysis": "sim_sensitivity.csv",
        "UCI-Based Targeting Scenario": "sim_uci_targeting.csv",
    }
    for title, filename in saved_files.items():
        with st.expander(title):
            path = TABLES / filename
            if path.exists():
                saved = pd.read_csv(path)
                st.dataframe(saved, use_container_width=True)
            else:
                st.warning(f"Missing report: {filename}")
