"""
CLINSIGHT - Streamlit Dashboard
Clinical Supply Chain Intelligence System
End-to-end pipeline: Data Validation -> Anomaly Detection ->
Monte Carlo Simulation -> LLM Intelligence Briefings
"""

import streamlit as st
import pandas as pd
from fpdf import FPDF
from contract_validator import validate_all_studies
from anomaly_detector import detect_all_anomalies
from monte_carlo import run_simulations
from explainer import generate_portfolio_briefings

# -- PAGE CONFIG ------------------------------------------------
st.set_page_config(
    page_title="CLINSIGHT - Clinical Supply Intelligence",
    layout="wide"
)

# -- HEADER -----------------------------------------------------
st.title("CLINSIGHT")
st.subheader("Clinical Supply Chain Intelligence System")
st.markdown(
    "*Automated anomaly detection · Monte Carlo simulation · "
    "LLM-powered intelligence briefings*"
)
st.divider()

# -- LOAD DATA --------------------------------------------------
@st.cache_data
def load_data():
    return {
        "study_config":    pd.read_csv("study_config.csv"),
        "site_profile":    pd.read_csv("site_profile.csv"),
        "site_enrollment": pd.read_csv("site_enrollment.csv"),
        "inventory":       pd.read_csv("inventory.csv"),
        "irt_dispensing":  pd.read_csv("irt_dispensing.csv"),
    }

data = load_data()

# -- PDF GENERATOR ----------------------------------------------
def generate_pdf(briefings):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(auto=True, margin=20)
    w = pdf.w - 40

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(w, 10, "CLINSIGHT - Intelligence Briefings", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        w, 8,
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        ln=True
    )
    pdf.ln(5)

    for b in briefings:
        # Study header
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(w, 10, f"{b.study_id} - {b.urgency}", ln=True)

        # Headline
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(w, 8, b.headline)
        pdf.ln(2)

        # Situation
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(w, 7, "Situation:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(w, 6, b.situation)
        pdf.ln(2)

        # Impact
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(w, 7, "Impact:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(w, 6, b.impact)
        pdf.ln(2)

        # Recommended Actions
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(w, 7, "Recommended Actions:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for i, action in enumerate(b.recommended_actions, 1):
            # Clean the action text - remove leading numbers if present
            clean_action = action.lstrip("0123456789. ")
            pdf.multi_cell(w, 6, f"{i}. {clean_action}")
            pdf.ln(1)

        pdf.ln(8)

    return bytes(pdf.output())

# -- PORTFOLIO OVERVIEW -----------------------------------------
st.markdown("### Portfolio Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Active Studies",
              len(data["study_config"]))
with col2:
    st.metric("Total Sites",
              len(data["site_profile"]))
with col3:
    st.metric("Total Patients Enrolled",
              int(data["site_enrollment"]["enrolled"].sum()))
with col4:
    st.metric("Total Drug Units",
              int(data["inventory"]["actual_units"].sum()))

st.divider()

# -- RUN PIPELINE -----------------------------------------------
st.markdown("### Run Intelligence Pipeline")
run_button = st.button(
    "Run Full CLINSIGHT Analysis",
    type="primary",
    use_container_width=True
)

if run_button:

    # -- LAYER 1 ------------------------------------------------
    with st.spinner("Layer 1 - Validating data contracts..."):
        layer1_results = validate_all_studies(
            data["study_config"],
            data["site_profile"],
            data["site_enrollment"],
            data["inventory"],
            data["irt_dispensing"]
        )

    st.markdown("#### Layer 1 - Data Contract Validation")
    col1, col2, col3 = st.columns(3)

    clear       = [r for r in layer1_results if r.status == "CLEAR"]
    warnings    = [r for r in layer1_results if r.status == "WARNING"]
    quarantined = [r for r in layer1_results if r.status == "QUARANTINED"]

    with col1:
        st.metric("Clear", len(clear), delta="Proceeding to analysis")
    with col2:
        st.metric("Warning", len(warnings), delta="Flagged for review")
    with col3:
        st.metric("Quarantined", len(quarantined), delta="Blocked")

    for r in layer1_results:
        if r.issues:
            with st.expander(
                f"{r.study_id} - {r.status} - {len(r.issues)} issue(s)"
            ):
                for issue in r.issues:
                    st.error(
                        f"{issue.rule} [{issue.severity}]: "
                        f"{issue.description}"
                    )
        else:
            st.success(f"{r.study_id} - All checks passed")

    st.divider()

    # -- LAYER 2 ------------------------------------------------
    cleared_studies = [
        r.study_id for r in layer1_results
        if r.status in ["CLEAR", "WARNING"]
    ]

    with st.spinner("Layer 2 - Detecting business anomalies..."):
        layer2_results = detect_all_anomalies(
            data["study_config"],
            data["site_enrollment"],
            data["inventory"],
            cleared_studies
        )

    all_anomalies = [a for r in layer2_results for a in r.anomalies]

    st.markdown("#### Layer 2 - Business Anomaly Detection")

    critical = [a for a in all_anomalies if a.severity == "CRITICAL"]
    high     = [a for a in all_anomalies if a.severity == "HIGH"]
    medium   = [a for a in all_anomalies if a.severity == "MEDIUM"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Critical Anomalies", len(critical))
    with col2:
        st.metric("High Anomalies", len(high))
    with col3:
        st.metric("Medium Anomalies", len(medium))

    for r in layer2_results:
        if r.has_anomalies:
            with st.expander(
                f"{r.study_id} - {len(r.anomalies)} anomaly(s) - "
                f"Highest: {r.highest_severity}"
            ):
                for a in r.anomalies:
                    if a.severity == "CRITICAL":
                        st.error(
                            f"{a.anomaly_type} - {a.description}"
                        )
                    elif a.severity == "HIGH":
                        st.warning(
                            f"{a.anomaly_type} - {a.description}"
                        )
                    else:
                        st.info(
                            f"{a.anomaly_type} - {a.description}"
                        )
                    st.caption(f"Recommended: {a.recommended_action}")
        else:
            st.success(f"{r.study_id} - No anomalies detected")

    st.divider()

    # -- LAYER 3 ------------------------------------------------
    with st.spinner("Layer 3 - Running Monte Carlo simulations..."):
        simulation_results = run_simulations(
            all_anomalies,
            data["site_enrollment"],
            data["inventory"]
        )

    st.markdown("#### Layer 3 - Monte Carlo Impact Simulation")
    st.caption("Based on 10,000 simulated scenarios per anomaly")

    for s in simulation_results:
        with st.expander(
            f"{s.study_id} - {s.anomaly_type} - Risk: {s.risk_level}"
        ):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("P10 Best Case", s.p10)
            with col2:
                st.metric("P50 Most Likely", s.p50)
            with col3:
                st.metric("P90 Worst Case", s.p90)
            st.caption(s.metric_label)
            st.info(s.interpretation)

    st.divider()

    # -- LAYER 4 ------------------------------------------------
    with st.spinner(
        "Layer 4 - Generating intelligence briefings..."
    ):
        briefings = generate_portfolio_briefings(
            layer2_results, simulation_results
        )

    st.markdown("#### Layer 4 - Intelligence Briefings")

    for b in briefings:
        urgency_color = (
            "red"    if b.urgency == "IMMEDIATE" else
            "orange" if b.urgency == "THIS WEEK" else
            "blue"
        )
        st.markdown(
            f"### {b.study_id} - "
            f":{urgency_color}[{b.urgency}]"
        )
        st.markdown(f"**{b.headline}**")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Situation**")
            st.write(b.situation)
        with col2:
            st.markdown("**Impact**")
            st.write(b.impact)

        st.markdown("**Recommended Actions**")
        for action in b.recommended_actions:
            st.markdown(f"- {action}")

        st.divider()

    st.success(
        f"CLINSIGHT analysis complete. "
        f"{len(all_anomalies)} anomalies detected across "
        f"{len([r for r in layer2_results if r.has_anomalies])} studies. "
        f"{len(briefings)} intelligence briefings generated."
    )

    # -- PDF DOWNLOAD -------------------------------------------
    if briefings:
        pdf_bytes = generate_pdf(briefings)
        st.download_button(
            label="Download Intelligence Report as PDF",
            data=pdf_bytes,
            file_name=f"CLINSIGHT_Report_"
                      f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )