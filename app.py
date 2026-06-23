"""
CLINSIGHT - Streamlit Dashboard v2.0
Clinical Supply Chain Intelligence System
"""

import streamlit as st
import pandas as pd
from fpdf import FPDF
from contract_validator import validate_all_studies
from anomaly_detector import detect_all_anomalies
from monte_carlo import run_simulations
from explainer import generate_portfolio_briefings

st.set_page_config(
    page_title="CLINSIGHT - Clinical Supply Intelligence",
    layout="wide"
)

st.title("CLINSIGHT")
st.subheader("Clinical Supply Chain Intelligence System")
st.markdown(
    "*Automated anomaly detection · Monte Carlo simulation · "
    "LLM-powered intelligence briefings*"
)
st.divider()


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


def generate_pdf(layer1_results, layer2_results, simulation_results, briefings):

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    LEFT  = 15
    RIGHT = 15
    pdf.set_margins(LEFT, 20, RIGHT)
    w = pdf.w - LEFT - RIGHT  # 180mm usable width

    # Colors
    NAVY   = (31,  56,  100)
    BLUE   = (46,  117, 182)
    RED    = (192, 0,   0)
    ORANGE = (255, 102, 0)
    GREEN  = (0,   112, 0)
    GRAY   = (128, 128, 128)
    LIGHT  = (240, 240, 240)
    WHITE  = (255, 255, 255)
    PURPLE = (150, 0,   150)

    # ── helpers ───────────────────────────────────────────────

    def ct(text):
        """Clean unicode characters unsupported by Helvetica."""
        if not text:
            return ""
        return (str(text)
                .replace("\u2014", "-").replace("\u2013", "-")
                .replace("\u2018", "'").replace("\u2019", "'")
                .replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2022", "-").replace("\u00e9", "e")
                .replace("\u00e0", "a").replace("\u00e8", "e")
                .replace("\u00ea", "e").replace("\u00f3", "o")
                .replace("\u00fa", "u").replace("\u00fc", "u")
                .replace("\u00e4", "a").replace("\u00f6", "o"))

    def go_left():
        """Reset x to left margin before multi_cell."""
        pdf.set_x(LEFT)

    def page_header(title, subtitle=""):
        pdf.set_fill_color(*NAVY)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 15)
        pdf.cell(w, 11, ct(title), ln=True, fill=True, align="C")
        if subtitle:
            pdf.set_fill_color(*BLUE)
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(w, 7, ct(subtitle), ln=True, fill=True, align="C")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

    def sec(text, color=BLUE):
        pdf.set_fill_color(*color)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(w, 7, f"  {ct(text)}", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    def sev_color(s):
        return RED if s == "CRITICAL" else ORANGE if s == "HIGH" else (200, 100, 0)

    def sta_color(s):
        return RED if s == "QUARANTINED" else ORANGE if s == "WARNING" else GREEN

    # ── PAGE 1 — COVER + SUMMARY ──────────────────────────────
    pdf.add_page()

    # Dark header band
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, pdf.w, 42, "F")
    pdf.set_y(7)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 11, "CLINSIGHT", ln=True, align="C")
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Clinical Supply Chain Intelligence Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(0, 6,
             f"Generated: {pd.Timestamp.now().strftime('%B %d, %Y  %H:%M')}  |  Confidential",
             ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(13)

    # Summary boxes
    sec("PORTFOLIO SUMMARY")
    clear_n = len([r for r in layer1_results if r.status == "CLEAR"])
    warn_n  = len([r for r in layer1_results if r.status == "WARNING"])
    quar_n  = len([r for r in layer1_results if r.status == "QUARANTINED"])
    anom_n  = sum(len(r.anomalies) for r in layer2_results)

    bw = (w - 8) / 5
    for label, val, col in [
        ("Total Studies", str(len(layer1_results)), NAVY),
        ("Clear",         str(clear_n),             GREEN),
        ("Warning",       str(warn_n),               ORANGE),
        ("Quarantined",   str(quar_n),               RED),
        ("Anomalies",     str(anom_n),               PURPLE),
    ]:
        x0 = pdf.get_x()
        y0 = pdf.get_y()
        pdf.set_fill_color(*col)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(bw, 13, val, align="C", fill=True)
        pdf.set_x(x0)
        pdf.set_y(y0 + 13)
        pdf.set_font("Helvetica", "", 7)
        pdf.cell(bw, 5, label, align="C", fill=True)
        pdf.set_y(y0)
        pdf.set_x(x0 + bw + 2)

    pdf.set_text_color(0, 0, 0)
    pdf.ln(22)
    pdf.ln(3)

    # Study status table
    sec("STUDY VALIDATION STATUS")
    cw = [44, 30, 22, 18, w - 114]
    pdf.set_fill_color(*NAVY)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 9)
    for i, h in enumerate(["Study ID", "Status", "Passed", "Failed", "Primary Issue"]):
        pdf.cell(cw[i], 7, h, border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    for idx, r in enumerate(layer1_results):
        bg = LIGHT if idx % 2 == 0 else WHITE
        pdf.set_fill_color(*bg)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(cw[0], 7, r.study_id, border=1, fill=True)
        pdf.set_text_color(*sta_color(r.status))
        pdf.cell(cw[1], 7, r.status, border=1, fill=True, align="C")
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(cw[2], 7, str(r.passed), border=1, fill=True, align="C")
        pdf.cell(cw[3], 7, str(r.failed), border=1, fill=True, align="C")
        issue_text = ct(r.issues[0].rule) if r.issues else "None"
        # truncate only if really long
        if len(issue_text) > 38:
            issue_text = issue_text[:36] + ".."
        pdf.cell(cw[4], 7, issue_text, border=1, fill=True)
        pdf.ln()

    # ── PAGE 2 — DATA CONTRACT VIOLATIONS ─────────────────────
    pdf.add_page()
    page_header("LAYER 1 - DATA CONTRACT VIOLATIONS",
                "Cross-system relationship validation results")

    violations = [r for r in layer1_results if r.issues]
    if not violations:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*GREEN)
        go_left()
        pdf.multi_cell(w, 8, "All data contract checks passed. No violations found.")
        pdf.set_text_color(0, 0, 0)
    else:
        for r in violations:
            # Study bar
            pdf.set_fill_color(*sta_color(r.status))
            pdf.set_text_color(*WHITE)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(w, 8,
                     f"  {r.study_id}  |  {r.status}  |  {len(r.issues)} violation(s)",
                     ln=True, fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)

            for issue in r.issues:
                # Severity badge + rule name
                pdf.set_fill_color(*sev_color(issue.severity))
                pdf.set_text_color(*WHITE)
                pdf.set_font("Helvetica", "B", 8)
                pdf.cell(26, 6, f"  {issue.severity}", fill=True)
                pdf.set_fill_color(*LIGHT)
                pdf.set_text_color(0, 0, 0)
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(w - 26, 6, f"  {ct(issue.rule)}", fill=True, ln=True)

                # Description
                pdf.set_font("Helvetica", "", 8)
                go_left()
                pdf.multi_cell(w, 5, ct(issue.description))
                pdf.ln(1)

                # Side-by-side comparison for site ID mismatch
                if "SITE_ID_MISMATCH" in issue.rule:
                    import re
                    half = (w - 2) / 2

                    pdf.set_fill_color(*NAVY)
                    pdf.set_text_color(*WHITE)
                    pdf.set_font("Helvetica", "B", 9)
                    go_left()
                    pdf.cell(half, 7, "  IRT Site IDs  (Received)",
                             border=1, fill=True, align="C")
                    pdf.cell(2,    7, "", border=0)
                    pdf.cell(half, 7, "  Site Profile IDs  (Expected)",
                             border=1, fill=True, align="C")
                    pdf.ln()
                    pdf.set_text_color(0, 0, 0)

                    irt_m  = re.search(r"IRT sent \d+ site IDs: \{([^}]+)\}",
                                       issue.description)
                    pro_m  = re.search(r"Site Profile has \d+ site IDs: \{([^}]+)\}",
                                       issue.description)
                    irt_ids  = irt_m.group(1).replace("'", "").split(", ") \
                               if irt_m else []
                    prof_ids = pro_m.group(1).replace("'", "").split(", ") \
                               if pro_m else []

                    for i in range(max(len(irt_ids), len(prof_ids))):
                        bg = LIGHT if i % 2 == 0 else WHITE
                        pdf.set_fill_color(*bg)
                        pdf.set_font("Helvetica", "", 9)
                        iv = irt_ids[i]  if i < len(irt_ids)  else "-"
                        pv = prof_ids[i] if i < len(prof_ids) else "-"
                        go_left()
                        pdf.set_text_color(*(RED if iv not in prof_ids else (0, 0, 0)))
                        pdf.cell(half, 6, f"  {iv}", border=1, fill=True)
                        pdf.set_text_color(0, 0, 0)
                        pdf.cell(2, 6, "", border=0)
                        pdf.set_text_color(*(RED if pv not in irt_ids else (0, 0, 0)))
                        pdf.cell(half, 6, f"  {pv}", border=1, fill=True)
                        pdf.set_text_color(0, 0, 0)
                        pdf.ln()

                    pdf.ln(2)
                    pdf.set_font("Helvetica", "I", 8)
                    pdf.set_text_color(*RED)
                    go_left()
                    pdf.multi_cell(w, 5,
                                   "Red = unmatched Site IDs. "
                                   "Dashboard will show blank data for these sites.")
                    pdf.set_text_color(0, 0, 0)

                pdf.ln(3)
            pdf.ln(2)

    # ── PAGE 3 — BUSINESS ANOMALIES ───────────────────────────
    pdf.add_page()
    page_header("LAYER 2 - BUSINESS ANOMALY DETECTION",
                "Operational issues detected in validated clinical supply data")

    anom_studies = [r for r in layer2_results if r.has_anomalies]
    if not anom_studies:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*GREEN)
        go_left()
        pdf.multi_cell(w, 8, "No business anomalies detected.")
        pdf.set_text_color(0, 0, 0)
    else:
        # Summary table
        sec("ANOMALY SUMMARY TABLE")
        aw = [40, 46, 24, w - 110]
        pdf.set_fill_color(*NAVY)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 9)
        for h in ["Study ID", "Anomaly Type", "Severity", "Key Metric"]:
            pdf.cell(aw[0 if h == "Study ID"
                        else 1 if h == "Anomaly Type"
                        else 2 if h == "Severity"
                        else 3],
                     7, h, border=1, fill=True, align="C")
        pdf.ln()
        pdf.set_text_color(0, 0, 0)

        # rebuild with index
        row_i = 0
        for r in anom_studies:
            for a in r.anomalies:
                bg = LIGHT if row_i % 2 == 0 else WHITE
                pdf.set_fill_color(*bg)
                pdf.set_font("Helvetica", "", 9)
                pdf.cell(aw[0], 6, a.study_id,           border=1, fill=True)
                pdf.cell(aw[1], 6, ct(a.anomaly_type),   border=1, fill=True)
                pdf.set_text_color(*sev_color(a.severity))
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(aw[2], 6, a.severity,
                         border=1, fill=True, align="C")
                pdf.set_text_color(0, 0, 0)
                pdf.set_font("Helvetica", "", 8)
                metric_txt = ct(a.metric)
                if len(metric_txt) > 52:
                    metric_txt = metric_txt[:50] + ".."
                pdf.cell(aw[3], 6, metric_txt, border=1, fill=True)
                pdf.ln()
                row_i += 1

        pdf.ln(4)
        sec("DETAILED FINDINGS")

        for r in anom_studies:
            hs = r.highest_severity
            bar_col = RED if hs == "CRITICAL" else ORANGE if hs == "HIGH" else GREEN
            pdf.set_fill_color(*bar_col)
            pdf.set_text_color(*WHITE)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(w, 8,
                     f"  {r.study_id}  |  {len(r.anomalies)} anomaly(s)"
                     f"  |  Highest: {hs}",
                     ln=True, fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)

            for a in r.anomalies:
                # Badge row
                pdf.set_fill_color(*sev_color(a.severity))
                pdf.set_text_color(*WHITE)
                pdf.set_font("Helvetica", "B", 8)
                pdf.cell(26, 6, f"  {a.severity}", fill=True)
                pdf.set_fill_color(*LIGHT)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(w - 26, 6, f"  {ct(a.anomaly_type)}", fill=True, ln=True)

                # Description
                pdf.set_font("Helvetica", "", 8)
                go_left()
                pdf.multi_cell(w, 5, ct(a.description))

                # Recommended action — full text, no truncation
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(*BLUE)
                go_left()
                pdf.multi_cell(w, 5, f"Action: {ct(a.recommended_action)}")
                pdf.set_text_color(0, 0, 0)
                pdf.ln(2)
            pdf.ln(3)

    # ── PAGE 4 — INTELLIGENCE BRIEFINGS ───────────────────────
    if briefings:
        pdf.add_page()
        page_header("LAYER 4 - INTELLIGENCE BRIEFINGS",
                    "LLM-generated executive summaries with recommended actions")

        uc_map = {"IMMEDIATE": RED, "THIS WEEK": ORANGE, "MONITOR": GREEN}

        for b in briefings:
            uc = uc_map.get(b.urgency, BLUE)
            pdf.set_fill_color(*uc)
            pdf.set_text_color(*WHITE)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(w, 8, f"  {b.study_id}  |  {b.urgency}",
                     ln=True, fill=True)
            pdf.set_text_color(0, 0, 0)

            pdf.set_font("Helvetica", "B", 10)
            go_left()
            pdf.multi_cell(w, 6, ct(b.headline))
            pdf.ln(1)

            for label, content in [
                ("Situation:", b.situation),
                ("Impact:",    b.impact),
            ]:
                pdf.set_font("Helvetica", "B", 9)
                go_left()
                pdf.cell(w, 6, label, ln=True)
                pdf.set_font("Helvetica", "", 9)
                go_left()
                pdf.multi_cell(w, 5, ct(content))
                pdf.ln(1)

            pdf.set_font("Helvetica", "B", 9)
            go_left()
            pdf.cell(w, 6, "Recommended Actions:", ln=True)
            pdf.set_font("Helvetica", "", 9)
            for action in b.recommended_actions:
                clean_a = ct(action.lstrip("0123456789. "))
                go_left()
                pdf.multi_cell(w, 5, f"  - {clean_a}")
            pdf.ln(5)

    # Footer on last page
    pdf.set_y(-12)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 5,
             "CLINSIGHT - Clinical Supply Chain Intelligence System  "
             "|  Confidential  |  For Internal Use Only",
             align="C")

    return bytes(pdf.output())


# -- PORTFOLIO OVERVIEW -----------------------------------------
st.markdown("### Portfolio Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Active Studies", len(data["study_config"]))
with col2:
    st.metric("Total Sites", len(data["site_profile"]))
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

    # LAYER 1
    with st.spinner("Layer 1 - Validating data contracts..."):
        layer1_results = validate_all_studies(
            data["study_config"], data["site_profile"],
            data["site_enrollment"], data["inventory"],
            data["irt_dispensing"]
        )

    st.markdown("#### Layer 1 - Data Contract Validation")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Clear",
                  len([r for r in layer1_results if r.status == "CLEAR"]),
                  delta="Proceeding")
    with c2:
        st.metric("Warning",
                  len([r for r in layer1_results if r.status == "WARNING"]),
                  delta="Flagged")
    with c3:
        st.metric("Quarantined",
                  len([r for r in layer1_results if r.status == "QUARANTINED"]),
                  delta="Blocked")

    for r in layer1_results:
        if r.issues:
            with st.expander(f"{r.study_id} - {r.status} - {len(r.issues)} issue(s)"):
                for issue in r.issues:
                    if issue.severity == "CRITICAL":
                        st.error(f"{issue.rule} [{issue.severity}]: {issue.description}")
                    else:
                        st.warning(f"{issue.rule} [{issue.severity}]: {issue.description}")
        else:
            st.success(f"{r.study_id} - All checks passed")

    st.divider()

    # LAYER 2
    cleared = [r.study_id for r in layer1_results
               if r.status in ["CLEAR", "WARNING"]]

    with st.spinner("Layer 2 - Detecting business anomalies..."):
        layer2_results = detect_all_anomalies(
            data["study_config"], data["site_enrollment"],
            data["inventory"], cleared
        )

    all_anomalies = [a for r in layer2_results for a in r.anomalies]

    st.markdown("#### Layer 2 - Business Anomaly Detection")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Critical", len([a for a in all_anomalies if a.severity == "CRITICAL"]))
    with c2:
        st.metric("High",     len([a for a in all_anomalies if a.severity == "HIGH"]))
    with c3:
        st.metric("Medium",   len([a for a in all_anomalies if a.severity == "MEDIUM"]))

    for r in layer2_results:
        if r.has_anomalies:
            with st.expander(f"{r.study_id} - {len(r.anomalies)} anomaly(s) - Highest: {r.highest_severity}"):
                for a in r.anomalies:
                    if a.severity == "CRITICAL":
                        st.error(f"{a.anomaly_type} - {a.description}")
                    elif a.severity == "HIGH":
                        st.warning(f"{a.anomaly_type} - {a.description}")
                    else:
                        st.info(f"{a.anomaly_type} - {a.description}")
                    st.caption(f"Recommended: {a.recommended_action}")
        else:
            st.success(f"{r.study_id} - No anomalies detected")

    st.divider()

    # LAYER 3
    with st.spinner("Layer 3 - Running Monte Carlo simulations..."):
        simulation_results = run_simulations(
            all_anomalies, data["site_enrollment"], data["inventory"]
        )

    st.markdown("#### Layer 3 - Monte Carlo Impact Simulation")
    st.caption("Based on 10,000 simulated scenarios per anomaly")

    for s in simulation_results:
        with st.expander(f"{s.study_id} - {s.anomaly_type} - Risk: {s.risk_level}"):
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("P10 Best Case",   s.p10)
            with c2: st.metric("P50 Most Likely",  s.p50)
            with c3: st.metric("P90 Worst Case",   s.p90)
            st.caption(s.metric_label)
            st.info(s.interpretation)

    st.divider()

    # LAYER 4
    with st.spinner("Layer 4 - Generating intelligence briefings..."):
        briefings = generate_portfolio_briefings(layer2_results, simulation_results)

    st.markdown("#### Layer 4 - Intelligence Briefings")
    for b in briefings:
        uc = ("red"    if b.urgency == "IMMEDIATE" else
              "orange" if b.urgency == "THIS WEEK"  else "blue")
        st.markdown(f"### {b.study_id} - :{uc}[{b.urgency}]")
        st.markdown(f"**{b.headline}**")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Situation**")
            st.write(b.situation)
        with c2:
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

    # PDF DOWNLOAD
    pdf_bytes = generate_pdf(
        layer1_results, layer2_results, simulation_results, briefings
    )
    st.download_button(
        label="Download Intelligence Report as PDF",
        data=pdf_bytes,
        file_name=f"CLINSIGHT_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )