"""
CLINSIGHT - Layer 4: LLM Intelligence Explainer
Takes anomalies + Monte Carlo simulation results and generates
plain English intelligence briefings with recommended actions.
Powered by LLaMA 3.3 via Groq API.
"""

import os
from groq import Groq
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List
from anomaly_detector import Anomaly, StudyAnomalyResult
from monte_carlo import SimulationResult

load_dotenv()
GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── DATA STRUCTURES ────────────────────────────────────────────

@dataclass
class IntelligenceBriefing:
    study_id:        str
    headline:        str
    situation:       str
    impact:          str
    recommended_actions: List[str]
    urgency:         str  # IMMEDIATE / THIS WEEK / MONITOR


# ── BRIEFING GENERATOR ─────────────────────────────────────────

def generate_study_briefing(study_id:     str,
                             anomalies:    List[Anomaly],
                             simulations:  List[SimulationResult]
                             ) -> IntelligenceBriefing:
    """
    Generate a plain English intelligence briefing for a study
    by combining anomaly findings with Monte Carlo projections.
    """

    # Build context for LLM
    anomaly_summary = "\n".join([
        f"- [{a.severity}] {a.anomaly_type}: {a.description} "
        f"Recommended action: {a.recommended_action}"
        for a in anomalies
    ])

    simulation_summary = "\n".join([
        f"- {s.anomaly_type}: {s.metric_label} — "
        f"P10={s.p10}, P50={s.p50}, P90={s.p90}. {s.interpretation}"
        for s in simulations
    ])

    highest_severity = "CRITICAL" if any(
        a.severity == "CRITICAL" for a in anomalies
    ) else "HIGH" if any(
        a.severity == "HIGH" for a in anomalies
    ) else "MEDIUM"

    prompt = f"""You are a senior clinical supply chain intelligence analyst.
You are writing a concise executive briefing for study {study_id}.

ANOMALIES DETECTED:
{anomaly_summary}

MONTE CARLO IMPACT PROJECTIONS:
{simulation_summary}

Write a structured intelligence briefing with exactly these four sections:

HEADLINE: One sentence capturing the most critical issue. Maximum 20 words.

SITUATION: Two to three sentences describing what is happening and why it matters. Be specific with numbers.

IMPACT: Two sentences describing the projected impact based on Monte Carlo results. Use P10/P50/P90 numbers.

ACTIONS: List exactly three specific recommended actions numbered 1, 2, 3. Each action should be concrete and executable today.

Keep the tone professional, urgent where needed, and data-driven.
Do not use bullet points except for the actions list.
Do not add any preamble or closing remarks."""

    response = GROQ_CLIENT.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )

    raw = response.choices[0].message.content.strip()

    # Parse structured response
    lines = raw.split("\n")
    headline = ""
    situation = ""
    impact = ""
    actions = []

    current_section = None
    for line in lines:
        line = line.strip()
        if line.startswith("HEADLINE:"):
            headline = line.replace("HEADLINE:", "").strip()
            current_section = "headline"
        elif line.startswith("SITUATION:"):
            situation = line.replace("SITUATION:", "").strip()
            current_section = "situation"
        elif line.startswith("IMPACT:"):
            impact = line.replace("IMPACT:", "").strip()
            current_section = "impact"
        elif line.startswith("ACTIONS:"):
            current_section = "actions"
        elif current_section == "situation" and line:
            situation += " " + line
        elif current_section == "impact" and line:
            impact += " " + line
        elif current_section == "actions" and line:
            if line[0].isdigit():
                actions.append(line)

    urgency = "IMMEDIATE" if highest_severity == "CRITICAL" else \
              "THIS WEEK" if highest_severity == "HIGH" else "MONITOR"

    return IntelligenceBriefing(
        study_id=study_id,
        headline=headline,
        situation=situation,
        impact=impact,
        recommended_actions=actions,
        urgency=urgency
    )


def generate_portfolio_briefings(layer2_results:      List[StudyAnomalyResult],
                                  simulation_results:  List[SimulationResult]
                                  ) -> List[IntelligenceBriefing]:
    """Generate briefings for all studies with anomalies."""
    briefings = []

    for study_result in layer2_results:
        if not study_result.has_anomalies:
            continue

        study_sims = [
            s for s in simulation_results
            if s.study_id == study_result.study_id
        ]

        print(f"Generating briefing for {study_result.study_id}...")
        briefing = generate_study_briefing(
            study_result.study_id,
            study_result.anomalies,
            study_sims
        )
        briefings.append(briefing)

    return briefings


def print_intelligence_briefings(briefings: List[IntelligenceBriefing]):
    """Print intelligence briefings in clean format."""
    print("\n" + "=" * 65)
    print("      CLINSIGHT - LAYER 4: INTELLIGENCE BRIEFINGS")
    print("=" * 65)

    for b in briefings:
        print(f"\n{'=' * 65}")
        print(f"STUDY: {b.study_id}  |  URGENCY: {b.urgency}")
        print(f"{'=' * 65}")
        print(f"\nHEADLINE: {b.headline}")
        print(f"\nSITUATION:\n{b.situation}")
        print(f"\nIMPACT:\n{b.impact}")
        print(f"\nRECOMMENDED ACTIONS:")
        for action in b.recommended_actions:
            print(f"  {action}")

    print(f"\n{'=' * 65}")
    print(f"Total briefings generated: {len(briefings)}")
    print("=" * 65)


if __name__ == "__main__":
    import pandas as pd
    from contract_validator import validate_all_studies
    from anomaly_detector import detect_all_anomalies
    from monte_carlo import run_simulations

    study_config    = pd.read_csv("study_config.csv")
    site_profile    = pd.read_csv("site_profile.csv")
    site_enrollment = pd.read_csv("site_enrollment.csv")
    inventory       = pd.read_csv("inventory.csv")
    irt_dispensing  = pd.read_csv("irt_dispensing.csv")

    print("CLINSIGHT - Running Full Pipeline...\n")

    # Layer 1
    layer1_results  = validate_all_studies(
        study_config, site_profile,
        site_enrollment, inventory, irt_dispensing
    )
    cleared_studies = [
        r.study_id for r in layer1_results
        if r.status in ["CLEAR", "WARNING"]
    ]

    # Layer 2
    layer2_results = detect_all_anomalies(
        study_config, site_enrollment,
        inventory, cleared_studies
    )
    all_anomalies = [
        a for r in layer2_results for a in r.anomalies
    ]

    # Layer 3
    simulation_results = run_simulations(
        all_anomalies, site_enrollment, inventory
    )

    # Layer 4
    print("Generating intelligence briefings...\n")
    briefings = generate_portfolio_briefings(
        layer2_results, simulation_results
    )

    print_intelligence_briefings(briefings)