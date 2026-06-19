"""
CLINSIGHT - Layer 3: Monte Carlo Simulation Engine
For every anomaly detected in Layer 2, simulates 10,000 possible
futures to produce P10/P50/P90 impact projections.
Answers: "How bad could this get?"
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List
from anomaly_detector import Anomaly

# Simulation config
N_SIMULATIONS = 10_000
np.random.seed(42)


# ── DATA STRUCTURES ────────────────────────────────────────────

@dataclass
class SimulationResult:
    study_id:      str
    anomaly_type:  str
    description:   str
    p10:           float
    p50:           float
    p90:           float
    metric_label:  str
    interpretation: str
    risk_level:    str  # LOW / MEDIUM / HIGH / CRITICAL


# ── SIMULATION ENGINES ─────────────────────────────────────────

def simulate_enrollment_impact(anomaly: Anomaly,
                                site_enrollment: pd.DataFrame
                                ) -> SimulationResult:
    """
    Simulate the range of possible enrollment outcomes given
    a site performing below target.
    Projects: weeks to reach target enrollment.
    """
    study_data = site_enrollment[site_enrollment["study_id"] == anomaly.study_id]

    # Extract current enrollment metrics
    total_enrolled = study_data["enrolled"].sum()
    total_target   = study_data["target"].sum()
    remaining      = max(0, total_target - total_enrolled)

    # Historical weekly enrollment rate with uncertainty
    avg_weekly_rate = total_enrolled / 26  # Assume 26 weeks running
    std_weekly_rate = avg_weekly_rate * 0.3  # 30% variability

    # Simulate 10,000 possible futures
    simulated_weeks = []
    for _ in range(N_SIMULATIONS):
        weekly_rate = max(0.1, np.random.normal(avg_weekly_rate, std_weekly_rate))
        weeks_needed = remaining / weekly_rate
        simulated_weeks.append(weeks_needed)

    simulated_weeks = np.array(simulated_weeks)
    p10 = np.percentile(simulated_weeks, 10)
    p50 = np.percentile(simulated_weeks, 50)
    p90 = np.percentile(simulated_weeks, 90)

    risk_level = "CRITICAL" if p90 > 52 else "HIGH" if p90 > 26 else "MEDIUM"

    return SimulationResult(
        study_id=anomaly.study_id,
        anomaly_type=anomaly.anomaly_type,
        description=anomaly.description,
        p10=round(p10, 1),
        p50=round(p50, 1),
        p90=round(p90, 1),
        metric_label="Weeks to complete enrollment",
        interpretation=(
            f"Best case: {round(p10,1)} weeks to full enrollment. "
            f"Most likely: {round(p50,1)} weeks. "
            f"Worst case: {round(p90,1)} weeks. "
            f"{'Study timeline at serious risk.' if p90 > 26 else 'Manageable with intervention.'}"
        ),
        risk_level=risk_level
    )


def simulate_inactivity_impact(anomaly: Anomaly,
                                site_enrollment: pd.DataFrame
                                ) -> SimulationResult:
    """
    Simulate patient dropout risk from prolonged site inactivity.
    Projects: number of patients at risk of missing doses.
    """
    study_data = site_enrollment[site_enrollment["study_id"] == anomaly.study_id]
    total_enrolled = study_data["enrolled"].sum()

    # Simulate dropout probability with uncertainty
    base_dropout_rate = 0.05   # 5% baseline monthly dropout
    inactivity_multiplier = np.random.uniform(1.5, 3.5, N_SIMULATIONS)
    dropout_rates = base_dropout_rate * inactivity_multiplier

    patients_at_risk = np.round(total_enrolled * dropout_rates).astype(int)

    p10 = np.percentile(patients_at_risk, 10)
    p50 = np.percentile(patients_at_risk, 50)
    p90 = np.percentile(patients_at_risk, 90)

    risk_level = "CRITICAL" if p90 > total_enrolled * 0.2 else "HIGH"

    return SimulationResult(
        study_id=anomaly.study_id,
        anomaly_type=anomaly.anomaly_type,
        description=anomaly.description,
        p10=round(p10, 1),
        p50=round(p50, 1),
        p90=round(p90, 1),
        metric_label="Patients at risk of missing doses",
        interpretation=(
            f"Best case: {int(p10)} patients affected. "
            f"Most likely: {int(p50)} patients at risk. "
            f"Worst case: {int(p90)} patients may miss doses. "
            f"Immediate site contact required to prevent patient impact."
        ),
        risk_level=risk_level
    )


def simulate_inventory_impact(anomaly: Anomaly,
                               inventory: pd.DataFrame,
                               site_enrollment: pd.DataFrame
                               ) -> SimulationResult:
    """
    Simulate days until stockout given current inventory levels.
    Projects: days of supply remaining under different consumption scenarios.
    """
    study_inv      = inventory[inventory["study_id"] == anomaly.study_id]
    study_enroll   = site_enrollment[site_enrollment["study_id"] == anomaly.study_id]

    total_units    = study_inv["actual_units"].sum()
    total_patients = study_enroll["enrolled"].sum()

    # Daily consumption rate with uncertainty
    daily_consumption_per_patient = np.random.uniform(0.8, 1.2, N_SIMULATIONS)
    total_daily_consumption = total_patients * daily_consumption_per_patient

    days_of_supply = total_units / total_daily_consumption

    p10 = np.percentile(days_of_supply, 10)
    p50 = np.percentile(days_of_supply, 50)
    p90 = np.percentile(days_of_supply, 90)

    risk_level = "CRITICAL" if p10 < 14 else "HIGH" if p10 < 30 else "MEDIUM"

    return SimulationResult(
        study_id=anomaly.study_id,
        anomaly_type=anomaly.anomaly_type,
        description=anomaly.description,
        p10=round(p10, 1),
        p50=round(p50, 1),
        p90=round(p90, 1),
        metric_label="Days of drug supply remaining",
        interpretation=(
            f"Best case: {round(p90,1)} days of supply. "
            f"Most likely: {round(p50,1)} days. "
            f"Worst case: {round(p10,1)} days — "
            f"{'STOCKOUT IMMINENT. Emergency resupply required.' if p10 < 14 else 'Monitor closely and initiate resupply.'}"
        ),
        risk_level=risk_level
    )


def simulate_expiry_impact(anomaly: Anomaly,
                            inventory: pd.DataFrame
                            ) -> SimulationResult:
    """
    Simulate units at risk of expiry waste under different
    dispensing acceleration scenarios.
    Projects: units wasted if no action taken.
    """
    study_inv   = inventory[inventory["study_id"] == anomaly.study_id]
    total_units = study_inv["actual_units"].sum()

    # Simulate dispensing acceleration scenarios
    acceleration_factor = np.random.uniform(0.8, 2.0, N_SIMULATIONS)
    units_dispensed     = np.minimum(
        total_units,
        total_units * acceleration_factor * 0.4
    )
    units_wasted = np.maximum(0, total_units - units_dispensed)

    p10 = np.percentile(units_wasted, 10)
    p50 = np.percentile(units_wasted, 50)
    p90 = np.percentile(units_wasted, 90)

    risk_level = "HIGH" if p50 > total_units * 0.3 else "MEDIUM"

    return SimulationResult(
        study_id=anomaly.study_id,
        anomaly_type=anomaly.anomaly_type,
        description=anomaly.description,
        p10=round(p10, 1),
        p50=round(p50, 1),
        p90=round(p90, 1),
        metric_label="Units at risk of expiry waste",
        interpretation=(
            f"Best case: {int(p10)} units wasted with aggressive dispensing. "
            f"Most likely: {int(p50)} units wasted. "
            f"Worst case: {int(p90)} units wasted if no action taken. "
            f"Immediate redistribution recommended."
        ),
        risk_level=risk_level
    )


# ── MAIN SIMULATOR ─────────────────────────────────────────────

def run_simulations(anomalies:       List[Anomaly],
                    site_enrollment: pd.DataFrame,
                    inventory:       pd.DataFrame
                    ) -> List[SimulationResult]:
    """
    Run Monte Carlo simulations for each detected anomaly.
    Maps anomaly type to the appropriate simulation engine.
    """
    results = []

    for anomaly in anomalies:
        if anomaly.anomaly_type == "ENROLLMENT_DROP":
            result = simulate_enrollment_impact(anomaly, site_enrollment)
        elif anomaly.anomaly_type == "SITE_INACTIVITY":
            result = simulate_inactivity_impact(anomaly, site_enrollment)
        elif anomaly.anomaly_type == "INVENTORY_SHORTAGE":
            result = simulate_inventory_impact(anomaly, inventory, site_enrollment)
        elif anomaly.anomaly_type == "EXPIRY_RISK":
            result = simulate_expiry_impact(anomaly, inventory)
        else:
            continue

        results.append(result)

    return results


def print_simulation_report(results: List[SimulationResult]):
    """Print Monte Carlo simulation report."""
    print("\n" + "=" * 65)
    print("    CLINSIGHT - LAYER 3: MONTE CARLO IMPACT SIMULATION")
    print(f"    Based on {N_SIMULATIONS:,} simulated scenarios per anomaly")
    print("=" * 65)

    current_study = None
    for r in results:
        if r.study_id != current_study:
            current_study = r.study_id
            print(f"\n── {r.study_id} ──────────────────────────────")

        print(f"\n  [{r.risk_level}] {r.anomaly_type}")
        print(f"  {r.description[:80]}...")
        print(f"\n  Monte Carlo Results ({r.metric_label}):")
        print(f"  P10 (Best case)   : {r.p10}")
        print(f"  P50 (Most likely) : {r.p50}")
        print(f"  P90 (Worst case)  : {r.p90}")
        print(f"\n  Interpretation: {r.interpretation}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    from contract_validator import validate_all_studies
    from anomaly_detector import detect_all_anomalies

    study_config    = pd.read_csv("study_config.csv")
    site_profile    = pd.read_csv("site_profile.csv")
    site_enrollment = pd.read_csv("site_enrollment.csv")
    inventory       = pd.read_csv("inventory.csv")
    irt_dispensing  = pd.read_csv("irt_dispensing.csv")

    print("CLINSIGHT - Running Layer 3 Monte Carlo Simulation...\n")

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

    print(f"Total anomalies to simulate: {len(all_anomalies)}\n")

    # Layer 3
    simulation_results = run_simulations(
        all_anomalies, site_enrollment, inventory
    )

    print_simulation_report(simulation_results)