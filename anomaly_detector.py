"""
CLINSIGHT - Layer 2: Business Anomaly Detector
Detects unexpected patterns in clean validated clinical supply data.
Only runs on studies that passed Layer 1 validation.
Detects: enrollment drops, site inactivity, inventory risk, expiry risk.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List


# ── DATA STRUCTURES ────────────────────────────────────────────

@dataclass
class Anomaly:
    study_id:    str
    anomaly_type: str
    severity:    str    # CRITICAL / HIGH / MEDIUM
    description: str
    metric:      str    # the specific number that triggered this
    recommended_action: str


@dataclass
class StudyAnomalyResult:
    study_id:  str
    anomalies: List[Anomaly] = field(default_factory=list)

    @property
    def has_anomalies(self):
        return len(self.anomalies) > 0

    @property
    def highest_severity(self):
        if any(a.severity == "CRITICAL" for a in self.anomalies):
            return "CRITICAL"
        elif any(a.severity == "HIGH" for a in self.anomalies):
            return "HIGH"
        elif any(a.severity == "MEDIUM" for a in self.anomalies):
            return "MEDIUM"
        return "NONE"


# ── ANOMALY DETECTION RULES ────────────────────────────────────

def detect_enrollment_drop(study_id: str,
                            site_enrollment: pd.DataFrame,
                            result: StudyAnomalyResult,
                            threshold_pct: float = 50.0):
    """
    Detect sites with enrollment significantly below target.
    Threshold: enrollment_pct below threshold triggers anomaly.
    """
    study_data = site_enrollment[site_enrollment["study_id"] == study_id]
    low_sites  = study_data[study_data["enrollment_pct"] < threshold_pct]

    if not low_sites.empty:
        for _, row in low_sites.iterrows():
            severity = "CRITICAL" if row["enrollment_pct"] < 20 else "HIGH"
            result.anomalies.append(Anomaly(
                study_id=study_id,
                anomaly_type="ENROLLMENT_DROP",
                severity=severity,
                description=(
                    f"Site {row['site_id']} ({row['country']}) is at "
                    f"{row['enrollment_pct']}% enrollment — "
                    f"{row['enrolled']} of {row['target']} patients enrolled."
                ),
                metric=f"{row['enrollment_pct']}% enrollment rate",
                recommended_action=(
                    f"Escalate to regional manager for {row['country']}. "
                    f"Review site activation barriers and consider "
                    f"activating a backup site in the {row['region']} region."
                )
            ))


def detect_site_inactivity(study_id: str,
                            site_enrollment: pd.DataFrame,
                            result: StudyAnomalyResult,
                            max_inactive_days: int = 14):
    """
    Detect sites with no enrollment activity for more than max_inactive_days.
    """
    study_data = site_enrollment[site_enrollment["study_id"] == study_id].copy()
    study_data["last_activity"] = pd.to_datetime(study_data["last_activity"])
    study_data["days_inactive"]  = (
        datetime.now() - study_data["last_activity"]
    ).dt.days

    inactive = study_data[study_data["days_inactive"] > max_inactive_days]

    if not inactive.empty:
        for _, row in inactive.iterrows():
            severity = "CRITICAL" if row["days_inactive"] > 30 else "HIGH"
            result.anomalies.append(Anomaly(
                study_id=study_id,
                anomaly_type="SITE_INACTIVITY",
                severity=severity,
                description=(
                    f"Site {row['site_id']} ({row['country']}) has had "
                    f"no enrollment activity for {row['days_inactive']} days. "
                    f"Last activity: {row['last_activity'].strftime('%Y-%m-%d')}."
                ),
                metric=f"{row['days_inactive']} days inactive",
                recommended_action=(
                    f"Contact site coordinator at {row['site_id']} immediately. "
                    f"Verify site is still operational and investigate "
                    f"cause of inactivity. Escalate if no response within 48 hours."
                )
            ))


def detect_inventory_shortage(study_id: str,
                               inventory: pd.DataFrame,
                               result: StudyAnomalyResult,
                               shortage_threshold: float = 30.0):
    """
    Detect depots with critically low inventory levels.
    Threshold: utilization_pct below shortage_threshold triggers anomaly.
    """
    study_inv = inventory[inventory["study_id"] == study_id]
    low_stock = study_inv[study_inv["utilization_pct"] < shortage_threshold]

    if not low_stock.empty:
        for _, row in low_stock.iterrows():
            severity = "CRITICAL" if row["utilization_pct"] < 15 else "HIGH"
            result.anomalies.append(Anomaly(
                study_id=study_id,
                anomaly_type="INVENTORY_SHORTAGE",
                severity=severity,
                description=(
                    f"Depot {row['depot_id']} ({row['depot_location']}) has "
                    f"critically low inventory: {row['actual_units']} units "
                    f"vs {row['expected_units']} expected "
                    f"({row['utilization_pct']}% utilization)."
                ),
                metric=f"{row['actual_units']} units remaining "
                       f"({row['utilization_pct']}% of expected)",
                recommended_action=(
                    f"Initiate emergency resupply to {row['depot_location']} depot. "
                    f"Contact CMO for expedited shipment. "
                    f"Review allocation from nearest depot with surplus stock."
                )
            ))


def detect_expiry_risk(study_id: str,
                        inventory: pd.DataFrame,
                        result: StudyAnomalyResult,
                        expiry_warning_days: int = 60):
    """
    Detect drug supply approaching expiry date within warning window.
    """
    study_inv = inventory[inventory["study_id"] == study_id].copy()
    study_inv["expiry_date"] = pd.to_datetime(study_inv["expiry_date"])
    study_inv["days_to_expiry"] = (
        study_inv["expiry_date"] - datetime.now()
    ).dt.days

    at_risk = study_inv[study_inv["days_to_expiry"] < expiry_warning_days]

    if not at_risk.empty:
        for _, row in at_risk.iterrows():
            severity = "CRITICAL" if row["days_to_expiry"] < 30 else "HIGH"
            result.anomalies.append(Anomaly(
                study_id=study_id,
                anomaly_type="EXPIRY_RISK",
                severity=severity,
                description=(
                    f"Drug {row['drug']} at {row['depot_location']} depot "
                    f"expires in {row['days_to_expiry']} days "
                    f"({row['expiry_date'].strftime('%Y-%m-%d')}). "
                    f"{row['actual_units']} units at risk."
                ),
                metric=f"{row['days_to_expiry']} days to expiry",
                recommended_action=(
                    f"Accelerate dispensing at sites served by "
                    f"{row['depot_location']} depot. "
                    f"Consider redistributing stock to higher-enrollment sites. "
                    f"Notify CMO to prepare replacement batch."
                )
            ))


# ── MAIN DETECTOR ──────────────────────────────────────────────

def detect_all_anomalies(study_config:    pd.DataFrame,
                          site_enrollment: pd.DataFrame,
                          inventory:       pd.DataFrame,
                          cleared_studies: List[str]
                          ) -> List[StudyAnomalyResult]:
    """
    Run all anomaly detection rules for each cleared study.
    Only processes studies that passed Layer 1 validation.
    """
    results = []

    for study_id in cleared_studies:
        result = StudyAnomalyResult(study_id=study_id)

        detect_enrollment_drop(study_id, site_enrollment, result)
        detect_site_inactivity(study_id, site_enrollment, result)
        detect_inventory_shortage(study_id, inventory, result)
        detect_expiry_risk(study_id, inventory, result)

        results.append(result)

    return results


def print_anomaly_report(results: List[StudyAnomalyResult]):
    """Print Layer 2 anomaly detection report."""
    print("\n" + "=" * 65)
    print("      CLINSIGHT - LAYER 2: BUSINESS ANOMALY DETECTION")
    print("=" * 65)

    clean   = [r for r in results if not r.has_anomalies]
    flagged = [r for r in results if r.has_anomalies]

    print(f"\nStudies Analysed  : {len(results)}")
    print(f"Clean             : {len(clean)}")
    print(f"Anomalies Found   : {len(flagged)}")
    print(f"Total Anomalies   : {sum(len(r.anomalies) for r in results)}")

    print("\n" + "-" * 65)
    print("ANOMALY BREAKDOWN:")
    print("-" * 65)

    for r in results:
        if r.has_anomalies:
            print(f"\n[{r.highest_severity}] {r.study_id} "
                  f"— {len(r.anomalies)} anomaly(s) detected")
            for a in r.anomalies:
                print(f"\n   Type     : {a.anomaly_type}")
                print(f"   Severity : {a.severity}")
                print(f"   Detail   : {a.description}")
                print(f"   Metric   : {a.metric}")
                print(f"   Action   : {a.recommended_action}")
        else:
            print(f"\n[CLEAN] {r.study_id} — No anomalies detected")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    from contract_validator import validate_all_studies

    study_config    = pd.read_csv("study_config.csv")
    site_profile    = pd.read_csv("site_profile.csv")
    site_enrollment = pd.read_csv("site_enrollment.csv")
    inventory       = pd.read_csv("inventory.csv")
    irt_dispensing  = pd.read_csv("irt_dispensing.csv")

    print("CLINSIGHT - Running Layer 2 Business Anomaly Detection...\n")

    # Only analyse studies that passed Layer 1
    layer1_results  = validate_all_studies(
        study_config, site_profile,
        site_enrollment, inventory, irt_dispensing
    )
    cleared_studies = [
        r.study_id for r in layer1_results
        if r.status in ["CLEAR", "WARNING"]
    ]

    print(f"Studies passed from Layer 1: {len(cleared_studies)}")
    print(f"Studies: {cleared_studies}\n")

    results = detect_all_anomalies(
        study_config, site_enrollment,
        inventory, cleared_studies
    )

    print_anomaly_report(results)