"""
CLINSIGHT - Data Generator
Generates synthetic clinical trial data mimicking real CTMS, IRT,
and supply chain data sources following CDISC standards.
Creates realistic scenarios including anomalies for testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# ── CONFIGURATION ──────────────────────────────────────────────
STUDIES = [
    {"study_id": "CLN-001-2025", "drug": "Compound-A", "target_enrollment": 120},
    {"study_id": "CLN-002-2025", "drug": "Compound-B", "target_enrollment": 80},
    {"study_id": "CLN-003-2025", "drug": "Compound-C", "target_enrollment": 60},
    {"study_id": "CLN-004-2025", "drug": "Compound-D", "target_enrollment": 100},
    {"study_id": "CLN-005-2025", "drug": "Compound-E", "target_enrollment": 40},
]

SITES = [
    {"site_id": "S001", "country": "USA",    "region": "Americas"},
    {"site_id": "S002", "country": "UK",     "region": "Europe"},
    {"site_id": "S003", "country": "Japan",  "region": "APAC"},
    {"site_id": "S004", "country": "India",  "region": "APAC"},
    {"site_id": "S005", "country": "Germany","region": "Europe"},
    {"site_id": "S006", "country": "China",  "region": "APAC"},
    {"site_id": "S007", "country": "Brazil", "region": "Americas"},
    {"site_id": "S008", "country": "France", "region": "Europe"},
]

DEPOTS = [
    {"depot_id": "D001", "location": "Hamburg",   "region": "Europe"},
    {"depot_id": "D002", "location": "Singapore", "region": "APAC"},
    {"depot_id": "D003", "location": "Chicago",   "region": "Americas"},
    {"depot_id": "D004", "location": "Mumbai",    "region": "APAC"},
]


def generate_study_config() -> pd.DataFrame:
    """Generate study configuration table - simulates study_config in Gold layer."""
    records = []
    start_date = datetime(2025, 1, 1)
    for s in STUDIES:
        records.append({
            "study_id":          s["study_id"],
            "drug":              s["drug"],
            "target_enrollment": s["target_enrollment"],
            "start_date":        start_date.strftime("%Y-%m-%d"),
            "planned_end_date":  (start_date + timedelta(days=730)).strftime("%Y-%m-%d"),
            "status":            "ACTIVE",
            "phase":             random.choice(["Phase II", "Phase III"]),
        })
        start_date += timedelta(days=90)
    return pd.DataFrame(records)


def generate_site_profile() -> pd.DataFrame:
    """Generate site profile table - simulates site_profile in Gold layer."""
    records = []
    for study in STUDIES:
        assigned_sites = random.sample(SITES, k=random.randint(4, 6))
        for site in assigned_sites:
            records.append({
                "study_id":        study["study_id"],
                "site_id":         site["site_id"],
                "country":         site["country"],
                "region":          site["region"],
                "activation_date": "2025-03-01",
                "status":          "ACTIVE",
                "latitude":        round(random.uniform(-60, 70), 4),
                "longitude":       round(random.uniform(-180, 180), 4),
            })
    return pd.DataFrame(records)


def generate_site_enrollment(site_profile: pd.DataFrame,
                              introduce_anomalies: bool = True) -> pd.DataFrame:
    """
    Generate site enrollment table - simulates CTMS enrollment data.
    Introduces realistic anomalies for testing Layer 2 detection.
    """
    records = []
    base_date = datetime.now() - timedelta(days=7)

    for _, row in site_profile.iterrows():
        study = next(s for s in STUDIES if s["study_id"] == row["study_id"])
        n_sites = site_profile[site_profile["study_id"] == row["study_id"]].shape[0]
        expected_per_site = study["target_enrollment"] // n_sites

        enrolled = random.randint(
            int(expected_per_site * 0.7),
            int(expected_per_site * 1.1)
        )
        last_activity = base_date + timedelta(days=random.randint(1, 30))

        if introduce_anomalies:
            # Anomaly 1 - Enrollment drop at APAC sites for CLN-002-2025
            if row["study_id"] == "CLN-002-2025" and row["region"] == "APAC":
                enrolled = random.randint(1, 5)

            # Anomaly 2 - Site inactivity at S003 for CLN-003-2025
            if row["study_id"] == "CLN-003-2025" and row["site_id"] == "S003":
                last_activity = base_date - timedelta(days=25)

        records.append({
            "study_id":       row["study_id"],
            "site_id":        row["site_id"],
            "country":        row["country"],
            "region":         row["region"],
            "enrolled":       enrolled,
            "target":         expected_per_site,
            "dropout_count":  random.randint(0, max(1, enrolled // 10)),
            "last_activity":  last_activity.strftime("%Y-%m-%d"),
            "enrollment_pct": round((enrolled / expected_per_site) * 100, 1),
        })

    return pd.DataFrame(records)


def generate_inventory(introduce_anomalies: bool = True) -> pd.DataFrame:
    """
    Generate inventory table - simulates IRT drug supply data.
    Introduces supply anomalies for testing.
    """
    records = []
    for study in STUDIES:
        for depot in DEPOTS:
            expected_units = random.randint(200, 500)
            actual_units = random.randint(
                int(expected_units * 0.85),
                int(expected_units * 1.15)
            )
            expiry_date = datetime(2026, 12, 31) + timedelta(days=random.randint(-180, 180))

            if introduce_anomalies:
                # Anomaly 3 - Critical inventory shortage at Singapore for CLN-004-2025
                if study["study_id"] == "CLN-004-2025" and depot["depot_id"] == "D002":
                    actual_units = random.randint(10, 30)

                # Anomaly 4 - Expiry risk at Hamburg for CLN-005-2025
                if study["study_id"] == "CLN-005-2025" and depot["depot_id"] == "D001":
                    expiry_date = datetime.now() + timedelta(days=random.randint(20, 45))

            records.append({
                "study_id":        study["study_id"],
                "depot_id":        depot["depot_id"],
                "depot_location":  depot["location"],
                "region":          depot["region"],
                "drug":            study["drug"],
                "expected_units":  expected_units,
                "actual_units":    actual_units,
                "expiry_date":     expiry_date.strftime("%Y-%m-%d"),
                "utilization_pct": round((actual_units / expected_units) * 100, 1),
            })

    return pd.DataFrame(records)


def generate_irt_dispensing(site_enrollment: pd.DataFrame,
                             introduce_anomalies: bool = True) -> pd.DataFrame:
    """
    Generate IRT drug dispensing records.
    Used in Layer 1 cross-system validation against CTMS enrollment.
    """
    records = []
    for _, row in site_enrollment.iterrows():
        dispensed = row["enrolled"]

        if introduce_anomalies:
            # Anomaly 5 - IRT dispensing mismatch at S007 for CLN-001-2025
            if row["study_id"] == "CLN-001-2025" and row["site_id"] == "S007":
                dispensed = max(0, row["enrolled"] - random.randint(5, 10))

        records.append({
            "study_id":        row["study_id"],
            "site_id":         row["site_id"],
            "enrolled_count":  row["enrolled"],
            "dispensed_count": dispensed,
            "match":           dispensed == row["enrolled"],
        })

    return pd.DataFrame(records)


def generate_all_data(save_csv: bool = True):
    """Generate all datasets and optionally save as CSV files."""
    print("Generating synthetic clinical trial data...\n")

    study_config    = generate_study_config()
    site_profile    = generate_site_profile()
    site_enrollment = generate_site_enrollment(site_profile)
    inventory       = generate_inventory()
    irt_dispensing  = generate_irt_dispensing(site_enrollment)

    if save_csv:
        study_config.to_csv("study_config.csv",      index=False)
        site_profile.to_csv("site_profile.csv",       index=False)
        site_enrollment.to_csv("site_enrollment.csv", index=False)
        inventory.to_csv("inventory.csv",             index=False)
        irt_dispensing.to_csv("irt_dispensing.csv",   index=False)
        print("Datasets saved successfully.\n")

    print(f"Studies generated:      {len(study_config)}")
    print(f"Site profiles:          {len(site_profile)}")
    print(f"Enrollment records:     {len(site_enrollment)}")
    print(f"Inventory records:      {len(inventory)}")
    print(f"IRT dispensing records: {len(irt_dispensing)}")
    print("\nAnomalies planted:")
    print("  - Enrollment drop at APAC sites for CLN-002-2025")
    print("  - Site inactivity at S003 for CLN-003-2025")
    print("  - Critical inventory shortage at Singapore for CLN-004-2025")
    print("  - Drug expiry risk at Hamburg for CLN-005-2025")
    print("  - IRT dispensing mismatch at S007 for CLN-001-2025")

    return {
        "study_config":    study_config,
        "site_profile":    site_profile,
        "site_enrollment": site_enrollment,
        "inventory":       inventory,
        "irt_dispensing":  irt_dispensing,
    }


if __name__ == "__main__":
    data = generate_all_data(save_csv=True)