"""
CLINSIGHT - Data Generator v2.0
Generates synthetic clinical trial data mimicking real CTMS, IRT,
and supply chain data sources following CDISC standards.
Includes real-world anomaly types based on actual clinical data issues.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

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
    {"site_id": "S001", "site_name": "Metro Research Center",     "country": "USA",     "region": "Americas"},
    {"site_id": "S002", "site_name": "Royal Clinical Institute",  "country": "UK",      "region": "Europe"},
    {"site_id": "S003", "site_name": "Osaka Medical Research",    "country": "Japan",   "region": "APAC"},
    {"site_id": "S004", "site_name": "Apollo Clinical Center",    "country": "India",   "region": "APAC"},
    {"site_id": "S005", "site_name": "Berlin Research Group",     "country": "Germany", "region": "Europe"},
    {"site_id": "S006", "site_name": "Shanghai Clinical Hub",     "country": "China",   "region": "APAC"},
    {"site_id": "S007", "site_name": "Sao Paulo Medical Center",  "country": "Brazil",  "region": "Americas"},
    {"site_id": "S008", "site_name": "Paris Research Institute",  "country": "France",  "region": "Europe"},
]

# IRT uses different site ID format for some studies - this causes mismatch anomaly
IRT_SITES_MISMATCH = [
    {"site_id": "IRT-001", "site_name": "Metro Research Center",    "country": "USA",   "region": "Americas"},
    {"site_id": "IRT-002", "site_name": "Royal Clinical Institute", "country": "UK",    "region": "Europe"},
    {"site_id": "IRT-003", "site_name": "Osaka Medical Research",   "country": "Japan", "region": "APAC"},
    {"site_id": "IRT-004", "site_name": "Apollo Clinical Center",   "country": "India", "region": "APAC"},
]

DEPOTS = [
    {"depot_id": "D001", "location": "Hamburg",   "region": "Europe"},
    {"depot_id": "D002", "location": "Singapore", "region": "APAC"},
    {"depot_id": "D003", "location": "Chicago",   "region": "Americas"},
    {"depot_id": "D004", "location": "Mumbai",    "region": "APAC"},
]


def generate_study_config() -> pd.DataFrame:
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
    records = []
    for study in STUDIES:
        assigned_sites = random.sample(SITES, k=random.randint(4, 6))
        for site in assigned_sites:
            records.append({
                "study_id":        study["study_id"],
                "site_id":         site["site_id"],
                "site_name":       site["site_name"],
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
        last_activity = base_date + timedelta(days=random.randint(1, 5))

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
            "site_name":      row["site_name"],
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
    records = []
    for study in STUDIES:
        # Anomaly - CLN-004-2025 has NO depot inventory at all (missing data)
        if introduce_anomalies and study["study_id"] == "CLN-004-2025":
            continue  # Skip entirely - simulates missing depot inventory

        for depot in DEPOTS:
            expected_units = random.randint(200, 500)
            actual_units = random.randint(
                int(expected_units * 0.85),
                int(expected_units * 1.15)
            )
            expiry_date = datetime(2026, 12, 31) + timedelta(days=random.randint(-180, 180))

            if introduce_anomalies:
                # Anomaly - Critical inventory shortage Singapore for CLN-003-2025
                if study["study_id"] == "CLN-003-2025" and depot["depot_id"] == "D002":
                    actual_units = random.randint(10, 30)

                # Anomaly - Expiry risk Hamburg for CLN-005-2025
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


def generate_irt_dispensing(site_profile: pd.DataFrame,
                             site_enrollment: pd.DataFrame,
                             introduce_anomalies: bool = True) -> pd.DataFrame:
    """
    Generate IRT dispensing records.
    For CLN-001-2025 - IRT uses completely different site ID format
    causing zero overlap with Site Profile - real world anomaly type.
    """
    records = []

    for study in STUDIES:
        study_profile = site_profile[site_profile["study_id"] == study["study_id"]]
        study_enrollment = site_enrollment[site_enrollment["study_id"] == study["study_id"]]

        # Anomaly - CLN-001-2025 IRT uses different site ID system
        if introduce_anomalies and study["study_id"] == "CLN-001-2025":
            # IRT sends completely different site IDs - zero overlap with Site Profile
            for irt_site in IRT_SITES_MISMATCH:
                enrolled = random.randint(15, 35)
                records.append({
                    "study_id":        study["study_id"],
                    "site_id":         irt_site["site_id"],  # IRT-001, IRT-002 etc
                    "site_name":       irt_site["site_name"],
                    "enrolled_count":  enrolled,
                    "dispensed_count": enrolled,
                    "match":           False,  # Will never match Site Profile
                    "source":          "IRT"
                })
            continue

        # Normal studies - IRT matches Site Profile
        for _, row in study_enrollment.iterrows():
            dispensed = row["enrolled"]

            # Small mismatch at S007 for CLN-002-2025
            if introduce_anomalies:
                if study["study_id"] == "CLN-002-2025" and row["site_id"] == "S007":
                    dispensed = max(0, row["enrolled"] - random.randint(3, 8))

            records.append({
                "study_id":        study["study_id"],
                "site_id":         row["site_id"],
                "site_name":       row["site_name"],
                "enrolled_count":  row["enrolled"],
                "dispensed_count": dispensed,
                "match":           dispensed == row["enrolled"],
                "source":          "IRT"
            })

    return pd.DataFrame(records)


def generate_all_data(save_csv: bool = True):
    print("Generating synthetic clinical trial data...\n")

    study_config    = generate_study_config()
    site_profile    = generate_site_profile()
    site_enrollment = generate_site_enrollment(site_profile)
    inventory       = generate_inventory()
    irt_dispensing  = generate_irt_dispensing(site_profile, site_enrollment)

    if save_csv:
        study_config.to_csv("study_config.csv",      index=False)
        site_profile.to_csv("site_profile.csv",       index=False)
        site_enrollment.to_csv("site_enrollment.csv", index=False)
        inventory.to_csv("inventory.csv",             index=False)
        irt_dispensing.to_csv("irt_dispensing.csv",   index=False)
        print("Datasets saved.\n")

    print(f"Studies:            {len(study_config)}")
    print(f"Site profiles:      {len(site_profile)}")
    print(f"Enrollment records: {len(site_enrollment)}")
    print(f"Inventory records:  {len(inventory)}")
    print(f"IRT records:        {len(irt_dispensing)}")
    print("\nAnomalies planted:")
    print("  - CLN-001-2025: IRT site IDs completely different from Site Profile (zero overlap)")
    print("  - CLN-002-2025: APAC enrollment drop + IRT dispensing mismatch at S007")
    print("  - CLN-003-2025: Site inactivity 25 days + critical inventory shortage Singapore")
    print("  - CLN-004-2025: Zero depot inventory records (missing data)")
    print("  - CLN-005-2025: Drug expiry risk at Hamburg depot")

    return {
        "study_config":    study_config,
        "site_profile":    site_profile,
        "site_enrollment": site_enrollment,
        "inventory":       inventory,
        "irt_dispensing":  irt_dispensing,
    }


if __name__ == "__main__":
    data = generate_all_data(save_csv=True)