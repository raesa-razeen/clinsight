"""
CLINSIGHT - Layer 1: Data Contract Validator
Validates cross-system relationships between CTMS, IRT, and supply data.
Each study is processed independently - one bad study never blocks others.
Statuses: CLEAR / WARNING / QUARANTINED
"""

import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List


# ── DATA STRUCTURES ────────────────────────────────────────────

@dataclass
class ValidationIssue:
    rule:        str
    severity:    str   # CRITICAL / HIGH / MEDIUM
    description: str
    affected:    str   # study_id + detail


@dataclass
class StudyValidationResult:
    study_id:  str
    status:    str   # CLEAR / WARNING / QUARANTINED
    issues:    List[ValidationIssue] = field(default_factory=list)
    passed:    int = 0
    failed:    int = 0

    def add_issue(self, issue: ValidationIssue):
        self.issues.append(issue)
        self.failed += 1
        if issue.severity == "CRITICAL":
            self.status = "QUARANTINED"
        elif issue.severity == "HIGH" and self.status == "CLEAR":
            self.status = "WARNING"

    def add_pass(self):
        self.passed += 1


# ── VALIDATION RULES ───────────────────────────────────────────

def validate_study_exists(study_id: str,
                           study_config: pd.DataFrame,
                           result: StudyValidationResult):
    """Rule 1 - Every study must exist in study configuration."""
    if study_id not in study_config["study_id"].values:
        result.add_issue(ValidationIssue(
            rule="STUDY_CONFIG_EXISTS",
            severity="CRITICAL",
            description=f"Study {study_id} not found in study configuration table.",
            affected=study_id
        ))
    else:
        result.add_pass()


def validate_site_profile_completeness(study_id: str,
                                        site_profile: pd.DataFrame,
                                        site_enrollment: pd.DataFrame,
                                        result: StudyValidationResult):
    """Rule 2 - Every site in enrollment must exist in site profile."""
    enrolled_sites = set(
        site_enrollment[site_enrollment["study_id"] == study_id]["site_id"]
    )
    profiled_sites = set(
        site_profile[site_profile["study_id"] == study_id]["site_id"]
    )
    missing = enrolled_sites - profiled_sites
    if missing:
        result.add_issue(ValidationIssue(
            rule="SITE_PROFILE_COMPLETENESS",
            severity="CRITICAL",
            description=f"{len(missing)} site(s) in enrollment have no site profile: {missing}",
            affected=f"{study_id} - sites {missing}"
        ))
    else:
        result.add_pass()


def validate_irt_ctms_match(study_id: str,
                             site_enrollment: pd.DataFrame,
                             irt_dispensing: pd.DataFrame,
                             result: StudyValidationResult):
    """Rule 3 - Every enrolled patient must have a dispensed kit in IRT."""
    enrollment = site_enrollment[site_enrollment["study_id"] == study_id]
    irt        = irt_dispensing[irt_dispensing["study_id"] == study_id]

    mismatches = []
    for _, row in enrollment.iterrows():
        irt_row = irt[irt["site_id"] == row["site_id"]]
        if irt_row.empty:
            mismatches.append(row["site_id"])
        elif not irt_row.iloc[0]["match"]:
            enrolled  = irt_row.iloc[0]["enrolled_count"]
            dispensed = irt_row.iloc[0]["dispensed_count"]
            mismatches.append(
                f"{row['site_id']} (enrolled={enrolled}, dispensed={dispensed})"
            )

    if mismatches:
        result.add_issue(ValidationIssue(
            rule="IRT_CTMS_MATCH",
            severity="HIGH",
            description=f"IRT dispensing mismatch at {len(mismatches)} site(s): {mismatches}",
            affected=f"{study_id}"
        ))
    else:
        result.add_pass()


def validate_inventory_exists(study_id: str,
                               inventory: pd.DataFrame,
                               result: StudyValidationResult):
    """Rule 4 - Every active study must have inventory records."""
    study_inventory = inventory[inventory["study_id"] == study_id]
    if study_inventory.empty:
        result.add_issue(ValidationIssue(
            rule="INVENTORY_EXISTS",
            severity="CRITICAL",
            description=f"No inventory records found for study {study_id}.",
            affected=study_id
        ))
    else:
        result.add_pass()


def validate_data_freshness(study_id: str,
                             site_enrollment: pd.DataFrame,
                             result: StudyValidationResult,
                             max_days_stale: int = 14):
    """Rule 5 - Enrollment data must have been updated within max_days_stale days."""
    study_data = site_enrollment[site_enrollment["study_id"] == study_id]
    if study_data.empty:
        result.add_pass()
        return

    latest = pd.to_datetime(study_data["last_activity"]).max()
    days_since = (datetime.now() - latest).days

    if days_since > max_days_stale:
        result.add_issue(ValidationIssue(
            rule="DATA_FRESHNESS",
            severity="HIGH",
            description=f"Enrollment data is {days_since} days old. "
                        f"Last activity: {latest.strftime('%Y-%m-%d')}. "
                        f"Threshold: {max_days_stale} days.",
            affected=study_id
        ))
    else:
        result.add_pass()


def validate_geocoding_completeness(study_id: str,
                                     site_profile: pd.DataFrame,
                                     result: StudyValidationResult):
    """Rule 6 - All sites must have valid geocoordinates."""
    study_sites = site_profile[site_profile["study_id"] == study_id]
    missing_geo = study_sites[
        study_sites["latitude"].isna() | study_sites["longitude"].isna()
    ]
    if not missing_geo.empty:
        result.add_issue(ValidationIssue(
            rule="GEOCODING_COMPLETENESS",
            severity="MEDIUM",
            description=f"{len(missing_geo)} site(s) missing geocoordinates.",
            affected=f"{study_id} - sites {list(missing_geo['site_id'])}"
        ))
    else:
        result.add_pass()

def validate_irt_site_id_match(study_id: str,
                                site_profile: pd.DataFrame,
                                irt_dispensing: pd.DataFrame,
                                result: StudyValidationResult):
    """
    Rule 7 - Site IDs in IRT must match Site IDs in Site Profile.
    Real world issue: IRT and CTMS sometimes use different site ID systems
    causing zero overlap and blank dashboards.
    """
    profile_sites = set(
        site_profile[site_profile["study_id"] == study_id]["site_id"]
    )
    irt_sites = set(
        irt_dispensing[irt_dispensing["study_id"] == study_id]["site_id"]
    )

    if not profile_sites or not irt_sites:
        result.add_pass()
        return

    matching = profile_sites.intersection(irt_sites)
    irt_only = irt_sites - profile_sites
    profile_only = profile_sites - irt_sites

    overlap_pct = (len(matching) / max(len(irt_sites), len(profile_sites))) * 100

    if overlap_pct == 0:
        result.add_issue(ValidationIssue(
            rule="IRT_SITE_ID_MISMATCH",
            severity="CRITICAL",
            description=(
                f"Zero overlap between IRT and Site Profile site IDs for {study_id}. "
                f"IRT sent {len(irt_sites)} site IDs: {irt_sites}. "
                f"Site Profile has {len(profile_sites)} site IDs: {profile_sites}. "
                f"Dashboard will show blank inventory for all sites."
            ),
            affected=f"{study_id} - all sites affected"
        ))
    elif overlap_pct < 50:
        result.add_issue(ValidationIssue(
            rule="IRT_SITE_ID_PARTIAL_MISMATCH",
            severity="HIGH",
            description=(
                f"Partial site ID mismatch for {study_id}. "
                f"Only {len(matching)} of {len(irt_sites)} IRT sites match Site Profile. "
                f"IRT only (unmatched): {irt_only}. "
                f"Profile only (no IRT data): {profile_only}."
            ),
            affected=f"{study_id} - {len(irt_only)} unmatched sites"
        ))
    else:
        result.add_pass()


def validate_depot_inventory_exists(study_id: str,
                                     inventory: pd.DataFrame,
                                     result: StudyValidationResult):
    """
    Rule 8 - Every active study must have depot inventory records.
    Real world issue: Source system sometimes fails to send depot
    inventory data resulting in blank dashboard displays.
    """
    study_inv = inventory[inventory["study_id"] == study_id]

    if study_inv.empty:
        result.add_issue(ValidationIssue(
            rule="DEPOT_INVENTORY_MISSING",
            severity="CRITICAL",
            description=(
                f"Zero depot inventory records found for {study_id}. "
                f"Depot Inventory table has no data for this study. "
                f"Dashboard will show blank depot inventory. "
                f"Source system may not have sent depot data."
            ),
            affected=f"{study_id} - all depots"
        ))
    else:
        result.add_pass()
# ── MAIN VALIDATOR ─────────────────────────────────────────────

def validate_all_studies(study_config:    pd.DataFrame,
                          site_profile:    pd.DataFrame,
                          site_enrollment: pd.DataFrame,
                          inventory:       pd.DataFrame,
                          irt_dispensing:  pd.DataFrame
                          ) -> List[StudyValidationResult]:
    """
    Run all validation rules for each study independently.
    One quarantined study never blocks others.
    """
    results = []

    for study_id in study_config["study_id"].unique():
        result = StudyValidationResult(study_id=study_id, status="CLEAR")

        # Run all rules independently
        validate_study_exists(study_id, study_config, result)
        validate_site_profile_completeness(study_id, site_profile, site_enrollment, result)
        validate_irt_ctms_match(study_id, site_enrollment, irt_dispensing, result)
        validate_inventory_exists(study_id, inventory, result)
        validate_data_freshness(study_id, site_enrollment, result)
        validate_geocoding_completeness(study_id, site_profile, result)
        validate_irt_site_id_match(study_id, site_profile, irt_dispensing, result)
        validate_depot_inventory_exists(study_id, inventory, result)

        results.append(result)

    return results


def print_validation_report(results: List[StudyValidationResult]):
    """Print a clean validation report to terminal."""
    print("\n" + "=" * 65)
    print("         CLINSIGHT - LAYER 1: DATA CONTRACT VALIDATION")
    print("=" * 65)

    clear       = [r for r in results if r.status == "CLEAR"]
    warnings    = [r for r in results if r.status == "WARNING"]
    quarantined = [r for r in results if r.status == "QUARANTINED"]

    print(f"\nTotal Studies Validated : {len(results)}")
    print(f"[CLEAR]       Proceeding : {len(clear)}")
    print(f"[WARNING]     Flagged    : {len(warnings)}")
    print(f"[QUARANTINED] Blocked    : {len(quarantined)}")

    print("\n" + "-" * 65)
    print("STUDY-LEVEL RESULTS:")
    print("-" * 65)

    for r in results:
        status_label = {
            "CLEAR":       "[CLEAR]      ",
            "WARNING":     "[WARNING]    ",
            "QUARANTINED": "[QUARANTINED]"
        }.get(r.status, r.status)

        print(f"\n{status_label} {r.study_id} "
              f"| Passed: {r.passed} | Failed: {r.failed}")

        for issue in r.issues:
            print(f"   [{issue.severity}] {issue.rule}")
            print(f"   {issue.description}")

    print("\n" + "=" * 65)

    if quarantined:
        print("QUARANTINED STUDIES - ACTION REQUIRED:")
        print("-" * 65)
        for r in quarantined:
            print(f"  - {r.study_id}: "
                  f"{r.issues[0].description if r.issues else 'Unknown issue'}")

    print("=" * 65)


if __name__ == "__main__":
    # Load synthetic datasets
    study_config    = pd.read_csv("study_config.csv")
    site_profile    = pd.read_csv("site_profile.csv")
    site_enrollment = pd.read_csv("site_enrollment.csv")
    inventory       = pd.read_csv("inventory.csv")
    irt_dispensing  = pd.read_csv("irt_dispensing.csv")

    print("CLINSIGHT - Running Layer 1 Data Contract Validation...\n")

    results = validate_all_studies(
        study_config, site_profile,
        site_enrollment, inventory, irt_dispensing
    )

    print_validation_report(results)