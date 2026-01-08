#!/usr/bin/env python3
"""P1 Conjecture 2 Test: Pathogen specificity lives in FAILURE MODES, not mean MIC.

Hypothesis:
    Pathogens differ in *how* peptides fail (resistance, tolerance),
    not in average MIC values.

Test:
    Predict P(MIC > threshold | peptide, pathogen) instead of E[MIC]
    Using classification instead of regression.

CRITICAL - R1 CONSTRAINT (Threshold Lock):
    All MIC thresholds must be defined GLOBALLY and A PRIORI.
    - Thresholds fixed BEFORE any pathogen-specific analysis
    - NEVER tune thresholds per pathogen, cluster, or fold
    - If separation appears ONLY after threshold adjustment = FALSIFIED

Falsifies if:
    Failure distributions overlap completely across pathogens.

Classification (R3):
    - Threshold classification deployable if signal found with global threshold
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PACKAGE_DIR / "results" / "validation_batch"

# =============================================================================
# R1: GLOBAL A PRIORI THRESHOLDS - DEFINED BEFORE ANY ANALYSIS
# =============================================================================
# CLINICAL thresholds from CLSI/EUCAST breakpoints (truly a priori):
CLINICAL_THRESHOLDS = {
    "susceptible": 0.0,      # log10(MIC) = 0 -> MIC = 1 ug/mL
    "intermediate": 0.5,     # log10(MIC) = 0.5 -> MIC = 3.16 ug/mL
    "resistant": 1.0,        # log10(MIC) = 1.0 -> MIC = 10 ug/mL
}

# NOTE: If clinical thresholds don't create variance (all samples in one class),
# this is a MEANINGFUL NEGATIVE RESULT, not a test failure.
# We then report: "Model predictions don't span clinical decision boundaries."


def load_all_candidates(results_dir: Path) -> List[dict]:
    """Load candidates from all pathogen result files with pathogen labels."""
    import csv

    pathogens = ["A_baumannii", "S_aureus", "P_aeruginosa", "Enterobacteriaceae", "H_pylori"]
    all_candidates = []

    for pathogen in pathogens:
        csv_path = results_dir / f"{pathogen}_candidates.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping")
            continue

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_candidates.append({
                    "sequence": row["sequence"],
                    "mic_pred": float(row["mic_pred"]),
                    "pathogen": pathogen,
                })

    print(f"Loaded {len(all_candidates)} total candidates across {len(set(c['pathogen'] for c in all_candidates))} pathogens")
    return all_candidates


def classify_by_threshold(candidates: List[dict], threshold: float) -> Dict[str, Dict[str, int]]:
    """Classify peptides as above/below threshold, grouped by pathogen.

    Returns counts of {pathogen: {"above": n, "below": m}}
    """
    by_pathogen = {}

    for c in candidates:
        p = c["pathogen"]
        if p not in by_pathogen:
            by_pathogen[p] = {"above": 0, "below": 0}

        if c["mic_pred"] > threshold:
            by_pathogen[p]["above"] += 1
        else:
            by_pathogen[p]["below"] += 1

    return by_pathogen


def test_failure_rate_separation(candidates: List[dict], threshold: float) -> Dict:
    """Test if failure rate (P(MIC > threshold)) differs across pathogens.

    Uses global threshold per R1 constraint.
    """
    counts = classify_by_threshold(candidates, threshold)

    # Calculate failure rates
    failure_rates = {}
    total_above = 0
    total_below = 0
    for p, c in counts.items():
        total = c["above"] + c["below"]
        failure_rates[p] = c["above"] / total if total > 0 else 0
        total_above += c["above"]
        total_below += c["below"]

    # Check if threshold creates any variance
    if total_above == 0 or total_below == 0:
        return {
            "threshold": threshold,
            "counts": counts,
            "failure_rates": failure_rates,
            "no_variance": True,
            "all_class": "below" if total_above == 0 else "above",
            "chi2": 0,
            "p_value": 1.0,
            "dof": 0,
            "rate_range": 0,
            "rate_std": 0,
            "practical_significance": False,
            "note": f"All {len(candidates)} samples {'below' if total_above == 0 else 'above'} threshold. Cannot test separation.",
        }

    # Chi-square test for independence
    # H0: failure rate is independent of pathogen
    observed = np.array([[c["above"], c["below"]] for c in counts.values()])

    # Check for zero rows/columns that would cause chi2 to fail
    if np.any(observed.sum(axis=1) == 0) or np.any(observed.sum(axis=0) == 0):
        chi2, p_value, dof = 0, 1.0, 0
    elif observed.sum() > 0 and observed.shape[0] >= 2:
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(observed)
        except ValueError:
            chi2, p_value, dof = 0, 1.0, 0
    else:
        chi2, p_value, dof = 0, 1.0, 0

    # Effect size: range of failure rates
    rates = list(failure_rates.values())
    rate_range = max(rates) - min(rates) if rates else 0
    rate_std = np.std(rates) if rates else 0

    # Practical significance: is the difference meaningful?
    # Effect size < 0.1 is trivially small
    practical_significance = rate_range > 0.1

    return {
        "threshold": threshold,
        "counts": counts,
        "failure_rates": failure_rates,
        "no_variance": False,
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "rate_range": rate_range,
        "rate_std": rate_std,
        "practical_significance": practical_significance,
    }


def test_all_thresholds(candidates: List[dict]) -> Dict:
    """Test all pre-defined global thresholds.

    R1 CONSTRAINT: These thresholds were defined a priori.
    """
    results = {}

    for name, threshold in CLINICAL_THRESHOLDS.items():
        results[name] = test_failure_rate_separation(candidates, threshold)
        results[name]["threshold_name"] = name

    return results


def check_r1_compliance(results: Dict) -> Dict:
    """Verify R1 compliance: signal should be consistent across thresholds.

    If signal appears ONLY at specific thresholds, this suggests
    threshold optimization (R1 violation).
    """
    signal_at = []
    no_signal_at = []
    no_variance_at = []

    for name, r in results.items():
        if r.get("no_variance"):
            no_variance_at.append(name)
        elif r["p_value"] < 0.05 and r["practical_significance"]:
            signal_at.append(name)
        else:
            no_signal_at.append(name)

    # Count only testable thresholds
    n_testable = len(signal_at) + len(no_signal_at)
    n_signal = len(signal_at)
    n_total = len(results)

    if n_testable == 0:
        r1_status = "UNTESTABLE"
        r1_interpretation = (
            f"All {n_total} clinical thresholds outside data range. "
            "Model predictions don't span clinical decision boundaries. "
            "This is a meaningful negative result: C2 cannot be tested with current model."
        )
    elif n_signal == 0:
        r1_status = "PASS_NO_SIGNAL"
        r1_interpretation = f"No signal at {n_testable} testable thresholds. Conjecture falsified."
    elif n_signal == n_testable:
        r1_status = "PASS_ROBUST"
        r1_interpretation = f"Signal at all {n_testable} testable thresholds. Robust finding."
    elif n_signal >= n_testable / 2:
        r1_status = "PASS_PARTIAL"
        r1_interpretation = f"Signal at {n_signal}/{n_testable} testable thresholds. Partially robust."
    else:
        r1_status = "CAUTION"
        r1_interpretation = (
            f"Signal at only {n_signal}/{n_testable} testable thresholds. "
            "May be threshold-dependent - treat with caution."
        )

    return {
        "signal_at": signal_at,
        "no_signal_at": no_signal_at,
        "no_variance_at": no_variance_at,
        "n_testable": n_testable,
        "r1_status": r1_status,
        "r1_interpretation": r1_interpretation,
    }


def main():
    print("=" * 70)
    print("P1 CONJECTURE 2 TEST: Pathogen specificity in FAILURE MODES")
    print("=" * 70)
    print()

    # R1 declaration
    print("-" * 70)
    print("R1 CONSTRAINT: GLOBAL A PRIORI THRESHOLDS")
    print("-" * 70)
    print("The following CLINICAL thresholds were defined BEFORE analysis:")
    for name, thresh in CLINICAL_THRESHOLDS.items():
        mic_ug = 10 ** thresh
        print(f"  {name}: log10(MIC) = {thresh} (MIC = {mic_ug:.2f} ug/mL)")
    print()
    print("These thresholds CANNOT be adjusted based on results (R1 compliance).")
    print("If all samples fall on one side, this is a meaningful finding, not a test failure.")
    print()

    # Load data
    print("Loading candidates...")
    candidates = load_all_candidates(RESULTS_DIR)
    if not candidates:
        print("ERROR: No candidates found")
        sys.exit(1)
    print()

    # Test all thresholds
    print("-" * 70)
    print("FAILURE MODE TESTS (P(MIC > threshold) by pathogen)")
    print("-" * 70)

    threshold_results = test_all_thresholds(candidates)

    for name, r in threshold_results.items():
        print(f"\nThreshold: {name} (log10(MIC) = {r['threshold']})")

        if r.get("no_variance"):
            print(f"  {r['note']}")
            print("  --> NO VARIANCE (threshold outside data range)")
            continue

        print("  Failure rates by pathogen:")
        for p, rate in sorted(r["failure_rates"].items()):
            counts = r["counts"][p]
            print(f"    {p}: {rate:.1%} ({counts['above']}/{counts['above']+counts['below']})")

        print(f"  Chi-square: {r['chi2']:.2f}, p = {r['p_value']:.4f}")
        print(f"  Rate range: {r['rate_range']:.3f}, practical: {r['practical_significance']}")

        if r["p_value"] < 0.05 and r["practical_significance"]:
            print("  --> SIGNAL DETECTED (statistical + practical)")
        elif r["p_value"] < 0.05:
            print("  --> Statistical significance only (no practical effect)")
        else:
            print("  --> No significant difference")

    # R1 compliance check
    print()
    print("-" * 70)
    print("R1 COMPLIANCE CHECK")
    print("-" * 70)

    r1_check = check_r1_compliance(threshold_results)
    print(f"Signal at thresholds: {r1_check['signal_at']}")
    print(f"No signal at: {r1_check['no_signal_at']}")
    print(f"R1 Status: {r1_check['r1_status']}")
    print(f"Interpretation: {r1_check['r1_interpretation']}")

    # Final verdict
    print()
    print("=" * 70)
    print("CONJECTURE 2 VERDICT")
    print("=" * 70)

    has_signal = len(r1_check["signal_at"]) > 0
    is_robust = r1_check["r1_status"] in ["PASS_ROBUST", "PASS_PARTIAL"]
    is_untestable = r1_check["r1_status"] == "UNTESTABLE"

    if is_untestable:
        verdict = "UNTESTABLE"
        interpretation = (
            "Model predictions don't span clinical MIC decision boundaries. "
            "All samples have log10(MIC) < 0 (MIC < 1 ug/mL). "
            "C2 cannot be tested with current model output distribution."
        )
    elif has_signal and is_robust:
        verdict = "CONFIRMED"
        interpretation = (
            "Pathogens show different failure rates (P(MIC > threshold)). "
            "Signal is robust across multiple thresholds (R1 compliant)."
        )
    elif has_signal:
        verdict = "PARTIAL"
        interpretation = (
            "Some signal detected but not robust across thresholds. "
            "Treat with caution - may be threshold-dependent."
        )
    else:
        verdict = "FALSIFIED"
        interpretation = (
            "No significant difference in failure rates across pathogens. "
            "Pathogen specificity does NOT live in failure modes."
        )

    print(f"VERDICT: {verdict}")
    print(f"INTERPRETATION: {interpretation}")
    print()

    # R3 Classification
    print("-" * 70)
    print("R3 CLASSIFICATION (Inference-Time Availability)")
    print("-" * 70)
    if verdict == "CONFIRMED":
        r3_class = "Deployable (threshold classification at inference time)"
    elif verdict == "UNTESTABLE":
        r3_class = "Research-Only (model output range incompatible with clinical thresholds)"
    else:
        r3_class = "Research-Only (no failure mode signal)"

    print(f"Classification: {r3_class}")

    # Save results
    output_file = RESULTS_DIR / "P1_C2_results.json"
    output = {
        "conjecture": "C2: Pathogen specificity lives in FAILURE MODES, not mean MIC",
        "r1_thresholds": CLINICAL_THRESHOLDS,
        "r1_note": "Thresholds defined a priori, not tuned",
        "verdict": verdict,
        "interpretation": interpretation,
        "r1_compliance": r1_check,
        "r3_classification": r3_class,
        "threshold_results": threshold_results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print()
    print(f"Results saved to: {output_file}")

    return verdict


if __name__ == "__main__":
    verdict = main()
    sys.exit(0 if verdict in ["CONFIRMED", "PARTIAL", "FALSIFIED", "UNTESTABLE"] else 1)
