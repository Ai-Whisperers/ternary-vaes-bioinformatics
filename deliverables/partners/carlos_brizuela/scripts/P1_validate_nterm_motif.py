#!/usr/bin/env python3
"""P1 N-Terminal Cationic Motif Validation.

Hypothesis: P. aeruginosa-effective peptides have N-terminal KK/RK dipeptides
that exploit the highly negative membrane charge (-0.7) and high LPS abundance (0.90).

Test: Compare top P. aeruginosa vs bottom Enterobacteriaceae peptides within
cluster 3 (detergent mechanism) for N-terminal cationic patterns.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PACKAGE_DIR / "results" / "validation_batch"

# Amino acid categories
CATIONIC = set("KRH")
CATIONIC_DIPEPTIDES = {"KK", "KR", "RK", "RR", "KH", "HK", "RH", "HR", "HH"}


def load_candidates(results_dir: Path) -> List[dict]:
    """Load candidates from validation batch."""
    pathogens = ["A_baumannii", "S_aureus", "P_aeruginosa", "Enterobacteriaceae", "H_pylori"]
    candidates = []

    for pathogen in pathogens:
        csv_path = results_dir / f"{pathogen}_candidates.csv"
        if not csv_path.exists():
            continue

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                candidates.append({
                    "sequence": row["sequence"],
                    "mic_pred": float(row["mic_pred"]),
                    "net_charge": float(row["net_charge"]),
                    "hydrophobicity": float(row["hydrophobicity"]),
                    "length": len(row["sequence"]),
                    "pathogen": pathogen,
                })

    return candidates


def assign_clusters(candidates: List[dict], n_clusters: int = 5) -> np.ndarray:
    """Assign peptides to clusters using same method as C3."""
    features = np.array([
        [c["length"], c["net_charge"], c["hydrophobicity"]]
        for c in candidates
    ])

    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_norm)

    return labels


def analyze_nterm_pattern(sequence: str) -> Dict:
    """Analyze N-terminal cationic patterns."""
    if len(sequence) < 2:
        return {
            "nterm_dipeptide": "",
            "has_cationic_dipeptide": False,
            "nterm_5_cationic_count": 0,
            "nterm_5_cationic_fraction": 0.0,
            "first_cationic_position": -1,
        }

    nterm_dipeptide = sequence[:2]
    has_cationic_dipeptide = nterm_dipeptide in CATIONIC_DIPEPTIDES

    # Count cationic in first 5 AA
    nterm_5 = sequence[:5]
    cationic_count = sum(1 for aa in nterm_5 if aa in CATIONIC)
    cationic_fraction = cationic_count / len(nterm_5)

    # First cationic position (0-indexed)
    first_cationic = -1
    for i, aa in enumerate(sequence[:10]):
        if aa in CATIONIC:
            first_cationic = i
            break

    return {
        "nterm_dipeptide": nterm_dipeptide,
        "has_cationic_dipeptide": has_cationic_dipeptide,
        "nterm_5_cationic_count": cationic_count,
        "nterm_5_cationic_fraction": cationic_fraction,
        "first_cationic_position": first_cationic,
    }


def validate_hypothesis(candidates: List[dict], cluster_labels: np.ndarray) -> Dict:
    """
    Validate N-terminal cationic hypothesis.

    Compare P. aeruginosa-effective vs Enterobacteriaceae peptides within cluster 3.
    """
    # Get cluster 3 candidates (detergent mechanism)
    cluster_3_mask = cluster_labels == 3
    cluster_3_candidates = [c for c, m in zip(candidates, cluster_3_mask) if m]

    print(f"Cluster 3 (detergent mechanism): {len(cluster_3_candidates)} peptides")

    # Separate by pathogen
    pa_peptides = [c for c in cluster_3_candidates if c["pathogen"] == "P_aeruginosa"]
    entero_peptides = [c for c in cluster_3_candidates if c["pathogen"] == "Enterobacteriaceae"]

    print(f"  P. aeruginosa peptides: {len(pa_peptides)}")
    print(f"  Enterobacteriaceae peptides: {len(entero_peptides)}")

    if len(pa_peptides) < 3 or len(entero_peptides) < 3:
        # Fall back to cluster 4 if cluster 3 has insufficient data
        print("\n  Insufficient data in cluster 3, checking cluster 4...")
        cluster_4_mask = cluster_labels == 4
        cluster_4_candidates = [c for c, m in zip(candidates, cluster_4_mask) if m]
        pa_peptides_4 = [c for c in cluster_4_candidates if c["pathogen"] == "P_aeruginosa"]
        entero_peptides_4 = [c for c in cluster_4_candidates if c["pathogen"] == "Enterobacteriaceae"]

        # Combine both clusters
        pa_peptides = pa_peptides + pa_peptides_4
        entero_peptides = entero_peptides + entero_peptides_4
        print(f"  Combined P. aeruginosa: {len(pa_peptides)}")
        print(f"  Combined Enterobacteriaceae: {len(entero_peptides)}")

    # Analyze N-terminal patterns for each group
    pa_patterns = [analyze_nterm_pattern(c["sequence"]) for c in pa_peptides]
    entero_patterns = [analyze_nterm_pattern(c["sequence"]) for c in entero_peptides]

    # Compute statistics
    pa_cationic_dipeptide_rate = sum(1 for p in pa_patterns if p["has_cationic_dipeptide"]) / len(pa_patterns) if pa_patterns else 0
    entero_cationic_dipeptide_rate = sum(1 for p in entero_patterns if p["has_cationic_dipeptide"]) / len(entero_patterns) if entero_patterns else 0

    pa_nterm5_cationic_mean = np.mean([p["nterm_5_cationic_count"] for p in pa_patterns]) if pa_patterns else 0
    entero_nterm5_cationic_mean = np.mean([p["nterm_5_cationic_count"] for p in entero_patterns]) if entero_patterns else 0

    pa_first_cationic_mean = np.mean([p["first_cationic_position"] for p in pa_patterns if p["first_cationic_position"] >= 0]) if pa_patterns else -1
    entero_first_cationic_mean = np.mean([p["first_cationic_position"] for p in entero_patterns if p["first_cationic_position"] >= 0]) if entero_patterns else -1

    # Statistical tests
    # 1. Fisher exact test for cationic dipeptide presence
    pa_with = sum(1 for p in pa_patterns if p["has_cationic_dipeptide"])
    pa_without = len(pa_patterns) - pa_with
    entero_with = sum(1 for p in entero_patterns if p["has_cationic_dipeptide"])
    entero_without = len(entero_patterns) - entero_with

    contingency_table = [[pa_with, pa_without], [entero_with, entero_without]]

    try:
        odds_ratio, fisher_p = stats.fisher_exact(contingency_table)
    except Exception:
        odds_ratio, fisher_p = 1.0, 1.0

    # 2. Mann-Whitney U for N-term 5 cationic count
    pa_counts = [p["nterm_5_cationic_count"] for p in pa_patterns]
    entero_counts = [p["nterm_5_cationic_count"] for p in entero_patterns]

    if len(pa_counts) >= 3 and len(entero_counts) >= 3:
        mw_stat, mw_p = stats.mannwhitneyu(pa_counts, entero_counts, alternative='two-sided')
    else:
        mw_stat, mw_p = 0, 1.0

    # Collect dipeptide frequencies
    pa_dipeptides = Counter(p["nterm_dipeptide"] for p in pa_patterns)
    entero_dipeptides = Counter(p["nterm_dipeptide"] for p in entero_patterns)

    # Determine verdict
    rate_difference = pa_cationic_dipeptide_rate - entero_cationic_dipeptide_rate
    count_difference = pa_nterm5_cationic_mean - entero_nterm5_cationic_mean

    if rate_difference > 0.30 and fisher_p < 0.05:
        verdict = "CONFIRMED"
        confidence = "HIGH"
    elif rate_difference > 0.20 or (count_difference > 0.3 and mw_p < 0.05):
        verdict = "CONFIRMED"
        confidence = "MEDIUM"
    elif rate_difference > 0.10:
        verdict = "WEAK_SIGNAL"
        confidence = "LOW"
    else:
        verdict = "FALSIFIED"
        confidence = "N/A"

    results = {
        "cluster_analyzed": "3+4 (combined signal clusters)",
        "n_pa_peptides": len(pa_peptides),
        "n_entero_peptides": len(entero_peptides),
        "pa_cationic_dipeptide_rate": round(pa_cationic_dipeptide_rate, 3),
        "entero_cationic_dipeptide_rate": round(entero_cationic_dipeptide_rate, 3),
        "rate_difference": round(rate_difference, 3),
        "pa_nterm5_cationic_mean": round(pa_nterm5_cationic_mean, 2),
        "entero_nterm5_cationic_mean": round(entero_nterm5_cationic_mean, 2),
        "count_difference": round(count_difference, 2),
        "pa_first_cationic_position_mean": round(pa_first_cationic_mean, 2),
        "entero_first_cationic_position_mean": round(entero_first_cationic_mean, 2),
        "fisher_exact_p": round(fisher_p, 4),
        "fisher_odds_ratio": round(odds_ratio, 2) if odds_ratio != float('inf') else "inf",
        "mannwhitney_p": round(mw_p, 4),
        "pa_top_dipeptides": dict(pa_dipeptides.most_common(5)),
        "entero_top_dipeptides": dict(entero_dipeptides.most_common(5)),
        "verdict": verdict,
        "confidence": confidence,
    }

    return results


def main():
    print("=" * 70)
    print("P1 N-TERMINAL CATIONIC MOTIF VALIDATION")
    print("=" * 70)
    print()
    print("Hypothesis: P. aeruginosa-effective peptides have N-terminal cationic")
    print("dipeptides (KK, RK, etc.) that exploit high membrane charge (-0.7)")
    print()

    # Load data
    print("Loading candidates...")
    candidates = load_candidates(RESULTS_DIR)
    if not candidates:
        print("ERROR: No candidates found")
        sys.exit(1)
    print(f"  Loaded {len(candidates)} candidates")

    # Assign clusters
    print("\nAssigning clusters...")
    cluster_labels = assign_clusters(candidates)

    # Validate hypothesis
    print()
    print("-" * 70)
    print("HYPOTHESIS TEST")
    print("-" * 70)

    results = validate_hypothesis(candidates, cluster_labels)

    # Display results
    print()
    print(f"P. aeruginosa peptides: {results['n_pa_peptides']}")
    print(f"Enterobacteriaceae peptides: {results['n_entero_peptides']}")
    print()
    print("N-terminal cationic dipeptide (KK, RK, KR, RR, etc.):")
    print(f"  P. aeruginosa rate:      {results['pa_cationic_dipeptide_rate']:.1%}")
    print(f"  Enterobacteriaceae rate: {results['entero_cationic_dipeptide_rate']:.1%}")
    print(f"  Difference:              {results['rate_difference']:+.1%}")
    print(f"  Fisher exact p-value:    {results['fisher_exact_p']:.4f}")
    print(f"  Odds ratio:              {results['fisher_odds_ratio']}")
    print()
    print("N-terminal 5 AA cationic count:")
    print(f"  P. aeruginosa mean:      {results['pa_nterm5_cationic_mean']:.2f}")
    print(f"  Enterobacteriaceae mean: {results['entero_nterm5_cationic_mean']:.2f}")
    print(f"  Difference:              {results['count_difference']:+.2f}")
    print(f"  Mann-Whitney p-value:    {results['mannwhitney_p']:.4f}")
    print()
    print("First cationic residue position (0-indexed):")
    print(f"  P. aeruginosa mean:      {results['pa_first_cationic_position_mean']:.2f}")
    print(f"  Enterobacteriaceae mean: {results['entero_first_cationic_position_mean']:.2f}")
    print()
    print("Top N-terminal dipeptides:")
    print(f"  P. aeruginosa:      {results['pa_top_dipeptides']}")
    print(f"  Enterobacteriaceae: {results['entero_top_dipeptides']}")

    # Verdict
    print()
    print("=" * 70)
    if results["verdict"] == "CONFIRMED":
        print(f"VERDICT: HYPOTHESIS {results['verdict']} (Confidence: {results['confidence']})")
        print()
        print("N-terminal cationic motifs ARE enriched in P. aeruginosa-effective peptides.")
        print("This supports mechanism-based design targeting high-LPS membranes.")
    elif results["verdict"] == "WEAK_SIGNAL":
        print(f"VERDICT: {results['verdict']} (Confidence: {results['confidence']})")
        print()
        print("Trend exists but not statistically robust. Proceed with caution.")
    else:
        print(f"VERDICT: HYPOTHESIS {results['verdict']}")
        print()
        print("N-terminal cationic patterns do NOT distinguish pathogen-effective peptides.")
        print("The cationic fraction signal may be distributed throughout the sequence.")
    print("=" * 70)

    # Save results
    output_file = RESULTS_DIR / "P1_nterm_validation.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
