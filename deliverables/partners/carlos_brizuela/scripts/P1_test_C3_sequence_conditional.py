#!/usr/bin/env python3
"""P1 Conjecture 3 Test: Pathogen specificity is SEQUENCE-CONDITIONAL, not global.

Hypothesis:
    Only specific peptide submanifolds exhibit pathogen differentiation.
    Global models fail because they average over regimes with different behavior.

Test:
    1. Cluster peptides by features (length, charge, hydrophobicity)
    2. Test pathogen separation WITHIN each cluster
    3. Identify any clusters with above-noise differentiation

Falsifies if:
    No cluster shows above-noise pathogen separation.

Classification (R3):
    - Cluster membership computable at inference → Deployable if signal found
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PACKAGE_DIR / "results" / "validation_batch"


def load_all_candidates(results_dir: Path) -> Tuple[List[dict], List[str]]:
    """Load candidates from all pathogen result files with pathogen labels."""
    import csv

    pathogens = ["A_baumannii", "S_aureus", "P_aeruginosa", "Enterobacteriaceae", "H_pylori"]
    all_candidates = []
    all_labels = []

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
                    "length": len(row["sequence"]),
                    "mic_pred": float(row["mic_pred"]),
                    "net_charge": float(row["net_charge"]),
                    "hydrophobicity": float(row["hydrophobicity"]),
                    "pathogen": pathogen,
                })
                all_labels.append(pathogen)

    print(f"Loaded {len(all_candidates)} total candidates across {len(pathogens)} pathogens")
    return all_candidates, all_labels


def cluster_by_features(candidates: List[dict], n_clusters: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster peptides by biophysical features.

    Features: length, net_charge, hydrophobicity
    """
    # Extract features
    X = np.array([
        [c["length"], c["net_charge"], c["hydrophobicity"]]
        for c in candidates
    ])

    # Standardize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_scaled = (X - X_mean) / X_std

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    return cluster_labels, X


def test_within_cluster_separation(
    candidates: List[dict],
    cluster_labels: np.ndarray,
    n_clusters: int,
) -> Dict:
    """Test pathogen separation within each cluster."""
    results = {}

    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_candidates = [c for c, m in zip(candidates, mask) if m]

        if len(cluster_candidates) < 10:
            results[f"cluster_{cluster_id}"] = {
                "n": len(cluster_candidates),
                "conclusion": "insufficient_data",
            }
            continue

        # Group by pathogen
        by_pathogen = {}
        for c in cluster_candidates:
            p = c["pathogen"]
            if p not in by_pathogen:
                by_pathogen[p] = []
            by_pathogen[p].append(c["mic_pred"])

        # Skip if only one pathogen represented
        if len(by_pathogen) < 2:
            results[f"cluster_{cluster_id}"] = {
                "n": len(cluster_candidates),
                "n_pathogens": len(by_pathogen),
                "conclusion": "single_pathogen",
            }
            continue

        # Kruskal-Wallis test
        groups = [np.array(v) for v in by_pathogen.values() if len(v) >= 3]
        if len(groups) < 2:
            results[f"cluster_{cluster_id}"] = {
                "n": len(cluster_candidates),
                "n_pathogens": len(by_pathogen),
                "conclusion": "insufficient_per_pathogen",
            }
            continue

        h_stat, p_value = stats.kruskal(*groups)

        # Calculate effect size (between vs within)
        pathogen_means = [np.mean(v) for v in by_pathogen.values()]
        pathogen_stds = [np.std(v) for v in by_pathogen.values() if len(v) > 1]
        between_std = np.std(pathogen_means) if len(pathogen_means) > 1 else 0
        within_std = np.mean(pathogen_stds) if pathogen_stds else 1

        effect_ratio = between_std / within_std if within_std > 0 else 0

        # Characterize cluster
        lengths = [c["length"] for c in cluster_candidates]
        charges = [c["net_charge"] for c in cluster_candidates]
        hydros = [c["hydrophobicity"] for c in cluster_candidates]

        results[f"cluster_{cluster_id}"] = {
            "n": len(cluster_candidates),
            "n_pathogens": len(by_pathogen),
            "pathogens": {p: len(v) for p, v in by_pathogen.items()},
            "kruskal_h": h_stat,
            "kruskal_p": p_value,
            "between_std": between_std,
            "within_std": within_std,
            "effect_ratio": effect_ratio,
            "cluster_profile": {
                "length": f"{np.mean(lengths):.1f} +/- {np.std(lengths):.1f}",
                "charge": f"{np.mean(charges):.1f} +/- {np.std(charges):.1f}",
                "hydrophobicity": f"{np.mean(hydros):.2f} +/- {np.std(hydros):.2f}",
            },
            "conclusion": determine_cluster_conclusion(p_value, effect_ratio),
        }

    return results


def determine_cluster_conclusion(p_value: float, effect_ratio: float) -> str:
    """Determine cluster conclusion based on statistical AND practical significance."""
    if p_value > 0.05:
        return "no_separation"
    elif effect_ratio < 0.5:
        return "statistical_only"  # p < 0.05 but no practical effect
    else:
        return "signal_found"  # Both statistical and practical significance


def summarize_results(cluster_results: Dict) -> Dict:
    """Summarize across all clusters."""
    signal_clusters = []
    no_signal_clusters = []
    inconclusive_clusters = []

    for cluster_id, result in cluster_results.items():
        conclusion = result.get("conclusion", "unknown")
        if conclusion == "signal_found":
            signal_clusters.append(cluster_id)
        elif conclusion in ["no_separation", "statistical_only"]:
            no_signal_clusters.append(cluster_id)
        else:
            inconclusive_clusters.append(cluster_id)

    n_testable = len(signal_clusters) + len(no_signal_clusters)

    if len(signal_clusters) > 0:
        verdict = "PARTIAL_SIGNAL"
        interpretation = (
            f"{len(signal_clusters)}/{n_testable} clusters show pathogen separation. "
            f"Specificity may be sequence-conditional."
        )
    else:
        verdict = "FALSIFIED"
        interpretation = (
            f"No cluster shows above-noise pathogen separation. "
            f"Conjecture 3 is falsified: specificity is NOT sequence-conditional."
        )

    return {
        "signal_clusters": signal_clusters,
        "no_signal_clusters": no_signal_clusters,
        "inconclusive_clusters": inconclusive_clusters,
        "verdict": verdict,
        "interpretation": interpretation,
    }


def main():
    print("=" * 70)
    print("P1 CONJECTURE 3 TEST: Pathogen specificity is SEQUENCE-CONDITIONAL")
    print("=" * 70)
    print()

    # Load data
    print("Loading candidates...")
    candidates, labels = load_all_candidates(RESULTS_DIR)
    if not candidates:
        print("ERROR: No candidates found")
        sys.exit(1)
    print()

    # Cluster
    n_clusters = 5
    print(f"Clustering into {n_clusters} clusters by (length, charge, hydrophobicity)...")
    cluster_labels, X = cluster_by_features(candidates, n_clusters=n_clusters)

    # Report cluster sizes
    for i in range(n_clusters):
        n = sum(cluster_labels == i)
        print(f"  Cluster {i}: {n} peptides")
    print()

    # Test within-cluster separation
    print("-" * 70)
    print("WITHIN-CLUSTER PATHOGEN SEPARATION TESTS")
    print("-" * 70)

    cluster_results = test_within_cluster_separation(candidates, cluster_labels, n_clusters)

    for cluster_id, result in cluster_results.items():
        print(f"\n{cluster_id}:")
        print(f"  N = {result['n']}")
        if result.get("conclusion") in ["insufficient_data", "single_pathogen", "insufficient_per_pathogen"]:
            print(f"  Conclusion: {result['conclusion']}")
            continue

        print(f"  Pathogens: {result['n_pathogens']}")
        print(f"  Profile: {result['cluster_profile']}")
        print(f"  Kruskal-Wallis H = {result['kruskal_h']:.2f}, p = {result['kruskal_p']:.4f}")
        print(f"  Effect ratio: {result['effect_ratio']:.3f} (< 0.5 = no practical effect)")
        print(f"  Conclusion: {result['conclusion']}")

    # Summary
    print()
    print("=" * 70)
    print("CONJECTURE 3 VERDICT")
    print("=" * 70)

    summary = summarize_results(cluster_results)

    print(f"Signal clusters: {summary['signal_clusters']}")
    print(f"No-signal clusters: {summary['no_signal_clusters']}")
    print(f"Inconclusive: {summary['inconclusive_clusters']}")
    print()
    print(f"VERDICT: {summary['verdict']}")
    print(f"INTERPRETATION: {summary['interpretation']}")
    print()

    # R3 Classification
    print("-" * 70)
    print("R3 CLASSIFICATION (Inference-Time Availability)")
    print("-" * 70)
    if summary["verdict"] == "PARTIAL_SIGNAL":
        r3_class = "Deployable (cluster membership computable for novel peptides)"
        r3_note = "Would require cluster-specific models"
    else:
        r3_class = "Research-Only (no signal found)"
        r3_note = "Negative result documented"

    print(f"Classification: {r3_class}")
    print(f"Note: {r3_note}")

    # Save results
    output_file = RESULTS_DIR / "P1_C3_results.json"
    output = {
        "conjecture": "C3: Pathogen specificity is SEQUENCE-CONDITIONAL, not global",
        "n_clusters": n_clusters,
        "verdict": summary["verdict"],
        "interpretation": summary["interpretation"],
        "r3_classification": r3_class,
        "cluster_results": cluster_results,
        "summary": summary,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print()
    print(f"Results saved to: {output_file}")

    return summary["verdict"]


if __name__ == "__main__":
    verdict = main()
    sys.exit(0 if verdict in ["PARTIAL_SIGNAL", "FALSIFIED"] else 1)
