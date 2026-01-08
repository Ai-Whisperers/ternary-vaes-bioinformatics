#!/usr/bin/env python3
"""P1 Mechanism Fingerprint Analysis: What does pathogen specificity actually encode?

Key question: Is the C3 signal "pathogen specificity" or "mechanism specificity"?

If mechanism fingerprints emerge, the model isn't learning "this peptide kills S. aureus"
- it's learning "this peptide uses mechanism X, which happens to work better on Gram+ membranes."

This reframes AMP design: optimize per-mechanism, not per-pathogen.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
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

# =============================================================================
# PATHOGEN PROPERTIES (for mechanism inference)
# =============================================================================

PATHOGEN_PROPERTIES = {
    "A_baumannii": {
        "gram": "negative",
        "gram_code": -1,
        "lps_abundance": 0.85,
        "membrane_charge": -0.6,
        "outer_membrane": True,
        "priority": "critical",
    },
    "P_aeruginosa": {
        "gram": "negative",
        "gram_code": -1,
        "lps_abundance": 0.90,
        "membrane_charge": -0.7,
        "outer_membrane": True,
        "priority": "critical",
    },
    "Enterobacteriaceae": {
        "gram": "negative",
        "gram_code": -1,
        "lps_abundance": 0.88,
        "membrane_charge": -0.55,
        "outer_membrane": True,
        "priority": "critical",
    },
    "S_aureus": {
        "gram": "positive",
        "gram_code": 1,
        "lps_abundance": 0.0,  # No LPS in Gram+
        "membrane_charge": -0.3,
        "outer_membrane": False,
        "priority": "high",
    },
    "H_pylori": {
        "gram": "negative",
        "gram_code": -1,
        "lps_abundance": 0.75,
        "membrane_charge": -0.4,
        "outer_membrane": True,
        "priority": "medium",
    },
}

# Known AMP mechanism indicators
MECHANISM_FEATURES = {
    "barrel_stave": {
        "length_range": (15, 25),
        "hydrophobicity_range": (-0.5, 0.5),
        "charge_range": (2, 6),
        "description": "Forms transmembrane pores, needs length for bilayer spanning",
    },
    "carpet": {
        "length_range": (10, 20),
        "hydrophobicity_range": (-1.0, 0.0),
        "charge_range": (3, 8),
        "description": "Covers membrane surface, high charge density matters",
    },
    "toroidal": {
        "length_range": (20, 30),
        "hydrophobicity_range": (0.0, 1.0),
        "charge_range": (2, 5),
        "description": "Creates toroidal pores with lipid participation",
    },
    "detergent": {
        "length_range": (8, 15),
        "hydrophobicity_range": (-0.5, 0.5),
        "charge_range": (1, 4),
        "description": "Micelle-like disruption, short peptides",
    },
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_candidates(results_dir: Path) -> List[dict]:
    """Load candidates from validation batch."""
    pathogens = list(PATHOGEN_PROPERTIES.keys())
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
                    "charge_density": float(row["net_charge"]) / len(row["sequence"]),
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


# =============================================================================
# MECHANISM FINGERPRINT ANALYSIS
# =============================================================================

def compute_cluster_pathogen_profile(
    candidates: List[dict],
    cluster_labels: np.ndarray,
    signal_clusters: List[int] = [1, 3, 4],
) -> Dict[int, Dict]:
    """Compute MIC profile per pathogen within each signal cluster."""
    profiles = {}

    for cluster_id in signal_clusters:
        mask = cluster_labels == cluster_id
        cluster_candidates = [c for c, m in zip(candidates, mask) if m]

        if len(cluster_candidates) < 10:
            continue

        # Group by pathogen
        by_pathogen = defaultdict(list)
        for c in cluster_candidates:
            by_pathogen[c["pathogen"]].append(c["mic_pred"])

        # Compute stats per pathogen
        pathogen_stats = {}
        for pathogen, mics in by_pathogen.items():
            if len(mics) >= 3:
                pathogen_stats[pathogen] = {
                    "n": len(mics),
                    "mean_mic": float(np.mean(mics)),
                    "std_mic": float(np.std(mics)),
                    "median_mic": float(np.median(mics)),
                    "gram": PATHOGEN_PROPERTIES[pathogen]["gram"],
                    "lps": PATHOGEN_PROPERTIES[pathogen]["lps_abundance"],
                    "membrane_charge": PATHOGEN_PROPERTIES[pathogen]["membrane_charge"],
                }

        # Rank pathogens by mean MIC (lower = more effective)
        ranked = sorted(pathogen_stats.items(), key=lambda x: x[1]["mean_mic"])
        for rank, (pathogen, stats) in enumerate(ranked):
            pathogen_stats[pathogen]["effectiveness_rank"] = rank + 1

        profiles[cluster_id] = {
            "n_total": len(cluster_candidates),
            "n_pathogens": len(pathogen_stats),
            "pathogen_stats": pathogen_stats,
            "ranking": [p for p, _ in ranked],
        }

    return profiles


def analyze_gram_correlation(profiles: Dict[int, Dict]) -> Dict:
    """Test if MIC correlates with Gram type within signal clusters."""
    results = {}

    for cluster_id, profile in profiles.items():
        gram_neg_mics = []
        gram_pos_mics = []

        for pathogen, pstats in profile["pathogen_stats"].items():
            if pstats["gram"] == "negative":
                gram_neg_mics.extend([pstats["mean_mic"]] * pstats["n"])
            else:
                gram_pos_mics.extend([pstats["mean_mic"]] * pstats["n"])

        if len(gram_neg_mics) >= 3 and len(gram_pos_mics) >= 3:
            # Mann-Whitney U test
            u_stat, p_value = stats.mannwhitneyu(gram_neg_mics, gram_pos_mics, alternative='two-sided')
            effect = np.mean(gram_neg_mics) - np.mean(gram_pos_mics)

            results[cluster_id] = {
                "gram_neg_mean": float(np.mean(gram_neg_mics)),
                "gram_pos_mean": float(np.mean(gram_pos_mics)),
                "difference": float(effect),
                "p_value": float(p_value),
                "gram_effect": "Gram+ easier" if effect > 0 else "Gram- easier",
                "significant": p_value < 0.05,
            }

    return results


def infer_mechanism_class(candidates: List[dict], cluster_labels: np.ndarray) -> Dict[int, Dict]:
    """Infer likely AMP mechanism based on cluster peptide properties."""
    mechanism_scores = {}

    for cluster_id in sorted(set(cluster_labels)):
        mask = cluster_labels == cluster_id
        cluster_candidates = [c for c, m in zip(candidates, mask) if m]

        if len(cluster_candidates) < 5:
            continue

        # Compute cluster averages
        avg_length = np.mean([c["length"] for c in cluster_candidates])
        avg_hydro = np.mean([c["hydrophobicity"] for c in cluster_candidates])
        avg_charge = np.mean([c["net_charge"] for c in cluster_candidates])

        # Score each mechanism
        scores = {}
        for mech_name, mech_props in MECHANISM_FEATURES.items():
            score = 0

            # Length match
            if mech_props["length_range"][0] <= avg_length <= mech_props["length_range"][1]:
                score += 1

            # Hydrophobicity match
            if mech_props["hydrophobicity_range"][0] <= avg_hydro <= mech_props["hydrophobicity_range"][1]:
                score += 1

            # Charge match
            if mech_props["charge_range"][0] <= avg_charge <= mech_props["charge_range"][1]:
                score += 1

            scores[mech_name] = score

        best_mechanism = max(scores, key=scores.get)

        mechanism_scores[cluster_id] = {
            "avg_length": round(avg_length, 1),
            "avg_hydrophobicity": round(avg_hydro, 2),
            "avg_charge": round(avg_charge, 1),
            "mechanism_scores": scores,
            "likely_mechanism": best_mechanism,
            "mechanism_description": MECHANISM_FEATURES[best_mechanism]["description"],
            "confidence": scores[best_mechanism] / 3.0,
        }

    return mechanism_scores


def analyze_sequence_motifs(
    candidates: List[dict],
    cluster_labels: np.ndarray,
    signal_clusters: List[int] = [1, 3, 4],
) -> Dict[int, Dict]:
    """Find sequence patterns that correlate with pathogen effectiveness."""
    motif_analysis = {}

    # Amino acid categories
    CATIONIC = set("KRH")
    HYDROPHOBIC = set("AILMFVWY")
    POLAR = set("NQST")

    for cluster_id in signal_clusters:
        mask = cluster_labels == cluster_id
        cluster_candidates = [c for c, m in zip(candidates, mask) if m]

        if len(cluster_candidates) < 10:
            continue

        # Group by best pathogen (lowest MIC)
        by_pathogen = defaultdict(list)
        for c in cluster_candidates:
            by_pathogen[c["pathogen"]].append(c)

        # Analyze sequence composition per pathogen group
        pathogen_motifs = {}
        for pathogen, peptides in by_pathogen.items():
            if len(peptides) < 3:
                continue

            # Compute AA composition
            all_seqs = "".join([p["sequence"] for p in peptides])
            total_aa = len(all_seqs)

            cationic_frac = sum(1 for aa in all_seqs if aa in CATIONIC) / total_aa
            hydrophobic_frac = sum(1 for aa in all_seqs if aa in HYDROPHOBIC) / total_aa
            polar_frac = sum(1 for aa in all_seqs if aa in POLAR) / total_aa

            # N-terminal analysis (first 5 AA)
            n_term = [p["sequence"][:5] for p in peptides]
            n_term_cationic = np.mean([sum(1 for aa in seq if aa in CATIONIC) for seq in n_term])

            pathogen_motifs[pathogen] = {
                "n_peptides": len(peptides),
                "cationic_fraction": round(cationic_frac, 3),
                "hydrophobic_fraction": round(hydrophobic_frac, 3),
                "polar_fraction": round(polar_frac, 3),
                "n_term_cationic_avg": round(n_term_cationic, 2),
                "gram": PATHOGEN_PROPERTIES[pathogen]["gram"],
            }

        motif_analysis[cluster_id] = pathogen_motifs

    return motif_analysis


def compute_design_implications(
    profiles: Dict[int, Dict],
    gram_results: Dict,
    mechanism_scores: Dict[int, Dict],
) -> Dict:
    """Synthesize findings into actionable AMP design principles."""
    implications = {
        "key_findings": [],
        "design_rules": [],
        "mechanism_to_pathogen_map": {},
        "optimization_strategy": None,
    }

    # Analyze Gram correlation across clusters
    gram_effects = []
    for cluster_id, gram_result in gram_results.items():
        if gram_result["significant"]:
            gram_effects.append(gram_result["gram_effect"])
            implications["key_findings"].append(
                f"Cluster {cluster_id}: {gram_result['gram_effect']} "
                f"(diff={gram_result['difference']:.3f}, p={gram_result['p_value']:.4f})"
            )

    # Check for consistent Gram pattern
    if gram_effects:
        gram_pos_easier = sum(1 for e in gram_effects if "Gram+" in e)
        gram_neg_easier = sum(1 for e in gram_effects if "Gram-" in e)

        if gram_pos_easier > gram_neg_easier:
            implications["design_rules"].append(
                "Short peptides (13-14 AA) are MORE effective against Gram+ bacteria"
            )
            implications["optimization_strategy"] = "gram_type_routing"
        elif gram_neg_easier > gram_pos_easier:
            implications["design_rules"].append(
                "Short peptides (13-14 AA) are MORE effective against Gram- bacteria"
            )
            implications["optimization_strategy"] = "gram_type_routing"
        else:
            implications["design_rules"].append(
                "Gram type effect is cluster-dependent - use mechanism-specific optimization"
            )
            implications["optimization_strategy"] = "mechanism_routing"

    # Map mechanisms to pathogen effectiveness
    for cluster_id, mech_info in mechanism_scores.items():
        if cluster_id in profiles:
            ranking = profiles[cluster_id]["ranking"]
            implications["mechanism_to_pathogen_map"][mech_info["likely_mechanism"]] = {
                "cluster": cluster_id,
                "best_against": ranking[0] if ranking else None,
                "worst_against": ranking[-1] if ranking else None,
                "confidence": mech_info["confidence"],
            }

    return implications


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("P1 MECHANISM FINGERPRINT ANALYSIS")
    print("What does pathogen specificity actually encode?")
    print("=" * 70)
    print()

    # Load data
    print("Loading candidates...")
    candidates = load_candidates(RESULTS_DIR)
    if not candidates:
        print("ERROR: No candidates found")
        sys.exit(1)
    print(f"  Loaded {len(candidates)} candidates")

    # Assign clusters
    print("\nAssigning clusters (same as C3)...")
    cluster_labels = assign_clusters(candidates)

    # Identify signal clusters from C3 results
    signal_clusters = [1, 3, 4]
    print(f"  Signal clusters: {signal_clusters}")

    # ==========================================================================
    # ANALYSIS 1: Pathogen MIC Profiles
    # ==========================================================================
    print()
    print("-" * 70)
    print("ANALYSIS 1: Pathogen MIC Profiles Within Signal Clusters")
    print("-" * 70)

    profiles = compute_cluster_pathogen_profile(candidates, cluster_labels, signal_clusters)

    for cluster_id, profile in profiles.items():
        print(f"\n[Cluster {cluster_id}] N={profile['n_total']}")
        print(f"  Pathogen effectiveness ranking (best → worst): {' → '.join(profile['ranking'])}")
        print()
        print(f"  {'Pathogen':<20} {'N':>4} {'Mean MIC':>10} {'Gram':>8} {'Rank':>6}")
        print("  " + "-" * 52)
        for pathogen, pstats in sorted(profile["pathogen_stats"].items(),
                                       key=lambda x: x[1]["effectiveness_rank"]):
            print(f"  {pathogen:<20} {pstats['n']:>4} {pstats['mean_mic']:>10.4f} "
                  f"{pstats['gram']:>8} {pstats['effectiveness_rank']:>6}")

    # ==========================================================================
    # ANALYSIS 2: Gram Type Correlation
    # ==========================================================================
    print()
    print("-" * 70)
    print("ANALYSIS 2: Does MIC Correlate With Gram Type?")
    print("-" * 70)

    gram_results = analyze_gram_correlation(profiles)

    for cluster_id, result in gram_results.items():
        sig = "***" if result["significant"] else ""
        print(f"\n[Cluster {cluster_id}]")
        print(f"  Gram- mean MIC: {result['gram_neg_mean']:.4f}")
        print(f"  Gram+ mean MIC: {result['gram_pos_mean']:.4f}")
        print(f"  Difference: {result['difference']:+.4f} ({result['gram_effect']}) {sig}")
        print(f"  p-value: {result['p_value']:.4f}")

    # ==========================================================================
    # ANALYSIS 3: Mechanism Inference
    # ==========================================================================
    print()
    print("-" * 70)
    print("ANALYSIS 3: Inferred AMP Mechanisms by Cluster")
    print("-" * 70)

    mechanism_scores = infer_mechanism_class(candidates, cluster_labels)

    for cluster_id in sorted(mechanism_scores.keys()):
        mech = mechanism_scores[cluster_id]
        is_signal = cluster_id in signal_clusters
        marker = "**SIGNAL**" if is_signal else ""

        print(f"\n[Cluster {cluster_id}] {marker}")
        print(f"  Properties: length={mech['avg_length']}, hydro={mech['avg_hydrophobicity']}, "
              f"charge={mech['avg_charge']}")
        print(f"  Likely mechanism: {mech['likely_mechanism']} (confidence={mech['confidence']:.0%})")
        print(f"  Description: {mech['mechanism_description']}")

    # ==========================================================================
    # ANALYSIS 4: Sequence Motifs
    # ==========================================================================
    print()
    print("-" * 70)
    print("ANALYSIS 4: Sequence Composition by Pathogen Target")
    print("-" * 70)

    motif_analysis = analyze_sequence_motifs(candidates, cluster_labels, signal_clusters)

    for cluster_id, pathogen_motifs in motif_analysis.items():
        print(f"\n[Cluster {cluster_id}]")
        print(f"  {'Pathogen':<20} {'N':>4} {'Cationic':>10} {'Hydrophobic':>12} {'N-term +':>10}")
        print("  " + "-" * 60)
        for pathogen, motifs in sorted(pathogen_motifs.items()):
            print(f"  {pathogen:<20} {motifs['n_peptides']:>4} "
                  f"{motifs['cationic_fraction']:>10.1%} {motifs['hydrophobic_fraction']:>12.1%} "
                  f"{motifs['n_term_cationic_avg']:>10.1f}")

    # ==========================================================================
    # SYNTHESIS: Design Implications
    # ==========================================================================
    print()
    print("=" * 70)
    print("SYNTHESIS: AMP DESIGN IMPLICATIONS")
    print("=" * 70)

    implications = compute_design_implications(profiles, gram_results, mechanism_scores)

    print("\nKey Findings:")
    for finding in implications["key_findings"]:
        print(f"  - {finding}")

    print("\nDesign Rules:")
    for rule in implications["design_rules"]:
        print(f"  - {rule}")

    print("\nMechanism → Pathogen Map:")
    for mech, mapping in implications["mechanism_to_pathogen_map"].items():
        if mapping["best_against"]:
            print(f"  - {mech}: best against {mapping['best_against']}, "
                  f"worst against {mapping['worst_against']} "
                  f"(cluster {mapping['cluster']}, confidence={mapping['confidence']:.0%})")

    print(f"\nRecommended Optimization Strategy: {implications['optimization_strategy']}")

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    output = {
        "profiles": {str(k): v for k, v in profiles.items()},
        "gram_correlation": {str(k): v for k, v in gram_results.items()},
        "mechanism_inference": {str(k): v for k, v in mechanism_scores.items()},
        "motif_analysis": {str(k): v for k, v in motif_analysis.items()},
        "design_implications": implications,
    }

    output_file = RESULTS_DIR / "P1_mechanism_fingerprint.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # ==========================================================================
    # VERDICT
    # ==========================================================================
    print()
    print("=" * 70)
    print("VERDICT: Is this pathogen specificity or mechanism specificity?")
    print("=" * 70)

    # Check if rankings are consistent across clusters
    rankings = [profiles[c]["ranking"] for c in signal_clusters if c in profiles]
    if len(rankings) >= 2:
        # Compare first-ranked pathogen across clusters
        first_ranked = [r[0] for r in rankings]
        if len(set(first_ranked)) == 1:
            print(f"\nCONSISTENT: All signal clusters show {first_ranked[0]} as most susceptible")
            print("→ This suggests a SHARED MECHANISM across clusters")
        else:
            print(f"\nDIVERGENT: Different clusters favor different pathogens")
            print(f"  Cluster rankings: {dict(zip(signal_clusters, first_ranked))}")
            print("→ This suggests MULTIPLE MECHANISMS captured by different clusters")

    return 0


if __name__ == "__main__":
    sys.exit(main())
