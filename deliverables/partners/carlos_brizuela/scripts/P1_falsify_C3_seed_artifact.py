#!/usr/bin/env python3
"""P1 C3 Falsification: Is the cluster separation a seed sequence artifact?

Hypothesis to FALSIFY:
    The cluster separation observed in C3 is due to different seed sequences
    used for each pathogen, NOT biological pathogen specificity.

Test Strategy:
    1. Map each candidate to its likely seed origin (by sequence similarity)
    2. Test: does seed origin predict cluster membership better than pathogen?
    3. If seed explains clusters → SEED ARTIFACT → C3 is FALSIFIED
    4. If pathogen explains clusters after controlling for seeds → SIGNAL SURVIVES

This is a RUTHLESS falsification. If seed artifact is found, the C3 signal dies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PACKAGE_DIR / "results" / "validation_batch"
CONFIG_DIR = PACKAGE_DIR / "configs"

# Known seed sequences per pathogen (from pathogens.json)
PATHOGEN_SEEDS = {
    "A_baumannii": [
        "GIGKFLHSAKKFGKAFVGEIMNS",
        "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
        "RLARIVVIRVAR"
    ],
    "P_aeruginosa": [
        "ILPWKWPWWPWRR",
        "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
        "RLKKTFFKIVKTVKW"
    ],
    "Enterobacteriaceae": [
        "KLAKLAKKLAKLAK",
        "GIGKFLHSAKKFGKAFVGEIMNS",
        "KRFRIRVRV"
    ],
    "S_aureus": [
        "KLAKLAKKLAKLAK",
        "KLWKKLKKALK",
        "FKCRRWQWRMKKLGAPS"
    ],
    "H_pylori": [
        "KLWKKLKKALK",
        "KLAKLAKKLAKLAK",
        "RLKKTFFKIV"
    ],
}

# All unique seeds
ALL_SEEDS = list(set(s for seeds in PATHOGEN_SEEDS.values() for s in seeds))


def load_all_candidates(results_dir: Path) -> List[dict]:
    """Load candidates from all pathogen result files."""
    import csv

    pathogens = ["A_baumannii", "S_aureus", "P_aeruginosa", "Enterobacteriaceae", "H_pylori"]
    all_candidates = []

    for pathogen in pathogens:
        csv_path = results_dir / f"{pathogen}_candidates.csv"
        if not csv_path.exists():
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

    return all_candidates


def sequence_similarity(seq1: str, seq2: str) -> float:
    """Calculate simple sequence similarity (shared amino acids / max length)."""
    if not seq1 or not seq2:
        return 0.0

    # Use longest common subsequence ratio
    from difflib import SequenceMatcher
    return SequenceMatcher(None, seq1, seq2).ratio()


def find_closest_seed(sequence: str, seeds: List[str]) -> Tuple[str, float]:
    """Find the seed sequence most similar to the given sequence."""
    best_seed = None
    best_sim = -1

    for seed in seeds:
        sim = sequence_similarity(sequence, seed)
        if sim > best_sim:
            best_sim = sim
            best_seed = seed

    return best_seed, best_sim


def assign_seed_origins(candidates: List[dict]) -> List[dict]:
    """Assign each candidate to its likely seed origin."""
    for c in candidates:
        seq = c["sequence"]
        pathogen = c["pathogen"]

        # First try pathogen-specific seeds
        pathogen_seeds = PATHOGEN_SEEDS.get(pathogen, [])
        if pathogen_seeds:
            closest, sim = find_closest_seed(seq, pathogen_seeds)
            c["closest_pathogen_seed"] = closest
            c["pathogen_seed_similarity"] = sim

        # Also find closest among ALL seeds
        closest_any, sim_any = find_closest_seed(seq, ALL_SEEDS)
        c["closest_any_seed"] = closest_any
        c["any_seed_similarity"] = sim_any

    return candidates


def cluster_by_features(candidates: List[dict], n_clusters: int = 5) -> np.ndarray:
    """Cluster peptides by biophysical features (same as C3 test)."""
    X = np.array([
        [c["length"], c["net_charge"], c["hydrophobicity"]]
        for c in candidates
    ])

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_scaled = (X - X_mean) / X_std

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(X_scaled)


def test_seed_vs_pathogen_clustering(candidates: List[dict], cluster_labels: np.ndarray) -> Dict:
    """Test whether cluster membership is explained by seeds or pathogens.

    Key test: If seeds explain clusters better than pathogens, C3 is a seed artifact.
    """
    n = len(candidates)

    # Extract labels
    pathogens = np.array([c["pathogen"] for c in candidates])
    closest_seeds = np.array([c.get("closest_any_seed", "unknown") for c in candidates])

    # Unique values
    unique_pathogens = list(set(pathogens))
    unique_seeds = list(set(closest_seeds))
    unique_clusters = list(set(cluster_labels))

    # Contingency tables
    # Cluster x Pathogen
    cluster_pathogen = np.zeros((len(unique_clusters), len(unique_pathogens)))
    for i, (cl, pa) in enumerate(zip(cluster_labels, pathogens)):
        cl_idx = unique_clusters.index(cl)
        pa_idx = unique_pathogens.index(pa)
        cluster_pathogen[cl_idx, pa_idx] += 1

    # Cluster x Seed
    cluster_seed = np.zeros((len(unique_clusters), len(unique_seeds)))
    for i, (cl, se) in enumerate(zip(cluster_labels, closest_seeds)):
        cl_idx = unique_clusters.index(cl)
        se_idx = unique_seeds.index(se)
        cluster_seed[cl_idx, se_idx] += 1

    # Chi-square tests (higher chi2 = stronger association)
    try:
        chi2_pathogen, p_pathogen, _, _ = stats.chi2_contingency(cluster_pathogen)
    except ValueError:
        chi2_pathogen, p_pathogen = 0, 1.0

    try:
        chi2_seed, p_seed, _, _ = stats.chi2_contingency(cluster_seed)
    except ValueError:
        chi2_seed, p_seed = 0, 1.0

    # Cramér's V (effect size for chi-square)
    def cramers_v(chi2, n, k, r):
        return np.sqrt(chi2 / (n * min(k - 1, r - 1))) if n > 0 and min(k, r) > 1 else 0

    v_pathogen = cramers_v(chi2_pathogen, n, len(unique_clusters), len(unique_pathogens))
    v_seed = cramers_v(chi2_seed, n, len(unique_clusters), len(unique_seeds))

    return {
        "chi2_pathogen": chi2_pathogen,
        "p_pathogen": p_pathogen,
        "cramers_v_pathogen": v_pathogen,
        "chi2_seed": chi2_seed,
        "p_seed": p_seed,
        "cramers_v_seed": v_seed,
        "n_unique_pathogens": len(unique_pathogens),
        "n_unique_seeds": len(unique_seeds),
        "n_clusters": len(unique_clusters),
    }


def test_within_seed_pathogen_separation(candidates: List[dict], cluster_labels: np.ndarray) -> Dict:
    """Test if pathogens separate WITHIN samples that share the same seed origin.

    This is the killer test:
    - If pathogen separation exists ONLY between different seeds → seed artifact
    - If pathogen separation exists WITHIN same seed → real signal
    """
    # Group by closest seed
    by_seed = defaultdict(list)
    for c, cl in zip(candidates, cluster_labels):
        seed = c.get("closest_any_seed", "unknown")
        by_seed[seed].append({**c, "cluster": cl})

    results = {}
    for seed, seed_candidates in by_seed.items():
        if len(seed_candidates) < 10:
            results[seed] = {"n": len(seed_candidates), "conclusion": "insufficient_data"}
            continue

        # Group by pathogen within this seed
        by_pathogen = defaultdict(list)
        for c in seed_candidates:
            by_pathogen[c["pathogen"]].append(c["mic_pred"])

        if len(by_pathogen) < 2:
            results[seed] = {
                "n": len(seed_candidates),
                "n_pathogens": len(by_pathogen),
                "conclusion": "single_pathogen",
            }
            continue

        # Kruskal-Wallis test for pathogen differences WITHIN this seed
        groups = [np.array(v) for v in by_pathogen.values() if len(v) >= 3]
        if len(groups) < 2:
            results[seed] = {
                "n": len(seed_candidates),
                "n_pathogens": len(by_pathogen),
                "conclusion": "insufficient_per_pathogen",
            }
            continue

        h_stat, p_value = stats.kruskal(*groups)

        # Effect size
        means = [np.mean(v) for v in by_pathogen.values()]
        stds = [np.std(v) for v in by_pathogen.values() if len(v) > 1]
        between_std = np.std(means) if len(means) > 1 else 0
        within_std = np.mean(stds) if stds else 1
        effect_ratio = between_std / within_std if within_std > 0 else 0

        has_signal = p_value < 0.05 and effect_ratio > 0.5

        results[seed] = {
            "n": len(seed_candidates),
            "n_pathogens": len(by_pathogen),
            "pathogens": list(by_pathogen.keys()),
            "kruskal_h": h_stat,
            "p_value": p_value,
            "effect_ratio": effect_ratio,
            "has_signal": has_signal,
            "conclusion": "signal" if has_signal else "no_signal",
        }

    return results


def main():
    print("=" * 70)
    print("P1 C3 FALSIFICATION: Is cluster separation a SEED ARTIFACT?")
    print("=" * 70)
    print()
    print("If seeds explain clusters better than pathogens → C3 is FALSIFIED")
    print("If pathogen signal persists after controlling for seeds → C3 SURVIVES")
    print()

    # Load data
    print("Loading candidates...")
    candidates = load_all_candidates(RESULTS_DIR)
    if not candidates:
        print("ERROR: No candidates found")
        sys.exit(1)
    print(f"Loaded {len(candidates)} candidates")
    print()

    # Assign seed origins
    print("Assigning seed origins...")
    candidates = assign_seed_origins(candidates)

    # Report seed distribution
    seed_counts = defaultdict(int)
    for c in candidates:
        seed_counts[c.get("closest_any_seed", "unknown")] += 1
    print("Seed distribution:")
    for seed, count in sorted(seed_counts.items(), key=lambda x: -x[1]):
        print(f"  {seed[:30]+'...' if len(seed) > 30 else seed}: {count}")
    print()

    # Cluster
    print("Clustering by features...")
    cluster_labels = cluster_by_features(candidates, n_clusters=5)
    print()

    # Test 1: Does seed explain clusters better than pathogen?
    print("-" * 70)
    print("TEST 1: Seed vs Pathogen association with clusters")
    print("-" * 70)

    assoc = test_seed_vs_pathogen_clustering(candidates, cluster_labels)

    print(f"Pathogen-Cluster association:")
    print(f"  Chi-square: {assoc['chi2_pathogen']:.2f}, p = {assoc['p_pathogen']:.4f}")
    print(f"  Cramér's V: {assoc['cramers_v_pathogen']:.3f}")
    print()
    print(f"Seed-Cluster association:")
    print(f"  Chi-square: {assoc['chi2_seed']:.2f}, p = {assoc['p_seed']:.4f}")
    print(f"  Cramér's V: {assoc['cramers_v_seed']:.3f}")
    print()

    seed_explains_better = assoc['cramers_v_seed'] > assoc['cramers_v_pathogen']
    print(f"Seed explains clusters better than pathogen: {seed_explains_better}")
    print(f"Ratio (V_seed / V_pathogen): {assoc['cramers_v_seed'] / max(assoc['cramers_v_pathogen'], 0.001):.2f}")
    print()

    # Test 2: Pathogen separation WITHIN same seed origin
    print("-" * 70)
    print("TEST 2: Pathogen separation WITHIN same seed origin")
    print("-" * 70)
    print("(If pathogens separate within same seed → real signal)")
    print()

    within_seed = test_within_seed_pathogen_separation(candidates, cluster_labels)

    signal_within_seeds = []
    no_signal_within_seeds = []

    for seed, result in within_seed.items():
        seed_short = seed[:25] + '...' if len(seed) > 25 else seed
        print(f"{seed_short}:")
        print(f"  N = {result['n']}")

        if result["conclusion"] in ["insufficient_data", "single_pathogen", "insufficient_per_pathogen"]:
            print(f"  {result['conclusion']}")
        else:
            print(f"  Pathogens: {result['n_pathogens']}")
            print(f"  Kruskal H = {result['kruskal_h']:.2f}, p = {result['p_value']:.4f}")
            print(f"  Effect ratio: {result['effect_ratio']:.3f}")
            print(f"  --> {'SIGNAL WITHIN SEED' if result['has_signal'] else 'No separation within seed'}")

            if result['has_signal']:
                signal_within_seeds.append(seed)
            else:
                no_signal_within_seeds.append(seed)
        print()

    # Final verdict
    print("=" * 70)
    print("C3 FALSIFICATION VERDICT")
    print("=" * 70)

    # Criteria for falsification:
    # 1. Seed explains clusters better than pathogen (Cramér's V ratio > 1.5)
    # 2. No pathogen separation within same seed origin

    seed_dominates = assoc['cramers_v_seed'] > 1.5 * assoc['cramers_v_pathogen']
    no_within_seed_signal = len(signal_within_seeds) == 0

    if seed_dominates and no_within_seed_signal:
        verdict = "FALSIFIED"
        interpretation = (
            "C3 cluster separation is a SEED ARTIFACT. "
            f"Seeds explain clusters {assoc['cramers_v_seed']/max(assoc['cramers_v_pathogen'],0.001):.1f}x better than pathogens, "
            f"and no pathogen separation exists within same seed origin."
        )
    elif seed_dominates:
        verdict = "LIKELY_ARTIFACT"
        interpretation = (
            f"Seeds explain clusters {assoc['cramers_v_seed']/max(assoc['cramers_v_pathogen'],0.001):.1f}x better than pathogens. "
            f"However, {len(signal_within_seeds)} seeds show within-seed pathogen separation. "
            "Treat with caution."
        )
    elif len(signal_within_seeds) > 0:
        verdict = "SURVIVES"
        interpretation = (
            f"Pathogen separation exists WITHIN same seed origin ({len(signal_within_seeds)} seeds). "
            "This cannot be explained by seed differences. C3 signal appears real."
        )
    else:
        verdict = "INCONCLUSIVE"
        interpretation = (
            "Neither seed nor pathogen clearly dominates cluster assignment. "
            "Insufficient evidence to confirm or falsify."
        )

    print(f"Seed dominates clusters: {seed_dominates}")
    print(f"Within-seed pathogen signal: {len(signal_within_seeds)} seeds")
    print()
    print(f"VERDICT: {verdict}")
    print(f"INTERPRETATION: {interpretation}")
    print()

    # R3 Classification update
    print("-" * 70)
    print("R3 CLASSIFICATION UPDATE")
    print("-" * 70)
    if verdict == "SURVIVES":
        r3_class = "Deployable (cluster-conditional models with real signal)"
    else:
        r3_class = "Research-Only (seed artifact, not biological signal)"
    print(f"C3 Classification: {r3_class}")

    # Save results
    output_file = RESULTS_DIR / "P1_C3_falsification.json"
    output = {
        "test": "C3 Seed Artifact Falsification",
        "verdict": verdict,
        "interpretation": interpretation,
        "r3_classification": r3_class,
        "seed_vs_pathogen": assoc,
        "within_seed_tests": {k: v for k, v in within_seed.items()},
        "signal_within_seeds": signal_within_seeds,
        "no_signal_within_seeds": no_signal_within_seeds,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print()
    print(f"Results saved to: {output_file}")

    return verdict


if __name__ == "__main__":
    verdict = main()
    sys.exit(0 if verdict in ["FALSIFIED", "SURVIVES", "LIKELY_ARTIFACT", "INCONCLUSIVE"] else 1)
