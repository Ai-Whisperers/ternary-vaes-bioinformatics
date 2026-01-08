#!/usr/bin/env python3
"""P1 C3 Enhanced Clustering: Cross-Domain Methodology Integration.

Imports from:
- Rojas (arbovirus): Hyperbolic variance for identifying "cryptically conserved" clusters
- Colbes (DDG): Arrow-flip threshold detection for regime routing

This script improves pathogen-specific signal detection by:
1. Using hyperbolic distance (not Euclidean) for clustering in Poincare ball
2. Computing dual metrics: hyperbolic variance vs feature entropy
3. Detecting thresholds where cluster-conditional outperforms global prediction
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PACKAGE_DIR / "results" / "validation_batch"
PROJECT_ROOT = PACKAGE_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# HYPERBOLIC GEOMETRY (from Rojas framework)
# =============================================================================

def poincare_distance(u: np.ndarray, v: np.ndarray, c: float = 1.0) -> float:
    """Compute hyperbolic distance in Poincare ball.

    d(u,v) = (2/sqrt(c)) * arctanh(sqrt(c) * ||-u + v||_M)

    where ||.||_M is the Mobius addition norm.
    """
    # Clamp to ball interior
    eps = 1e-5
    u = u / (np.linalg.norm(u) + eps) * min(np.linalg.norm(u), 1 - eps)
    v = v / (np.linalg.norm(v) + eps) * min(np.linalg.norm(v), 1 - eps)

    # Mobius addition: -u + v
    u_sq = np.sum(u ** 2)
    v_sq = np.sum(v ** 2)
    uv = np.sum(u * v)

    num = (1 + 2 * c * (-uv) + c * v_sq) * u + (1 - c * u_sq) * v
    denom = 1 + 2 * c * (-uv) + c ** 2 * u_sq * v_sq

    mobius_add = num / (denom + eps)
    mobius_norm = np.linalg.norm(mobius_add)

    # Clamp for numerical stability
    mobius_norm = np.clip(mobius_norm, 0, 1 - eps)

    dist = (2 / np.sqrt(c)) * np.arctanh(np.sqrt(c) * mobius_norm)
    return float(dist)


def frechet_mean_poincare(points: np.ndarray, c: float = 1.0, max_iter: int = 100) -> np.ndarray:
    """Compute Frechet mean (centroid) in Poincare ball.

    Uses iterative gradient descent in hyperbolic space.
    """
    if len(points) == 0:
        return np.zeros(points.shape[1])

    # Initialize at Euclidean mean (projected to ball)
    mean = np.mean(points, axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0.95:
        mean = mean / norm * 0.95

    for _ in range(max_iter):
        # Compute gradient: sum of log maps from mean to each point
        grad = np.zeros_like(mean)
        for p in points:
            # Simplified: use direction toward point weighted by distance
            d = poincare_distance(mean, p, c)
            direction = p - mean
            if np.linalg.norm(direction) > 1e-8:
                direction = direction / np.linalg.norm(direction)
            grad += d * direction

        grad /= len(points)

        # Update mean with small step
        mean = mean + 0.1 * grad

        # Project back to ball
        norm = np.linalg.norm(mean)
        if norm > 0.95:
            mean = mean / norm * 0.95

    return mean


def compute_hyperbolic_variance(embeddings: np.ndarray, c: float = 1.0) -> float:
    """Compute variance of distances from hyperbolic centroid.

    Low variance = tightly clustered in hyperbolic space
    High variance = spread out
    """
    if len(embeddings) < 2:
        return 0.0

    centroid = frechet_mean_poincare(embeddings, c)
    distances = [poincare_distance(emb, centroid, c) for emb in embeddings]
    return float(np.var(distances))


def compute_feature_entropy(values: np.ndarray, n_bins: int = 10) -> float:
    """Compute Shannon entropy of feature distribution."""
    hist, _ = np.histogram(values, bins=n_bins, density=True)
    hist = hist[hist > 0]  # Remove zeros
    if len(hist) == 0:
        return 0.0
    hist = hist / hist.sum()  # Normalize
    return float(-np.sum(hist * np.log2(hist + 1e-10)))


# =============================================================================
# HYPERBOLIC KMEANS CLUSTERING
# =============================================================================

def hyperbolic_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = 5,
    max_iter: int = 100,
    c: float = 1.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """KMeans clustering using hyperbolic distance in Poincare ball.

    Returns:
        labels: Cluster assignment for each point
        centroids: Cluster centroids in hyperbolic space
        metrics: Per-cluster hyperbolic variance and other metrics
    """
    np.random.seed(random_state)
    n_samples, n_features = embeddings.shape

    # Initialize centroids using k-means++ style (but with hyperbolic distance)
    centroid_indices = [np.random.randint(n_samples)]
    for _ in range(1, n_clusters):
        # Compute distance to nearest centroid for each point
        dists = np.array([
            min(poincare_distance(embeddings[i], embeddings[ci], c)
                for ci in centroid_indices)
            for i in range(n_samples)
        ])
        # Select next centroid proportional to squared distance
        probs = dists ** 2
        probs = probs / probs.sum()
        next_idx = np.random.choice(n_samples, p=probs)
        centroid_indices.append(next_idx)

    centroids = embeddings[centroid_indices].copy()

    # Iterate
    for iteration in range(max_iter):
        # Assign points to nearest centroid
        labels = np.array([
            np.argmin([poincare_distance(emb, c_pt, c) for c_pt in centroids])
            for emb in embeddings
        ])

        # Update centroids
        new_centroids = []
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                new_centroids.append(frechet_mean_poincare(embeddings[mask], c))
            else:
                new_centroids.append(centroids[k])
        new_centroids = np.array(new_centroids)

        # Check convergence
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    # Compute per-cluster metrics
    metrics = {}
    for k in range(n_clusters):
        mask = labels == k
        cluster_embs = embeddings[mask]

        if len(cluster_embs) > 1:
            hyp_var = compute_hyperbolic_variance(cluster_embs, c)
            # Also compute Euclidean variance for comparison
            euc_var = float(np.mean(np.var(cluster_embs, axis=0)))
        else:
            hyp_var = 0.0
            euc_var = 0.0

        metrics[k] = {
            "n": int(mask.sum()),
            "hyperbolic_variance": hyp_var,
            "euclidean_variance": euc_var,
        }

    return labels, centroids, metrics


# =============================================================================
# ARROW-FLIP THRESHOLD DETECTION (from Colbes framework)
# =============================================================================

def compute_pathogen_separation(
    candidates: List[dict],
    mask: np.ndarray,
    cluster_labels: Optional[np.ndarray] = None,
) -> float:
    """Compute pathogen separation score for a subset of candidates.

    Uses Kruskal-Wallis H statistic normalized by sample size.
    Higher = better separation.
    """
    subset = [c for c, m in zip(candidates, mask) if m]
    if len(subset) < 10:
        return 0.0

    # Group MIC values by pathogen
    by_pathogen = defaultdict(list)
    for c in subset:
        by_pathogen[c["pathogen"]].append(c["mic_pred"])

    # Need at least 2 pathogens with data
    pathogens_with_data = [p for p, vals in by_pathogen.items() if len(vals) >= 3]
    if len(pathogens_with_data) < 2:
        return 0.0

    # Kruskal-Wallis test
    groups = [by_pathogen[p] for p in pathogens_with_data]
    try:
        h_stat, p_value = stats.kruskal(*groups)
        # Normalize by sample size for comparability
        effect_size = h_stat / len(subset)
        return float(effect_size) if p_value < 0.05 else 0.0
    except Exception:
        return 0.0


def detect_arrow_flip_thresholds(
    candidates: List[dict],
    cluster_labels: np.ndarray,
    features: List[str] = None,
) -> Dict[str, Dict]:
    """Find thresholds where cluster-conditional outperforms global prediction.

    Implements Colbes methodology: grid search over feature values to find
    discontinuities in prediction accuracy.
    """
    if features is None:
        features = ["length", "net_charge", "hydrophobicity"]

    # Add charge_density if not present
    for c in candidates:
        if "charge_density" not in c:
            c["charge_density"] = c["net_charge"] / c["length"] if c["length"] > 0 else 0

    if "charge_density" not in features:
        features.append("charge_density")

    thresholds = {}
    global_mask = np.ones(len(candidates), dtype=bool)
    global_separation = compute_pathogen_separation(candidates, global_mask, cluster_labels)

    for feature in features:
        try:
            values = np.array([c[feature] for c in candidates])
        except KeyError:
            continue

        best_threshold = None
        best_improvement = 0.0
        best_above_sep = 0.0
        best_below_sep = 0.0

        # Grid search over percentiles
        for percentile in range(10, 91, 5):
            threshold = np.percentile(values, percentile)

            below = values < threshold
            above = ~below

            if below.sum() < 20 or above.sum() < 20:
                continue

            sep_below = compute_pathogen_separation(candidates, below, cluster_labels)
            sep_above = compute_pathogen_separation(candidates, above, cluster_labels)

            # Improvement = best of splits - global
            improvement = max(sep_below, sep_above) - global_separation

            if improvement > best_improvement:
                best_improvement = improvement
                best_threshold = float(threshold)
                best_below_sep = sep_below
                best_above_sep = sep_above

        thresholds[feature] = {
            "threshold": best_threshold,
            "improvement": best_improvement,
            "below_separation": best_below_sep,
            "above_separation": best_above_sep,
            "global_separation": global_separation,
            "is_significant": best_improvement > 0.02,  # 2% improvement threshold
        }

    return thresholds


# =============================================================================
# DUAL METRIC ANALYSIS (Rojas insight: hyperbolic variance vs entropy)
# =============================================================================

def analyze_dual_metrics(
    candidates: List[dict],
    cluster_labels: np.ndarray,
    embeddings: Optional[np.ndarray] = None,
) -> Dict[int, Dict]:
    """Analyze clusters using dual metrics: hyperbolic variance AND feature entropy.

    Cryptically conserved = LOW hyperbolic variance + HIGH feature entropy
    These clusters have the strongest pathogen-specific signal.
    """
    results = {}

    for k in sorted(set(cluster_labels)):
        mask = cluster_labels == k
        cluster_candidates = [c for c, m in zip(candidates, mask) if m]

        if len(cluster_candidates) < 5:
            continue

        # Feature entropy (how diverse are the peptide features?)
        lengths = [c["length"] for c in cluster_candidates]
        charges = [c["net_charge"] for c in cluster_candidates]
        hydros = [c["hydrophobicity"] for c in cluster_candidates]

        entropy_length = compute_feature_entropy(np.array(lengths))
        entropy_charge = compute_feature_entropy(np.array(charges))
        entropy_hydro = compute_feature_entropy(np.array(hydros))
        mean_entropy = (entropy_length + entropy_charge + entropy_hydro) / 3

        # Hyperbolic variance (if embeddings available)
        if embeddings is not None:
            cluster_embs = embeddings[mask]
            hyp_var = compute_hyperbolic_variance(cluster_embs)
        else:
            # Fallback: use Euclidean variance of raw features
            features = np.array([[c["length"], c["net_charge"], c["hydrophobicity"]]
                                 for c in cluster_candidates])
            hyp_var = float(np.mean(np.var(features, axis=0)))

        # Pathogen separation (from original C3)
        pathogen_counts = defaultdict(int)
        mic_by_pathogen = defaultdict(list)
        for c in cluster_candidates:
            pathogen_counts[c["pathogen"]] += 1
            mic_by_pathogen[c["pathogen"]].append(c["mic_pred"])

        # Kruskal-Wallis for pathogen separation
        groups = [v for v in mic_by_pathogen.values() if len(v) >= 3]
        if len(groups) >= 2:
            h_stat, p_value = stats.kruskal(*groups)
            between_std = np.std([np.mean(v) for v in groups])
            within_std = np.mean([np.std(v) for v in groups])
            effect_ratio = between_std / (within_std + 1e-8)
        else:
            h_stat, p_value = 0, 1
            effect_ratio = 0

        # Classify cluster type
        if hyp_var < 0.05 and mean_entropy > 0.5:
            cluster_type = "CRYPTIC"  # Low var, high entropy = strongest signal
        elif hyp_var < 0.05 and mean_entropy < 0.5:
            cluster_type = "CONSERVED"  # Low var, low entropy
        elif hyp_var > 0.1:
            cluster_type = "VARIABLE"  # High variance
        else:
            cluster_type = "MIXED"

        results[k] = {
            "n": len(cluster_candidates),
            "n_pathogens": len(pathogen_counts),
            "hyperbolic_variance": round(hyp_var, 4),
            "mean_entropy": round(mean_entropy, 3),
            "entropy_breakdown": {
                "length": round(entropy_length, 3),
                "charge": round(entropy_charge, 3),
                "hydrophobicity": round(entropy_hydro, 3),
            },
            "kruskal_h": round(h_stat, 2),
            "p_value": p_value,
            "effect_ratio": round(effect_ratio, 3),
            "cluster_type": cluster_type,
            "has_pathogen_signal": p_value < 0.01 and effect_ratio > 0.5,
        }

    return results


# =============================================================================
# DATA LOADING
# =============================================================================

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


def try_load_vae_embeddings(candidates: List[dict]) -> Optional[np.ndarray]:
    """Try to load VAE embeddings for candidates."""
    try:
        from scripts.predict_mic import PeptideMICPredictor

        checkpoint = PACKAGE_DIR / "checkpoints_definitive" / "best_production.pt"
        if not checkpoint.exists():
            print("  Checkpoint not found, using fallback features")
            return None

        predictor = PeptideMICPredictor(checkpoint_path=str(checkpoint), verbose=False)

        embeddings = []
        for c in candidates:
            result = predictor.predict(c["sequence"])
            if result.latent_vector is not None:
                embeddings.append(result.latent_vector)
            else:
                # Fallback: use normalized raw features
                embeddings.append(np.array([
                    c["length"] / 30,
                    c["net_charge"] / 10,
                    c["hydrophobicity"],
                ]))

        return np.array(embeddings)

    except Exception as e:
        print(f"  Could not load VAE embeddings: {e}")
        return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("P1 C3 ENHANCED CLUSTERING: Cross-Domain Methodology Integration")
    print("=" * 70)
    print()
    print("Imports from:")
    print("  - Rojas (arbovirus): Hyperbolic variance for 'cryptic conservation'")
    print("  - Colbes (DDG): Arrow-flip threshold detection for regime routing")
    print()

    # Load data
    print("Loading candidates...")
    candidates = load_candidates(RESULTS_DIR)
    if not candidates:
        print("ERROR: No candidates found")
        sys.exit(1)
    print(f"  Loaded {len(candidates)} candidates")

    # Try to load VAE embeddings
    print("\nLoading VAE embeddings...")
    embeddings = try_load_vae_embeddings(candidates)
    using_vae = embeddings is not None

    if not using_vae:
        print("  Using fallback: raw features (length, charge, hydrophobicity)")
        embeddings = np.array([
            [c["length"] / 30, c["net_charge"] / 10, c["hydrophobicity"]]
            for c in candidates
        ])

    # Normalize embeddings
    emb_mean = embeddings.mean(axis=0)
    emb_std = embeddings.std(axis=0) + 1e-8
    embeddings_norm = (embeddings - emb_mean) / emb_std

    # ==========================================================================
    # COMPARISON: Euclidean vs Hyperbolic Clustering
    # ==========================================================================
    print()
    print("-" * 70)
    print("COMPARISON: Euclidean vs Hyperbolic Clustering")
    print("-" * 70)

    # Euclidean KMeans (original C3)
    print("\nEuclidean KMeans (original C3)...")
    kmeans_euc = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels_euc = kmeans_euc.fit_predict(embeddings_norm)

    # Hyperbolic KMeans (Rojas method)
    print("Hyperbolic KMeans (Rojas method)...")
    labels_hyp, centroids_hyp, metrics_hyp = hyperbolic_kmeans(
        embeddings_norm, n_clusters=5, random_state=42
    )

    # ==========================================================================
    # DUAL METRIC ANALYSIS
    # ==========================================================================
    print()
    print("-" * 70)
    print("DUAL METRIC ANALYSIS (Hyperbolic Variance vs Feature Entropy)")
    print("-" * 70)

    print("\n[Euclidean Clusters]")
    dual_euc = analyze_dual_metrics(candidates, labels_euc, embeddings_norm)
    for k, m in sorted(dual_euc.items()):
        print(f"  Cluster {k}: N={m['n']:3d}, HypVar={m['hyperbolic_variance']:.4f}, "
              f"Entropy={m['mean_entropy']:.2f}, Type={m['cluster_type']:9s}, "
              f"Signal={'YES' if m['has_pathogen_signal'] else 'NO'}")

    print("\n[Hyperbolic Clusters]")
    dual_hyp = analyze_dual_metrics(candidates, labels_hyp, embeddings_norm)
    for k, m in sorted(dual_hyp.items()):
        print(f"  Cluster {k}: N={m['n']:3d}, HypVar={m['hyperbolic_variance']:.4f}, "
              f"Entropy={m['mean_entropy']:.2f}, Type={m['cluster_type']:9s}, "
              f"Signal={'YES' if m['has_pathogen_signal'] else 'NO'}")

    # Count cryptic clusters (strongest signal type)
    cryptic_euc = sum(1 for m in dual_euc.values() if m["cluster_type"] == "CRYPTIC")
    cryptic_hyp = sum(1 for m in dual_hyp.values() if m["cluster_type"] == "CRYPTIC")
    signal_euc = sum(1 for m in dual_euc.values() if m["has_pathogen_signal"])
    signal_hyp = sum(1 for m in dual_hyp.values() if m["has_pathogen_signal"])

    print(f"\nCryptic clusters: Euclidean={cryptic_euc}, Hyperbolic={cryptic_hyp}")
    print(f"Signal clusters:  Euclidean={signal_euc}, Hyperbolic={signal_hyp}")

    # ==========================================================================
    # ARROW-FLIP THRESHOLD DETECTION
    # ==========================================================================
    print()
    print("-" * 70)
    print("ARROW-FLIP THRESHOLD DETECTION (Colbes method)")
    print("-" * 70)

    # Use hyperbolic clusters for threshold detection
    thresholds = detect_arrow_flip_thresholds(candidates, labels_hyp)

    print("\nFeature         | Threshold | Improvement | Significant?")
    print("-" * 60)
    for feat, t in sorted(thresholds.items(), key=lambda x: -x[1]["improvement"]):
        thresh = f"{t['threshold']:.2f}" if t['threshold'] else "N/A"
        imp = f"+{t['improvement']:.3f}" if t['improvement'] > 0 else f"{t['improvement']:.3f}"
        sig = "YES" if t["is_significant"] else "NO"
        print(f"{feat:15s} | {thresh:9s} | {imp:11s} | {sig}")

    # Find best threshold for regime routing
    best_feature = max(thresholds, key=lambda f: thresholds[f]["improvement"])
    best_threshold = thresholds[best_feature]

    print(f"\nBest arrow-flip: {best_feature} @ {best_threshold['threshold']:.2f}")
    print(f"  Improvement over global: {best_threshold['improvement']:.3f}")

    # ==========================================================================
    # REGIME ROUTING RULES
    # ==========================================================================
    print()
    print("-" * 70)
    print("REGIME ROUTING RULES")
    print("-" * 70)

    if best_threshold["is_significant"]:
        thresh_val = best_threshold["threshold"]
        print(f"\nIF {best_feature} > {thresh_val:.2f}:")
        print(f"    USE cluster_conditional (hyperbolic clusters with signal)")
        print(f"ELSE:")
        print(f"    USE global_model")

        # Identify which clusters have signal
        signal_clusters = [k for k, m in dual_hyp.items() if m["has_pathogen_signal"]]
        print(f"\nSignal clusters: {signal_clusters}")
    else:
        print("\nNo significant arrow-flip threshold found.")
        print("Recommendation: Use cluster-conditional for ALL short peptides (clusters 1, 3, 4)")

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    output_file = RESULTS_DIR / "P1_C3_enhanced_results.json"

    # Convert numpy int keys to Python int for JSON serialization
    dual_euc_clean = {int(k): v for k, v in dual_euc.items()}
    dual_hyp_clean = {int(k): v for k, v in dual_hyp.items()}

    results = {
        "using_vae_embeddings": using_vae,
        "n_candidates": len(candidates),
        "comparison": {
            "euclidean_signal_clusters": int(signal_euc),
            "hyperbolic_signal_clusters": int(signal_hyp),
            "euclidean_cryptic_clusters": int(cryptic_euc),
            "hyperbolic_cryptic_clusters": int(cryptic_hyp),
        },
        "dual_metrics_euclidean": dual_euc_clean,
        "dual_metrics_hyperbolic": dual_hyp_clean,
        "arrow_flip_thresholds": thresholds,
        "best_threshold": {
            "feature": best_feature,
            "value": best_threshold["threshold"],
            "improvement": best_threshold["improvement"],
            "is_significant": best_threshold["is_significant"],
        },
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print()
    print(f"Results saved to: {output_file}")

    # ==========================================================================
    # VERDICT
    # ==========================================================================
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    if signal_hyp > signal_euc:
        print("Hyperbolic clustering OUTPERFORMS Euclidean")
    elif signal_hyp == signal_euc:
        print("Hyperbolic clustering MATCHES Euclidean")
    else:
        print("Euclidean clustering still better (unexpected)")

    if cryptic_hyp > 0:
        print(f"Found {cryptic_hyp} CRYPTICALLY CONSERVED cluster(s) - strongest pathogen signal")

    if best_threshold["is_significant"]:
        print(f"Arrow-flip detected at {best_feature}={best_threshold['threshold']:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
