#!/usr/bin/env python3
"""Gene Ontology / Functional Validation of Replacement Calculus.

This script tests the central hypothesis:
> If two amino acids share functional roles, there should exist a
> low-cost morphism between them in the groupoid.

Validation approach:
1. Build hybrid groupoid from codon embeddings
2. Compute functional similarity between all amino acid pairs
3. Test correlations between:
   - Morphism existence vs functional similarity
   - Path cost vs functional distance
   - Groupoid clusters vs functional clusters
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import json
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score, precision_recall_curve, adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')

from functional_profiles import (
    AMINO_ACID_PROFILES,
    compute_functional_similarity_matrix,
    compute_functional_distance_matrix,
    cluster_by_function,
    get_catalytic_amino_acids,
)

from replacement_calculus.groupoids import find_escape_path, analyze_groupoid_structure

# Import hybrid groupoid builder
sys.path.insert(0, str(Path(__file__).parent.parent / 'integration'))
from hybrid_groupoid import (
    load_codon_embeddings,
    build_hybrid_groupoid,
    HybridValidityConfig,
)


# =============================================================================
# Validation Functions
# =============================================================================

def extract_groupoid_structure(
    groupoid,
    aa_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract morphism and path cost matrices from groupoid.

    Returns:
        morphism_matrix: Binary matrix (1 if morphism exists)
        cost_matrix: Path cost matrix (inf if no path)
        amino_acids: List of AA codes in order
    """
    amino_acids = sorted(aa_to_idx.keys())
    n = len(amino_acids)

    morphism_matrix = np.zeros((n, n))
    cost_matrix = np.full((n, n), np.inf)

    for i, aa1 in enumerate(amino_acids):
        for j, aa2 in enumerate(amino_acids):
            if aa1 == aa2:
                cost_matrix[i, j] = 0
                continue

            idx1 = aa_to_idx[aa1]
            idx2 = aa_to_idx[aa2]

            # Check direct morphism
            if groupoid.has_morphism(idx1, idx2):
                morphism_matrix[i, j] = 1

            # Find path and compute cost
            path = find_escape_path(groupoid, idx1, idx2)
            if path:
                cost = sum(m.cost for m in path)
                cost_matrix[i, j] = cost

    return morphism_matrix, cost_matrix, amino_acids


def validate_hypothesis_1(
    morphism_matrix: np.ndarray,
    similarity_matrix: np.ndarray,
    amino_acids: List[str],
) -> Dict:
    """H1: Functional Similarity Predicts Path Existence.

    Test: Does having a morphism correlate with high functional similarity?
    """
    n = len(amino_acids)

    # Extract upper triangle (avoid diagonal and duplicates)
    morphisms = []
    similarities = []

    for i in range(n):
        for j in range(i+1, n):
            morphisms.append(morphism_matrix[i, j] or morphism_matrix[j, i])
            similarities.append(similarity_matrix[i, j])

    morphisms = np.array(morphisms)
    similarities = np.array(similarities)

    # Point-biserial correlation (binary vs continuous)
    from scipy.stats import pointbiserialr
    corr, p_value = pointbiserialr(morphisms, similarities)

    # ROC-AUC
    if len(np.unique(morphisms)) > 1:
        auc = roc_auc_score(morphisms, similarities)
    else:
        auc = 0.5

    # Threshold analysis
    results = {
        'correlation': float(corr),
        'p_value': float(p_value),
        'auc_roc': float(auc),
        'n_pairs': len(morphisms),
        'n_with_morphism': int(morphisms.sum()),
        'n_without_morphism': int(len(morphisms) - morphisms.sum()),
    }

    # Mean similarity by morphism presence
    if morphisms.sum() > 0:
        results['mean_sim_with_morphism'] = float(np.mean(similarities[morphisms == 1]))
    if (morphisms == 0).sum() > 0:
        results['mean_sim_without_morphism'] = float(np.mean(similarities[morphisms == 0]))

    return results


def validate_hypothesis_2(
    cost_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    amino_acids: List[str],
) -> Dict:
    """H2: Path Cost Predicts Functional Distance.

    Test: Does lower path cost correspond to lower functional distance?
    """
    n = len(amino_acids)

    # Extract pairs with finite paths
    costs = []
    distances = []
    pairs = []

    for i in range(n):
        for j in range(i+1, n):
            if np.isfinite(cost_matrix[i, j]):
                costs.append(cost_matrix[i, j])
                distances.append(distance_matrix[i, j])
                pairs.append((amino_acids[i], amino_acids[j]))

    if len(costs) < 5:
        return {'error': 'Not enough pairs with paths', 'n_valid': len(costs)}

    costs = np.array(costs)
    distances = np.array(distances)

    # Correlations
    spearman_r, spearman_p = spearmanr(costs, distances)
    pearson_r, pearson_p = pearsonr(costs, distances)

    # Binned analysis
    median_cost = np.median(costs)
    low_cost_pairs = [(p, d) for c, d, p in zip(costs, distances, pairs) if c <= median_cost]
    high_cost_pairs = [(p, d) for c, d, p in zip(costs, distances, pairs) if c > median_cost]

    results = {
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'n_pairs': len(costs),
        'median_cost': float(median_cost),
    }

    if low_cost_pairs:
        results['mean_dist_low_cost'] = float(np.mean([d for _, d in low_cost_pairs]))
    if high_cost_pairs:
        results['mean_dist_high_cost'] = float(np.mean([d for _, d in high_cost_pairs]))

    # Top examples
    sorted_pairs = sorted(zip(costs, distances, pairs), key=lambda x: x[0])
    results['lowest_cost_pairs'] = [
        {'pair': p, 'cost': float(c), 'distance': float(d)}
        for c, d, p in sorted_pairs[:5]
    ]
    results['highest_cost_pairs'] = [
        {'pair': p, 'cost': float(c), 'distance': float(d)}
        for c, d, p in sorted_pairs[-5:]
    ]

    return results


def validate_hypothesis_3(
    groupoid,
    aa_to_idx: Dict[str, int],
    n_clusters: int = 5,
) -> Dict:
    """H3: GO-Derived Clusters Match Groupoid Structure.

    Test: Does clustering by function match groupoid connectivity?
    """
    amino_acids = sorted(aa_to_idx.keys())
    n = len(amino_acids)

    # Functional clusters
    functional_clusters = cluster_by_function(n_clusters)

    # Convert to label array
    functional_labels = np.zeros(n, dtype=int)
    for cluster_idx, (cluster_name, members) in enumerate(functional_clusters.items()):
        for aa in members:
            if aa in amino_acids:
                functional_labels[amino_acids.index(aa)] = cluster_idx

    # Groupoid-based clustering via path distances
    _, cost_matrix, _ = extract_groupoid_structure(groupoid, aa_to_idx)

    # Replace inf with large value for clustering
    cost_matrix_finite = cost_matrix.copy()
    max_finite = cost_matrix[np.isfinite(cost_matrix)].max()
    cost_matrix_finite[~np.isfinite(cost_matrix_finite)] = max_finite * 2

    # Make symmetric (take min of both directions)
    cost_matrix_sym = np.minimum(cost_matrix_finite, cost_matrix_finite.T)

    # Cluster using cost matrix
    condensed = squareform(cost_matrix_sym)
    Z = linkage(condensed, method='average')
    groupoid_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1

    # Adjusted Rand Index
    ari = adjusted_rand_score(functional_labels, groupoid_labels)

    # Cluster compositions
    func_cluster_contents = {}
    for name, members in functional_clusters.items():
        func_cluster_contents[name] = members

    groupoid_cluster_contents = {}
    for i in range(n_clusters):
        members = [amino_acids[j] for j in range(n) if groupoid_labels[j] == i]
        groupoid_cluster_contents[f"groupoid_cluster_{i}"] = members

    return {
        'adjusted_rand_index': float(ari),
        'n_clusters': n_clusters,
        'functional_clusters': func_cluster_contents,
        'groupoid_clusters': groupoid_cluster_contents,
    }


def validate_hypothesis_4(
    groupoid,
    aa_to_idx: Dict[str, int],
    amino_acids: List[str],
) -> Dict:
    """H4: Escape Paths Predict Annotation Transfer.

    Test: If A→B has low-cost path and A is catalytic, is B also catalytic?
    """
    # Get catalytic amino acids
    catalytic = set(get_catalytic_amino_acids())

    _, cost_matrix, _ = extract_groupoid_structure(groupoid, aa_to_idx)

    # For each pair, check if path cost predicts shared catalytic property
    predictions = []
    labels = []

    for i, aa1 in enumerate(amino_acids):
        for j, aa2 in enumerate(amino_acids):
            if i == j or not np.isfinite(cost_matrix[i, j]):
                continue

            # Prediction: low cost = same catalytic status
            cost = cost_matrix[i, j]

            # Ground truth: both catalytic or both non-catalytic
            same_catalytic = (aa1 in catalytic) == (aa2 in catalytic)

            predictions.append(-cost)  # Negative because lower cost = more similar
            labels.append(int(same_catalytic))

    if len(predictions) < 10:
        return {'error': 'Not enough pairs', 'n_pairs': len(predictions)}

    predictions = np.array(predictions)
    labels = np.array(labels)

    # ROC-AUC
    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, predictions)
    else:
        auc = 0.5

    # Precision at different recall levels
    precision, recall, thresholds = precision_recall_curve(labels, predictions)

    return {
        'auc_roc': float(auc),
        'n_pairs': len(predictions),
        'n_same_catalytic': int(labels.sum()),
        'catalytic_aas': list(catalytic),
        'precision_at_recall_50': float(precision[np.argmin(np.abs(recall - 0.5))]),
        'precision_at_recall_80': float(precision[np.argmin(np.abs(recall - 0.8))]),
    }


def validate_enzyme_class_prediction(
    groupoid,
    aa_to_idx: Dict[str, int],
    amino_acids: List[str],
) -> Dict:
    """Validate if path structure predicts enzyme class enrichment.

    For each EC class, check if amino acids enriched in that class
    are close in the groupoid.
    """
    ec_classes = ['ec1_oxidoreductase', 'ec2_transferase', 'ec3_hydrolase',
                  'ec4_lyase', 'ec5_isomerase', 'ec6_ligase']

    _, cost_matrix, _ = extract_groupoid_structure(groupoid, aa_to_idx)

    results = {}

    for ec in ec_classes:
        # Get enrichment values
        enrichments = {aa: getattr(AMINO_ACID_PROFILES[aa], ec) for aa in amino_acids}

        # Find top 5 AAs for this EC class
        top_aas = sorted(enrichments.items(), key=lambda x: -x[1])[:5]
        top_aa_names = [aa for aa, _ in top_aas]

        # Compute average path cost within top AAs
        within_costs = []
        for i, aa1 in enumerate(top_aa_names):
            for j, aa2 in enumerate(top_aa_names):
                if i < j:
                    idx1 = amino_acids.index(aa1)
                    idx2 = amino_acids.index(aa2)
                    if np.isfinite(cost_matrix[idx1, idx2]):
                        within_costs.append(cost_matrix[idx1, idx2])

        # Compute average path cost to other AAs
        between_costs = []
        other_aas = [aa for aa in amino_acids if aa not in top_aa_names]
        for aa1 in top_aa_names:
            for aa2 in other_aas:
                idx1 = amino_acids.index(aa1)
                idx2 = amino_acids.index(aa2)
                if np.isfinite(cost_matrix[idx1, idx2]):
                    between_costs.append(cost_matrix[idx1, idx2])

        results[ec] = {
            'top_aas': top_aa_names,
            'mean_within_cost': float(np.mean(within_costs)) if within_costs else None,
            'mean_between_cost': float(np.mean(between_costs)) if between_costs else None,
        }

        if within_costs and between_costs:
            # Effect size: are within-class AAs closer?
            results[ec]['separation'] = float(np.mean(between_costs) - np.mean(within_costs))

    return results


# =============================================================================
# Main Validation
# =============================================================================

def run_full_validation() -> Dict:
    """Run complete functional validation suite."""
    results = {}

    print("=" * 70)
    print("V5: GENE ONTOLOGY / FUNCTIONAL VALIDATION")
    print("=" * 70)

    # Step 1: Build groupoid
    print("\n1. Building hybrid groupoid...")
    embeddings, _ = load_codon_embeddings()

    config = HybridValidityConfig(
        max_embedding_distance=3.5,
        max_size_diff=40.0,
        max_hydrophobicity_diff=5.0,
        require_charge_compatible=False,
        require_polarity_compatible=False,
    )

    groupoid, aa_to_idx = build_hybrid_groupoid(embeddings, config)
    print(f"   Objects: {groupoid.n_objects()}")
    print(f"   Morphisms: {groupoid.n_morphisms()}")

    # Step 2: Compute functional matrices
    print("\n2. Computing functional similarity matrices...")
    similarity_matrix, amino_acids = compute_functional_similarity_matrix()
    distance_matrix, _ = compute_functional_distance_matrix()
    print(f"   Amino acids: {len(amino_acids)}")

    # Step 3: Extract groupoid structure
    print("\n3. Extracting groupoid structure...")
    morphism_matrix, cost_matrix, _ = extract_groupoid_structure(groupoid, aa_to_idx)

    n_paths = np.sum(np.isfinite(cost_matrix) & (cost_matrix > 0))
    print(f"   Pairs with paths: {n_paths}")

    # Step 4: Validate H1
    print("\n" + "=" * 70)
    print("HYPOTHESIS 1: Functional Similarity Predicts Path Existence")
    print("=" * 70)

    h1_results = validate_hypothesis_1(morphism_matrix, similarity_matrix, amino_acids)
    results['H1_morphism_similarity'] = h1_results

    print(f"   Correlation: r = {h1_results['correlation']:.4f} (p = {h1_results['p_value']:.2e})")
    print(f"   ROC-AUC: {h1_results['auc_roc']:.4f}")
    if 'mean_sim_with_morphism' in h1_results:
        print(f"   Mean similarity (with morphism): {h1_results['mean_sim_with_morphism']:.3f}")
    if 'mean_sim_without_morphism' in h1_results:
        print(f"   Mean similarity (no morphism): {h1_results['mean_sim_without_morphism']:.3f}")

    # Step 5: Validate H2
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: Path Cost Predicts Functional Distance")
    print("=" * 70)

    h2_results = validate_hypothesis_2(cost_matrix, distance_matrix, amino_acids)
    results['H2_cost_distance'] = h2_results

    if 'error' not in h2_results:
        print(f"   Spearman r = {h2_results['spearman_r']:.4f} (p = {h2_results['spearman_p']:.2e})")
        print(f"   Pearson r = {h2_results['pearson_r']:.4f} (p = {h2_results['pearson_p']:.2e})")
        print(f"   Mean distance (low cost): {h2_results.get('mean_dist_low_cost', 'N/A'):.3f}")
        print(f"   Mean distance (high cost): {h2_results.get('mean_dist_high_cost', 'N/A'):.3f}")

        print("\n   Lowest cost pairs (closest in groupoid):")
        for item in h2_results['lowest_cost_pairs']:
            print(f"     {item['pair'][0]}-{item['pair'][1]}: cost={item['cost']:.2f}, func_dist={item['distance']:.2f}")

        print("\n   Highest cost pairs (farthest in groupoid):")
        for item in h2_results['highest_cost_pairs']:
            print(f"     {item['pair'][0]}-{item['pair'][1]}: cost={item['cost']:.2f}, func_dist={item['distance']:.2f}")

    # Step 6: Validate H3
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: Functional Clusters Match Groupoid Structure")
    print("=" * 70)

    h3_results = validate_hypothesis_3(groupoid, aa_to_idx)
    results['H3_cluster_match'] = h3_results

    print(f"   Adjusted Rand Index: {h3_results['adjusted_rand_index']:.4f}")
    print("\n   Functional clusters:")
    for name, members in h3_results['functional_clusters'].items():
        print(f"     {name}: {', '.join(members)}")
    print("\n   Groupoid-derived clusters:")
    for name, members in h3_results['groupoid_clusters'].items():
        print(f"     {name}: {', '.join(members)}")

    # Step 7: Validate H4
    print("\n" + "=" * 70)
    print("HYPOTHESIS 4: Escape Paths Predict Annotation Transfer")
    print("=" * 70)

    h4_results = validate_hypothesis_4(groupoid, aa_to_idx, amino_acids)
    results['H4_annotation_transfer'] = h4_results

    if 'error' not in h4_results:
        print(f"   ROC-AUC (catalytic prediction): {h4_results['auc_roc']:.4f}")
        print(f"   Precision at 50% recall: {h4_results['precision_at_recall_50']:.4f}")
        print(f"   Precision at 80% recall: {h4_results['precision_at_recall_80']:.4f}")
        print(f"   Catalytic amino acids: {', '.join(h4_results['catalytic_aas'])}")

    # Step 8: Enzyme class analysis
    print("\n" + "=" * 70)
    print("ENZYME CLASS ANALYSIS")
    print("=" * 70)

    ec_results = validate_enzyme_class_prediction(groupoid, aa_to_idx, amino_acids)
    results['enzyme_class_analysis'] = ec_results

    for ec, data in ec_results.items():
        ec_name = ec.replace('_', ' ').title()
        print(f"\n   {ec_name}:")
        print(f"     Top AAs: {', '.join(data['top_aas'])}")
        if data['mean_within_cost'] is not None:
            print(f"     Within-class cost: {data['mean_within_cost']:.2f}")
        if data['mean_between_cost'] is not None:
            print(f"     Between-class cost: {data['mean_between_cost']:.2f}")
        if data.get('separation') is not None:
            sep = data['separation']
            direction = "CLOSER" if sep > 0 else "FARTHER"
            print(f"     Separation: {sep:+.2f} ({direction} within class)")

    # Step 9: Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print("\n   | Hypothesis | Metric | Value | Interpretation |")
    print("   |------------|--------|-------|----------------|")
    print(f"   | H1: Similarity→Morphism | AUC | {h1_results['auc_roc']:.3f} | {'SUPPORTS' if h1_results['auc_roc'] > 0.6 else 'WEAK'} |")
    if 'error' not in h2_results:
        print(f"   | H2: Cost→Distance | Spearman | {h2_results['spearman_r']:.3f} | {'SUPPORTS' if h2_results['spearman_r'] > 0.3 else 'WEAK'} |")
    print(f"   | H3: Cluster Match | ARI | {h3_results['adjusted_rand_index']:.3f} | {'SUPPORTS' if h3_results['adjusted_rand_index'] > 0.3 else 'WEAK'} |")
    if 'error' not in h4_results:
        print(f"   | H4: Annotation Transfer | AUC | {h4_results['auc_roc']:.3f} | {'SUPPORTS' if h4_results['auc_roc'] > 0.6 else 'WEAK'} |")

    # Save results
    output_path = Path(__file__).parent / 'functional_validation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n   Results saved to: {output_path}")
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_full_validation()
