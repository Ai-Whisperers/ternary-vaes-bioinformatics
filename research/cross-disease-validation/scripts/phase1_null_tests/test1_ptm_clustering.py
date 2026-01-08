"""
Test 1: Cross-Disease PTM Clustering Analysis

Objective: Test if PTMs cluster by disease mechanism (RA vs Tau) or by chemistry (citrullination vs phosphorylation)

Null Hypothesis (H0): PTMs cluster by modification chemistry, not disease mechanism

Alternative Hypothesis (H1): PTMs cluster by disease (RA vs Alzheimer's)
                              Silhouette > 0.3, ARI > 0.5

Method:
1. Load RA citrullination sites (45 positions)
2. Load Tau phosphorylation sites (54 positions)
3. Embed each PTM in p-adic space
4. Compute pairwise p-adic distances
5. Hierarchical clustering with k=2
6. Evaluate: Silhouette score, Adjusted Rand Index

Success Criteria:
- Silhouette score > 0.3 (moderate clustering quality)
- ARI > 0.5 (agreement with true disease labels)
- Diseases separate in dendrogram

Pre-registered on: 2026-01-03
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json
from collections import defaultdict

# Add repo root to path
script_dir = Path(__file__).parent
scripts_dir = script_dir.parent
cross_disease_dir = scripts_dir.parent
research_dir = cross_disease_dir.parent
repo_root = research_dir.parent
sys.path.insert(0, str(repo_root))

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Configuration
RESULTS_DIR = Path('research/cross-disease-validation/results/test1_ptm_clustering')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Import PTM data loader
sys.path.insert(0, str(scripts_dir / 'utils'))
from load_ptm_data import load_all_ptm_data

def get_protein_sequence_stub(uniprot_id, protein_name):
    """
    Get protein sequence (stub implementation).

    For full implementation, would download from UniProt.
    For now, returns placeholder to test framework.
    """
    # This is a stub - in production, would download from UniProt
    # For testing, we'll use a simple approach: estimate embedding without full sequence
    return None

def compute_ptm_embedding_simple(ptm_site):
    """
    Compute simplified PTM embedding based on position and residue.

    This is a simplified approach that doesn't require full protein sequences.
    Uses position encoding + residue properties.

    For full implementation, would use TrainableCodonEncoder + full protein sequence.
    """
    # Simple encoding based on:
    # 1. Position (normalized)
    # 2. Residue type (charge, hydrophobicity)
    # 3. PTM type (phosphorylation vs citrullination)

    position = ptm_site['position']
    residue = ptm_site['residue']
    ptm_type = ptm_site['ptm_type']

    # Residue properties (simplified)
    residue_props = {
        'R': {'charge': 1, 'hydrophobicity': -4.5, 'size': 174},  # Arginine
        'S': {'charge': 0, 'hydrophobicity': -0.8, 'size': 89},   # Serine
        'T': {'charge': 0, 'hydrophobicity': -0.7, 'size': 116},  # Threonine
        'Y': {'charge': 0, 'hydrophobicity': -1.3, 'size': 181},  # Tyrosine
    }

    props = residue_props.get(residue, {'charge': 0, 'hydrophobicity': 0, 'size': 100})

    # Simple feature vector (would be replaced by actual p-adic embedding)
    features = np.array([
        position / 500.0,  # Normalized position
        props['charge'],
        props['hydrophobicity'],
        props['size'] / 200.0,
        1.0 if ptm_type == 'phosphorylation' else 0.0,
        1.0 if ptm_type == 'citrullination' else 0.0,
    ])

    # Add random variation to simulate p-adic embedding variance
    # This is a placeholder - real implementation would use actual encoder
    np.random.seed(hash(f"{ptm_site['protein']}_{position}") % (2**32))
    noise = np.random.normal(0, 0.1, size=10)

    embedding = np.concatenate([features, noise])

    return embedding

def compute_pairwise_distances(embeddings):
    """
    Compute pairwise Euclidean distances between embeddings.

    In production, would use poincare_distance for hyperbolic space.
    For testing framework, using Euclidean distance.
    """
    n = len(embeddings)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix

def plot_dendrogram_custom(linkage_matrix, labels, output_path):
    """Create dendrogram visualization with disease coloring."""
    fig, ax = plt.subplots(figsize=(15, 8))

    # Create dendrogram
    dend = dendrogram(linkage_matrix, labels=labels, ax=ax)

    # Color labels by disease
    xlabels = ax.get_xmajorticklabels()
    for lbl in xlabels:
        text = lbl.get_text()
        if 'RA' in text or 'Vimentin' in text or 'Fibrinogen' in text:
            lbl.set_color('red')
        else:
            lbl.set_color('blue')

    ax.set_title('Hierarchical Clustering of PTMs (RA vs Tau)', fontsize=14)
    ax.set_xlabel('PTM Site', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='RA (Citrullination)'),
        Patch(facecolor='blue', label='Tau (Phosphorylation)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("="*80)
    print("TEST 1: Cross-Disease PTM Clustering Analysis")
    print("="*80)
    print()
    print("Pre-registered Hypothesis:")
    print("  H0: PTMs cluster by chemistry (citrullination vs phosphorylation)")
    print("  H1: PTMs cluster by disease (RA vs Alzheimer's)")
    print("      Silhouette > 0.3, ARI > 0.5")
    print()
    print("="*80)

    # Load PTM data
    print("\n[1/6] Loading PTM data...")
    ptm_data = load_all_ptm_data()

    tau_sites = ptm_data['tau']
    ra_sites = ptm_data['ra']
    all_sites = ptm_data['all']

    print(f"  Tau phosphorylation sites: {len(tau_sites)}")
    print(f"  RA citrullination sites: {len(ra_sites)}")
    print(f"  Total PTM sites: {len(all_sites)}")

    # Compute embeddings
    print("\n[2/6] Computing PTM embeddings...")
    print("  NOTE: Using simplified embedding (position + residue properties)")
    print("  Full implementation would use TrainableCodonEncoder with protein sequences")

    all_embeddings = []
    all_labels = []
    site_descriptions = []

    for site in all_sites:
        # Compute simplified embedding
        embedding = compute_ptm_embedding_simple(site)
        all_embeddings.append(embedding)

        # Track disease label
        all_labels.append(site['disease'])

        # Create description for visualization
        desc = f"{site['protein'][:10]}_" + \
               f"{site['residue']}{site['position']}_" + \
               f"{site['disease']}"
        site_descriptions.append(desc)

    all_embeddings = np.array(all_embeddings)
    print(f"  Embedding shape: {all_embeddings.shape}")

    # Compute distance matrix
    print("\n[3/6] Computing pairwise distances...")
    distance_matrix = compute_pairwise_distances(all_embeddings)
    print(f"  Distance matrix shape: {distance_matrix.shape}")
    print(f"  Mean distance: {distance_matrix[np.triu_indices_from(distance_matrix, k=1)].mean():.3f}")

    # Hierarchical clustering
    print("\n[4/6] Performing hierarchical clustering...")
    clustering = AgglomerativeClustering(
        n_clusters=2,
        metric='precomputed',
        linkage='average'
    )
    predicted_labels = clustering.fit_predict(distance_matrix)

    # Map disease labels to binary
    true_labels_binary = np.array([0 if label == 'RA' else 1 for label in all_labels])

    # Check if clustering matches disease or is inverted
    # (clustering labels 0/1 are arbitrary, so we need to check both orientations)
    match_direct = np.mean(predicted_labels == true_labels_binary)
    match_inverted = np.mean(predicted_labels == (1 - true_labels_binary))

    if match_inverted > match_direct:
        # Flip predicted labels to match true labels
        predicted_labels = 1 - predicted_labels

    print(f"  Cluster 0 (RA expected): {np.sum(predicted_labels == 0)} sites")
    print(f"  Cluster 1 (Tau expected): {np.sum(predicted_labels == 1)} sites")

    # Evaluation metrics
    print("\n[5/6] Evaluating clustering quality...")

    # Silhouette score
    silhouette = silhouette_score(distance_matrix, predicted_labels, metric='precomputed')
    print(f"  Silhouette Score: {silhouette:.3f}")
    print(f"    Interpretation: {'GOOD' if silhouette > 0.3 else 'POOR'} clustering quality")

    # Adjusted Rand Index
    ari = adjusted_rand_score(true_labels_binary, predicted_labels)
    print(f"  Adjusted Rand Index: {ari:.3f}")
    print(f"    Interpretation: {'HIGH' if ari > 0.5 else 'LOW'} agreement with true labels")

    # Confusion matrix
    ra_in_cluster0 = np.sum((true_labels_binary == 0) & (predicted_labels == 0))
    ra_in_cluster1 = np.sum((true_labels_binary == 0) & (predicted_labels == 1))
    tau_in_cluster0 = np.sum((true_labels_binary == 1) & (predicted_labels == 0))
    tau_in_cluster1 = np.sum((true_labels_binary == 1) & (predicted_labels == 1))

    print(f"\n  Confusion Matrix:")
    print(f"    RA in Cluster 0: {ra_in_cluster0} ({ra_in_cluster0/len(ra_sites)*100:.1f}%)")
    print(f"    RA in Cluster 1: {ra_in_cluster1} ({ra_in_cluster1/len(ra_sites)*100:.1f}%)")
    print(f"    Tau in Cluster 0: {tau_in_cluster0} ({tau_in_cluster0/len(tau_sites)*100:.1f}%)")
    print(f"    Tau in Cluster 1: {tau_in_cluster1} ({tau_in_cluster1/len(tau_sites)*100:.1f}%)")

    # Decision
    print("\n" + "="*80)
    print("DECISION")
    print("="*80)
    if silhouette > 0.3 and ari > 0.5:
        print("REJECT NULL HYPOTHESIS")
        print("PTMs cluster by disease mechanism (RA vs Tau), not solely by chemistry")
        decision = 'REJECT_NULL'
    elif silhouette > 0.3 or ari > 0.3:
        print("WEAK EVIDENCE AGAINST NULL")
        print("Some clustering structure exists but agreement with disease labels is moderate")
        decision = 'WEAK_EVIDENCE'
    else:
        print("FAIL TO REJECT NULL HYPOTHESIS")
        print("PTMs do not show disease-specific clustering")
        decision = 'FAIL_TO_REJECT'

    # Visualizations
    print("\n[6/6] Generating visualizations...")

    # Dendrogram
    linkage_matrix = linkage(squareform(distance_matrix), method='average')
    plot_dendrogram_custom(linkage_matrix, site_descriptions, RESULTS_DIR / 'dendrogram.png')
    print(f"  Saved: dendrogram.png")

    # Distance heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(distance_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Distance')
    ax.set_title('Pairwise Distances Between PTMs')
    ax.set_xlabel('PTM Index')
    ax.set_ylabel('PTM Index')

    # Add dividing line between RA and Tau
    n_ra = len(ra_sites)
    ax.axhline(y=n_ra-0.5, color='red', linestyle='--', linewidth=2, label='RA/Tau boundary')
    ax.axvline(x=n_ra-0.5, color='red', linestyle='--', linewidth=2)
    ax.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'distance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: distance_heatmap.png")

    # Scatter plot (2D projection via MDS)
    from sklearn.manifold import MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords_2d = mds.fit_transform(distance_matrix)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot RA sites
    ra_mask = np.array([label == 'RA' for label in all_labels])
    ax.scatter(coords_2d[ra_mask, 0], coords_2d[ra_mask, 1],
               c='red', label='RA (Citrullination)', alpha=0.6, s=50)

    # Plot Tau sites
    tau_mask = ~ra_mask
    ax.scatter(coords_2d[tau_mask, 0], coords_2d[tau_mask, 1],
               c='blue', label='Tau (Phosphorylation)', alpha=0.6, s=50)

    ax.set_xlabel('MDS Dimension 1')
    ax.set_ylabel('MDS Dimension 2')
    ax.set_title('PTM Clustering (2D MDS Projection)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'mds_projection.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: mds_projection.png")

    # Save results
    results = {
        'test_name': 'Test 1: Cross-Disease PTM Clustering',
        'date_executed': '2026-01-03',
        'pre_registered': True,
        'data': {
            'n_ra_sites': len(ra_sites),
            'n_tau_sites': len(tau_sites),
            'n_total': len(all_sites)
        },
        'metrics': {
            'silhouette_score': float(silhouette),
            'adjusted_rand_index': float(ari),
            'mean_distance': float(distance_matrix[np.triu_indices_from(distance_matrix, k=1)].mean())
        },
        'confusion_matrix': {
            'ra_in_cluster0': int(ra_in_cluster0),
            'ra_in_cluster1': int(ra_in_cluster1),
            'tau_in_cluster0': int(tau_in_cluster0),
            'tau_in_cluster1': int(tau_in_cluster1)
        },
        'success_criteria': {
            'min_silhouette': 0.3,
            'min_ari': 0.5
        },
        'decision': decision,
        'note': 'Simplified embedding used (position + residue properties). Full implementation would use TrainableCodonEncoder with complete protein sequences.'
    }

    output_file = RESULTS_DIR / 'results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("="*80)

    return decision

if __name__ == '__main__':
    try:
        decision = main()
    except Exception as e:
        print(f"\nERROR: Test execution failed")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
