#!/usr/bin/env python3
"""
Compare contact prediction signal across checkpoints for small proteins.

Tests whether ceiling-hierarchy checkpoint improves signal over high-richness.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry import poincare_distance
from src.models.ternary_vae import TernaryVAEV5_11_PartialFreeze
from src.core import TERNARY

# Key test proteins (subset from 04_small_protein_conjecture.py)
SMALL_PROTEINS = {
    'chignolin': {
        'name': 'Chignolin', 'length': 10, 'constraints': 'none',
        'sequence': 'GYDPETGTWG',
        'codons': ['GGC', 'TAC', 'GAC', 'CCC', 'GAG', 'ACC', 'GGC', 'ACC', 'TGG', 'GGC'],
        'ca_coords': np.array([
            [1.458, -0.517, 0.034], [2.634, 2.951, -0.466], [0.193, 4.825, 1.587],
            [-2.261, 2.235, 1.667], [-0.718, -0.929, 2.792], [2.580, -1.047, 4.513],
            [2.046, 2.562, 5.274], [-1.318, 3.002, 6.680], [-2.826, -0.360, 6.042],
            [-0.029, -2.544, 5.221],
        ])
    },
    'trp_cage': {
        'name': 'Trp-cage TC5b', 'length': 20, 'constraints': 'hydrophobic_core',
        'sequence': 'NLYIQWLKDGGPSSGRPPPS',
        'codons': ['AAC', 'CTG', 'TAC', 'ATC', 'CAG', 'TGG', 'CTG', 'AAG', 'GAC', 'GGC',
                   'GGC', 'CCC', 'AGC', 'AGC', 'GGC', 'CGC', 'CCC', 'CCC', 'CCC', 'AGC'],
        'ca_coords': np.array([
            [-8.284, 1.757, 5.847], [-7.665, 5.476, 5.618], [-7.853, 6.821, 2.090],
            [-5.306, 9.318, 1.063], [-2.566, 7.447, 2.796], [-3.928, 5.040, 5.449],
            [-2.101, 2.107, 4.109], [-4.139, -0.706, 2.667], [-1.792, -2.684, 0.772],
            [-2.665, -2.155, -2.905], [0.376, -0.123, -3.516], [0.596, 3.055, -1.541],
            [4.198, 2.212, -0.674], [4.858, 0.133, 2.315], [2.596, -2.783, 3.077],
            [4.756, -5.428, 1.481], [2.903, -7.115, -1.263], [3.697, -4.619, -4.050],
            [0.030, -4.701, -5.171], [-1.389, -1.197, -4.795],
        ])
    },
    'insulin_b': {
        'name': 'Insulin B-chain', 'length': 30, 'constraints': 'disulfide',
        'sequence': 'FVNQHLCGSHLVEALYLVCGERGFFYTPKT',
        'codons': ['TTT', 'GTG', 'AAC', 'CAG', 'CAC', 'CTG', 'TGC', 'GGC', 'AGC', 'CAC',
                   'CTG', 'GTG', 'GAG', 'GCC', 'CTG', 'TAC', 'CTG', 'GTG', 'TGC', 'GGC',
                   'GAG', 'CGC', 'GGC', 'TTC', 'TTC', 'TAC', 'ACC', 'CCC', 'AAG', 'ACC'],
        'ca_coords': np.array([
            [12.86, 3.91, 5.44], [10.41, 1.04, 5.87], [11.06, -2.54, 4.74],
            [8.15, -4.90, 4.73], [8.09, -6.97, 1.63], [5.30, -9.45, 1.13],
            [5.93, -11.16, -2.17], [3.77, -14.24, -2.23], [5.11, -17.64, -1.28],
            [3.62, -18.50, 1.97], [4.18, -15.02, 3.36], [2.57, -13.82, 6.53],
            [3.97, -10.33, 7.18], [1.66, -8.64, 9.76], [2.21, -4.92, 10.21],
            [-0.25, -3.05, 12.22], [-0.45, 0.76, 11.53], [-2.61, 2.40, 14.04],
            [-2.01, 6.14, 14.10], [-4.33, 8.63, 12.24], [-3.21, 11.88, 10.71],
            [-5.07, 13.41, 7.87], [-3.04, 16.59, 7.33], [-4.27, 18.21, 4.16],
            [-1.82, 21.00, 3.59], [-2.52, 22.65, 0.24], [0.67, 24.59, -0.68],
            [1.15, 26.10, -4.07], [4.59, 27.54, -4.82], [5.08, 29.17, -8.07],
        ])
    },
    'villin_hp35': {
        'name': 'Villin HP35', 'length': 35, 'constraints': 'hydrophobic_core',
        'sequence': 'LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF',
        'codons': ['CTG', 'AGC', 'GAC', 'GAG', 'GAC', 'TTC', 'AAG', 'GCC', 'GTG', 'TTC',
                   'GGC', 'ATG', 'ACC', 'CGC', 'AGC', 'GCC', 'TTC', 'GCC', 'AAC', 'CTG',
                   'CCC', 'CTG', 'TGG', 'AAG', 'CAG', 'CAG', 'AAC', 'CTG', 'AAG', 'AAG',
                   'GAG', 'AAG', 'GGC', 'CTG', 'TTC'],
        'ca_coords': np.array([
            [24.040, 19.584, 7.913], [23.093, 16.012, 8.580], [21.893, 14.823, 11.916],
            [18.188, 14.576, 11.626], [16.478, 11.229, 11.717], [14.128, 11.023, 14.603],
            [10.590, 10.021, 13.689], [9.627, 12.555, 11.074], [12.065, 14.904, 9.689],
            [10.635, 17.422, 7.345], [13.310, 19.950, 6.723], [12.397, 22.396, 4.085],
            [14.942, 24.952, 3.183], [13.723, 27.471, 0.712], [16.026, 29.730, -1.179],
            [14.413, 31.439, -4.179], [16.692, 34.418, -4.496], [14.824, 37.603, -4.890],
            [16.880, 39.377, -7.493], [14.510, 41.978, -8.829], [15.954, 43.064, -12.201],
            [13.017, 45.361, -13.041], [13.588, 45.620, -16.778], [10.275, 47.360, -17.655],
            [9.853, 46.574, -21.330], [6.279, 47.700, -22.106], [5.159, 45.917, -25.232],
            [1.579, 46.986, -25.524], [-0.082, 44.262, -27.531], [-3.541, 45.339, -28.525],
            [-5.344, 42.343, -30.076], [-8.954, 43.210, -30.711], [-10.794, 40.096, -31.757],
            [-14.398, 40.673, -32.561], [-15.983, 37.345, -33.315],
        ])
    },
    'gb1': {
        'name': 'Protein G B1', 'length': 56, 'constraints': 'hydrophobic_core',
        'sequence': 'MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE',
        'codons': ['ATG', 'ACC', 'TAC', 'AAG', 'CTG', 'ATC', 'CTG', 'AAC', 'GGC', 'AAG',
                   'ACC', 'CTG', 'AAG', 'GGC', 'GAG', 'ACC', 'ACC', 'ACC', 'GAG', 'GCC',
                   'GTG', 'GAC', 'GCC', 'GCC', 'ACC', 'GCC', 'GAG', 'AAG', 'GTG', 'TTC',
                   'AAG', 'CAG', 'TAC', 'GCC', 'AAC', 'GAC', 'AAC', 'GGC', 'GTG', 'GAC',
                   'GGC', 'GAG', 'TGG', 'ACC', 'TAC', 'GAC', 'GAC', 'GCC', 'ACC', 'AAG',
                   'ACC', 'TTC', 'ACC', 'GTG', 'ACC', 'GAG'],
        'ca_coords': np.array([
            [16.671, 43.549, 34.916], [17.088, 44.071, 31.201], [15.063, 41.265, 29.947],
            [17.159, 38.243, 30.434], [16.513, 37.621, 34.134], [19.393, 39.947, 34.876],
            [19.109, 42.006, 31.695], [21.927, 39.820, 30.582], [21.461, 36.628, 32.552],
            [23.173, 37.963, 35.680], [26.128, 40.159, 34.749], [25.336, 40.739, 31.062],
            [28.310, 38.562, 30.209], [28.159, 35.259, 32.013], [30.070, 35.997, 35.163],
            [33.039, 38.352, 34.491], [32.314, 39.360, 30.863], [35.369, 37.230, 30.071],
            [35.369, 33.936, 31.926], [37.195, 34.352, 35.202], [40.345, 36.462, 34.571],
            [40.082, 37.491, 30.893], [43.498, 35.894, 30.093], [44.049, 32.569, 31.867],
            [45.680, 32.835, 35.269], [48.939, 34.728, 34.638], [49.233, 35.976, 30.998],
            [52.733, 34.635, 30.354], [53.492, 31.222, 31.738], [55.132, 30.997, 35.202],
            [58.508, 32.710, 34.635], [58.999, 34.228, 31.103], [62.368, 32.666, 30.241],
            [63.355, 29.358, 31.800], [64.904, 29.103, 35.261], [68.293, 30.807, 34.706],
            [68.980, 32.435, 31.247], [72.243, 30.677, 30.362], [73.294, 27.340, 31.880],
            [74.652, 27.163, 35.439], [77.909, 29.096, 34.866], [78.785, 30.631, 31.364],
            [81.823, 28.554, 30.561], [82.802, 25.168, 31.995], [84.169, 24.862, 35.522],
            [87.454, 26.815, 34.964], [88.285, 28.490, 31.553], [91.241, 26.299, 30.669],
            [92.193, 22.972, 32.178], [93.476, 22.612, 35.724], [96.747, 24.570, 35.211],
            [97.648, 26.274, 31.835], [100.605, 24.076, 30.875], [101.586, 20.688, 32.316],
            [102.891, 20.362, 35.880], [106.162, 22.326, 35.363],
        ])
    },
}


def load_codon_mapping():
    """Load codon to embedding position mapping."""
    map_path = Path(__file__).parent.parent / 'embeddings' / 'codon_mapping_3adic.json'
    with open(map_path) as f:
        return json.load(f)['codon_to_position']


def extract_embeddings_from_checkpoint(checkpoint_path):
    """Extract embeddings directly from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)

    encoder_weight = state_dict.get('encoder_A.net.0.weight')
    hidden_dim = encoder_weight.shape[0] if encoder_weight is not None else 64

    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=16, hidden_dim=hidden_dim, max_radius=0.99,
        curvature=1.0, use_controller=False, use_dual_projection=True,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    all_indices = torch.arange(TERNARY.N_OPERATIONS)
    all_ternary = TERNARY.to_ternary(all_indices)

    with torch.no_grad():
        out = model(all_ternary.float(), compute_control=False)

    return out['z_B_hyp']


def compute_contact_map(coords, threshold=8.0, min_seq_sep=4):
    """Compute binary contact map from Cα coordinates."""
    n = len(coords)
    dist_matrix = squareform(pdist(coords))
    contact_map = (dist_matrix < threshold).astype(float)
    for i in range(n):
        for j in range(n):
            if abs(i - j) < min_seq_sep:
                contact_map[i, j] = 0
    return contact_map


def evaluate_protein(protein_id, z_hyp, codon_to_pos):
    """Evaluate contact prediction for a single protein."""
    protein = SMALL_PROTEINS[protein_id]
    codons = protein['codons']
    coords = protein['ca_coords']

    n = min(len(codons), len(coords))
    codons = codons[:n]
    coords = coords[:n]

    if n < 8:
        return None

    contact_map = compute_contact_map(coords)
    n_contacts = int(contact_map.sum() / 2)

    if n_contacts == 0:
        return None

    hyp_dists = []
    contacts = []

    for i in range(n):
        for j in range(i + 4, n):
            if codons[i] not in codon_to_pos or codons[j] not in codon_to_pos:
                continue

            idx_i = codon_to_pos[codons[i]]
            idx_j = codon_to_pos[codons[j]]

            d = poincare_distance(
                z_hyp[idx_i:idx_i+1],
                z_hyp[idx_j:idx_j+1],
                c=1.0
            ).item()

            hyp_dists.append(d)
            contacts.append(contact_map[i, j])

    if not hyp_dists or sum(contacts) == 0:
        return None

    hyp_dists = np.array(hyp_dists)
    contacts = np.array(contacts)

    from sklearn.metrics import roc_auc_score

    try:
        auc = roc_auc_score(contacts, -hyp_dists)
    except ValueError:
        auc = 0.5

    contact_dists = hyp_dists[contacts == 1]
    noncontact_dists = hyp_dists[contacts == 0]

    if len(contact_dists) > 0 and len(noncontact_dists) > 0:
        pooled_std = np.sqrt((contact_dists.var() + noncontact_dists.var()) / 2)
        cohens_d = (contact_dists.mean() - noncontact_dists.mean()) / pooled_std if pooled_std > 0 else 0
    else:
        cohens_d = 0

    return {
        'name': protein['name'],
        'length': n,
        'constraints': protein['constraints'],
        'n_contacts': n_contacts,
        'auc': auc,
        'cohens_d': cohens_d,
    }


def main():
    print("=" * 80)
    print("CHECKPOINT COMPARISON: Small Proteins")
    print("=" * 80)
    print()

    checkpoints = [
        ('v5_11_structural_best.pt', 'Ceiling Hierarchy'),
        ('homeostatic_rich_best.pt', 'High Richness'),
    ]

    ckpt_dir = Path(__file__).parent.parent / 'checkpoints'
    codon_to_pos = load_codon_mapping()

    # Sort proteins by size
    sorted_proteins = sorted(SMALL_PROTEINS.keys(),
                            key=lambda x: SMALL_PROTEINS[x]['length'])

    # Focus on key proteins
    test_proteins = ['chignolin', 'trp_cage', 'insulin_a', 'insulin_b',
                     'villin_hp35', 'crambin', 'gb1', 'bpti']

    all_results = {}

    for ckpt_file, description in checkpoints:
        ckpt_path = ckpt_dir / ckpt_file
        if not ckpt_path.exists():
            print(f"Skipping {ckpt_file} (not found)")
            continue

        print(f"\n{'='*60}")
        print(f"CHECKPOINT: {description}")
        print(f"{'='*60}")

        z_hyp = extract_embeddings_from_checkpoint(ckpt_path)
        results = []

        for pid in test_proteins:
            if pid not in SMALL_PROTEINS:
                continue

            protein = SMALL_PROTEINS[pid]
            result = evaluate_protein(pid, z_hyp, codon_to_pos)

            if result:
                results.append(result)
                print(f"  {protein['name']:<30} AUC={result['auc']:.3f}  d={result['cohens_d']:+.3f}")

        all_results[description] = results

        if results:
            mean_auc = np.mean([r['auc'] for r in results])
            print(f"\n  MEAN AUC = {mean_auc:.4f}")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Protein':<25} {'Ceil.Hier':>10} {'High.Rich':>10} {'Δ':>8}")
    print("-" * 60)

    if len(all_results) == 2:
        ceil_results = {r['name']: r for r in all_results['Ceiling Hierarchy']}
        rich_results = {r['name']: r for r in all_results['High Richness']}

        for name in ceil_results:
            if name in rich_results:
                ceil_auc = ceil_results[name]['auc']
                rich_auc = rich_results[name]['auc']
                delta = ceil_auc - rich_auc
                marker = "<<<" if delta > 0.05 else (">>>" if delta < -0.05 else "")
                print(f"{name:<25} {ceil_auc:>10.3f} {rich_auc:>10.3f} {delta:>+8.3f} {marker}")

        # Aggregate
        ceil_mean = np.mean([r['auc'] for r in all_results['Ceiling Hierarchy']])
        rich_mean = np.mean([r['auc'] for r in all_results['High Richness']])
        print("-" * 60)
        print(f"{'MEAN':<25} {ceil_mean:>10.3f} {rich_mean:>10.3f} {ceil_mean - rich_mean:>+8.3f}")

        if ceil_mean > rich_mean:
            print("\n>>> CEILING HIERARCHY wins for contact prediction")
        else:
            print("\n>>> HIGH RICHNESS wins for contact prediction")


if __name__ == '__main__':
    main()
