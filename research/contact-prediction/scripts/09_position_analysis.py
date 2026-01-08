#!/usr/bin/env python3
"""
POSITION ANALYSIS: Test if contact prediction signal varies by sequence position.

Hypothesis: N-terminal regions may show different signals than C-terminal regions
due to co-translational folding effects.
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


def load_data():
    """Load embeddings and protein database."""
    emb_path = Path(__file__).parent.parent / 'embeddings' / 'v5_11_3_embeddings.pt'
    emb_data = torch.load(emb_path, map_location='cpu', weights_only=False)
    z_hyp = emb_data['z_B_hyp']

    map_path = Path(__file__).parent.parent / 'embeddings' / 'codon_mapping_3adic.json'
    with open(map_path) as f:
        codon_to_pos = json.load(f)['codon_to_position']

    # Import proteins
    script06_path = Path(__file__).parent / '06_expanded_small_proteins.py'
    import importlib.util
    spec = importlib.util.spec_from_file_location("script06", script06_path)
    script06 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script06)

    return z_hyp, codon_to_pos, script06.PROTEINS


def analyze_position_effects(z_hyp, codon_to_pos, proteins):
    """Analyze contact prediction by position in sequence."""
    from sklearn.metrics import roc_auc_score

    # Collect all contacts with position metadata
    all_n_term = []  # Contacts where both residues in N-terminal half
    all_c_term = []  # Contacts where both residues in C-terminal half
    all_cross = []   # Contacts spanning N and C terminal halves

    for pid, protein in proteins.items():
        codons = protein['codons']
        coords = protein['ca_coords']
        n = min(len(codons), len(coords))

        if n < 20:
            continue

        codons = codons[:n]
        coords = coords[:n]

        dist_matrix = squareform(pdist(coords))
        contact_map = (dist_matrix < 8.0).astype(float)

        midpoint = n // 2

        for i in range(n):
            for j in range(i + 4, n):
                if codons[i] not in codon_to_pos or codons[j] not in codon_to_pos:
                    continue

                idx_i = codon_to_pos[codons[i]]
                idx_j = codon_to_pos[codons[j]]

                hyp_dist = poincare_distance(
                    z_hyp[idx_i:idx_i+1],
                    z_hyp[idx_j:idx_j+1],
                    c=1.0
                ).item()

                is_contact = contact_map[i, j]

                # Classify by position
                if i < midpoint and j < midpoint:
                    all_n_term.append((hyp_dist, is_contact))
                elif i >= midpoint and j >= midpoint:
                    all_c_term.append((hyp_dist, is_contact))
                else:
                    all_cross.append((hyp_dist, is_contact))

    return all_n_term, all_c_term, all_cross


def main():
    print("=" * 80)
    print("POSITION ANALYSIS: N-term vs C-term Contact Prediction")
    print("=" * 80)
    print()

    z_hyp, codon_to_pos, proteins = load_data()

    all_n_term, all_c_term, all_cross = analyze_position_effects(
        z_hyp, codon_to_pos, proteins
    )

    from sklearn.metrics import roc_auc_score

    print("Results by Sequence Position:")
    print("-" * 60)

    for name, data in [
        ('N-terminal (first half)', all_n_term),
        ('C-terminal (second half)', all_c_term),
        ('Cross-terminal', all_cross),
    ]:
        if len(data) < 20:
            print(f"  {name}: insufficient data")
            continue

        dists = np.array([d[0] for d in data])
        contacts = np.array([d[1] for d in data])

        if contacts.sum() == 0 or contacts.sum() == len(contacts):
            print(f"  {name}: no variance in contacts")
            continue

        auc = roc_auc_score(contacts, -dists)
        n_contacts = int(contacts.sum())

        print(f"\n  {name}:")
        print(f"    n_pairs = {len(data)}, n_contacts = {n_contacts}")
        print(f"    AUC = {auc:.4f}")

    # Statistical comparison
    print()
    print("=" * 80)
    print("Statistical Comparison")
    print("=" * 80)

    if len(all_n_term) >= 20 and len(all_c_term) >= 20:
        n_dists = np.array([d[0] for d in all_n_term])
        c_dists = np.array([d[0] for d in all_c_term])

        t_stat, p_val = stats.ttest_ind(n_dists, c_dists)
        print(f"\n  Mean hyperbolic distance comparison:")
        print(f"    N-terminal: {np.mean(n_dists):.4f}")
        print(f"    C-terminal: {np.mean(c_dists):.4f}")
        print(f"    t-test: t = {t_stat:.4f}, p = {p_val:.4f}")

    # Contact rate by position
    print()
    print("=" * 80)
    print("Contact Rate by Position")
    print("=" * 80)

    for name, data in [
        ('N-terminal', all_n_term),
        ('C-terminal', all_c_term),
        ('Cross-terminal', all_cross),
    ]:
        if len(data) < 10:
            continue
        contacts = np.array([d[1] for d in data])
        rate = contacts.mean()
        print(f"  {name}: contact rate = {rate:.4f} ({int(contacts.sum())}/{len(contacts)})")


if __name__ == '__main__':
    main()
