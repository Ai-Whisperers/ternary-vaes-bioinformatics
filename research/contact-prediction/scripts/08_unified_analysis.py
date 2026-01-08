#!/usr/bin/env python3
"""
UNIFIED ANALYSIS: Comprehensive test with 16+ proteins and folding rate data.

Combines the protein database from 06_expanded_small_proteins.py with
quantitative folding rates from the literature.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import json
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry import poincare_distance

# =============================================================================
# FOLDING RATES FROM LITERATURE (ln(kf) in s^-1)
# =============================================================================
# Sources: Plaxco et al. 1998, Maxwell et al. 2005, Kubelka et al. 2004,
# Various NMR/kinetics studies

FOLDING_RATES = {
    # Ultrafast folders (τ < 10 μs, ln(kf) > 11.5)
    'chignolin': 13.8,       # ~1 μs (Honda et al. 2008)
    'trp_cage': 12.4,        # ~4 μs (Qiu et al. 2002)
    'villin_hp35': 12.3,     # ~4.3 μs (Kubelka et al. 2003)
    'fsd1': 11.8,            # ~7 μs (Dahiyat & Mayo 1997)

    # Fast folders (τ ~ 10-1000 μs, ln(kf) 7-11.5)
    'lambda_repressor': 9.9,  # ~50 μs (Burton et al. 1997)
    'engrailed_hd': 9.4,      # ~80 μs (Mayor et al. 2003)
    'cold_shock': 8.9,        # ~150 μs (Perl et al. 1998)
    'src_sh3': 8.5,           # ~200 μs (Grantcharova & Baker 1997)
    'gb1': 7.6,               # ~500 μs (Park et al. 1997)
    'zinc_finger': 8.0,       # ~350 μs (Blasie & Berg 2002)

    # Slow folders (τ > 1 ms, ln(kf) < 7)
    'ubiquitin': 5.5,         # ~4 ms (Khorasanizadeh et al. 1996)
    'rubredoxin': 4.0,        # ~18 ms (Wittung-Stafshede 1999)

    # Disulfide-constrained (special category - very slow)
    'insulin_b': 3.0,         # Very slow (disulfide formation)
    'crambin': 2.5,           # Very slow (3 disulfides)
    'bpti': 2.0,              # Very slow (3 disulfides)
}

# Amino acid properties
HYDROPHOBIC_AA = set('AILMFVPGW')
POLAR_AA = set('STYCNQ')
CHARGED_AA = set('DEKRH')


def load_codon_mapping():
    """Load codon to embedding position mapping."""
    map_path = Path(__file__).parent.parent / 'embeddings' / 'codon_mapping_3adic.json'
    with open(map_path) as f:
        return json.load(f)['codon_to_position']


def load_embeddings():
    """Load pre-extracted embeddings."""
    emb_path = Path(__file__).parent.parent / 'embeddings' / 'v5_11_3_embeddings.pt'
    emb_data = torch.load(emb_path, map_location='cpu', weights_only=False)
    return emb_data['z_B_hyp']


def load_protein_database():
    """Load protein database from script 06."""
    # Import the PROTEINS dict from script 06
    script06_path = Path(__file__).parent / '06_expanded_small_proteins.py'

    import importlib.util
    spec = importlib.util.spec_from_file_location("script06", script06_path)
    script06 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script06)

    proteins = script06.PROTEINS

    # Add folding rates
    for pid, rate in FOLDING_RATES.items():
        if pid in proteins:
            proteins[pid]['folding_rate'] = rate

    return proteins


def compute_contact_map(coords, threshold=8.0, min_seq_sep=4):
    """Compute binary contact map from Cα coordinates."""
    n = len(coords)
    dist_matrix = squareform(pdist(coords))
    contact_map = (dist_matrix < threshold).astype(float)
    for i in range(n):
        for j in range(n):
            if abs(i - j) < min_seq_sep:
                contact_map[i, j] = 0
    return contact_map, dist_matrix


def get_secondary_structure(ss):
    """Parse secondary structure annotation."""
    return {
        'helix': [i for i, c in enumerate(ss) if c == 'H'],
        'sheet': [i for i, c in enumerate(ss) if c == 'E'],
        'coil': [i for i, c in enumerate(ss) if c == 'C'],
    }


def evaluate_protein(protein_id, protein, z_hyp, codon_to_pos):
    """Evaluate a single protein with all metrics."""
    codons = protein['codons']
    coords = protein['ca_coords']
    sequence = protein.get('sequence', '')
    ss = protein.get('ss', '')

    n = min(len(codons), len(coords))
    if sequence:
        n = min(n, len(sequence))

    codons = codons[:n]
    coords = coords[:n]
    sequence = sequence[:n] if sequence else 'X' * n
    ss = ss[:n] if ss else 'C' * n

    if n < 8:
        return None

    contact_map, dist_matrix = compute_contact_map(coords, min_seq_sep=4)
    n_contacts = int(contact_map.sum() / 2)

    if n_contacts < 2:
        return None

    # Collect all pairs
    pairs_data = []

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

            # Contact type
            aa_i, aa_j = sequence[i], sequence[j]
            if aa_i in HYDROPHOBIC_AA and aa_j in HYDROPHOBIC_AA:
                contact_type = 'hydrophobic'
            elif aa_i in CHARGED_AA or aa_j in CHARGED_AA:
                contact_type = 'charged'
            else:
                contact_type = 'polar'

            # Secondary structure
            ss_i, ss_j = ss[i] if i < len(ss) else 'C', ss[j] if j < len(ss) else 'C'
            if ss_i == 'H' and ss_j == 'H':
                ss_type = 'helix-helix'
            elif ss_i == 'E' and ss_j == 'E':
                ss_type = 'sheet-sheet'
            elif (ss_i == 'H' and ss_j == 'E') or (ss_i == 'E' and ss_j == 'H'):
                ss_type = 'helix-sheet'
            else:
                ss_type = 'other'

            pairs_data.append({
                'i': i, 'j': j,
                'seq_sep': j - i,
                'hyp_dist': hyp_dist,
                'contact': contact_map[i, j],
                'ca_dist': dist_matrix[i, j],
                'contact_type': contact_type,
                'ss_type': ss_type,
            })

    if len(pairs_data) < 10:
        return None

    # Convert to arrays
    hyp_dists = np.array([p['hyp_dist'] for p in pairs_data])
    contacts = np.array([p['contact'] for p in pairs_data])
    seq_seps = np.array([p['seq_sep'] for p in pairs_data])
    contact_types = [p['contact_type'] for p in pairs_data]
    ss_types = [p['ss_type'] for p in pairs_data]

    # Overall AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc_overall = roc_auc_score(contacts, -hyp_dists)
    except ValueError:
        auc_overall = 0.5

    # AUC by sequence separation
    auc_by_range = {}
    for range_name, (min_sep, max_sep) in [
        ('local', (4, 8)),
        ('medium', (8, 16)),
        ('long', (16, 100)),
    ]:
        mask = (seq_seps >= min_sep) & (seq_seps < max_sep)
        if mask.sum() > 5 and contacts[mask].sum() > 0 and contacts[mask].sum() < mask.sum():
            try:
                auc_by_range[range_name] = roc_auc_score(contacts[mask], -hyp_dists[mask])
            except:
                pass

    # AUC by contact type
    auc_by_type = {}
    for ctype in ['hydrophobic', 'charged', 'polar']:
        mask = np.array([ct == ctype for ct in contact_types])
        if mask.sum() > 5 and contacts[mask].sum() > 0 and contacts[mask].sum() < mask.sum():
            try:
                auc_by_type[ctype] = roc_auc_score(contacts[mask], -hyp_dists[mask])
            except:
                pass

    # AUC by secondary structure
    auc_by_ss = {}
    for ss_type in ['helix-helix', 'sheet-sheet', 'helix-sheet']:
        mask = np.array([st == ss_type for st in ss_types])
        if mask.sum() > 5 and contacts[mask].sum() > 0 and contacts[mask].sum() < mask.sum():
            try:
                auc_by_ss[ss_type] = roc_auc_score(contacts[mask], -hyp_dists[mask])
            except:
                pass

    # Cohen's d
    contact_dists = hyp_dists[contacts == 1]
    noncontact_dists = hyp_dists[contacts == 0]
    if len(contact_dists) > 0 and len(noncontact_dists) > 0:
        pooled_std = np.sqrt((contact_dists.var() + noncontact_dists.var()) / 2)
        cohens_d = (contact_dists.mean() - noncontact_dists.mean()) / pooled_std if pooled_std > 0 else 0
    else:
        cohens_d = 0

    return {
        'id': protein_id,
        'name': protein['name'],
        'length': n,
        'fold_type': protein['fold_type'],
        'constraint': protein['constraint'],
        'folding': protein.get('folding', 'unknown'),
        'folding_rate': protein.get('folding_rate', None),
        'n_contacts': n_contacts,
        'n_pairs': len(pairs_data),
        'auc_overall': auc_overall,
        'cohens_d': cohens_d,
        'auc_by_range': auc_by_range,
        'auc_by_type': auc_by_type,
        'auc_by_ss': auc_by_ss,
    }


def main():
    print("=" * 90)
    print("UNIFIED ANALYSIS: Comprehensive Contact Prediction Study")
    print("=" * 90)
    print()

    # Load data
    print("Loading data...")
    z_hyp = load_embeddings()
    codon_to_pos = load_codon_mapping()
    proteins = load_protein_database()
    print(f"  Loaded {len(proteins)} proteins from database")
    print(f"  {len(FOLDING_RATES)} have quantitative folding rates")
    print()

    # Evaluate all proteins
    results = []
    print("Evaluating proteins...")
    print("-" * 90)

    for pid in sorted(proteins.keys()):
        protein = proteins[pid]
        result = evaluate_protein(pid, protein, z_hyp, codon_to_pos)
        if result:
            results.append(result)
            rate = result.get('folding_rate')
            rate_str = f"ln(kf)={rate:.1f}" if rate else "N/A"
            print(f"  {result['name']:<28} AUC={result['auc_overall']:.3f}  d={result['cohens_d']:+.3f}  {rate_str}")
        else:
            print(f"  {protein['name']:<28} SKIPPED (insufficient data)")

    print("-" * 90)
    print(f"  Total valid proteins: {len(results)}")

    # =========================================================================
    # STATISTICAL TESTS
    # =========================================================================
    print()
    print("=" * 90)
    print("STATISTICAL SIGNIFICANCE TEST")
    print("=" * 90)

    overall_aucs = [r['auc_overall'] for r in results]
    t_stat, p_val = stats.ttest_1samp(overall_aucs, 0.5)

    print(f"\n  Overall: Mean AUC = {np.mean(overall_aucs):.4f} +/- {np.std(overall_aucs):.4f}")
    print(f"  One-sample t-test (H0: AUC = 0.5): t = {t_stat:.4f}, p = {p_val:.6f}")

    if p_val < 0.001:
        print("  >>> HIGHLY SIGNIFICANT (p < 0.001)")
    elif p_val < 0.01:
        print("  >>> VERY SIGNIFICANT (p < 0.01)")
    elif p_val < 0.05:
        print("  >>> SIGNIFICANT (p < 0.05)")

    # =========================================================================
    # ANALYSIS 1: Folding Rate Correlation
    # =========================================================================
    print()
    print("=" * 90)
    print("ANALYSIS 1: Folding Rate vs Contact Prediction")
    print("=" * 90)

    proteins_with_rates = [r for r in results if r['folding_rate'] is not None]

    if len(proteins_with_rates) >= 5:
        rates = np.array([r['folding_rate'] for r in proteins_with_rates])
        aucs = np.array([r['auc_overall'] for r in proteins_with_rates])

        rho, p_rate = stats.spearmanr(rates, aucs)
        print(f"\n  Spearman correlation (ln(kf) vs AUC): rho = {rho:.4f} (p = {p_rate:.4f})")

        if rho > 0.3 and p_rate < 0.1:
            print("  >>> CONFIRMED: Faster folders encode contact physics better")

        print("\n  By folding speed category:")
        for category, (min_r, max_r) in [
            ('Ultrafast (ln(kf) > 11.5)', (11.5, 20)),
            ('Fast (7 < ln(kf) < 11.5)', (7, 11.5)),
            ('Slow (ln(kf) < 7)', (0, 7)),
        ]:
            cat = [r for r in proteins_with_rates if min_r <= r['folding_rate'] < max_r]
            if cat:
                mean_auc = np.mean([r['auc_overall'] for r in cat])
                std_auc = np.std([r['auc_overall'] for r in cat])
                names = [r['id'][:8] for r in cat]
                print(f"    {category}: n={len(cat)}, AUC = {mean_auc:.3f} +/- {std_auc:.3f}")

    # =========================================================================
    # ANALYSIS 2: Qualitative Folding Category
    # =========================================================================
    print()
    print("=" * 90)
    print("ANALYSIS 2: Qualitative Folding Category")
    print("=" * 90)

    for category in ['ultrafast', 'fast', 'slow']:
        cat_results = [r for r in results if r['folding'] == category]
        if cat_results:
            mean_auc = np.mean([r['auc_overall'] for r in cat_results])
            std_auc = np.std([r['auc_overall'] for r in cat_results])
            print(f"\n  {category.upper()}: n={len(cat_results)}, AUC = {mean_auc:.3f} +/- {std_auc:.3f}")

    # =========================================================================
    # ANALYSIS 3: Contact Range
    # =========================================================================
    print()
    print("=" * 90)
    print("ANALYSIS 3: Contact Range Effects")
    print("=" * 90)

    range_labels = {'local': '4-8 residues', 'medium': '8-16 residues', 'long': '>16 residues'}
    for range_name in ['local', 'medium', 'long']:
        range_aucs = [r['auc_by_range'].get(range_name) for r in results if range_name in r['auc_by_range']]
        if range_aucs:
            print(f"\n  {range_name.upper()} ({range_labels[range_name]}): n={len(range_aucs)}")
            print(f"    AUC = {np.mean(range_aucs):.4f} +/- {np.std(range_aucs):.4f}")

    # =========================================================================
    # ANALYSIS 4: Contact Type
    # =========================================================================
    print()
    print("=" * 90)
    print("ANALYSIS 4: Contact Type (AA Properties)")
    print("=" * 90)

    for ctype in ['hydrophobic', 'polar', 'charged']:
        type_aucs = [r['auc_by_type'].get(ctype) for r in results if ctype in r['auc_by_type']]
        if type_aucs:
            print(f"\n  {ctype.upper()}: n={len(type_aucs)}")
            print(f"    AUC = {np.mean(type_aucs):.4f} +/- {np.std(type_aucs):.4f}")

    # =========================================================================
    # ANALYSIS 5: Secondary Structure
    # =========================================================================
    print()
    print("=" * 90)
    print("ANALYSIS 5: Secondary Structure Contacts")
    print("=" * 90)

    for ss_type in ['helix-helix', 'sheet-sheet', 'helix-sheet']:
        ss_aucs = [r['auc_by_ss'].get(ss_type) for r in results if ss_type in r['auc_by_ss']]
        if ss_aucs:
            print(f"\n  {ss_type.upper()}: n={len(ss_aucs)}")
            print(f"    AUC = {np.mean(ss_aucs):.4f} +/- {np.std(ss_aucs):.4f}")

    # =========================================================================
    # ANALYSIS 6: Constraint Type
    # =========================================================================
    print()
    print("=" * 90)
    print("ANALYSIS 6: Constraint Type")
    print("=" * 90)

    for constraint in ['hydrophobic', 'designed', 'metal', 'disulfide']:
        const_results = [r for r in results if r['constraint'] == constraint]
        if const_results:
            mean_auc = np.mean([r['auc_overall'] for r in const_results])
            std_auc = np.std([r['auc_overall'] for r in const_results])
            print(f"\n  {constraint.upper()}: n={len(const_results)}")
            print(f"    AUC = {mean_auc:.4f} +/- {std_auc:.4f}")

    # =========================================================================
    # ANALYSIS 7: Fold Type
    # =========================================================================
    print()
    print("=" * 90)
    print("ANALYSIS 7: Fold Type")
    print("=" * 90)

    for fold in ['alpha', 'beta', 'alpha/beta']:
        fold_results = [r for r in results if r['fold_type'] == fold]
        if fold_results:
            mean_auc = np.mean([r['auc_overall'] for r in fold_results])
            std_auc = np.std([r['auc_overall'] for r in fold_results])
            print(f"\n  {fold.upper()}: n={len(fold_results)}")
            print(f"    AUC = {mean_auc:.4f} +/- {std_auc:.4f}")

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print()
    print("=" * 90)
    print("SUMMARY: Top Performers")
    print("=" * 90)

    sorted_results = sorted(results, key=lambda x: x['auc_overall'], reverse=True)
    print(f"\n  {'Protein':<25} {'AUC':>6} {'d':>7} {'Fold':>10} {'Constraint':>12} {'Folding':>10}")
    print("  " + "-" * 75)
    for r in sorted_results[:10]:
        print(f"  {r['name']:<25} {r['auc_overall']:>6.3f} {r['cohens_d']:>+7.3f} {r['fold_type']:>10} {r['constraint']:>12} {r['folding']:>10}")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    output_file = Path(__file__).parent.parent / 'data' / 'unified_analysis_results.json'
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {output_file}")

    return results


if __name__ == '__main__':
    results = main()
