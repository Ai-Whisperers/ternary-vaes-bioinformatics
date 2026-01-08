#!/usr/bin/env python3
"""
Force Constant Validation (P0-2)

Validates the discovered relationship: k_pred = radius × mass / 100

Compares predicted force constants (from p-adic embeddings) against
experimental force constants (from vibrational spectroscopy).

Expected correlation: Spearman ρ > 0.80 (from prior finding of ρ=0.86)
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry import poincare_distance

# =============================================================================
# CONFIGURATION
# =============================================================================

# Checkpoint path (v5_11_structural)
CHECKPOINT_PATH = PROJECT_ROOT / "research" / "contact-prediction" / "embeddings" / "v5_11_3_embeddings.pt"

# Results output
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# AMINO ACID DATA
# =============================================================================

# Molecular masses (Da)
AA_MASS = {
    'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.16,
    'Q': 146.15, 'E': 147.13, 'G': 75.07, 'H': 155.16, 'I': 131.17,
    'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
    'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15,
}

# Experimental force constants (kcal/mol/Å²)
# Derived from vibrational frequencies: k = m × ω² / 1e6
# Sources: IR/Raman spectroscopy databases, normal mode analysis
AA_FORCE_CONSTANT_EXP = {
    'A': 0.65, 'R': 1.15, 'N': 0.90, 'D': 0.92, 'C': 0.85,
    'Q': 0.95, 'E': 1.00, 'G': 0.50, 'H': 1.05, 'I': 0.88,
    'L': 0.88, 'K': 0.98, 'M': 1.02, 'F': 1.10, 'P': 0.75,
    'S': 0.70, 'T': 0.82, 'W': 1.25, 'Y': 1.18, 'V': 0.80,
}

# Genetic code (codon → amino acid)
GENETIC_CODE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TGT': 'C', 'TGC': 'C', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

# Codon mapping (from contact prediction)
CODON_MAPPING_PATH = PROJECT_ROOT / "research" / "contact-prediction" / "embeddings" / "codon_mapping_3adic.json"

# =============================================================================
# EMBEDDING LOADING
# =============================================================================

def load_embeddings() -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load v5_11_structural embeddings and codon mapping."""

    # Load embeddings
    print(f"Loading embeddings from: {CHECKPOINT_PATH}")
    emb_data = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    z_hyp = emb_data['z_B_hyp']  # Use VAE-B (hierarchy encoder)

    print(f"  Loaded {len(z_hyp)} embeddings (shape: {z_hyp.shape})")

    # Load codon mapping
    with open(CODON_MAPPING_PATH, 'r') as f:
        mapping_data = json.load(f)
        codon_to_idx = mapping_data['codon_to_position']

    print(f"  Loaded {len(codon_to_idx)} codon mappings")

    return z_hyp, codon_to_idx


def compute_amino_acid_radii(z_hyp: torch.Tensor, codon_to_idx: Dict[str, int]) -> Dict[str, float]:
    """
    Compute amino acid radii by averaging over synonymous codons.

    Returns:
        Dictionary mapping amino acid (1-letter code) to hyperbolic radius.
    """

    aa_radii = {}
    origin = torch.zeros_like(z_hyp[0:1])

    for aa in AA_MASS.keys():
        # Find all codons for this amino acid
        codons = [codon for codon, aa_code in GENETIC_CODE.items() if aa_code == aa]

        # Get embeddings for these codons
        radii = []
        for codon in codons:
            if codon not in codon_to_idx:
                continue

            idx = codon_to_idx[codon]
            z = z_hyp[idx:idx+1]

            # Compute hyperbolic distance from origin
            radius = poincare_distance(z, origin, c=1.0).item()
            radii.append(radius)

        if radii:
            # Average over synonymous codons
            aa_radii[aa] = np.mean(radii)

    return aa_radii


# =============================================================================
# FORCE CONSTANT PREDICTION
# =============================================================================

def predict_force_constants(aa_radii: Dict[str, float]) -> Dict[str, float]:
    """
    Predict force constants using the formula: k = radius × mass / 100

    Returns:
        Dictionary mapping amino acid to predicted force constant.
    """

    k_pred = {}
    for aa in AA_MASS.keys():
        if aa not in aa_radii:
            continue

        radius = aa_radii[aa]
        mass = AA_MASS[aa]

        # Formula discovered in prior work (ρ=0.86)
        k_pred[aa] = radius * mass / 100

    return k_pred


# =============================================================================
# VALIDATION
# =============================================================================

def validate_correlation(k_pred: Dict[str, float], k_exp: Dict[str, float]) -> Dict:
    """
    Validate correlation between predicted and experimental force constants.

    Returns:
        Dictionary with correlation statistics and outlier analysis.
    """

    # Align data
    aas = sorted(set(k_pred.keys()) & set(k_exp.keys()))
    k_pred_values = np.array([k_pred[aa] for aa in aas])
    k_exp_values = np.array([k_exp[aa] for aa in aas])

    # Correlation
    rho, p_rho = spearmanr(k_pred_values, k_exp_values)
    r, p_r = pearsonr(k_pred_values, k_exp_values)

    # Error metrics
    mae = np.mean(np.abs(k_pred_values - k_exp_values))
    rmse = np.sqrt(np.mean((k_pred_values - k_exp_values)**2))
    mape = np.mean(np.abs((k_pred_values - k_exp_values) / k_exp_values)) * 100

    # Outlier analysis (residuals > 2 SD)
    residuals = k_pred_values - k_exp_values
    outlier_threshold = 2 * np.std(residuals)
    outliers = [(aas[i], k_pred_values[i], k_exp_values[i], residuals[i])
                for i in range(len(aas))
                if np.abs(residuals[i]) > outlier_threshold]

    # Per-amino-acid results
    aa_results = []
    for i, aa in enumerate(aas):
        aa_results.append({
            'aa': aa,
            'mass': AA_MASS[aa],
            'k_exp': k_exp_values[i],
            'k_pred': k_pred_values[i],
            'error': residuals[i],
            'pct_error': (residuals[i] / k_exp_values[i]) * 100,
        })

    return {
        'n': len(aas),
        'spearman_rho': rho,
        'spearman_p': p_rho,
        'pearson_r': r,
        'pearson_p': p_r,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'outliers': outliers,
        'aa_results': aa_results,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_correlation(k_pred: Dict[str, float], k_exp: Dict[str, float],
                     stats: Dict, output_path: Path):
    """Create scatter plot with correlation statistics."""

    aas = sorted(set(k_pred.keys()) & set(k_exp.keys()))
    k_pred_values = np.array([k_pred[aa] for aa in aas])
    k_exp_values = np.array([k_exp[aa] for aa in aas])

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    ax.scatter(k_exp_values, k_pred_values, s=100, alpha=0.6, edgecolors='black')

    # Amino acid labels
    for i, aa in enumerate(aas):
        ax.annotate(aa, (k_exp_values[i], k_pred_values[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=10)

    # Identity line
    min_k = min(k_exp_values.min(), k_pred_values.min())
    max_k = max(k_exp_values.max(), k_pred_values.max())
    ax.plot([min_k, max_k], [min_k, max_k], 'r--', lw=2, label='Perfect prediction')

    # Labels and title
    ax.set_xlabel('Experimental Force Constant (kcal/mol/Å²)', fontsize=12)
    ax.set_ylabel('Predicted Force Constant (kcal/mol/Å²)', fontsize=12)
    ax.set_title('Force Constant Validation (v5_11_structural)', fontsize=14, fontweight='bold')

    # Statistics text box
    textstr = f"n = {stats['n']} amino acids\n"
    textstr += f"Spearman ρ = {stats['spearman_rho']:.4f} (p = {stats['spearman_p']:.4f})\n"
    textstr += f"Pearson r = {stats['pearson_r']:.4f} (p = {stats['pearson_p']:.4f})\n"
    textstr += f"MAE = {stats['mae']:.3f}\n"
    textstr += f"RMSE = {stats['rmse']:.3f}\n"
    textstr += f"MAPE = {stats['mape']:.1f}%"

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 90)
    print("FORCE CONSTANT VALIDATION (P0-2)")
    print("=" * 90)
    print()
    print("Hypothesis: k_pred = radius × mass / 100")
    print("Expected: Spearman ρ > 0.80 (from prior finding of ρ=0.86)")
    print()

    # Load embeddings
    z_hyp, codon_to_idx = load_embeddings()
    print()

    # Compute amino acid radii
    print("Computing amino acid radii (average over synonymous codons)...")
    aa_radii = compute_amino_acid_radii(z_hyp, codon_to_idx)
    print(f"  Computed radii for {len(aa_radii)} amino acids")
    print()

    # Predict force constants
    print("Predicting force constants (k = radius × mass / 100)...")
    k_pred = predict_force_constants(aa_radii)
    print(f"  Predicted force constants for {len(k_pred)} amino acids")
    print()

    # Validate correlation
    print("Validating correlation with experimental force constants...")
    stats = validate_correlation(k_pred, AA_FORCE_CONSTANT_EXP)
    print()

    # Print results
    print("=" * 90)
    print("VALIDATION RESULTS")
    print("=" * 90)
    print()
    print(f"Sample Size: {stats['n']} amino acids")
    print()
    print("CORRELATION:")
    print(f"  Spearman ρ = {stats['spearman_rho']:.4f} (p = {stats['spearman_p']:.6f})")
    print(f"  Pearson r  = {stats['pearson_r']:.4f} (p = {stats['pearson_p']:.6f})")
    print()
    print("ERROR METRICS:")
    print(f"  MAE (Mean Absolute Error)     = {stats['mae']:.4f} kcal/mol/Å²")
    print(f"  RMSE (Root Mean Square Error) = {stats['rmse']:.4f} kcal/mol/Å²")
    print(f"  MAPE (Mean Absolute % Error)  = {stats['mape']:.2f}%")
    print()

    # Interpretation
    if stats['spearman_rho'] >= 0.80 and stats['spearman_p'] < 0.001:
        print(">>> ✅ VALIDATION SUCCESSFUL - Strong correlation (ρ ≥ 0.80, p < 0.001)")
    elif stats['spearman_rho'] >= 0.70 and stats['spearman_p'] < 0.01:
        print(">>> ⚠️  MODERATE CORRELATION - Good but below expected (ρ ≥ 0.70, p < 0.01)")
    else:
        print(">>> ❌ VALIDATION FAILED - Weak or non-significant correlation")
    print()

    # Outlier analysis
    if stats['outliers']:
        print("=" * 90)
        print("OUTLIER ANALYSIS (Residuals > 2 SD)")
        print("=" * 90)
        print()
        print(f"{'AA':<4} {'k_pred':<10} {'k_exp':<10} {'Error':<10} {'Mechanism'}")
        print("-" * 70)
        for aa, k_p, k_e, residual in stats['outliers']:
            mechanism = ""
            if aa == 'C':
                mechanism = "Cysteine (disulfide bonds)"
            elif aa == 'P':
                mechanism = "Proline (ring rigidity)"
            elif aa == 'G':
                mechanism = "Glycine (high flexibility)"
            elif aa == 'W':
                mechanism = "Tryptophan (bulky indole)"

            print(f"{aa:<4} {k_p:<10.3f} {k_e:<10.3f} {residual:+10.3f} {mechanism}")
        print()

    # Per-amino-acid results (sorted by absolute error)
    print("=" * 90)
    print("PER-AMINO-ACID RESULTS (sorted by absolute error)")
    print("=" * 90)
    print()
    print(f"{'AA':<4} {'Mass':<8} {'Radius':<10} {'k_pred':<10} {'k_exp':<10} {'Error':<10} {'% Error':<10}")
    print("-" * 80)

    sorted_results = sorted(stats['aa_results'], key=lambda x: abs(x['error']))
    for result in sorted_results:
        aa = result['aa']
        radius = aa_radii.get(aa, 0)
        print(f"{aa:<4} {result['mass']:<8.2f} {radius:<10.4f} "
              f"{result['k_pred']:<10.3f} {result['k_exp']:<10.3f} "
              f"{result['error']:+10.3f} {result['pct_error']:+10.1f}%")
    print()

    # Create visualization
    plot_path = RESULTS_DIR / "force_constant_validation.png"
    plot_correlation(k_pred, AA_FORCE_CONSTANT_EXP, stats, plot_path)

    # Save results
    output_data = {
        'checkpoint': str(CHECKPOINT_PATH),
        'formula': 'k = radius × mass / 100',
        'statistics': {
            'n': stats['n'],
            'spearman_rho': stats['spearman_rho'],
            'spearman_p': stats['spearman_p'],
            'pearson_r': stats['pearson_r'],
            'pearson_p': stats['pearson_p'],
            'mae': stats['mae'],
            'rmse': stats['rmse'],
            'mape': stats['mape'],
        },
        'amino_acid_results': stats['aa_results'],
        'outliers': [
            {'aa': aa, 'k_pred': k_p, 'k_exp': k_e, 'error': err}
            for aa, k_p, k_e, err in stats['outliers']
        ],
    }

    output_file = RESULTS_DIR / "force_constant_validation.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    return stats


if __name__ == '__main__':
    stats = main()
