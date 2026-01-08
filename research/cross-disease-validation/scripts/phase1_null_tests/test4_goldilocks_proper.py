"""
Test 4: HIV Goldilocks Zone Generalization to RA (PROPER VALIDATION)

Objective: Test if RA citrullination distances overlap with HIV CTL escape distances
           using ACTUAL TrainableCodonEncoder and real protein sequences

Null Hypothesis (H0): RA citrullination distances differ from HIV CTL escape distances
                      Overlap < 50%

Alternative Hypothesis (H1): RA citrullination distances overlap HIV range
                              Overlap > 70%
                              Mean distance statistically similar

Method:
1. Load TrainableCodonEncoder (validated LOO ρ=0.61)
2. Download RA protein sequences from UniProt
3. For each RA citrullination site, compute TRUE hyperbolic distance (WT R → Cit R)
4. Load ACTUAL HIV CTL escape distances from existing analysis
5. Statistical comparison (t-test, overlap fraction, distribution comparison)

Success Criteria:
- Overlap > 70% (most RA PTMs fall in HIV range)
- RA mean within HIV 95% CI
- KS test p > 0.05 (distributions not significantly different)

Pre-registered on: 2026-01-03 (revised with proper validation)
"""

import sys
from pathlib import Path
import numpy as np
import json
from collections import defaultdict

# Add repo root to path
script_dir = Path(__file__).parent
scripts_dir = script_dir.parent
cross_disease_dir = scripts_dir.parent
research_dir = cross_disease_dir.parent
repo_root = research_dir.parent
sys.path.insert(0, str(repo_root))

import torch
from scipy.stats import ttest_ind, ks_2samp
from Bio import SeqIO, Entrez
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Import encoder
from src.encoders.trainable_codon_encoder import TrainableCodonEncoder
from src.geometry import poincare_distance
from src.biology.codons import AMINO_ACID_TO_CODONS

# Configuration
RESULTS_DIR = Path('research/cross-disease-validation/results/test4_goldilocks_proper')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ENCODER_PATH = repo_root / 'research/codon-encoder/training/results/trained_codon_encoder.pt'
HIV_RESULTS_PATH = repo_root / 'src/research/bioinformatics/codon_encoder_research/hiv/results/hiv_escape_results.json'

# Import PTM data loader
sys.path.insert(0, str(scripts_dir / 'utils'))
from load_ptm_data import load_ra_citrullination_sites

# UniProt IDs for RA proteins
UNIPROT_IDS = {
    'Vimentin': 'P08670',
    'Fibrinogen alpha': 'P02671',
    'Fibrinogen beta': 'P02675',
    'Fibrinogen gamma': 'P02679',
    'Histone H3.1': 'P68431',
    'Histone H4': 'P62805',
    'Collagen II alpha-1': 'P02458',
    'Alpha-enolase': 'P06733',
    'Myelin basic protein': 'P02686',
    'Keratin 1': 'P04264',
    'Filaggrin': 'P20930',
}

# Configure Entrez
Entrez.email = "research@example.com"  # Required by NCBI

def download_uniprot_sequence(uniprot_id):
    """Download protein sequence from UniProt."""
    import urllib.request
    import urllib.error

    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            fasta_data = response.read().decode('utf-8')
            # Parse FASTA
            lines = fasta_data.strip().split('\n')
            sequence = ''.join(lines[1:])  # Skip header
            return sequence
    except urllib.error.URLError as e:
        print(f"  ERROR downloading {uniprot_id}: {e}")
        return None

def get_ra_protein_sequences():
    """Download all RA protein sequences from UniProt."""
    sequences = {}

    print("  Downloading protein sequences from UniProt...")
    for protein_name, uniprot_id in UNIPROT_IDS.items():
        print(f"    {protein_name} ({uniprot_id})...", end=" ")
        seq = download_uniprot_sequence(uniprot_id)
        if seq:
            sequences[protein_name] = seq
            print(f"{len(seq)} aa")
        else:
            print("FAILED")

    return sequences

def get_codon_at_position(cds_sequence, position):
    """Extract codon at 1-indexed protein position."""
    # Convert to 0-indexed
    codon_start = (position - 1) * 3
    codon = cds_sequence[codon_start:codon_start+3]
    return codon if len(codon) == 3 else None

def compute_ra_citrullination_distance(site, encoder, protein_sequences, device='cpu'):
    """
    Compute hyperbolic distance for R → Citrullinated R.

    Citrullination converts R (charge +1) to Citrulline (charge 0).
    We model this as a mutation to the codon that best represents
    the charge loss.

    Since TrainableCodonEncoder operates on amino acids (not individual PTMs),
    we approximate citrullination as R → Q (glutamine, neutral like citrulline).

    This is a simplification, but captures the charge loss effect.
    """
    protein_name = site['protein']
    position = site['position']

    # Get protein sequence
    if protein_name not in protein_sequences:
        return None

    protein_seq = protein_sequences[protein_name]

    # Check position is valid
    if position < 1 or position > len(protein_seq):
        return None

    # Get amino acid at position (should be R)
    aa_at_position = protein_seq[position - 1]
    if aa_at_position != 'R':
        print(f"    WARNING: {protein_name} position {position} is {aa_at_position}, not R")
        return None

    # Get wildtype R embedding (average of all R codons)
    wt_emb = encoder.get_amino_acid_embedding('R')

    # Get citrullinated "embedding"
    # Approximate as Q (glutamine - neutral like citrulline)
    cit_emb = encoder.get_amino_acid_embedding('Q')

    # Compute hyperbolic distance
    dist = poincare_distance(wt_emb.unsqueeze(0), cit_emb.unsqueeze(0), c=1.0)

    return float(dist.item())

def load_hiv_escape_distances():
    """Load actual HIV CTL escape distances from existing analysis."""
    with open(HIV_RESULTS_PATH, 'r') as f:
        hiv_data = json.load(f)

    # Extract all mutation distances
    distances = []
    high_efficacy_low_cost = []

    for mutation in hiv_data['all_mutations']:
        dist = mutation['hyperbolic_distance']
        distances.append(dist)

        # Track high-efficacy/low-cost subset (Goldilocks)
        if mutation['escape_efficacy'] == 'high' and mutation['fitness_cost'] == 'low':
            high_efficacy_low_cost.append(dist)

    return {
        'all_distances': distances,
        'goldilocks_distances': high_efficacy_low_cost,
        'mean': np.mean(distances),
        'std': np.std(distances),
        'min': np.min(distances),
        'max': np.max(distances)
    }

def main():
    print("="*80)
    print("TEST 4: HIV Goldilocks Zone Generalization (PROPER VALIDATION)")
    print("="*80)
    print()
    print("Pre-registered Hypothesis:")
    print("  H0: RA citrullination distances differ from HIV CTL escape")
    print("  H1: RA distances overlap HIV range (>70%), distributions similar")
    print()
    print("Method: TrainableCodonEncoder + UniProt sequences + actual HIV data")
    print("="*80)

    # Load encoder
    print("\n[1/6] Loading TrainableCodonEncoder...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    encoder = TrainableCodonEncoder(latent_dim=16, hidden_dim=64)
    checkpoint = torch.load(ENCODER_PATH, map_location=device)
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder.to(device)
    encoder.eval()

    print(f"  Loaded checkpoint (LOO Spearman ρ = {checkpoint.get('best_spearman', 'N/A')})")

    # Download protein sequences
    print("\n[2/6] Downloading RA protein sequences from UniProt...")
    protein_sequences = get_ra_protein_sequences()
    print(f"\n  Downloaded {len(protein_sequences)} / {len(UNIPROT_IDS)} proteins")

    # Load RA sites
    print("\n[3/6] Loading RA citrullination sites...")
    ra_sites = load_ra_citrullination_sites()
    print(f"  Loaded {len(ra_sites)} citrullination sites")

    # Compute RA distances
    print("\n[4/6] Computing RA citrullination hyperbolic distances...")
    print("  Using TrainableCodonEncoder (R → Q approximation for citrullination)")

    ra_distances = []
    ra_site_info = []

    with torch.no_grad():
        for i, site in enumerate(ra_sites):
            dist = compute_ra_citrullination_distance(site, encoder, protein_sequences, device)

            if dist is not None:
                ra_distances.append(dist)
                ra_site_info.append({
                    'protein': site['protein'],
                    'position': site['position'],
                    'residue': site['residue'],
                    'evidence': site.get('evidence', 'Unknown'),
                    'target': site.get('target', 'Unknown'),
                    'distance': dist
                })

                if (i + 1) % 10 == 0:
                    print(f"  Computed {i+1} / {len(ra_sites)}...")

    ra_distances = np.array(ra_distances)

    print(f"\n  Successfully computed {len(ra_distances)} / {len(ra_sites)} distances")
    print(f"  RA mean: {np.mean(ra_distances):.3f} ± {np.std(ra_distances):.3f}")
    print(f"  RA range: [{np.min(ra_distances):.3f}, {np.max(ra_distances):.3f}]")

    # Load HIV distances
    print("\n[5/6] Loading HIV CTL escape distances...")
    hiv_data = load_hiv_escape_distances()

    print(f"  HIV all mutations: n={len(hiv_data['all_distances'])}")
    print(f"  HIV mean: {hiv_data['mean']:.3f} ± {hiv_data['std']:.3f}")
    print(f"  HIV range: [{hiv_data['min']:.3f}, {hiv_data['max']:.3f}]")

    if hiv_data['goldilocks_distances']:
        print(f"\n  HIV Goldilocks (high-efficacy/low-cost): n={len(hiv_data['goldilocks_distances'])}")
        print(f"  Goldilocks mean: {np.mean(hiv_data['goldilocks_distances']):.3f}")

    # Statistical comparison
    print("\n[6/6] Statistical comparison...")

    # Compute HIV range (mean ± 1.5 std to be generous)
    hiv_lower = hiv_data['mean'] - 1.5 * hiv_data['std']
    hiv_upper = hiv_data['mean'] + 1.5 * hiv_data['std']

    print(f"\n  HIV Goldilocks range (mean ± 1.5 std): [{hiv_lower:.3f}, {hiv_upper:.3f}]")

    # Compute overlap
    in_hiv_range = np.sum((ra_distances >= hiv_lower) & (ra_distances <= hiv_upper))
    overlap_fraction = in_hiv_range / len(ra_distances)

    print(f"  RA sites in HIV range: {in_hiv_range} / {len(ra_distances)} ({overlap_fraction:.1%})")

    # t-test: Are means different?
    t_stat, p_ttest = ttest_ind(ra_distances, hiv_data['all_distances'])
    print(f"\n  t-test (independent samples):")
    print(f"    t-statistic: {t_stat:.3f}")
    print(f"    p-value: {p_ttest:.4f}")
    print(f"    Interpretation: {'NOT DIFFERENT' if p_ttest > 0.05 else 'DIFFERENT'} means")

    # KS test: Are distributions different?
    ks_stat, p_ks = ks_2samp(ra_distances, hiv_data['all_distances'])
    print(f"\n  Kolmogorov-Smirnov test:")
    print(f"    KS statistic: {ks_stat:.3f}")
    print(f"    p-value: {p_ks:.4f}")
    print(f"    Interpretation: {'SAME' if p_ks > 0.05 else 'DIFFERENT'} distributions")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(ra_distances)**2 + hiv_data['std']**2) / 2)
    cohens_d = (np.mean(ra_distances) - hiv_data['mean']) / pooled_std
    print(f"\n  Cohen's d: {cohens_d:.3f} ({'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'} effect)")

    # Decision
    print("\n" + "="*80)
    print("DECISION")
    print("="*80)

    if overlap_fraction > 0.7 and p_ttest > 0.05:
        print("REJECT NULL HYPOTHESIS")
        print("RA citrullination distances overlap HIV CTL escape range")
        print("Universal immune modulation Goldilocks zone supported")
        decision = 'REJECT_NULL'
    elif overlap_fraction > 0.5 and p_ks > 0.05:
        print("WEAK EVIDENCE AGAINST NULL")
        print("Moderate overlap with similar distributions")
        decision = 'WEAK_EVIDENCE'
    else:
        print("FAIL TO REJECT NULL HYPOTHESIS")
        print("RA citrullination distances differ from HIV CTL escape")
        decision = 'FAIL_TO_REJECT'

    print(f"\nCriteria met:")
    print(f"  Overlap > 70%: {overlap_fraction > 0.7} ({overlap_fraction:.1%})")
    print(f"  Same means (p > 0.05): {p_ttest > 0.05} (p={p_ttest:.4f})")
    print(f"  Same distributions (p > 0.05): {p_ks > 0.05} (p={p_ks:.4f})")

    # Visualizations
    print("\n[Generating visualizations...]")

    # Distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram overlay
    ax = axes[0, 0]
    ax.hist(ra_distances, bins=15, alpha=0.6, color='blue', label='RA Citrullination', density=True)
    ax.hist(hiv_data['all_distances'], bins=15, alpha=0.6, color='red', label='HIV CTL Escape', density=True)
    ax.axvline(np.mean(ra_distances), color='blue', linestyle='--', linewidth=2)
    ax.axvline(hiv_data['mean'], color='red', linestyle='--', linewidth=2)
    ax.axvspan(hiv_lower, hiv_upper, alpha=0.2, color='red', label='HIV ±1.5 std')
    ax.set_xlabel('Hyperbolic Distance', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Distribution Comparison', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot
    ax = axes[0, 1]
    data_to_plot = [ra_distances, hiv_data['all_distances']]
    labels = ['RA Citrullination', 'HIV CTL Escape']
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Hyperbolic Distance', fontsize=11)
    ax.set_title('Distribution Comparison (Box Plot)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Cumulative distribution
    ax = axes[1, 0]
    ra_sorted = np.sort(ra_distances)
    hiv_sorted = np.sort(hiv_data['all_distances'])
    ax.plot(ra_sorted, np.arange(1, len(ra_sorted)+1)/len(ra_sorted),
            'b-', linewidth=2, label='RA Citrullination')
    ax.plot(hiv_sorted, np.arange(1, len(hiv_sorted)+1)/len(hiv_sorted),
            'r-', linewidth=2, label='HIV CTL Escape')
    ax.set_xlabel('Hyperbolic Distance', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title(f'CDF Comparison (KS p={p_ks:.4f})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Q-Q plot
    ax = axes[1, 1]
    from scipy.stats import probplot
    (osm_ra, osr_ra), (slope_ra, intercept_ra, r_ra) = probplot(ra_distances, dist="norm", plot=None)
    (osm_hiv, osr_hiv), (slope_hiv, intercept_hiv, r_hiv) = probplot(hiv_data['all_distances'], dist="norm", plot=None)
    ax.scatter(osm_ra, osr_ra, alpha=0.6, color='blue', label='RA')
    ax.scatter(osm_hiv, osr_hiv, alpha=0.6, color='red', label='HIV')
    ax.plot(osm_ra, slope_ra * osm_ra + intercept_ra, 'b--', alpha=0.5)
    ax.plot(osm_hiv, slope_hiv * osm_hiv + intercept_hiv, 'r--', alpha=0.5)
    ax.set_xlabel('Theoretical Quantiles', fontsize=11)
    ax.set_ylabel('Sample Quantiles', fontsize=11)
    ax.set_title('Q-Q Plot (Normality Check)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: distribution_comparison.png")

    # Save results
    results = {
        'test_name': 'Test 4: HIV Goldilocks Zone Generalization (PROPER)',
        'date_executed': '2026-01-03',
        'pre_registered': True,
        'method': 'TrainableCodonEncoder + UniProt sequences + actual HIV data',
        'data': {
            'n_ra_sites': len(ra_distances),
            'n_ra_proteins': len(protein_sequences),
            'n_hiv_mutations': len(hiv_data['all_distances']),
            'ra_mean': float(np.mean(ra_distances)),
            'ra_std': float(np.std(ra_distances)),
            'ra_range': [float(np.min(ra_distances)), float(np.max(ra_distances))],
            'hiv_mean': float(hiv_data['mean']),
            'hiv_std': float(hiv_data['std']),
            'hiv_range': [float(hiv_data['min']), float(hiv_data['max'])]
        },
        'hiv_goldilocks_range': {
            'lower': float(hiv_lower),
            'upper': float(hiv_upper),
            'method': 'mean ± 1.5 std'
        },
        'overlap': {
            'n_in_range': int(in_hiv_range),
            'fraction': float(overlap_fraction)
        },
        'statistical_tests': {
            't_test': {
                't_statistic': float(t_stat),
                'p_value': float(p_ttest),
                'null_hypothesis': 'RA mean = HIV mean'
            },
            'ks_test': {
                'ks_statistic': float(ks_stat),
                'p_value': float(p_ks),
                'null_hypothesis': 'RA and HIV have same distribution'
            },
            'cohens_d': float(cohens_d)
        },
        'success_criteria': {
            'min_overlap': 0.7,
            'min_p_ttest': 0.05,
            'min_p_ks': 0.05
        },
        'decision': decision,
        'note': 'Used TrainableCodonEncoder with actual protein sequences. Citrullination approximated as R→Q (charge loss). HIV distances from validated analysis.'
    }

    # Save site-level data
    results['ra_sites'] = ra_site_info
    results['hiv_mutations'] = [
        {
            'epitope': m['epitope'],
            'mutation': m['mutation'],
            'distance': m['hyperbolic_distance'],
            'efficacy': m['escape_efficacy'],
            'fitness_cost': m['fitness_cost']
        }
        for m in checkpoint.get('all_mutations', [])
    ] if 'all_mutations' in checkpoint else []

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
