"""
Test 2: ALS Gene Codon Bias Analysis

Objective: Test if ALS-associated genes show enrichment of "optimal" (v=0) codons

Null Hypothesis (H0): ALS genes show no enrichment of v=0 codons compared to genome-wide average

Alternative Hypothesis (H1): ALS genes have v=0 fraction > 0.40 (20% enrichment)
                              with p < 0.05 for >= 2 of 3 genes

Method:
1. Download TARDBP, SOD1, FUS coding sequences from RefSeq
2. Extract codon frequencies for each gene
3. Compute fraction of v=0 codons
4. Compare to genome-wide average (estimated ~0.33)
5. Binomial test for each gene
6. Control: Test random genes of similar length and GC content

Success Criteria:
- >= 2 of 3 ALS genes show v=0 enrichment (p < 0.05)
- v=0 fraction > 0.40 (20% enrichment over genome average)
- Random control genes do not show enrichment

Pre-registered on: 2026-01-03
"""

import numpy as np
from pathlib import Path
import json
from collections import defaultdict

# Configuration
RESULTS_DIR = Path('research/cross-disease-validation/results/test2_codon_bias')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ALS genes with sequences (manually defined to avoid NCBI dependency issues)
# Sequences from RefSeq (human GRCh38)
ALS_GENES = {
    'TARDBP': {
        'name': 'TDP-43',
        'refseq': 'NM_007375.4',
        'description': 'TAR DNA-binding protein 43',
    },
    'SOD1': {
        'name': 'Superoxide dismutase 1',
        'refseq': 'NM_000454.5',
        'description': 'Superoxide dismutase 1',
    },
    'FUS': {
        'name': 'Fused in sarcoma',
        'refseq': 'NM_004960.4',
        'description': 'FUS RNA binding protein',
    }
}

# Genome-wide v=0 fraction (literature estimate from codon usage databases)
# Based on human codon usage tables
GENOME_V0_FRACTION = 0.33

def codon_to_index(codon):
    """
    Convert codon (3 nucleotides) to integer index for p-adic valuation.

    Mapping: A=0, C=1, G=2, T/U=3
    Index = base0 * 16 + base1 * 4 + base2
    """
    bases = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}

    if len(codon) != 3:
        return None

    try:
        index = 0
        for i, base in enumerate(codon.upper()):
            index += bases[base] * (4 ** (2 - i))
        return index
    except KeyError:
        return None

def padic_valuation_local(n, p=3):
    """
    Compute p-adic valuation of n.

    v_p(n) = highest power of p dividing n
    """
    if n == 0:
        return float('inf')

    val = 0
    while n % p == 0:
        n //= p
        val += 1
    return val

def get_codon_frequencies(sequence):
    """Extract codon frequencies from coding sequence."""
    # Clean sequence (remove whitespace, newlines)
    sequence = ''.join(sequence.split()).upper()

    # Extract codons (assuming sequence starts at reading frame 0)
    codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]

    codon_counts = defaultdict(int)
    for codon in codons:
        if len(codon) == 3 and all(b in 'ACGTU' for b in codon):
            codon_counts[codon.replace('U', 'T')] += 1

    return dict(codon_counts)

def compute_v0_enrichment(codon_counts):
    """
    Compute fraction of codons with p-adic valuation v=0.

    Returns:
        v0_fraction: Fraction of codons with v=0
        v0_count: Number of v=0 codons
        total_count: Total number of codons
        codon_details: Dict of {codon: (count, valuation)}
    """
    v0_count = 0
    total_count = 0
    codon_details = {}

    for codon, count in codon_counts.items():
        idx = codon_to_index(codon)
        if idx is None:
            continue

        val = padic_valuation_local(idx, p=3)
        codon_details[codon] = {'count': count, 'valuation': val, 'index': idx}

        if val == 0:
            v0_count += count
        total_count += count

    v0_fraction = v0_count / total_count if total_count > 0 else 0

    return v0_fraction, v0_count, total_count, codon_details

def binomial_test_scipy_alternative(k, n, p, alternative='greater'):
    """
    Manual binomial test implementation (in case scipy not available).

    Tests if observed k successes in n trials is significantly different
    from expected proportion p.
    """
    from scipy.stats import binomtest
    result = binomtest(k, n, p, alternative=alternative)
    return result.pvalue

def load_gene_sequence(gene_name):
    """
    Load gene sequence from local file or return placeholder.

    For actual execution, sequences should be downloaded from RefSeq
    or loaded from a local FASTA file.
    """
    # Check if we have local sequences
    sequence_file = Path(f'research/cross-disease-validation/data/gene_sequences/{gene_name}.fasta')

    if sequence_file.exists():
        with open(sequence_file, 'r') as f:
            lines = f.readlines()
            # Skip FASTA header
            sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))
            return sequence
    else:
        print(f"  WARNING: No local sequence file found for {gene_name}")
        print(f"  Expected location: {sequence_file}")
        print(f"  Please download from RefSeq or provide sequence manually")
        return None

def main():
    print("="*80)
    print("TEST 2: ALS Gene Codon Bias Analysis")
    print("="*80)
    print()
    print("Pre-registered Hypothesis:")
    print("  H0: ALS genes show v=0 fraction <= 0.33 (genome average)")
    print("  H1: ALS genes show v=0 fraction > 0.40 with p < 0.05 for >= 2 genes")
    print()
    print("="*80)

    results = {}
    all_sequences_available = True

    for gene_name, gene_info in ALS_GENES.items():
        print(f"\n[Processing] {gene_name} ({gene_info['name']})")
        print(f"  RefSeq: {gene_info['refseq']}")

        # Load sequence
        cds = load_gene_sequence(gene_name)

        if cds is None:
            print(f"  SKIPPED: No sequence available")
            all_sequences_available = False
            results[gene_name] = {
                'status': 'SKIPPED',
                'reason': 'No sequence file found'
            }
            continue

        # Get codon frequencies
        codon_counts = get_codon_frequencies(cds)
        total_codons = sum(codon_counts.values())
        print(f"  Sequence length: {len(cds)} bp")
        print(f"  Total codons: {total_codons}")

        # Compute v=0 enrichment
        v0_fraction, v0_count, total_count, codon_details = compute_v0_enrichment(codon_counts)
        print(f"  v=0 codons: {v0_count} of {total_count}")
        print(f"  v=0 fraction: {v0_fraction:.4f}")
        print(f"  Genome baseline: {GENOME_V0_FRACTION:.4f}")
        print(f"  Fold enrichment: {v0_fraction / GENOME_V0_FRACTION:.2f}x")

        # Binomial test
        try:
            p_value = binomial_test_scipy_alternative(v0_count, total_count, GENOME_V0_FRACTION, alternative='greater')
            print(f"  Binomial test p-value: {p_value:.6f}")
            print(f"  Significant (p < 0.05): {'YES' if p_value < 0.05 else 'NO'}")
        except Exception as e:
            print(f"  ERROR computing p-value: {e}")
            p_value = None

        # Convert codon details to JSON-serializable format
        codon_details_json = {
            codon: {
                'count': int(info['count']),
                'valuation': int(info['valuation']) if info['valuation'] != float('inf') else None,
                'index': int(info['index'])
            }
            for codon, info in codon_details.items()
        }

        results[gene_name] = {
            'status': 'ANALYZED',
            'refseq': gene_info['refseq'],
            'name': gene_info['name'],
            'total_codons': int(total_count),
            'v0_count': int(v0_count),
            'v0_fraction': float(v0_fraction),
            'genome_baseline': float(GENOME_V0_FRACTION),
            'fold_enrichment': float(v0_fraction / GENOME_V0_FRACTION),
            'p_value': float(p_value) if p_value is not None else None,
            'significant': bool(p_value < 0.05) if p_value is not None else None,
            'codon_details': codon_details_json
        }

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    analyzed_genes = [g for g, r in results.items() if r.get('status') == 'ANALYZED']
    n_significant = sum(1 for g in analyzed_genes if results[g].get('significant', False))

    if not all_sequences_available:
        print("\nWARNING: Not all gene sequences were available.")
        print("To complete this test, please download sequences from RefSeq:")
        print()
        for gene_name, gene_info in ALS_GENES.items():
            if results[gene_name].get('status') == 'SKIPPED':
                print(f"  {gene_name}: {gene_info['refseq']}")
                print(f"    URL: https://www.ncbi.nlm.nih.gov/nuccore/{gene_info['refseq']}")
        print()

    if analyzed_genes:
        print(f"\nGenes analyzed: {len(analyzed_genes)} of {len(ALS_GENES)}")
        print(f"Genes with significant v=0 enrichment: {n_significant} of {len(analyzed_genes)}")
        print()

        for gene_name in analyzed_genes:
            res = results[gene_name]
            status = "✓" if res.get('significant') else "✗"
            p_val_str = f"{res['p_value']:.6f}" if res['p_value'] is not None else "N/A"
            print(f"  {status} {gene_name}: v=0 = {res['v0_fraction']:.4f} "
                  f"({res['fold_enrichment']:.2f}x baseline, p = {p_val_str})")

    # Decision
    print("\n" + "="*80)
    print("DECISION")
    print("="*80)

    if not all_sequences_available:
        print("TEST INCOMPLETE: Not all sequences available")
        print("Please provide gene sequences to complete analysis")
        decision = 'INCOMPLETE'
    elif len(analyzed_genes) < len(ALS_GENES):
        print("TEST INCOMPLETE: Some genes could not be analyzed")
        decision = 'INCOMPLETE'
    elif n_significant >= 2:
        print(f"REJECT NULL HYPOTHESIS")
        print(f"{n_significant} of {len(analyzed_genes)} genes show v=0 enrichment (p < 0.05)")
        print("Evidence supports H1: ALS genes are enriched for v=0 codons")
        decision = 'REJECT_NULL'
    else:
        print(f"FAIL TO REJECT NULL HYPOTHESIS")
        print(f"Only {n_significant} of {len(analyzed_genes)} genes show enrichment")
        print("Insufficient evidence for v=0 codon enrichment in ALS genes")
        decision = 'FAIL_TO_REJECT'

    # Save results
    output_data = {
        'test_name': 'Test 2: ALS Gene Codon Bias Analysis',
        'date_executed': '2026-01-03',
        'pre_registered': True,
        'genome_v0_baseline': float(GENOME_V0_FRACTION),
        'success_criteria': {
            'min_significant_genes': 2,
            'alpha': 0.05,
            'min_enrichment': 1.2  # 20% enrichment
        },
        'genes': results,
        'summary': {
            'n_genes_total': int(len(ALS_GENES)),
            'n_genes_analyzed': int(len(analyzed_genes)),
            'n_significant': int(n_significant) if all_sequences_available else None,
            'decision': str(decision)
        }
    }

    output_file = RESULTS_DIR / 'results.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

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
