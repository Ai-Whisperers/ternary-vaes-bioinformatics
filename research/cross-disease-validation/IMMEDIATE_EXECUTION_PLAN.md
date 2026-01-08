# Immediate Execution Plan: Phase 1 Null Hypothesis Tests

**Doc-Type:** Execution Plan · Version 1.0 · Created 2026-01-03

---

## Purpose

Define executable steps for Phase 1 computational validation using ONLY existing data. No new data acquisition required. Each test is a standalone script with clear inputs, outputs, and success criteria.

---

## Test 1: PTM Clustering Analysis

### Hypothesis

H0: PTMs cluster by modification chemistry (citrullination vs phosphorylation), not disease mechanism (RA vs Tau)

### Data Required

**Already Available:**
- RA citrullination sites: `src/research/bioinformatics/codon_encoder_research/rheumatoid_arthritis/`
- Tau phosphorylation sites: `src/research/bioinformatics/codon_encoder_research/neurodegeneration/alzheimers/data/tau_phospho_database.py`
- TrainableCodonEncoder checkpoint: `research/codon-encoder/training/results/trained_codon_encoder.pt`

### Script Outline

```python
# scripts/phase1_null_tests/test1_ptm_clustering.py

"""
Test 1: Cross-Disease PTM Clustering

Objective: Determine if PTMs cluster by disease (RA vs Tau) or by chemistry (Cit vs Phospho)

Method:
1. Load RA citrullination sites (47 positions on Vimentin, Fibrinogen, etc.)
2. Load Tau phosphorylation sites (47 positions)
3. Embed each PTM in p-adic space using PTMMapper
4. Compute pairwise p-adic distances
5. Hierarchical clustering with k=2 (two diseases)
6. Evaluate: Silhouette score, Adjusted Rand Index

Success Criteria:
- Silhouette > 0.3 (moderate clustering quality)
- ARI > 0.5 (agreement with true disease labels)
- Diseases separate in dendrogram
"""

import numpy as np
import torch
from pathlib import Path

# Imports
from src.encoders import TrainableCodonEncoder
from research.codon_encoder.pipelines.ptm_mapping import PTMMapper
from src.geometry import poincare_distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Configuration
CHECKPOINT_PATH = Path('research/codon-encoder/training/results/trained_codon_encoder.pt')
RESULTS_DIR = Path('research/cross-disease-validation/results/test1_ptm_clustering')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_ra_citrullination_sites():
    """Load RA citrullination sites from research directory."""
    # TODO: Parse from RA research scripts or create consolidated database
    # For now, placeholder structure
    ra_ptms = [
        {'protein': 'Vimentin', 'position': 71, 'residue': 'R', 'ptm_type': 'citrullination'},
        # ... add all 47 sites from RA analysis
    ]
    return ra_ptms

def load_tau_phospho_sites():
    """Load Tau phosphorylation sites from database."""
    from src.research.bioinformatics.codon_encoder_research.neurodegeneration.alzheimers.data.tau_phospho_database import TAU_PHOSPHO_SITES

    tau_ptms = []
    for position, site_info in TAU_PHOSPHO_SITES.items():
        tau_ptms.append({
            'protein': 'Tau',
            'position': position,
            'residue': site_info['residue'],
            'ptm_type': 'phosphorylation'
        })
    return tau_ptms

def compute_ptm_embedding(mapper, protein_seq, position, ptm_type):
    """Compute p-adic embedding for PTM."""
    # Get WT embedding
    wt_embedding = mapper.encoder.encode_sequence(protein_seq)

    # Compute PTM impact (simplified - actual implementation in ptm_mapping.py)
    ptm_impact = mapper.compute_ptm_impact(protein_seq, position, ptm_type)

    return ptm_impact

def compute_pairwise_padic_distances(embeddings):
    """Compute pairwise p-adic distances between embeddings."""
    n = len(embeddings)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            dist = poincare_distance(embeddings[i], embeddings[j], c=1.0).item()
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix

def main():
    print("="*80)
    print("TEST 1: Cross-Disease PTM Clustering Analysis")
    print("="*80)

    # Load encoder
    print("\n[1/6] Loading TrainableCodonEncoder...")
    encoder = TrainableCodonEncoder(latent_dim=16, hidden_dim=64)
    checkpoint = torch.load(CHECKPOINT_PATH)
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder.eval()

    mapper = PTMMapper(encoder)

    # Load PTM sites
    print("\n[2/6] Loading PTM sites...")
    ra_sites = load_ra_citrullination_sites()
    tau_sites = load_tau_phospho_sites()

    print(f"  RA citrullination sites: {len(ra_sites)}")
    print(f"  Tau phosphorylation sites: {len(tau_sites)}")

    # Compute embeddings
    print("\n[3/6] Computing PTM embeddings...")
    all_embeddings = []
    all_labels = []

    for site in ra_sites:
        # Get protein sequence (from UniProt or local database)
        protein_seq = get_protein_sequence(site['protein'])
        emb = compute_ptm_embedding(mapper, protein_seq, site['position'], site['ptm_type'])
        all_embeddings.append(emb)
        all_labels.append('RA')

    for site in tau_sites:
        tau_seq = get_tau_sequence()  # Tau sequence
        emb = compute_ptm_embedding(mapper, tau_seq, site['position'], site['ptm_type'])
        all_embeddings.append(emb)
        all_labels.append('Tau')

    # Compute distance matrix
    print("\n[4/6] Computing pairwise p-adic distances...")
    distance_matrix = compute_pairwise_padic_distances(all_embeddings)

    # Clustering
    print("\n[5/6] Hierarchical clustering...")
    clustering = AgglomerativeClustering(
        n_clusters=2,
        metric='precomputed',
        linkage='average'
    )
    predicted_labels = clustering.fit_predict(distance_matrix)

    # Evaluation
    print("\n[6/6] Evaluating clustering quality...")

    # Silhouette score
    silhouette = silhouette_score(distance_matrix, predicted_labels, metric='precomputed')

    # Adjusted Rand Index
    true_labels_binary = [0 if label == 'RA' else 1 for label in all_labels]
    ari = adjusted_rand_score(true_labels_binary, predicted_labels)

    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"  Interpretation: {'GOOD' if silhouette > 0.3 else 'POOR'} clustering quality")
    print(f"\nAdjusted Rand Index: {ari:.3f}")
    print(f"  Interpretation: {'HIGH' if ari > 0.5 else 'LOW'} agreement with true labels")

    # Decision
    print("\n" + "="*80)
    print("DECISION")
    print("="*80)
    if silhouette > 0.3 and ari > 0.5:
        print("REJECT NULL HYPOTHESIS")
        print("PTMs cluster by disease mechanism (RA vs Tau), not solely by chemistry")
    else:
        print("FAIL TO REJECT NULL HYPOTHESIS")
        print("PTMs do not show disease-specific clustering")

    # Visualization
    print("\n[Saving] Dendrogram and distance heatmap...")

    # Dendrogram
    linkage_matrix = linkage(distance_matrix, method='average')
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, labels=all_labels)
    plt.title('Hierarchical Clustering of PTMs (RA Citrullination vs Tau Phosphorylation)')
    plt.xlabel('PTM Site')
    plt.ylabel('P-adic Distance')
    plt.savefig(RESULTS_DIR / 'dendrogram.png', dpi=300, bbox_inches='tight')

    # Distance heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(distance_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='P-adic Distance')
    plt.title('Pairwise P-adic Distances Between PTMs')
    plt.xlabel('PTM Index')
    plt.ylabel('PTM Index')
    plt.savefig(RESULTS_DIR / 'distance_heatmap.png', dpi=300, bbox_inches='tight')

    # Save results
    results = {
        'silhouette_score': float(silhouette),
        'adjusted_rand_index': float(ari),
        'n_ra_sites': len(ra_sites),
        'n_tau_sites': len(tau_sites),
        'decision': 'REJECT_NULL' if (silhouette > 0.3 and ari > 0.5) else 'FAIL_TO_REJECT'
    }

    import json
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR}")
    print("="*80)

if __name__ == '__main__':
    main()
```

### Expected Runtime

30-60 minutes (embedding computation is the bottleneck)

### Deliverables

```
research/cross-disease-validation/results/test1_ptm_clustering/
├── results.json                # Quantitative results
├── dendrogram.png             # Hierarchical clustering visualization
├── distance_heatmap.png       # Pairwise distance matrix
└── test1_report.md            # Interpretation and conclusion
```

---

## Test 2: Codon Bias in ALS Genes

### Hypothesis

H0: ALS genes (TARDBP, SOD1, FUS) show no enrichment of v=0 codons compared to genome-wide average

### Data Required

**Already Available:**
- RefSeq coding sequences (downloadable via Entrez)
- p-adic valuation function (`src/core/padic_math.py`)

**No Patient Data Needed:** This test uses published gene sequences only

### Script Outline

```python
# scripts/phase1_null_tests/test2_codon_bias.py

"""
Test 2: ALS Gene Codon Bias Analysis

Objective: Test if ALS-associated genes show enrichment of "optimal" (v=0) codons

Method:
1. Download TARDBP, SOD1, FUS coding sequences from RefSeq
2. Extract codon frequencies for each gene
3. Compute fraction of v=0 codons
4. Compare to genome-wide average (estimated ~0.33)
5. Binomial test for each gene
6. Control: Test random genes of similar length and GC content

Success Criteria:
- ≥2 of 3 ALS genes show v=0 enrichment (p < 0.05)
- v=0 fraction > 0.40 (20% enrichment over genome average)
- Random control genes do not show enrichment
"""

import numpy as np
from Bio import Entrez, SeqIO
from src.core.padic_math import padic_valuation
from scipy.stats import binomtest
from pathlib import Path

# Configuration
Entrez.email = "your_email@example.com"  # Required for NCBI
RESULTS_DIR = Path('research/cross-disease-validation/results/test2_codon_bias')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ALS genes (RefSeq accessions)
ALS_GENES = {
    'TARDBP': 'NM_007375.4',  # TDP-43
    'SOD1': 'NM_000454.5',
    'FUS': 'NM_004960.4'
}

# Genome-wide v=0 fraction (literature estimate)
GENOME_V0_FRACTION = 0.33

def download_coding_sequence(accession):
    """Download coding sequence from RefSeq."""
    handle = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
    record = SeqIO.read(handle, "fasta")
    handle.close()
    return str(record.seq)

def codon_to_index(codon):
    """Convert codon (3 nucleotides) to integer index."""
    # Simplified mapping: A=0, C=1, G=2, T=3
    bases = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
    index = 0
    for i, base in enumerate(codon.upper()):
        index += bases[base] * (4 ** (2 - i))
    return index

def get_codon_frequencies(sequence):
    """Extract codon frequencies from coding sequence."""
    codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]
    codon_counts = {}
    for codon in codons:
        if len(codon) == 3 and all(b in 'ACGT' for b in codon.upper()):
            codon_counts[codon.upper()] = codon_counts.get(codon.upper(), 0) + 1
    return codon_counts

def compute_v0_fraction(codon_counts):
    """Compute fraction of codons with p-adic valuation v=0."""
    v0_count = 0
    total_count = 0

    for codon, count in codon_counts.items():
        idx = codon_to_index(codon)
        val = padic_valuation(idx, p=3)

        if val == 0:
            v0_count += count
        total_count += count

    return v0_count / total_count if total_count > 0 else 0

def main():
    print("="*80)
    print("TEST 2: ALS Gene Codon Bias Analysis")
    print("="*80)

    results = {}

    for gene_name, accession in ALS_GENES.items():
        print(f"\n[Processing] {gene_name} ({accession})")

        # Download sequence
        print("  Downloading sequence from RefSeq...")
        cds = download_coding_sequence(accession)

        # Get codon frequencies
        codon_counts = get_codon_frequencies(cds)
        total_codons = sum(codon_counts.values())
        print(f"  Total codons: {total_codons}")

        # Compute v=0 fraction
        v0_fraction = compute_v0_fraction(codon_counts)
        print(f"  v=0 fraction: {v0_fraction:.3f}")

        # Binomial test
        n_v0 = int(v0_fraction * total_codons)
        binom_result = binomtest(n_v0, total_codons, GENOME_V0_FRACTION, alternative='greater')
        p_value = binom_result.pvalue

        print(f"  Binomial test p-value: {p_value:.4f}")
        print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")

        results[gene_name] = {
            'accession': accession,
            'total_codons': total_codons,
            'v0_fraction': float(v0_fraction),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    n_significant = sum(1 for r in results.values() if r['significant'])
    print(f"Genes with significant v=0 enrichment: {n_significant} of {len(ALS_GENES)}")

    for gene_name, res in results.items():
        status = "✓" if res['significant'] else "✗"
        print(f"  {status} {gene_name}: v=0 = {res['v0_fraction']:.3f} (p = {res['p_value']:.4f})")

    # Decision
    print("\n" + "="*80)
    print("DECISION")
    print("="*80)
    if n_significant >= 2:
        print("REJECT NULL HYPOTHESIS")
        print(f"{n_significant} of {len(ALS_GENES)} genes show v=0 enrichment (p < 0.05)")
    else:
        print("FAIL TO REJECT NULL HYPOTHESIS")
        print(f"Only {n_significant} of {len(ALS_GENES)} genes show enrichment")

    # Save results
    import json
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump({
            'genome_v0_baseline': GENOME_V0_FRACTION,
            'genes': results,
            'n_significant': n_significant,
            'decision': 'REJECT_NULL' if n_significant >= 2 else 'FAIL_TO_REJECT'
        }, f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR}")
    print("="*80)

if __name__ == '__main__':
    main()
```

### Expected Runtime

5-10 minutes (RefSeq download + computation)

### Deliverables

```
research/cross-disease-validation/results/test2_codon_bias/
├── results.json              # Quantitative results for all genes
└── test2_report.md           # Interpretation
```

---

## Test 3: Dengue Serotype Distance vs DHF Rates

### Hypothesis

H0: NS1 p-adic distances between serotypes are uncorrelated with observed DHF rates from literature

### Data Required

**Already Available:**
- Dengue NS1 sequences (Paraguay 2011-2024): `deliverables/partners/alejandra_rojas/results/pan_arbovirus_primers/`
- Literature DHF rates (to be compiled from published studies)

### Literature DHF Rate Compilation

Compile from published epidemiological studies:

| Reference | Primary | Secondary | DHF Rate | Notes |
|-----------|---------|-----------|----------|-------|
| Halstead 2007, Lancet Infect Dis | DENV-1 | DENV-2 | 15% | Classic ADE study |
| Halstead 2007 | DENV-2 | DENV-1 | 8% | Asymmetric risk |
| Kliks 1989, J Infect Dis | DENV-1 | DENV-3 | 12% | Thailand cohort |
| Guzman 2013, Trans R Soc Trop Med | DENV-2 | DENV-3 | 18% | Cuba outbreak |
| Guzman 2013 | DENV-3 | DENV-2 | 10% | Asymmetric |
| Kouri 1991, Bull PAHO | DENV-1 | DENV-4 | 7% | Lower risk |
| ... | ... | ... | ... | Compile ≥10 combinations |

### Script Outline

```python
# scripts/phase1_null_tests/test3_dengue_dhf.py

"""
Test 3: Dengue Serotype Distance vs DHF Correlation

Objective: Test if NS1 p-adic distances predict DHF severity from literature

Method:
1. Load consensus NS1 sequences for DENV-1, DENV-2, DENV-3, DENV-4
2. Encode each NS1 using TrainableCodonEncoder
3. Compute pairwise p-adic distances (6 combinations)
4. Load literature DHF rates for primary→secondary serotype combinations
5. Spearman correlation between distance and DHF rate
6. Control: Correlate with serotype prevalence (confound check)

Success Criteria:
- Spearman ρ > 0.6 (moderate-to-strong correlation)
- p < 0.05 (statistically significant)
"""

# Implementation similar to Test 1/2
# Key addition: Literature DHF rate database
```

### Expected Runtime

15-20 minutes

---

## Test 4: HIV Goldilocks Zone Generalization

### Hypothesis

H0: RA Goldilocks PTM distances differ from HIV CTL escape optimal range (5.8-6.9)

### Data Required

**Already Available:**
- RA citrullination Goldilocks PTMs (from RA analysis)
- HIV CTL escape distance range: 5.8-6.9 (from existing HIV research)

### Script Outline

```python
# scripts/phase1_null_tests/test4_goldilocks_generalization.py

"""
Test 4: Goldilocks Zone Generalization Test

Objective: Test if HIV immune escape "Goldilocks zone" applies to RA

Method:
1. Load RA Goldilocks PTMs (successfully modulate immune response)
2. Compute p-adic distances for WT→Citrullinated transitions
3. Test overlap with HIV range (5.8-6.9)
4. Statistical test: Is RA mean within HIV range?

Success Criteria:
- >70% of RA Goldilocks PTMs fall in HIV range
- RA mean distance within HIV 95% CI
"""

# Implementation straightforward
```

### Expected Runtime

10-15 minutes

---

## Test 5: Contact Prediction for Disease PPIs

### Hypothesis

H0: P-adic contact prediction fails for disease-relevant protein interactions (AUC ≤ 0.55)

### Data Required

**Partially Available:**
- TDP-43 + hnRNP A1 interaction (literature-documented)
- PDB structure for TDP-43 RRM domain (4BS2)
- Known interface residues (from literature or structural analysis)

**Challenge:** May need to identify known disease PPI interfaces from literature

### Script Outline

```python
# scripts/phase1_null_tests/test5_contact_prediction_ppi.py

"""
Test 5: Contact Prediction Extension to Disease PPIs

Objective: Test if contact prediction works for disease-relevant interactions

Method:
1. Load TDP-43 and hnRNP A1 sequences
2. Predict inter-protein contacts using existing contact prediction framework
3. Load known interface from PDB or literature
4. Compute AUC (predicted vs known contacts)
5. Compare to random protein (negative control)

Success Criteria:
- AUC > 0.65 for true PPI
- AUC(true PPI) - AUC(random) > 0.15
"""

# Implementation uses existing contact prediction framework
```

### Expected Runtime

20-30 minutes (contact prediction is compute-intensive)

---

## Execution Order

### Week 1: Test 2 (Codon Bias)
**Rationale:** Simplest test, no dependencies, quick validation

### Week 2: Test 1 (PTM Clustering)
**Rationale:** Requires PTM data consolidation, but most critical test

### Week 3: Test 4 (Goldilocks Generalization)
**Rationale:** Quick test, validates HIV framework extension

### Week 4: Test 3 (Dengue DHF)
**Rationale:** Requires literature compilation, important for Rojas package

### Week 5: Test 5 (Contact Prediction)
**Rationale:** Most complex, may require PDB analysis

---

## Documentation Standards

### For Each Test

**Required Outputs:**
1. `results.json` - Quantitative results with decision
2. Visualizations (plots, dendrograms, heatmaps)
3. `testN_report.md` - Interpretation with:
   - Executive summary (1 paragraph)
   - Methods (reproducible detail)
   - Results (statistical tests, effect sizes)
   - Decision (reject null or fail to reject)
   - Limitations (confounds, assumptions)
   - Next steps (if test succeeds or fails)

**Transparency Requirements:**
- Report exact p-values (not "p < 0.05")
- Report effect sizes (Cohen's d, Spearman ρ, etc.)
- Include negative results (e.g., "Test 3 failed to reject null")
- Document code commits BEFORE running tests (prevent p-hacking)

---

## Phase 1 Completion Criteria

### After All 5 Tests

**Aggregate Report:**
```
research/cross-disease-validation/results/phase1_summary.md

Contents:
- Tests Passed: N of 5
- Tests Failed: 5 - N
- Decision Matrix (see NULL_HYPOTHESIS_TESTS.md)
- Overall Conclusion:
  - ≥3 passed → "Computational evidence supports conjecture, proceed to Phase 2"
  - 2 passed → "Weak evidence, refine hypotheses before Phase 2"
  - ≤1 passed → "Conjecture lacks computational support, pivot or abandon"
- Next Steps based on outcome
```

**Go/No-Go Decision:**
- Document decision in `PHASE1_DECISION.md`
- If Go: Draft Phase 2 data acquisition plan
- If No-Go: Analyze failures, propose refinements or alternative hypotheses

---

## Support Scripts (To Be Created)

### Data Loading Utilities

```python
# scripts/utils/load_ra_ptms.py
def load_ra_citrullination_sites():
    """Consolidate RA PTM data from research directory."""
    pass

# scripts/utils/load_tau_ptms.py
def load_tau_phospho_sites():
    """Load Tau phosphorylation database."""
    pass

# scripts/utils/protein_sequences.py
def get_protein_sequence(protein_name):
    """Fetch protein sequence from UniProt or local cache."""
    pass
```

### Visualization Utilities

```python
# scripts/utils/visualization.py
def plot_dendrogram(linkage_matrix, labels, output_path):
    """Standardized dendrogram plotting."""
    pass

def plot_distance_heatmap(distance_matrix, labels, output_path):
    """Standardized heatmap plotting."""
    pass
```

---

## Next Immediate Action

1. Create `scripts/phase1_null_tests/` directory
2. Implement Test 2 (simplest, validates framework)
3. Run Test 2, document results
4. Create `test2_report.md` with interpretation
5. Commit code and results to version control
6. Proceed to Test 1 (most critical)

---

**Status:** Ready for execution
**Estimated Phase 1 Completion:** 5 weeks (1 test per week)
**Commitment:** Report all results transparently, accept negative outcomes
