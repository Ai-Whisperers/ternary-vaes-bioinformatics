# Null Hypothesis Tests: Cross-Disease Validation

**Doc-Type:** Statistical Testing Framework · Version 1.0 · Created 2026-01-03

---

## Purpose

Define falsifiable null hypotheses for each component of the cross-disease conjecture. Tests are designed to be executable with existing data before acquiring new datasets.

---

## Test 1: PTM Clustering is Not Disease-Specific

### H0: PTMs Cluster by Modification Chemistry, Not Disease Mechanism

**Data:**
- RA citrullination sites: 47 positions (existing)
- Tau phosphorylation sites: 47 positions (existing)
- PTM types: Citrullination (RA), Phosphorylation (Tau)

**Method:**
```python
# Embed all 94 PTM sites in p-adic space
from research.codon_encoder.pipelines.ptm_mapping import PTMMapper
from src.encoders import TrainableCodonEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

encoder = TrainableCodonEncoder.load('research/codon-encoder/training/results/trained_codon_encoder.pt')
mapper = PTMMapper(encoder)

# RA citrullination sites (Vimentin, Fibrinogen, etc.)
ra_ptms = load_ra_citrullination_sites()  # 47 sites
ra_embeddings = [mapper.compute_ptm_impact(protein, pos, 'citrullination') for protein, pos in ra_ptms]

# Tau phosphorylation sites
tau_ptms = load_tau_phospho_sites()  # 47 sites
tau_embeddings = [mapper.compute_ptm_impact('Tau', pos, 'phosphorylation') for pos in tau_ptms]

# Concatenate
all_embeddings = ra_embeddings + tau_embeddings
true_labels = ['RA'] * 47 + ['Tau'] * 47

# Compute pairwise p-adic distances
from src.geometry import poincare_distance
distance_matrix = compute_pairwise_distances(all_embeddings, metric=poincare_distance)

# Hierarchical clustering
clustering = AgglomerativeClustering(n_clusters=2, metric='precomputed', linkage='average')
predicted_labels = clustering.fit_predict(distance_matrix)

# Evaluate clustering quality
silhouette = silhouette_score(distance_matrix, predicted_labels, metric='precomputed')
ari = adjusted_rand_score(true_labels, predicted_labels)
```

**Null Hypothesis:**
- PTMs cluster randomly with respect to disease (RA vs Tau)
- Silhouette score ≤ 0.2 (poor clustering)
- Adjusted Rand Index ≤ 0.3 (low agreement with true labels)

**Alternative Hypothesis:**
- PTMs cluster by disease mechanism
- Silhouette score > 0.3 (moderate clustering)
- ARI > 0.5 (moderate-to-strong agreement)

**Falsification Criterion:**
- If silhouette < 0.2 AND ARI < 0.3 → **Reject conjecture** (PTMs do not cluster by disease)
- If silhouette > 0.3 AND ARI > 0.5 → **Support conjecture** (disease-specific clustering)

**Confound Analysis:**
- Control test: Cluster by modification type (citrullination vs phosphorylation) instead of disease
- If modification-based clustering has higher silhouette, chemistry dominates over disease

---

## Test 2: Codon Bias is Not Tissue-Specific for ALS Genes

### H0: ALS Genes Show No Enrichment of v=0 Codons in Motor Neurons

**Data:**
- TARDBP (TDP-43), SOD1, FUS coding sequences (RefSeq, available)
- Human genome-wide codon usage (available)
- Optional: GTEx motor cortex RNA-seq (requires download, defer to Phase 2)

**Method (Without GTEx):**
```python
from Bio import SeqIO
from src.core.padic_math import padic_valuation
from scipy.stats import chi2_contingency
import numpy as np

# Download ALS gene coding sequences
als_genes = {
    'TARDBP': 'NM_007375',  # TDP-43
    'SOD1': 'NM_000454',
    'FUS': 'NM_004960'
}

def get_codon_usage(sequence):
    """Extract codon frequencies from coding sequence."""
    codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]
    codon_counts = {}
    for codon in codons:
        if len(codon) == 3:
            codon_counts[codon] = codon_counts.get(codon, 0) + 1
    return codon_counts

def compute_valuation_enrichment(codon_usage):
    """Compute enrichment of v=0 codons."""
    v0_count = sum(count for codon, count in codon_usage.items()
                   if padic_valuation(codon_to_index(codon), p=3) == 0)
    total_count = sum(codon_usage.values())
    return v0_count / total_count

# Compare to genome-wide average
genome_v0_fraction = 0.33  # Approximately 1/3 of codons have v=0 (literature estimate)

als_v0_fractions = {}
for gene, accession in als_genes.items():
    seq = download_refseq(accession)
    codon_usage = get_codon_usage(seq)
    v0_frac = compute_valuation_enrichment(codon_usage)
    als_v0_fractions[gene] = v0_frac

    # Binomial test: is v0_frac significantly > genome_v0_fraction?
    from scipy.stats import binomial_test
    n_codons = sum(codon_usage.values())
    n_v0 = int(v0_frac * n_codons)
    p_value = binomial_test(n_v0, n_codons, genome_v0_fraction, alternative='greater')

    print(f"{gene}: v=0 fraction = {v0_frac:.3f}, p = {p_value:.4f}")
```

**Null Hypothesis:**
- ALS genes have v=0 codon fraction ≤ genome-wide average (0.33)
- p > 0.05 for all 3 genes (no enrichment)

**Alternative Hypothesis:**
- ALS genes have v=0 fraction > 0.40 (20% enrichment)
- p < 0.05 for ≥2 of 3 genes

**Falsification Criterion:**
- If p > 0.05 for all genes → **Reject H1** (no codon bias at gene level)
- Note: This tests gene-level bias, not tissue-specific bias (requires GTEx)

**Confound Analysis:**
- Control: Test random genes (same length, same GC content) for v=0 enrichment
- If random genes also show enrichment, it's a genome-wide pattern, not ALS-specific

---

## Test 3: Dengue Serotype Distances Do Not Predict DHF Severity

### H0: NS1 P-adic Distances Uncorrelated with Observed DHF Rates

**Data:**
- Dengue NS1 sequences for 4 serotypes (Paraguay data available)
- Published DHF rates for serotype combinations (literature)

**DHF Rate Data (Literature):**
| Primary | Secondary | Observed DHF Rate | Source |
|---------|-----------|-------------------|---------|
| DENV-1 | DENV-2 | 0.15 (15%) | Halstead 2007 |
| DENV-2 | DENV-1 | 0.08 (8%) | Halstead 2007 |
| DENV-1 | DENV-3 | 0.12 (12%) | Kliks 1989 |
| DENV-2 | DENV-3 | 0.18 (18%) | Guzman 2013 |
| DENV-3 | DENV-2 | 0.10 (10%) | Guzman 2013 |
| ... | ... | ... | Multiple sources |

**Method:**
```python
from Bio import SeqIO
from src.encoders import TrainableCodonEncoder
from src.geometry import poincare_distance
from scipy.stats import spearmanr
import pandas as pd

# Encode NS1 sequences for all 4 serotypes
encoder = TrainableCodonEncoder.load('research/codon-encoder/training/results/trained_codon_encoder.pt')

serotypes = ['DENV1', 'DENV2', 'DENV3', 'DENV4']
ns1_embeddings = {}

for serotype in serotypes:
    # Use consensus sequence or representative Paraguay isolate
    ns1_seq = load_ns1_sequence(f'dengue_{serotype}_NS1_Paraguay.fasta')
    embedding = encoder.encode_sequence(ns1_seq)
    ns1_embeddings[serotype] = embedding

# Compute pairwise distances
distances = {}
for i, s1 in enumerate(serotypes):
    for s2 in serotypes[i+1:]:
        dist = poincare_distance(ns1_embeddings[s1], ns1_embeddings[s2], c=1.0)
        distances[(s1, s2)] = dist

# Load literature DHF rates
dhf_data = load_literature_dhf_rates()  # DataFrame with (primary, secondary, dhf_rate)

# Match distances to DHF rates
matched_data = []
for _, row in dhf_data.iterrows():
    primary, secondary, dhf_rate = row['primary'], row['secondary'], row['dhf_rate']

    # Get distance (order-independent)
    if (primary, secondary) in distances:
        dist = distances[(primary, secondary)]
    elif (secondary, primary) in distances:
        dist = distances[(secondary, primary)]
    else:
        continue

    matched_data.append({'distance': dist, 'dhf_rate': dhf_rate})

matched_df = pd.DataFrame(matched_data)

# Correlation test
rho, p_value = spearmanr(matched_df['distance'], matched_df['dhf_rate'])
print(f"Spearman ρ = {rho:.3f}, p = {p_value:.4f}")
```

**Null Hypothesis:**
- Spearman ρ ≤ 0.3 (weak-to-no correlation)
- p > 0.05 (not statistically significant)

**Alternative Hypothesis:**
- Spearman ρ > 0.6 (moderate-to-strong correlation)
- p < 0.05

**Falsification Criterion:**
- If ρ < 0.3 OR p > 0.05 → **Reject H3** (distances do not predict DHF)
- If ρ > 0.6 AND p < 0.05 → **Support H3** (distances correlate with severity)

**Confound Analysis:**
- Control: Correlate distance with serotype prevalence (if prevalent serotypes also show high DHF, confound)
- Alternative metric: Use E protein distance instead of NS1 (if E shows stronger correlation, NS1 may not be key)

---

## Test 4: HIV "Goldilocks Zone" Does Not Generalize to RA

### H0: RA PTM Goldilocks Distances Differ from HIV CTL Escape Distances

**Data:**
- HIV CTL escape optimal distance: 5.8-6.9 (from existing analysis)
- RA citrullination Goldilocks PTMs (from existing validation)

**Method:**
```python
from research.codon_encoder.benchmarks.immunology import load_ra_goldilocks_ptms
from src.research.bioinformatics.codon_encoder_research.hiv import load_hiv_ctl_escape

# Load RA Goldilocks PTMs (those that successfully modulate immune response without causing pathology)
ra_goldilocks = load_ra_goldilocks_ptms()  # List of (protein, position, wildtype, mutant)

# Compute p-adic distances for RA Goldilocks mutations
ra_distances = []
for protein, pos, wt, mut in ra_goldilocks:
    wt_embedding = encoder.encode_codon_at_position(protein, pos, wt)
    mut_embedding = encoder.encode_codon_at_position(protein, pos, mut)
    dist = poincare_distance(wt_embedding, mut_embedding, c=1.0)
    ra_distances.append(dist)

# HIV CTL escape distance range
hiv_goldilocks_range = (5.8, 6.9)

# Test overlap
ra_mean = np.mean(ra_distances)
ra_std = np.std(ra_distances)
ra_in_hiv_range = sum(hiv_goldilocks_range[0] <= d <= hiv_goldilocks_range[1] for d in ra_distances)
ra_fraction_overlap = ra_in_hiv_range / len(ra_distances)

print(f"RA Goldilocks: mean={ra_mean:.2f}, std={ra_std:.2f}")
print(f"Fraction in HIV range: {ra_fraction_overlap:.2%}")

# Statistical test: Is RA mean within HIV range?
from scipy.stats import ttest_1samp
hiv_midpoint = 6.35
t_stat, p_value = ttest_1samp(ra_distances, hiv_midpoint)
```

**Null Hypothesis:**
- RA Goldilocks distances are outside HIV range (mean < 5.8 OR mean > 6.9)
- Overlap < 50% (most RA PTMs fall outside HIV Goldilocks zone)

**Alternative Hypothesis:**
- RA Goldilocks distances overlap HIV range (mean within 5.8-6.9)
- Overlap > 70% (most RA PTMs in HIV zone)

**Falsification Criterion:**
- If overlap < 50% → **Reject generalization** (HIV zone is HIV-specific)
- If overlap > 70% → **Support generalization** (universal Goldilocks zone exists)

**Confound Analysis:**
- RA and HIV both involve adaptive immune responses, so overlap may reflect shared immune biology, not universal PPI failure
- Test with non-immune PTMs (e.g., Tau phosphorylation in neurons) - if those also fall in 5.8-6.9, stronger support for universality

---

## Test 5: Contact Prediction Does Not Extend to Disease PPIs

### H0: P-adic Contact Prediction Fails for Disease-Relevant Protein Interactions

**Data:**
- Known ALS PPI: TDP-43 + hnRNP A1 (validated interaction, PDB 4BS2 shows RRM domain)
- Known PD PPI: Alpha-synuclein + 14-3-3 (validated interaction)
- Negative control: TDP-43 + random non-interacting protein

**Method:**
```python
from research.contact_prediction.scripts.01_test_real_protein import predict_contacts
from sklearn.metrics import roc_auc_score

# Predict contacts between TDP-43 and hnRNP A1
tdp43_seq = get_uniprot_sequence('Q13148')
hnrnp_a1_seq = get_uniprot_sequence('P09651')

# Generate contact predictions (pairwise residue distances)
contact_scores = predict_contacts(tdp43_seq, hnrnp_a1_seq, encoder)

# Load known interface from PDB or literature
# Example: TDP-43 RRM1 (residues 103-175) interacts with hnRNP A1 RRM1 (residues 15-90)
true_interface = load_known_interface('TDP43', 'hnRNP_A1')  # Binary matrix

# Compute AUC
auc = roc_auc_score(true_interface.flatten(), contact_scores.flatten())
print(f"TDP-43 + hnRNP A1 contact prediction AUC = {auc:.3f}")

# Negative control: TDP-43 + non-interacting protein
random_protein_seq = get_random_protein_sequence(length=400)
random_contact_scores = predict_contacts(tdp43_seq, random_protein_seq, encoder)
random_auc = roc_auc_score(np.zeros_like(random_contact_scores).flatten(), random_contact_scores.flatten())
```

**Null Hypothesis:**
- Contact prediction AUC ≤ 0.55 (barely better than random)
- No difference between true PPI and random protein (AUC_true - AUC_random < 0.1)

**Alternative Hypothesis:**
- AUC > 0.65 (validated for small proteins)
- True PPI shows significantly higher AUC than random (AUC_true - AUC_random > 0.15)

**Falsification Criterion:**
- If AUC < 0.55 → **Reject extension** (contact prediction does not work for disease PPIs)
- If AUC > 0.65 AND delta > 0.15 → **Support extension** (method generalizes beyond small proteins)

**Confound Analysis:**
- TDP-43 and hnRNP A1 are both RNA-binding proteins with disordered regions
- Contact prediction may fail due to disorder, not because p-adic geometry is invalid
- Test with structured PPI (e.g., SOD1 dimer, PDB 1SPD) as positive control

---

## Combined Decision Matrix

| Test | Null Result | Alternative Result | Implication |
|------|-------------|-------------------|-------------|
| **Test 1: PTM Clustering** | Silhouette < 0.2 | Silhouette > 0.3 | Disease-specific mechanisms exist |
| **Test 2: Codon Bias** | p > 0.05 for all genes | p < 0.05 for ≥2 genes | ALS genes use optimal codons |
| **Test 3: Dengue DHF** | ρ < 0.3 or p > 0.05 | ρ > 0.6 and p < 0.05 | NS1 distance predicts severity |
| **Test 4: Goldilocks Zone** | Overlap < 50% | Overlap > 70% | Universal immune evasion zone |
| **Test 5: Contact Prediction** | AUC < 0.55 | AUC > 0.65 | Method extends to disease PPIs |

**Minimum Support Threshold:**
- If ≥3 of 5 tests support alternative hypothesis → Conjecture has computational evidence
- If ≤2 of 5 tests support alternative → Conjecture lacks support, refine or reject

---

## Statistical Power Analysis

### Test 1: PTM Clustering (n=94 sites)

**Effect Size:**
- Small effect: Cohen's d = 0.2 → Requires n > 200 for 80% power
- Medium effect: Cohen's d = 0.5 → Requires n > 50 for 80% power
- Large effect: Cohen's d = 0.8 → Requires n > 30 for 80% power

**Current Power:** With n=94, can detect medium-to-large effects (d ≥ 0.5) with 80% power

### Test 2: Codon Bias (n=3 genes)

**Limitation:** Low sample size limits power
- Binomial test for each gene separately (within-gene n = ~400 codons, adequate)
- Cross-gene generalization requires more ALS genes (expand to OPTN, VCP, UBQLN2, etc.)

### Test 3: Dengue DHF (n=6-12 serotype combinations)

**Limitation:** Small n for correlation
- With n=12, can detect ρ > 0.7 with 80% power (very large effect)
- Cannot detect ρ < 0.5 reliably (need n > 30 for medium effects)

**Mitigation:** Use multiple isolates per serotype, treat as clustered data

### Test 4: Goldilocks Zone (n=~20 RA Goldilocks PTMs)

**Limitation:** Small n limits precision of overlap estimate
- 95% CI for 50% overlap with n=20: [28%, 72%] (wide)
- Need n > 50 for tight CI

**Mitigation:** Expand to other RA PTMs beyond Goldilocks (all 47 citrullination sites)

### Test 5: Contact Prediction (n=1 PPI)

**Limitation:** Single example, no generalization
- Need n > 10 validated disease PPIs to assess method performance
- Current test is proof-of-concept, not validation

**Mitigation:** Test multiple PPIs (SOD1 dimer, alpha-synuclein oligomer, Dengue NS1 hexamer)

---

## Pre-Registration Protocol

To prevent p-hacking and ensure scientific rigor, pre-register hypotheses before running tests:

**Pre-Registration Checklist:**
1. Define null and alternative hypotheses (done above)
2. Specify test statistics and thresholds (done above)
3. Commit code to version control BEFORE running tests
4. Document all tests run, including failures (transparent reporting)
5. Do not modify thresholds post-hoc to achieve significance

**Registered Predictions:**
- Test 1: Silhouette > 0.3, ARI > 0.5
- Test 2: ≥2 of 3 genes show p < 0.05
- Test 3: Spearman ρ > 0.6, p < 0.05
- Test 4: Overlap > 70%
- Test 5: AUC > 0.65

**Reporting Standard:**
- Report all 5 tests regardless of outcome
- If 0-1 support alternative: "Conjecture lacks computational evidence"
- If 2 support alternative: "Weak evidence, requires validation"
- If 3-4 support alternative: "Moderate evidence, proceed to structural validation"
- If 5 support alternative: "Strong evidence, high-priority for experimental validation"

---

## Execution Timeline

### Week 1: Test 1 (PTM Clustering)
- Load RA + Tau PTM data
- Compute embeddings and distances
- Run clustering analysis
- Document results (silhouette, ARI, dendrogram)

### Week 2: Test 2 (Codon Bias)
- Download ALS gene sequences
- Compute v=0 enrichment
- Binomial tests for each gene
- Compare to random gene controls

### Week 3: Test 3 (Dengue DHF)
- Extract NS1 sequences (Paraguay data)
- Compute serotype distances
- Load literature DHF rates
- Spearman correlation test

### Week 4: Test 4 (Goldilocks Zone)
- Load RA Goldilocks PTM data
- Compute distances
- Test overlap with HIV range
- Statistical comparison

### Week 5: Test 5 (Contact Prediction)
- Identify known disease PPIs with structural data
- Run contact prediction
- Compute AUC vs random controls
- Document successes and failures

**Total Duration:** 5 weeks (1 test per week, allows thorough analysis and documentation)

---

## Success Criteria for Phase 1 (Computational Validation)

**Go Decision (Proceed to Phase 2):**
- ≥3 of 5 tests reject null hypothesis
- Effect sizes are medium-to-large (Cohen's d > 0.5, ρ > 0.5)
- Results consistent with published literature where applicable

**No-Go Decision (Refine or Abandon):**
- ≤2 of 5 tests reject null hypothesis
- Effect sizes are small (d < 0.3, ρ < 0.3)
- Results contradict established biological knowledge

**Partial Success (Refocus):**
- If Tests 1-2 succeed but 3-5 fail: Focus on neurological diseases (ALS/PD), deprioritize Dengue
- If Tests 3-4 succeed but 1-2 fail: Focus on Dengue immune mechanisms, deprioritize codon bias
- If Test 5 fails: Do not claim PPI prediction capability, limit to DDG and contacts within proteins

---

**Version:** 1.0 · **Status:** Pre-registered hypotheses
**Next Steps:** Execute Test 1 (PTM clustering) with existing RA + Tau data
