# Current Validation Opportunities - Ready to Execute

**Doc-Type:** Research Planning · Version 1.0 · 2026-01-03 · AI Whisperers

**Purpose:** Identify validations executable NOW with existing validated tools and datasets

**Status:** ACTIVE - Prioritized by data availability and tool readiness

---

## Validation Priority Matrix

| Validation | Data Ready? | Tools Ready? | Effort | Impact | Priority |
|-----------|-------------|--------------|--------|--------|----------|
| **1. Contact Prediction (Small Proteins)** | ✓ YES | ✓ YES | 1 week | High | **P0** |
| **2. Force Constant Validation** | ✓ YES | ✓ YES | 1 week | High | **P0** |
| **3. HIV Escape Mechanism Analysis** | ✓ YES | ✓ YES | 1 week | Medium | **P1** |
| **4. DDG Cross-Validation Stratified** | ✓ YES | ✓ YES | 1 week | Medium | **P1** |
| **5. Dengue E Protein vs NS1** | ✓ YES | ✓ YES | 2 weeks | Medium | **P1** |
| **6. Small Protein Conjecture** | ✓ YES | ✓ YES | 2 weeks | High | **P1** |
| **7. Arrow Flip Extended Analysis** | ✓ YES | ✓ YES | 1 week | Medium | **P2** |
| **8. Within-PTM Clustering** | ⚠ PARTIAL | ✓ YES | 3 weeks | High | **P2** |

**P0:** Execute immediately (this week)
**P1:** Execute next (weeks 2-3)
**P2:** Execute after P0/P1 complete (weeks 4-6)

---

## P0 Validation 1: Contact Prediction - Small Protein Validation

### What We Already Have

**Checkpoint:** `research/contact-prediction/checkpoints/v5_11_structural_best.pt`
- **Validated:** Insulin B-chain (30aa, AUC=0.6737), Lambda Repressor (AUC=0.814)
- **Status:** LOO validation complete, embedding extraction done

**Framework:** `research/contact-prediction/scripts/01_test_real_protein.py`
- **Input:** Protein sequence, known contacts (from PDB)
- **Output:** AUC-ROC, Cohen's d, contact precision/recall

**Embeddings:** `research/contact-prediction/embeddings/v5_11_3_embeddings.pt`
- **64 codons × 16 dimensions**
- **Codon mapping:** `codon_mapping_3adic.json`

### What We Can Validate Now

**Test on 5-10 additional small proteins (<100 residues):**

| Protein | PDB | Size | Known Contacts | Availability |
|---------|-----|------|----------------|--------------|
| Trp-cage | 1L2Y | 20aa | ~15 contacts | ✓ PDB |
| WW domain | 1PIN | 40aa | ~30 contacts | ✓ PDB |
| Villin headpiece | 1VII | 35aa | ~25 contacts | ✓ PDB |
| Protein G (B1 domain) | 1PGB | 56aa | ~50 contacts | ✓ PDB |
| Ubiquitin | 1UBQ | 76aa | ~80 contacts | ✓ PDB |
| Crambin | 1CRN | 46aa | ~40 contacts | ✓ PDB |

**Execution:**

```python
# For each protein
python research/contact-prediction/scripts/01_test_real_protein.py \
    --protein Trp_cage \
    --pdb_id 1L2Y \
    --checkpoint research/contact-prediction/checkpoints/v5_11_structural_best.pt \
    --output results/trp_cage_contacts.json

# Expected AUC: 0.55-0.75 (based on small protein conjecture)
```

**Expected Results:**
- **Hypothesis:** Small fast-folders show AUC > 0.55 (above random)
- **Mean AUC across 5 proteins:** 0.58-0.65
- **Compare to Phase 1 Test 5 (SOD1):** AUC=0.45 (failed), validates that small proteins work

**Effort:** 1 week (download PDBs, extract contacts, run predictions, generate report)

**Deliverable:** `research/contact-prediction/SMALL_PROTEIN_VALIDATION_REPORT.md`

---

## P0 Validation 2: Force Constant Prediction

### What We Already Have

**Discovery:** P-adic radial distance encodes force constants (ρ=0.86)
- **Finding:** `k = radius × mass / 100` (from codon-encoder research)
- **Validation:** Spearman ρ=0.86 on existing amino acid data
- **Document:** `research/codon-encoder/extraction/ANALYSIS_SUMMARY.md`

**Embeddings:** v5_11_structural hyperbolic embeddings
- **Radii:** `radius = poincare_distance(z_hyp, origin, c=1.0)`
- **Mass data:** Available for all 20 amino acids

**Ground Truth:** Experimental vibrational frequencies (literature)
- **Source:** Amino acid normal mode analysis (computational chemistry papers)
- **Availability:** Published data for common amino acids

### What We Can Validate Now

**Test force constant predictions vs experimental data:**

1. **Extract radii from v5_11_structural embeddings** (all 20 amino acids)
2. **Compute predicted force constants:** `k_pred = radius × mass / 100`
3. **Compare to experimental vibrational frequencies:** `ω_exp = √(k_exp / m)`
4. **Validate correlation:** `k_pred vs k_exp`

**Expected Results:**
- **Spearman ρ > 0.80** (validates force constant encoding)
- **Identifies outliers:** Which amino acids deviate? (Cysteine disulfides, Proline rigidity?)

**Execution:**

```python
# Load embeddings
checkpoint = torch.load('research/contact-prediction/checkpoints/v5_11_structural_best.pt')
z_hyp = checkpoint['z_B_hyp']  # Hyperbolic embeddings

# Compute radii
origin = torch.zeros_like(z_hyp)
radii = poincare_distance(z_hyp, origin, c=1.0)

# Compute force constants
masses = load_amino_acid_masses()  # From src.core.padic_math
k_pred = radii * masses / 100

# Load experimental data
k_exp = load_experimental_force_constants()  # From literature

# Validate
from scipy.stats import spearmanr
rho, pval = spearmanr(k_pred, k_exp)
print(f"Force constant correlation: ρ={rho:.3f}, p={pval:.4f}")
```

**Effort:** 1 week (literature search for experimental data, validation script, analysis)

**Deliverable:** `research/codon-encoder/FORCE_CONSTANT_VALIDATION_REPORT.md`

---

## P1 Validation 3: HIV Escape Mechanism Analysis

### What We Already Have

**Data:** `src/research/bioinformatics/codon_encoder_research/hiv/results/hiv_escape_results.json`
- **9 CTL escape mutations** with hyperbolic distances
- **Mean distance:** 6.204 ± 0.598
- **Range:** [5.264, 7.170]
- **Metadata:** Escape efficacy (high/moderate/low), fitness cost (low/moderate)

**Checkpoint:** TrainableCodonEncoder (LOO ρ=0.61)
- **Location:** `research/codon-encoder/training/results/trained_codon_encoder.pt`
- **Validated:** S669 DDG benchmark

### What We Can Validate Now

**Analyze mechanisms underlying Goldilocks zone (5.8-6.9):**

1. **Stratify mutations by escape efficacy:**
   - High efficacy: Do they cluster at specific distance?
   - Low efficacy: Are they outside Goldilocks zone?

2. **Correlate distance with fitness cost:**
   - Low fitness cost: Are they at lower end of range (5.8-6.2)?
   - High fitness cost: Are they at higher end (6.5-7.0)?

3. **Physicochemical analysis:**
   - Which property changes dominate? (Hydrophobicity, charge, size)
   - Do hydrophobic changes have different distance distribution than charge changes?

**Expected Results:**
- **High-efficacy/low-cost mutations cluster at 6.0-6.5** (Goldilocks sweet spot)
- **High-cost mutations at >6.8** (too disruptive)
- **Low-efficacy mutations at <5.5 or >7.2** (outside zone)

**Execution:**

```python
import json
with open('hiv/results/hiv_escape_results.json') as f:
    data = json.load(f)

# Stratify by efficacy
high_efficacy = [m for m in data['mutations'] if m['escape_efficacy'] == 'high']
low_efficacy = [m for m in data['mutations'] if m['escape_efficacy'] == 'low']

# Compute statistics
high_eff_distances = [m['hyperbolic_distance'] for m in high_efficacy]
low_eff_distances = [m['hyperbolic_distance'] for m in low_efficacy]

print(f"High efficacy: {np.mean(high_eff_distances):.2f} ± {np.std(high_eff_distances):.2f}")
print(f"Low efficacy: {np.mean(low_eff_distances):.2f} ± {np.std(low_eff_distances):.2f}")

# Test if significantly different
from scipy.stats import mannwhitneyu
u_stat, pval = mannwhitneyu(high_eff_distances, low_eff_distances)
print(f"Mann-Whitney U test: p={pval:.4f}")
```

**Effort:** 1 week (load data, stratified analysis, visualization, report)

**Deliverable:** `src/research/bioinformatics/codon_encoder_research/hiv/HIV_GOLDILOCKS_MECHANISM_REPORT.md`

---

## P1 Validation 4: DDG Cross-Validation Stratified

### What We Already Have

**Checkpoint:** TrainableCodonEncoder (LOO ρ=0.61 on S669)
- **Location:** `research/codon-encoder/training/results/trained_codon_encoder.pt`
- **S669 Dataset:** 52 proteins, 669 mutations (available in `research/codon-encoder/training/data/`)

**Validation:** Bootstrap CI [0.341, 0.770], permutation p < 0.001
- **Document:** `deliverables/partners/jose_colbes/validation/results/SCIENTIFIC_VALIDATION_REPORT.md`

### What We Can Validate Now

**Stratify DDG performance by protein/mutation properties:**

1. **By protein size:**
   - Small (<100aa): ρ = ?
   - Medium (100-200aa): ρ = ?
   - Large (>200aa): ρ = ?

2. **By mutation type:**
   - Hydrophobic→Hydrophobic: ρ = ?
   - Charged→Neutral: ρ = ?
   - Small→Large: ρ = ?

3. **By structural context (from Arrow Flip analysis):**
   - Buried (RSA <0.25): ρ = ? (expected: higher)
   - Surface (RSA >0.5): ρ = ? (expected: lower)
   - EC1 metal-binding: ρ = ? (expected: lower, structure-dependent)

**Expected Results:**
- **Small proteins:** ρ=0.65-0.75 (better, simpler structure)
- **Large proteins:** ρ=0.45-0.55 (worse, complex structure)
- **Buried mutations:** ρ=0.68 (Arrow Flip finding, 2x better)
- **Surface mutations:** ρ=0.35-0.45 (worse, more context-dependent)

**Execution:**

```python
# Load S669 dataset
s669_data = pd.read_csv('research/codon-encoder/training/data/s669.csv')

# Stratify by protein size
small_proteins = s669_data[s669_data['protein_length'] < 100]
large_proteins = s669_data[s669_data['protein_length'] > 200]

# Compute LOO correlation for each stratum
from sklearn.model_selection import LeaveOneOut
from scipy.stats import spearmanr

rho_small = compute_loo_correlation(small_proteins)
rho_large = compute_loo_correlation(large_proteins)

print(f"Small proteins: ρ={rho_small:.3f}")
print(f"Large proteins: ρ={rho_large:.3f}")
```

**Effort:** 1 week (stratified analysis, bootstrap CIs for each stratum, report)

**Deliverable:** `deliverables/partners/jose_colbes/validation/DDG_STRATIFIED_ANALYSIS.md`

---

## P1 Validation 5: Dengue E Protein vs NS1

### What We Already Have

**Phase 1 Test 3 Result:** NS1 ρ=-0.33 (p=0.29) - FAILED
- **Document:** `research/cross-disease-validation/results/test3_dengue_dhf/TEST3_REPORT.md`
- **Data:** Dengue serotype sequences (DENV-1, 2, 3, 4)

**alejandra_rojas Package:** P-adic embedding functions
- **Location:** `deliverables/partners/alejandra_rojas/src/geometry.py`
- **Function:** `compute_padic_embedding(sequence)` - ready to use

**Hypothesis Refinement:** E protein (envelope) is correct ADE target, not NS1 (non-structural)

### What We Can Validate Now

**Re-run Test 3 with E protein:**

1. **Extract E protein sequences** (positions 936-2421 in dengue genome)
2. **Compute p-adic embeddings** using alejandra_rojas tools
3. **Correlate with DHF rates** from literature (Halstead 2007, Guzman 2013)
4. **Compare NS1 (ρ=-0.33) vs E protein (expected: ρ=0.5-0.7)**

**Expected Results:**
- **E protein correlation:** ρ=0.55-0.75 (validates correct protein)
- **Mechanism:** E protein epitopes drive ADE (antibody-dependent enhancement)
- **NS1 was wrong target:** Confirms Phase 1 test design flaw

**Execution:**

```python
from deliverables.partners.alejandra_rojas.src.geometry import compute_padic_embedding

# Extract E protein sequences (positions 936-2421)
denv_serotypes = {
    'DENV-1': extract_e_protein(denv1_genome),
    'DENV-2': extract_e_protein(denv2_genome),
    'DENV-3': extract_e_protein(denv3_genome),
    'DENV-4': extract_e_protein(denv4_genome),
}

# Compute embeddings
embeddings = {}
for serotype, sequence in denv_serotypes.items():
    embeddings[serotype] = compute_padic_embedding(sequence)

# Pairwise distances
distances = []
for s1, s2 in itertools.combinations(denv_serotypes.keys(), 2):
    dist = hyperbolic_distance(embeddings[s1], embeddings[s2])
    dhf_diff = abs(dhf_rates[s1] - dhf_rates[s2])
    distances.append({'pair': (s1, s2), 'distance': dist, 'dhf_diff': dhf_diff})

# Correlate
from scipy.stats import spearmanr
rho, pval = spearmanr([d['distance'] for d in distances],
                       [d['dhf_diff'] for d in distances])

print(f"E protein correlation: ρ={rho:.3f}, p={pval:.4f}")
```

**Effort:** 2 weeks (extract E protein, literature DHF rates, analysis, report)

**Deliverable:** `research/cross-disease-validation/results/test3_dengue_e_protein/TEST3_E_PROTEIN_REPORT.md`

---

## P1 Validation 6: Small Protein Conjecture

### What We Already Have

**Finding:** Contact prediction works for small proteins (Insulin B, Lambda Repressor) but fails for complex proteins (SOD1)
- **Small proteins:** AUC=0.586-0.814
- **SOD1 (complex):** AUC=0.451

**Checkpoint:** v5_11_structural (AUC=0.6737 on Insulin B-chain)

**Literature:** Fast-folder databases (Protein G, Trp-cage, WW domain, etc.)

### What We Can Validate Now

**Define strict inclusion criteria for "small protein" conjecture:**

**Proposed Criteria:**
1. **Size:** 10-100 residues
2. **Folding rate:** >1000 s⁻¹ (fast folder)
3. **Structure:** Single domain, no complex topology
4. **No metal-binding sites** (Cu, Zn, Fe, Mn)
5. **No disulfide bonds** OR ≤1 disulfide
6. **Secondary structure:** Alpha-helix OR beta-sheet (not complex mix)

**Test on 10 proteins:**

| Protein | Size | Fast Folder? | Metal? | Disulfides | Expected AUC |
|---------|------|--------------|--------|------------|--------------|
| Trp-cage | 20aa | ✓ YES | ✗ NO | 0 | 0.60-0.75 |
| WW domain | 40aa | ✓ YES | ✗ NO | 0 | 0.55-0.70 |
| Villin headpiece | 35aa | ✓ YES | ✗ NO | 0 | 0.58-0.72 |
| Protein G | 56aa | ✓ YES | ✗ NO | 0 | 0.55-0.68 |
| Ubiquitin | 76aa | ⚠ MODERATE | ✗ NO | 0 | 0.50-0.60 |
| SOD1 | 153aa | ✗ NO | ✓ YES (Cu/Zn) | ✓ YES | 0.45 (confirmed) |
| Crambin | 46aa | ✓ YES | ✗ NO | 3 | 0.50-0.60 (disulfides) |

**Expected Results:**
- **Strict criteria proteins (n=5):** Mean AUC=0.60-0.70
- **Borderline proteins (n=2):** Mean AUC=0.50-0.60
- **Exclusion criteria proteins (n=3):** Mean AUC<0.50

**Effort:** 2 weeks (curate protein list, extract contacts, run predictions, define criteria)

**Deliverable:** `research/contact-prediction/SMALL_PROTEIN_CONJECTURE_REPORT.md`

---

## P2 Validation 7: Arrow Flip Extended Analysis

### What We Already Have

**Finding:** Position type (buried vs surface) modifies hydrophobicity threshold for regime selection
- **Buried (RSA <0.25):** Threshold=3.5, hybrid advantage=+0.565
- **Surface (RSA >0.5):** Threshold=5.5, hybrid advantage=+0.01
- **Document:** `research/codon-encoder/replacement_calculus/docs/V5_EXPERIMENTAL_VALIDATION.md`

**Dataset:** GO (Gene Ontology) mutations with DDG ground truth + position annotations

### What We Can Validate Now

**Extend analysis to more position types:**

1. **Interface positions** (protein-protein interaction sites)
2. **Active site positions** (catalytic residues)
3. **Allosteric sites** (regulatory regions)
4. **Disordered regions** (IUPred score >0.5)

**Hypothesis:** Each position type has different "arrow flip" threshold

**Expected Results:**
- **Active sites:** Very low threshold (structure-critical, simple predictor wins)
- **Allosteric sites:** High threshold (conformational changes, hybrid wins)
- **Disordered regions:** No clear threshold (high uncertainty)

**Execution:**

```python
# Load GO dataset with position annotations
go_data = load_go_dataset()

# Stratify by position type
active_site = go_data[go_data['position_type'] == 'active_site']
interface = go_data[go_data['position_type'] == 'interface']
disordered = go_data[go_data['disorder_score'] > 0.5]

# Compute arrow flip threshold for each
threshold_active = compute_arrow_flip_threshold(active_site)
threshold_interface = compute_arrow_flip_threshold(interface)

print(f"Active site threshold: {threshold_active:.2f}")
print(f"Interface threshold: {threshold_interface:.2f}")
```

**Effort:** 1 week (position annotation, stratified analysis, report)

**Deliverable:** `research/codon-encoder/replacement_calculus/ARROW_FLIP_EXTENDED_REPORT.md`

---

## P2 Validation 8: Within-PTM Clustering

### What We Already Have

**Phase 1 Test 1 Result:** PTMs cluster by type (RA citrullination vs Tau phosphorylation)
- **Silhouette score:** 0.42 (moderate clustering)
- **Issue:** PTM type confounded with disease (RA=citrul, Tau=phos)

**Data Available:**
- **Tau phosphorylation sites:** `src/research/bioinformatics/codon_encoder_research/alzheimers/data/tau_phospho_database.py`
- **RA citrullination sites:** Literature (need to extract verified sites)

### What We Can Validate Now

**Test within-PTM-type clustering:**

**Hypothesis 1: Phosphorylation Sites Cluster by Disease**
- **Tau phosphorylation (Alzheimer's)** vs **TDP-43 phosphorylation (ALS)**
- **Expected:** Weak clustering (same PTM chemistry, different proteins)

**Hypothesis 2: Citrullination Sites Cluster by Disease**
- **RA citrullination** vs **MS citrullination** (need to extract from literature)
- **Expected:** Moderate clustering (same PTM, different autoimmune targets)

**Partial Data:**
- **Tau phospho:** ✓ Available (n=54 sites)
- **TDP-43 phospho:** ⚠ Need to extract from literature (n≈20-30 sites, 1 week)
- **RA citrul:** ⚠ Need to extract verified sites (n=12 from Phase 1 Test 4 proper)
- **MS citrul:** ⚠ Need to extract from literature (n≈15-20 sites, 1 week)

**Effort:** 3 weeks (1 week TDP-43 curation, 1 week MS curation, 1 week clustering analysis)

**Deliverable:** `research/cross-disease-validation/results/test1_within_ptm_clustering/WITHIN_PTM_CLUSTERING_REPORT.md`

**Why P2:** Requires moderate literature curation (2 weeks) before execution

---

## Execution Timeline

### Week 1 (P0 Focus)
- **Day 1-3:** Contact Prediction - Small Proteins (download PDBs, extract contacts)
- **Day 4-5:** Force Constant Validation (literature search, validation script)
- **Weekend:** Run predictions, generate preliminary results

### Week 2 (P0 Completion + P1 Start)
- **Day 1-2:** Complete Contact Prediction analysis (AUC calculations, report writing)
- **Day 3-4:** Complete Force Constant analysis (correlation, outlier identification)
- **Day 5:** Start HIV Escape Mechanism Analysis (stratified analysis)

### Week 3 (P1 Focus)
- **Day 1-2:** Complete HIV Escape Mechanism (visualization, report)
- **Day 3-4:** DDG Cross-Validation Stratified (bootstrap CIs, stratification)
- **Day 5:** Start Dengue E Protein extraction

### Week 4 (P1 Completion)
- **Day 1-3:** Complete Dengue E Protein (literature DHF rates, correlation)
- **Day 4-5:** Start Small Protein Conjecture (curate protein list, define criteria)

### Weeks 5-6 (P1 Completion + P2)
- **Week 5:** Complete Small Protein Conjecture (run predictions, define criteria)
- **Week 6:** P2 validations (Arrow Flip extended, Within-PTM clustering setup)

---

## Expected Outcomes

### Immediate Impact (P0, Week 1-2)

**Contact Prediction:**
- Validates small protein conjecture (AUC=0.58-0.65 mean across 5-10 proteins)
- Defines strict inclusion criteria (size, metal-binding, disulfides)
- Confirms SOD1 failure was due to complexity (not fundamental flaw)

**Force Constants:**
- Validates ρ=0.86 finding on independent experimental data
- Identifies outliers (Cys, Pro) with mechanistic explanations
- Demonstrates p-adic radii encode physical properties

### Medium Impact (P1, Weeks 2-4)

**HIV Escape:**
- Explains Goldilocks zone mechanism (efficacy vs fitness tradeoff)
- Stratifies by physicochemical property changes
- Provides mechanistic understanding (not just empirical finding)

**DDG Stratified:**
- Quantifies TrainableCodonEncoder performance by context
- Validates Arrow Flip findings (buried>surface, 2x improvement)
- Identifies where sequence-only fails (large proteins, surface mutations)

**Dengue E Protein:**
- Rescues Phase 1 Test 3 failure (ρ=0.5-0.7 vs -0.33 for NS1)
- Validates hypothesis refinement approach
- Demonstrates importance of correct protein target

**Small Protein Conjecture:**
- Defines actionable inclusion criteria for contact prediction
- Provides clear guidelines for future applications
- Validates fast-folder hypothesis

---

## Success Criteria

**Minimum (4 weeks):**
- ✓ 2 of 2 P0 validations complete (Contact Prediction, Force Constants)
- ✓ 3 of 4 P1 validations complete (HIV, DDG, Dengue OR Small Protein)
- ✓ Deliverables: 5+ validation reports

**Stretch (6 weeks):**
- ✓ All P0 and P1 validations complete (6 total)
- ✓ 1 of 2 P2 validations complete (Arrow Flip OR Within-PTM)
- ✓ Deliverables: 7+ validation reports

---

## Deliverables Summary

| Validation | Deliverable File | Effort | Timeline |
|-----------|------------------|--------|----------|
| **P0-1: Contact Prediction** | `research/contact-prediction/SMALL_PROTEIN_VALIDATION_REPORT.md` | 1 week | Week 1-2 |
| **P0-2: Force Constants** | `research/codon-encoder/FORCE_CONSTANT_VALIDATION_REPORT.md` | 1 week | Week 1-2 |
| **P1-3: HIV Escape** | `hiv/HIV_GOLDILOCKS_MECHANISM_REPORT.md` | 1 week | Week 2-3 |
| **P1-4: DDG Stratified** | `jose_colbes/validation/DDG_STRATIFIED_ANALYSIS.md` | 1 week | Week 3 |
| **P1-5: Dengue E Protein** | `test3_dengue_e_protein/TEST3_E_PROTEIN_REPORT.md` | 2 weeks | Week 3-4 |
| **P1-6: Small Protein** | `research/contact-prediction/SMALL_PROTEIN_CONJECTURE_REPORT.md` | 2 weeks | Week 4-5 |
| **P2-7: Arrow Flip** | `replacement_calculus/ARROW_FLIP_EXTENDED_REPORT.md` | 1 week | Week 6 |
| **P2-8: Within-PTM** | `test1_within_ptm_clustering/WITHIN_PTM_CLUSTERING_REPORT.md` | 3 weeks | Week 4-6 |

---

## Next Immediate Action (P0-1: Contact Prediction)

**Start This Week:**

1. **Download PDB files** (30 minutes):
   - Trp-cage (1L2Y)
   - WW domain (1PIN)
   - Villin headpiece (1VII)
   - Protein G (1PGB)
   - Crambin (1CRN)

2. **Extract contact maps** (2 hours):
   - Parse PDB with BioPython
   - Compute Cα distances
   - Threshold <8Å = contact
   - Save contact lists

3. **Run contact prediction** (1 hour):
   - Load v5_11_structural checkpoint
   - Extract codon embeddings
   - Compute pairwise distances
   - Predict contacts (distance < threshold)

4. **Compute AUC** (30 minutes):
   - True contacts vs predicted contacts
   - ROC curve, AUC-ROC
   - Cohen's d, precision/recall

**Expected by end of Week 1:** Preliminary AUC results for 5 small proteins

---

**Version:** 1.0 · **Date:** 2026-01-03 · **Status:** ACTIVE - Ready for Immediate Execution
**Timeline:** 4-6 weeks for all P0-P1 validations
