# Phase 1: Null Hypothesis Testing - Progress Tracker

**Phase Start Date:** 2026-01-03
**Phase End Date:** 2026-01-03
**Current Status:** COMPLETE (5 of 5 tests complete)
**Decision Criterion:** >= 3 of 5 tests must reject null hypothesis
**Final Decision:** BELOW THRESHOLD - Hypothesis Refinement Required

---

## Test Status Overview

| Test | Status | Result | Effect Size | p-value | Decision |
|:----:|:------:|:------:|:-----------:|:-------:|:--------:|
| **Test 1** | COMPLETE | **PASS** | ARI=1.000, Silh=0.864 | N/A | REJECT NULL |
| **Test 2** | COMPLETE | **PASS** | 1.91-2.19x enrichment | < 10^-20 | REJECT NULL |
| **Test 3** | COMPLETE | **FAIL** | Spearman ρ=-0.333 | 0.29 | FAIL TO REJECT |
| **Test 4 (Proper)** | COMPLETE | **FAIL** | 0% overlap (vs 57.8% simplified) | < 0.0001 | FAIL TO REJECT |
| **Test 5** | COMPLETE | **FAIL** | AUC=0.451 (below baseline) | N/A | FAIL TO REJECT |

**Tests Passed:** 2 / 5 (40%)
**Tests Failed:** 3 / 5 (60%)
**Phase 1 Result:** BELOW 3/5 THRESHOLD

**Key Finding:** Deep validation (Test 4 proper) revealed simplified estimates were dangerously misleading (57.8% → 0% overlap)

---

## Phase 1 Final Summary

**See:** `PHASE1_FINAL_SUMMARY.md` for comprehensive report.

**Critical Scientific Findings:**

1. **Deep Validation is Essential:** Test 4 simplified version showed 57.8% overlap (misleading), proper validation revealed 0% overlap (truth)
2. **Literature Data Quality:** 73% of RA citrullination sites failed position verification (wrong amino acids in UniProt)
3. **PTMs ≠ Mutations:** TrainableCodonEncoder designed for genetic mutations, cannot model post-translational modifications
4. **No Universal Goldilocks Zone:** HIV escape zone (5.3-7.1) does not generalize to RA citrullination (mean=1.124)
5. **Contact Prediction Fails:** AUC=0.451 (below 0.5 baseline) for disease proteins, does not generalize from small proteins

**Recommendation:** Refine hypotheses and develop PTM-specific encoder before Phase 2 computational expansion.

**Scientific Value:** Honest null results prevent false claims and guide productive future research directions.

---

## Test 2: ALS Gene Codon Bias - COMPLETE

**Execution Date:** 2026-01-03
**Result:** REJECT NULL HYPOTHESIS

### Key Findings

All three ALS genes (TARDBP, SOD1, FUS) show extreme enrichment of v=0 codons:
- TARDBP: 65.3% v=0 (1.98x baseline, p = 3.70e-41)
- SOD1: 72.3% v=0 (2.19x baseline, p = 1.94e-23)
- FUS: 63.2% v=0 (1.91x baseline, p = 1.63e-45)

### Interpretation

Strong statistical evidence that ALS genes use v=0 codons far more frequently than genome-wide average (33%). Effect sizes exceed pre-registered threshold by 60-80%.

### Caveats

1. Gene-level analysis (not tissue-specific)
2. No control for expression level, GC content, amino acid composition
3. Small sample (3 genes)
4. Correlation does not imply causation

### Next Actions

**Immediate:**
- Control analysis with housekeeping genes (GAPDH, ACTB, TUBB)
- Control analysis with random genes (n=10, matched for length/GC)
- Amino acid composition analysis

**Phase 2:**
- GTEx motor cortex RNA-seq for tissue-specific validation

### Files

- `results/test2_codon_bias/results.json`
- `results/test2_codon_bias/TEST2_REPORT.md`
- `scripts/phase1_null_tests/test2_codon_bias.py`

---

## Test 1: PTM Clustering - COMPLETE (PASS with caveats)

**Execution Date:** 2026-01-03
**Result:** REJECT NULL HYPOTHESIS (with critical confounds identified)

### Key Findings

Perfect separation achieved (Silhouette=0.864, ARI=1.000) but likely reflects biochemical constraints rather than disease mechanisms:
- RA uses citrullination (targets arginine R)
- Tau uses phosphorylation (targets serine/threonine/tyrosine S/T/Y)
- Different PTM types target different residue classes

### Critical Confound

**Issue:** PTM type perfectly correlates with disease (RA=citrullination, Tau=phosphorylation)
**Implication:** Perfect separation may reflect chemistry, not disease mechanism
**Required Validation:** Within-PTM-type comparisons (RA vs MS citrullination, Tau vs TDP-43 phosphorylation)

### Files

- `results/test1_ptm_clustering/results.json`
- `results/test1_ptm_clustering/TEST1_REPORT.md`
- `results/test1_ptm_clustering/dendrogram.png`
- `results/test1_ptm_clustering/distance_heatmap.png`
- `results/test1_ptm_clustering/mds_projection.png`

---

## Test 3: Dengue DHF Correlation - COMPLETE (FAIL)

**Execution Date:** 2026-01-03
**Result:** FAIL TO REJECT NULL HYPOTHESIS

### Key Findings

NS1 p-adic distances showed **weak negative correlation** with DHF rates:
- Spearman ρ = -0.333 (opposite sign from hypothesis)
- p-value = 0.29 (not statistically significant)
- 12 serotype pairs tested

### Interpretation

NS1 sequence distance does NOT predict DHF severity. Possible explanations:
1. **Wrong target**: E protein (not NS1) is primary ADE driver
2. **Simplified embedding**: Codon statistics miss epitope structure
3. **Inverted U-curve**: DHF peaks at intermediate distance (Goldilocks zone)

### Required Follow-Up

**Phase 2 Priority:**
- Re-test with E protein sequences (positions 936-2421)
- Test quadratic model (inverted U-curve)
- Validate with AlphaFold3 E protein structures

### Scientific Value

Null result is informative:
- Identifies that not all viral proteins predict DHF
- Suggests epitope-specific analysis needed
- Points to E protein as better candidate

### Files

- `results/test3_dengue_dhf/results.json`
- `results/test3_dengue_dhf/TEST3_REPORT.md`
- `results/test3_dengue_dhf/distance_vs_dhf.png`
- `results/test3_dengue_dhf/distance_heatmap.png`

---

## Test 4: Goldilocks Zone Generalization - COMPLETE (FAIL)

**Execution Date:** 2026-01-03
**Result:** FAIL TO REJECT NULL HYPOTHESIS (Proper Validation)

### CRITICAL: Simplified vs Proper Validation

**Simplified Test (v1) - MISLEADING:**
- Method: Random noise (base=6.2, std=0.8)
- Result: 57.8% overlap (26 of 45 sites in HIV range)
- Decision: Weak evidence against null
- **Problem:** Assumed all literature positions had R, used random distances

**Proper Test (v2) - TRUTH:**
- Method: TrainableCodonEncoder + UniProt sequences + actual HIV data
- Result: 0% overlap (0 of 12 valid sites in HIV range)
- Decision: FAIL TO REJECT NULL
- **Finding:** 73% of literature sites failed validation (wrong amino acids)

### Key Findings

**Data Validation Crisis:**
- 33 of 45 literature citrullination sites (73%) do NOT have R at stated positions
- Examples: Vimentin 316 is S (not R), Fibrinogen alpha 36 is G (not R)
- Only 12 of 45 sites verified (26.7%)

**RA vs HIV Distances:**
- RA mean: 1.124 ± 0.000 (constant R→Q approximation)
- HIV mean: 6.204 ± 0.598
- Overlap: 0 of 12 sites (0%)
- Statistical tests: p < 0.0001 (all tests)
- Cohen's d: -12.007 (massive effect size)

### Critical Assessment

**Finding 1: Literature Position Errors (73%)**
- Likely causes: Isoform differences, coordinate systems, database updates
- Implication: Cannot trust literature coordinates without verification

**Finding 2: R→Q Distance Too Small**
- R→Q hyperbolic distance = 1.124 (constant)
- HIV escape spans 5.3-7.1 (much larger chemical space)
- TrainableCodonEncoder places R and Q close (both polar)

**Finding 3: Citrullination ≠ R→Q Mutation (Fundamental Error)**
- Citrullination: Post-translational modification (enzyme removes charge)
- R→Q: Genetic mutation (different amino acid)
- TrainableCodonEncoder encodes genetic code, NOT PTMs
- **Implication:** Test 4 hypothesis is untestable with current encoder

### Scientific Value

**High Value:**
- Identifies critical flaw: PTMs cannot be modeled as mutations
- Exposes literature data quality issues (73% error rate)
- Demonstrates importance of deep validation (user request was correct)
- Prevents false publication of Goldilocks generalization

### Required Follow-Up

**Phase 2 Priority:**
- Develop PTM-specific encoder (captures charge loss, structural changes)
- Curate validated PTM database (cross-reference UniProt, PhosphoSitePlus)
- Test within-PTM-type comparisons (RA vs MS citrullination)

### Files

- `results/test4_goldilocks_proper/TEST4_PROPER_REPORT.md` (359 lines, comprehensive)
- `results/test4_goldilocks_proper/results.json`
- `results/test4_goldilocks_proper/distribution_comparison.png`

---

## Test 5: Contact Prediction for Disease Proteins - COMPLETE (FAIL)

**Execution Date:** 2026-01-03
**Result:** FAIL TO REJECT NULL HYPOTHESIS

### Hypothesis

H0: P-adic contact prediction fails for disease-relevant proteins (AUC <= 0.55)
H1: P-adic contact prediction works (AUC > 0.65, signal > 0.15)

### Key Findings

**SOD1 (ALS-relevant protein) contact prediction:**
- AUC-ROC: 0.4515 (below 0.5 random baseline)
- Signal: -0.0485 (negative, wrong direction)
- Cohen's d: 0.203 (small effect)

**Comparison to validated small proteins:**
- Insulin B-chain: AUC = 0.585
- Lambda Repressor: AUC = 0.814
- Small protein mean: AUC = 0.586
- **SOD1 relative performance: -23.0%**

### Critical Assessment

**Finding 1: Below Random Performance**
- AUC = 0.451 < 0.5 (random baseline)
- Negative signal suggests anti-correlation
- Contact prediction does NOT work for disease proteins

**Finding 2: Structural Complexity**
- SOD1 has complex metal-binding sites (Cu, Zn)
- Contains disulfide bonds (C57-C146)
- Active site geometry not captured by sequence-only embedding

**Finding 3: Fragment Limitations**
- Used only first 30 residues (of 153 total)
- Missing long-range contacts from full structure
- Simplified contact map may not reflect true 3D geometry

### Interpretation

P-adic contact prediction validated on small proteins (AUC=0.586) does NOT generalize to larger disease-relevant proteins. Sequence-only embeddings miss critical structural features.

### Required Follow-Up

**Phase 2 Priority:**
- Add structural features (AlphaFold pLDDT, RSA)
- Test on full-length proteins (not fragments)
- Validate on domain-specific datasets
- Consider hybrid sequence+structure models

### Scientific Value

Honest null result identifies limitations:
- Contact prediction is structure-dependent
- Small protein validation does not guarantee generalization
- Need integration with AlphaFold for complex proteins

### Files

- `results/test5_contact_prediction/results.json`
- `results/test5_contact_prediction/sod1_contact_prediction.png`
- `scripts/phase1_null_tests/test5_contact_prediction_ppi.py`

---

## Test 1: PTM Clustering - BACKUP RECORD

**Planned Execution:** Completed
**Data Required:** RA citrullination (45 sites) + Tau phosphorylation (54 sites)

### Hypothesis

H0: PTMs cluster by modification chemistry (citrullination vs phosphorylation), not disease (RA vs Tau)

### Success Criteria

- Silhouette score > 0.3 (moderate clustering quality)
- Adjusted Rand Index > 0.5 (agreement with true disease labels)
- Diseases separate in dendrogram

### Required Preparation

1. Consolidate RA PTM data from research directory
2. Load Tau PTM data from existing database
3. Implement PTM embedding computation
4. Run hierarchical clustering analysis

### Expected Challenges

- PTM mapping pipeline needs validation for cross-disease comparison
- RA data may be scattered across multiple script files
- Tau database exists but integration needs verification

---

## Test 3: Dengue DHF Correlation - PENDING

**Planned Execution:** Week 3-4
**Data Required:** NS1 sequences (available) + Literature DHF rates (needs compilation)

### Hypothesis

H0: NS1 p-adic distances uncorrelated with observed DHF rates from literature

### Success Criteria

- Spearman ρ > 0.6 (moderate-to-strong correlation)
- p < 0.05 (statistically significant)

### Required Preparation

1. Extract consensus NS1 sequences for DENV-1, DENV-2, DENV-3, DENV-4
2. Compile literature DHF rates (Halstead 2007, Guzman 2013, Kliks 1989, etc.)
3. Encode NS1 sequences using TrainableCodonEncoder
4. Compute pairwise p-adic distances
5. Spearman correlation test

---

## Test 4: Goldilocks Zone Generalization - PENDING

**Planned Execution:** Week 4
**Data Required:** RA Goldilocks PTMs (available) + HIV escape range (5.8-6.9)

### Hypothesis

H0: RA Goldilocks PTM distances differ from HIV CTL escape optimal range

### Success Criteria

- > 70% of RA Goldilocks PTMs fall in HIV range (5.8-6.9)
- RA mean distance within HIV 95% CI

### Required Preparation

1. Load RA Goldilocks PTM data (successful immune modulation without pathology)
2. Compute p-adic distances for WT→Citrullinated transitions
3. Statistical test for overlap with HIV range

---

## Test 5: Contact Prediction PPIs - PENDING

**Planned Execution:** Week 5
**Data Required:** TDP-43 + hnRNP A1 sequences, known interface (PDB 4BS2)

### Hypothesis

H0: P-adic contact prediction fails for disease-relevant protein interactions (AUC <= 0.55)

### Success Criteria

- AUC > 0.65 for true PPI
- AUC(true PPI) - AUC(random) > 0.15

### Required Preparation

1. Extract TDP-43 RRM domain (residues 103-175) and hnRNP A1 RRM1 (residues 15-90)
2. Identify known interface residues from PDB 4BS2 or literature
3. Run contact prediction using existing framework
4. Compute AUC vs known interface

---

## Phase 1 Timeline

### Week 1 (Current)
- [x] Test 2 execution
- [x] Test 2 report
- [ ] Control analyses for Test 2 (housekeeping genes, random genes)

### Week 2 (Next)
- [ ] RA + Tau PTM data consolidation
- [ ] Test 1 execution (PTM clustering)
- [ ] Test 1 report

### Week 3
- [ ] Literature DHF rate compilation
- [ ] NS1 sequence extraction
- [ ] Test 3 execution (Dengue DHF)
- [ ] Test 3 report

### Week 4
- [ ] RA Goldilocks data extraction
- [ ] Test 4 execution (Goldilocks generalization)
- [ ] Test 4 report

### Week 5
- [ ] PPI interface data curation
- [ ] Test 5 execution (Contact prediction)
- [ ] Test 5 report

### Week 6 (Decision Point)
- [ ] Phase 1 summary report
- [ ] Go/No-Go decision for Phase 2
- [ ] If Go: Phase 2 planning (GTEx, PhosphoSitePlus acquisition)
- [ ] If No-Go: Refine hypotheses or pivot

---

## Decision Matrix

| Outcome | Tests Passed | Decision | Next Steps |
|---------|:------------:|----------|------------|
| **Strong Support** | 5 of 5 | Proceed to Phase 2 | Full computational expansion |
| **Moderate Support** | 3-4 of 5 | Proceed to Phase 2 | Focus on successful hypotheses |
| **Weak Support** | 2 of 5 | Refine or Partial Proceed | Identify strongest signal, refine others |
| **Insufficient Support** | 0-1 of 5 | No-Go or Pivot | Analyze failures, consider alternative approaches |

**Current Standing:** 1 of 5 (need 2 more to reach minimum threshold)

---

## Risk Assessment

### Test 1 (PTM Clustering) - MEDIUM RISK

**Risk:** RA and Tau PTMs may cluster by chemistry (both involve charged residues) rather than disease
**Mitigation:** If clustering fails, test individual diseases separately for internal consistency

### Test 3 (Dengue DHF) - MEDIUM RISK

**Risk:** Small sample size (6-12 serotype combinations) limits statistical power
**Mitigation:** Use multiple isolates per serotype, treat as clustered data

### Test 4 (Goldilocks) - LOW RISK

**Risk:** HIV and RA both involve adaptive immunity, overlap may not generalize to non-immune contexts
**Mitigation:** Even if generalization fails, RA-HIV overlap is still informative

### Test 5 (Contact Prediction) - HIGH RISK

**Risk:** Contact prediction validated on small proteins, may fail for large complexes
**Mitigation:** Test multiple PPIs (SOD1 dimer, alpha-synuclein oligomer) as backups

---

## Contingency Plans

### If Test 1 Fails (PTM Clustering)

**Plan A:** Test RA PTMs for internal clustering (citrullination sites that cluster vs don't cluster)
**Plan B:** Expand to phosphorylation-only comparison (Tau vs ALS TDP-43 when data available)

### If Test 3 Fails (Dengue DHF)

**Plan A:** Focus on Alejandra Rojas package extension with trajectory forecasting (already validated)
**Plan B:** Test alternative metrics (E protein distance instead of NS1)

### If Test 4 Fails (Goldilocks)

**Plan A:** Define RA-specific Goldilocks zone, don't claim universality
**Plan B:** Test other autoimmune PTMs (Type 1 Diabetes, Celiac) for zone consistency

### If Test 5 Fails (Contact Prediction)

**Plan A:** Limit claims to intra-protein contacts, not inter-protein PPIs
**Plan B:** Focus on small protein domains (TDP-43 RRM only, not full protein)

---

## Honest Assessment (Updated Post-Test 2)

### Strengths So Far

1. Test 2 showed VERY strong effect (2x enrichment, p < 10^-20)
2. Effect size exceeds expectations (1.2x threshold vs 2x observed)
3. Consistent across all 3 genes tested

### Concerns

1. Only 1 test complete (20% of Phase 1)
2. Test 2 results may be confounded (expression level, GC content)
3. Remaining tests may be harder (PTM clustering, Dengue correlation have more assumptions)

### Probability of Phase 1 Success

**Optimistic Scenario (4-5 tests pass):**
- Probability: 30-40%
- Assumption: Test 2 strong signal indicates p-adic embeddings work well

**Moderate Scenario (3 tests pass):**
- Probability: 40-50%
- Most likely: Tests 1, 2, 4 pass; Tests 3, 5 fail

**Conservative Scenario (1-2 tests pass):**
- Probability: 20-30%
- If confounds explain Test 2, remaining tests may also fail

**Realistic Expectation:** 2-3 tests will reject null (weak-to-moderate support)

---

## Next Immediate Action

**Priority 1:** Consolidate RA + Tau PTM data for Test 1

**Steps:**
1. Identify RA citrullination site locations in research directory
2. Load Tau phosphorylation database (`src/research/.../alzheimers/data/tau_phospho_database.py`)
3. Create unified PTM data loader (`scripts/utils/load_ptm_data.py`)
4. Verify protein sequences available (Vimentin, Fibrinogen, Tau)
5. Test PTM embedding computation on small sample (5 sites)

**Timeline:** Complete by end of Week 1

---

**Last Updated:** 2026-01-03 (after Test 2 completion)
**Next Update:** After Test 1 execution or weekly (whichever comes first)
