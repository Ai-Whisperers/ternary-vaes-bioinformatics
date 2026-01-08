# Cross-Disease Validation Research

**Research Area:** P-adic Codon Embeddings for Neurological and Immune Diseases
**Status:** Hypothesis Testing Phase
**Approach:** Computational validation → Structural validation → Experimental validation

---

## Research Question

Can p-adic codon embeddings reveal shared mechanisms across neurological diseases (ALS, Parkinson's) and immune diseases (Dengue DHF) through analysis of codon usage bias, PTM accumulation patterns, and protein interaction failures?

---

## Conjecture Statement

Neurological tissue-specific codon expression patterns drive PTM accumulation that can be detected and validated across diseases using p-adic hyperbolic embeddings, enabling cross-disease mechanism discovery.

**Decomposed Hypotheses:**
1. H1: Neurological tissues show distinct codon usage bias for disease-associated genes
2. H2: Disease-specific PTMs cluster in p-adic embedding space by mechanism
3. H3: Protein interaction failures across diseases share characteristic p-adic distances

---

## Scientific Approach

### Phase 1: Computational Validation (Current)

Test null hypotheses using existing data without acquiring new datasets:
- RA citrullination (47 sites) + Tau phosphorylation (47 sites) clustering
- ALS gene codon bias vs genome-wide averages
- Dengue serotype distances vs literature DHF rates
- HIV Goldilocks zone generalization to RA
- Contact prediction extension to disease PPIs

**Success Criteria:** ≥3 of 5 tests reject null hypothesis with medium-to-large effect sizes

### Phase 2: Expanded Computational Analysis

Acquire public datasets to test disease-specific predictions:
- GTEx motor cortex RNA-seq for tissue-specific codon bias
- PhosphoSitePlus TDP-43/alpha-synuclein PTMs
- GEO ALS/Parkinson's patient transcriptomics
- IEDB immune epitope data

**Success Criteria:** ALS/PD patterns consistent with RA/Tau, Dengue predictions replicate

### Phase 3: Structural Validation

Validate computational predictions with structural ground truth:
- AlphaFold3 predictions for PTM variants
- RMSD correlation with p-adic radial shifts
- Contact map validation for protein interactions
- Comparison to experimental structures (PDB, cryo-EM)

**Success Criteria:** ρ > 0.5 correlation between p-adic predictions and structural changes

### Phase 4: Experimental Validation (Collaboration)

Validate predictions with wet-lab experiments:
- Ribosome profiling for codon occupancy
- PTM stoichiometry measurements
- Protein aggregation assays
- Antibody binding assays (Dengue ADE)

**Success Criteria:** Experimental results confirm computational predictions

---

## Documents in This Directory

### Core Research Documents

| Document | Purpose | Status |
|----------|---------|:------:|
| **README.md** (this file) | Research overview and navigation | Complete |
| **CONJECTURE_ASSESSMENT.md** | Honest assessment of hypotheses, assumptions, limitations | Complete |
| **NULL_HYPOTHESIS_TESTS.md** | Falsifiable predictions with statistical tests | Complete |
| **RESEARCH_PLAN_NEUROLOGICAL_DISEASES.md** | Comprehensive technical roadmap (6 phases) | Reference |
| **DATA_ACQUISITION_GUIDE.md** | Data sources for Phase 2 expansion | Reference |
| **EXECUTIVE_SUMMARY.md** | High-level overview for stakeholders | Reference |

### Analysis Scripts (To Be Created)

```
scripts/
├── phase1_null_tests/
│   ├── test1_ptm_clustering.py           # RA + Tau clustering analysis
│   ├── test2_codon_bias.py               # ALS gene v=0 enrichment
│   ├── test3_dengue_dhf.py               # Serotype distance vs DHF rates
│   ├── test4_goldilocks_generalization.py # HIV vs RA overlap
│   └── test5_contact_prediction_ppi.py   # Disease PPI validation
├── phase2_expansion/
│   ├── gtex_codon_bias_analysis.py       # Tissue-specific analysis
│   ├── tdp43_ptm_sweep.py                # ALS PTM database
│   ├── alpha_synuclein_ptm_sweep.py      # PD PTM database
│   └── unified_ptm_clustering.py         # Cross-disease PTM analysis
└── phase3_structural/
    ├── alphafold3_ptm_validation.py      # PTM structure prediction
    ├── contact_map_validation.py         # PPI interface validation
    └── rmsd_correlation_analysis.py      # Structure vs p-adic correlation
```

---

## Current Infrastructure

### Production-Ready Tools

| Tool | Capability | Validation Status | Limitation |
|------|------------|:-----------------:|------------|
| **TrainableCodonEncoder** | Sequence → p-adic embedding | LOO ρ=0.61 (DDG) | Trained on S669 stability |
| **PTM Mapping Pipeline** | PTM impact prediction | RA citrullination validated | Only RA tested |
| **Contact Prediction** | Residue-residue contacts | AUC 0.67 (small proteins) | Fails for slow folders |
| **HIV Immune Escape** | Escape distance analysis | 77.8% boundary crossing | HIV-specific context |
| **HLA-Peptide Binding** | Epitope prediction | RA autoantigen validated | Class II MHC only |

### Available Data

| Dataset | Size | Use Case | Limitations |
|---------|------|----------|-------------|
| RA citrullination (47 sites) | 47 PTMs | Cross-disease PTM clustering | Single disease, single modification |
| Tau phosphorylation (47 sites) | 47 PTMs | Neurodegenerative PTM patterns | Alzheimer's, not ALS/PD |
| HIV CTL escape (9 mutations) | 9 variants | Goldilocks zone validation | Immune-specific, viral context |
| Contact prediction (15 proteins) | 15 structures | PPI prediction validation | Small proteins only |
| Dengue Paraguay (2011-2024) | ~200 genomes | Serotype trajectory analysis | No DHF severity metadata |

---

## Falsification Criteria

### What Would Disprove the Conjecture?

**Hypothesis H1 (Codon Bias):**
- ALS genes show no v=0 enrichment (p > 0.05 for all genes)
- GTEx analysis shows no tissue-specific bias after controlling for expression level
- Ribosome profiling shows identical codon occupancy across tissues

**Hypothesis H2 (PTM Clustering):**
- Silhouette score < 0.2 (random clustering)
- PTMs cluster by chemistry (phospho vs glycosyl), not disease
- AlphaFold3 structures show no correlation between p-adic shifts and RMSD (ρ < 0.3)

**Hypothesis H3 (Cross-Disease Validation):**
- Dengue serotype distances uncorrelated with DHF rates (ρ < 0.3, p > 0.05)
- "Failed handshake" distances span wide range (std > 3.0) across diseases
- Contact prediction fails for disease PPIs (AUC < 0.55)

**Global Falsification:**
- ≤2 of 5 null hypothesis tests reject null (weak computational evidence)
- Phase 2 expansion contradicts Phase 1 results (not reproducible)
- AlphaFold3 validation shows p-adic predictions are uncorrelated with structure (ρ < 0.3)

---

## Critical Assumptions

### Validated Assumptions

1. P-adic embeddings encode some biophysical information (DDG ρ=0.61, force constant ρ=0.86)
2. Contact prediction works for small fast-folding proteins (AUC 0.67-0.81)
3. RA PTM analysis identifies Goldilocks zones (validated with AlphaFold3)

### Unvalidated Assumptions (Require Testing)

1. v=0 codons are universally "optimal" for translation (literature-supported but context-dependent)
2. PTM clustering in p-adic space implies functional similarity (logical leap)
3. HIV immune escape Goldilocks zone (5.8-6.9) generalizes to non-immune interactions (speculation)
4. Codon usage drives PTM accumulation, not vice versa (causal direction unclear)
5. GTEx RNA-seq represents true codon usage (technical assumption, needs ribosome profiling)

### Known Confounds

1. Codon bias may reflect GC content, not functional optimization
2. PTMs in disease may be consequence, not cause (directionality)
3. Dengue ADE is multifactorial (antibody titer, Fc receptors, host genetics)
4. RNA-seq is post-mortem tissue (degradation, stress responses)
5. Batch effects between studies can create false tissue differences

---

## Honest Limitations

### What This Framework Cannot Do

1. **Prove causality** - Computational analysis is correlational only
2. **Replace experimental validation** - AlphaFold3 is a model, not ground truth
3. **Generalize beyond training data** - TrainableCodonEncoder trained on S669 DDG
4. **Account for context-dependence** - Tissue microenvironment, genetic background, temporal dynamics
5. **Predict low-stoichiometry PTMs** - Mass spec detection bias in databases

### What Would Strengthen the Conjecture

1. Ribosome profiling data (actual translation rates, not mRNA levels)
2. PTM stoichiometry data (what fraction is modified in vivo?)
3. Longitudinal patient data (track changes over disease progression)
4. Functional validation (mutagenesis, aggregation assays, immune assays)
5. Cross-species validation (mouse models, non-human Dengue strains)

---

## Execution Plan

### Phase 1: Null Hypothesis Testing (5 weeks)

**Week 1:** Test 1 - PTM clustering (RA + Tau)
**Week 2:** Test 2 - Codon bias (ALS genes)
**Week 3:** Test 3 - Dengue DHF correlation
**Week 4:** Test 4 - Goldilocks generalization
**Week 5:** Test 5 - Contact prediction PPIs

**Deliverable:** Null hypothesis test results report (transparent reporting of all outcomes)

**Decision Point:**
- If ≥3 tests reject null → Proceed to Phase 2
- If ≤2 tests reject null → Refine conjecture or focus on strongest signal

### Phase 2: Expanded Computational Analysis (8-12 weeks)

Acquire public datasets, expand to ALS/PD-specific analyses

**Decision Point:**
- If ALS/PD patterns consistent with Phase 1 → Proceed to Phase 3
- If contradictory results → Investigate confounds, may need to pivot

### Phase 3: Structural Validation (8-12 weeks)

AlphaFold3 predictions, correlation with p-adic embeddings

**Decision Point:**
- If ρ > 0.5 correlation → Strong computational evidence, prepare publication
- If ρ < 0.3 correlation → P-adic embeddings do not capture PTM/PPI effects, limit claims

### Phase 4: Experimental Validation (Collaboration-Dependent)

Seek collaborations for wet-lab validation

---

## Success Metrics

### Minimum Viable Publication

**Criteria:**
- Phase 1 complete with ≥3 of 5 tests rejecting null
- Phase 2 replicates Phase 1 findings with new datasets
- Transparent reporting of failures and limitations

**Publication Venue:** PLoS Computational Biology, Bioinformatics, BMC Bioinformatics

### High-Impact Publication

**Criteria:**
- All phases complete (Phases 1-3)
- AlphaFold3 validation shows ρ > 0.5 correlation
- Clinical application demonstrated (e.g., Dengue DHF risk tool)
- Experimental collaborators validate key predictions

**Publication Venue:** Nature Communications, Cell Systems, PNAS (computational biology section)

### Negative Result Publication

**Criteria:**
- Phase 1 complete with ≤2 tests rejecting null
- Thorough analysis of why conjecture failed
- Lessons learned for p-adic embedding applications

**Publication Venue:** PLoS ONE, F1000Research (negative results accepted)

---

## Reproducibility Standards

### Code and Data Sharing

**Pre-Registration:**
- Commit analysis scripts to version control BEFORE running tests
- Document all hypotheses and thresholds in advance (done in NULL_HYPOTHESIS_TESTS.md)

**Transparency:**
- Report all tests run, including failures
- Do not modify thresholds post-hoc to achieve significance
- Include code and processed data with publication

**Open Science:**
- Preprint on bioRxiv before journal submission
- Code on GitHub with MIT or Apache 2.0 license
- Data on Zenodo with DOI

### Computational Environment

**Software Versions:**
```
Python: 3.10+
PyTorch: 2.0+
geoopt: 0.5.0
scikit-learn: 1.3+
scipy: 1.11+
BioPython: 1.81+
```

**Random Seeds:**
- Set seeds for all stochastic processes (clustering, bootstrapping)
- Document in Methods section

**Hardware:**
- CPU-based analyses (p-adic distance computations are fast)
- GPU needed only for encoder inference (batch processing)

---

## Contact and Collaboration

### Internal Team

**Lead Researcher:** Responsible for overall execution, manuscript writing
**Bioinformatician:** Data acquisition, pipeline development
**Computational Biologist:** Statistical analysis, AlphaFold validation

### External Collaborations (Sought)

**Neurologist:** ALS/Parkinson's clinical context, patient data interpretation
**Immunologist:** Dengue DHF pathogenesis, ADE mechanism validation
**Structural Biologist:** Experimental structure comparison, AlphaFold validation
**Wet-Lab Partners:** PTM stoichiometry, aggregation assays, antibody binding

### Funding Opportunities

**Federal:**
- NIH R01 (neurological diseases)
- NSF Computational Biology

**Foundations:**
- Michael J. Fox Foundation (Parkinson's)
- ALS Association
- WHO/PAHO (Dengue surveillance)

**Industry:**
- Pharma partnerships for therapeutic target validation

---

## Ethical Considerations

### Use of Patient Data

**Public Datasets:**
- GTEx, GEO data are de-identified and publicly available
- Proper attribution to original studies required

**Future Patient Data:**
- Requires IRB approval if acquiring new patient samples
- Informed consent for genetic/transcriptomic analysis

### Clinical Translation

**Dengue DHF Risk Tool:**
- Not a diagnostic tool (research use only)
- Requires clinical validation before deployment
- Ethical obligation to share with public health agencies if validated

**ALS/PD Biomarkers:**
- Genetic information is sensitive (discrimination concerns)
- Counseling required for clinical implementation

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-03 | 1.0 | Initial research framework with honest assessment |

---

## Next Steps

1. Execute Phase 1, Test 1 (PTM clustering) using existing RA + Tau data
2. Document results regardless of outcome (transparent reporting)
3. Continue through all 5 null hypothesis tests
4. Compile results into Phase 1 report
5. Make Go/No-Go decision for Phase 2 based on objective criteria

---

**Research Philosophy:** Rigorous skepticism, transparent reporting, falsifiable predictions, and willingness to reject the conjecture if data does not support it.
