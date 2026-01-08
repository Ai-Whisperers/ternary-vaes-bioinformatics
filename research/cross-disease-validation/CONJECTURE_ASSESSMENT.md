# Conjecture Assessment: Neurology, Codon Expression, and PTM Accumulation

**Doc-Type:** Scientific Assessment · Version 1.0 · Created 2026-01-03

---

## Core Conjecture (User-Stated)

"Neurology and nervous system overall plays a significant role in codons expression, thus in PTMs accumulation, thus we can exploit deeply each disease data to identify the proteins that are more sensitive or even candidates for each disease, and then cross-validate to improve our understanding of these diseases, specially Dengue"

---

## Formal Hypothesis Decomposition

### H1: Tissue-Specific Codon Usage Hypothesis

**Claim:** Neurological tissues (motor neurons, dopaminergic neurons) exhibit distinct codon usage bias compared to other tissues, affecting translation efficiency and protein quality control.

**Mechanistic Pathway:**
```
Tissue Type → tRNA Pool Composition → Codon Usage Optimization → Translation Kinetics → Co-translational Folding → PTM Timing
```

**Testable Predictions:**
1. ALS-associated genes (TARDBP, SOD1, FUS) show enrichment of "optimal" codons (p-adic valuation v=0) in motor neuron RNA-seq vs other tissues
2. Parkinson's-associated genes (SNCA, LRRK2) show similar enrichment in substantia nigra vs cerebellum
3. Codon usage bias correlates with protein aggregation propensity (slow translation at non-optimal codons → misfolding)

**Assumptions to Validate:**
- p-adic valuation v=0 codons are "optimal" for translation (needs citation)
- GTEx RNA-seq accurately represents in vivo codon usage (technical assumption)
- Codon bias affects translation speed (literature-supported but context-dependent)
- Translation speed affects co-translational folding (supported for some proteins, not universal)

**Known Confounds:**
- Codon usage can reflect GC content bias (genomic constraint, not selection)
- Tissue-specific bias may reflect gene expression level, not functional optimization
- Motor neurons vs substantia nigra may differ due to developmental lineage, not functional need
- RNA-seq captures steady-state mRNA, not actively translating ribosomes (ribosome profiling needed for true codon occupancy)

**Null Hypothesis:**
- Codon usage in neurological tissues is indistinguishable from other brain regions after controlling for expression level and GC content

---

### H2: PTM Accumulation Hypothesis

**Claim:** Post-translational modifications accumulate in disease-specific patterns that cluster in p-adic embedding space, revealing shared mechanisms across diseases.

**Mechanistic Pathway:**
```
Codon-Encoded Structure → Native Fold → PTM Site Accessibility → Kinase/Enzyme Recognition → PTM Addition → Structural Change → Disease Phenotype
```

**Testable Predictions:**
1. TDP-43 phosphorylation sites (S403, S404, S409, S410) cluster in p-adic space and separate from non-pathological sites
2. Alpha-synuclein S129 phosphorylation shifts embedding radially in Poincaré ball
3. Dengue NS1 glycosylation patterns differ between mild vs severe cases and correlate with p-adic distance from wildtype
4. Cross-disease PTMs (RA citrullination, Tau phosphorylation, ALS TDP-43 phosphorylation) cluster by modification type or disease mechanism

**Assumptions to Validate:**
- PTM mapping pipeline (`ptm_mapping.py`) accurately captures PTM-induced structural changes (validated for RA, needs extension)
- P-adic embeddings preserve PTM site accessibility information (unproven)
- PTM clustering in p-adic space implies functional similarity (logical leap)
- Disease-associated PTMs are causal, not consequence (directionality unclear)

**Known Confounds:**
- PTMs often occur post-aggregation (e.g., TDP-43 phosphorylation in inclusions may be secondary)
- Kinase promiscuity means clustering may reflect kinase expression, not substrate structure
- PTM databases (PhosphoSitePlus) are biased toward studied proteins (detection bias)
- Mass spectrometry PTM detection is incomplete (low stoichiometry modifications missed)

**Null Hypothesis:**
- PTMs cluster by modification chemistry (phosphorylation, glycosylation, etc.) regardless of disease context
- P-adic embeddings do not distinguish pathological from non-pathological PTMs better than random

---

### H3: Cross-Disease Validation Hypothesis

**Claim:** Dengue DHF immune "handshake" failures parallel neurological protein-protein interaction failures, enabling cross-disease validation.

**Mechanistic Pathway:**
```
Immune (Dengue): Non-neutralizing Antibody + NS1 → Failed Neutralization → ADE → DHF
Neurological (ALS): Misfolded TDP-43 + Chaperone → Failed Refolding → Aggregation
Neurological (PD): Alpha-synuclein + Clearance → Failed Autophagy → Lewy Bodies
```

**Testable Predictions:**
1. "Failed handshake" interactions show similar p-adic distances across diseases (5.8-6.9 from HIV CTL escape)
2. Successful interactions cluster at different distances (either closer for tight binding, or farther for avoidance)
3. Dengue serotype combinations with high DHF rates show NS1 p-adic distances in "Goldilocks zone"
4. ALS genetic modifiers (SMN, FUS) that fail to interact with TDP-43 show similar distances

**Assumptions to Validate:**
- HIV CTL escape "Goldilocks zone" (5.8-6.9) is generalizable to non-immune interactions (huge assumption)
- Protein-protein interaction failure is mechanistically equivalent to antibody-antigen mismatch (oversimplification)
- P-adic distance encodes interaction specificity (unproven for most PPIs)
- Dengue ADE is primarily driven by NS1 structure, not antibody isotype or Fc receptor polymorphisms (known to be multifactorial)

**Known Confounds:**
- ADE in Dengue involves multiple factors: antibody titer, affinity maturation, Fc receptor genotype, viral load, host immune status
- Protein aggregation in ALS/PD is multifactorial: age, oxidative stress, proteasome capacity, genetic background
- "Handshake failure" is a metaphor, not a quantifiable biophysical parameter
- Contact prediction AUC 0.67 means 33% false positive rate - not sufficient for clinical decisions

**Null Hypothesis:**
- P-adic distances between disease proteins are indistinguishable from random protein pairs
- Cross-disease clustering reflects shared biochemistry (e.g., all phosphoproteins cluster), not disease mechanism

---

## What We Can Test Computationally NOW

### Available Data (No New Acquisition Needed)

| Dataset | Status | Use Case | Limitations |
|---------|:------:|----------|-------------|
| TrainableCodonEncoder embeddings | READY | Codon-level analysis for any sequence | Trained on S669 DDG, may not generalize to PTMs |
| RA PTM database (47 citrullination sites) | READY | PTM clustering validation | Only citrullination, single disease |
| Tau PTM database (47 phospho-sites) | READY | Neurodegenerative PTM patterns | Alzheimer's, not ALS/PD |
| HIV immune escape data (9 CTL mutations) | READY | "Goldilocks zone" validation | HIV-specific, viral evolution context |
| Contact prediction (15 small proteins) | READY | Protein-protein interaction proxy | Only fast-folding proteins, not disease-relevant |
| Dengue sequences (Paraguay 2011-2024) | READY | Serotype trajectory analysis | No DHF severity metadata |
| V5.11.3 checkpoint embeddings (pre-extracted) | READY | All 19,683 ternary operations embedded | No biological validation yet |

### Computational Experiments (Executable Today)

**Experiment 1: PTM Clustering Null Test**
```
Hypothesis: RA citrullination + Tau phosphorylation cluster by disease or by modification type?
Method: Embed all 94 PTM sites in p-adic space, compute pairwise distances, hierarchical clustering
Validation: Silhouette score, dendrogram inspection
Falsification Criterion: If silhouette < 0.2 OR diseases mix randomly, reject H2
```

**Experiment 2: Codon Bias in Public Sequences**
```
Hypothesis: ALS genes show v=0 codon enrichment
Method: Download TARDBP/SOD1/FUS coding sequences, compute codon frequencies, compare to genome-wide average
Validation: Chi-squared test for codon usage
Falsification Criterion: If p > 0.05 for all 3 genes, reject H1 (or refine to "no genome-wide bias exists")
```

**Experiment 3: Dengue Serotype Distance vs Literature DHF Rates**
```
Hypothesis: Serotype combinations with high DHF rates show specific p-adic distances
Method: Encode NS1 from all 4 serotypes, compute pairwise distances, correlate with published DHF rates from literature
Validation: Spearman correlation
Falsification Criterion: If ρ < 0.3 AND p > 0.05, reject H3 for Dengue
```

**Experiment 4: HIV "Goldilocks Zone" Generalization**
```
Hypothesis: Optimal escape distance (5.8-6.9) applies to non-immune contexts
Method: Test if RA PTMs that successfully modulate immune response (from Goldilocks validation) fall in this range
Validation: Distribution overlap test
Falsification Criterion: If <50% of RA Goldilocks PTMs fall in HIV range, reject generalization
```

---

## Critical Assumptions Requiring Scrutiny

### Assumption 1: P-adic Embeddings Preserve Biologically Relevant Structure

**Evidence For:**
- TrainableCodonEncoder achieves ρ=0.61 DDG prediction (sequence-only)
- Contact prediction AUC 0.67 for small proteins (better than random)
- Force constant correlation ρ=0.86 (dimension 13 encodes physics)

**Evidence Against:**
- DDG prediction lags Rosetta (0.69) which uses structure
- Contact prediction fails for slow-folding proteins (ln(kf) < 7)
- Hierarchy ceiling -0.8321 means 17% of variance unexplained by valuation alone
- No validation for PTM effects (all current work is on wildtype sequences)

**Test:**
- Compare p-adic PTM predictions to AlphaFold3 RMSD (structure ground truth)
- If RMSD correlation < 0.5, p-adic embeddings do not preserve PTM structural changes

### Assumption 2: Codon Usage Drives Disease, Not Vice Versa

**Causal Direction Ambiguity:**
```
Possibility A (Hypothesis): Codon Usage → Translation Stress → Misfolding → Disease
Possibility B (Confound): Disease → Cell Stress → Altered tRNA Pools → Apparent Codon Bias in RNA-seq
```

**Test:**
- If codon bias appears AFTER disease onset (temporal analysis in longitudinal data), causality is reversed
- If codon bias is germline-encoded (DNA-level), supports Possibility A
- Without ribosome profiling (actual translation rates), cannot distinguish

### Assumption 3: Cross-Disease Mechanisms Are Shared, Not Convergent

**Shared Mechanism:**
```
Common upstream cause → Same intermediate pathway → Similar outcomes in different tissues
Example: P-adic valuation → Protein instability → Aggregation (ALS) OR Immune evasion (Dengue)
```

**Convergent Evolution:**
```
Independent causes → Different pathways → Superficially similar outcomes
Example: TDP-43 aggregation (prion-like) vs NS1 glycan shield (viral strategy) both evade clearance, but unrelated mechanisms
```

**Test:**
- If p-adic clustering shows statistical significance BUT mechanistic follow-up (AlphaFold, biochemistry) reveals distinct biophysics, clustering is spurious
- Requires experimental validation (beyond computational scope)

### Assumption 4: GTEx/GEO Data Represent True Biological State

**Technical Confounds:**
- RNA-seq is post-mortem tissue (ALS patients) - RNA degradation, stress responses
- GTEx is healthy controls - may not reflect disease-associated codon bias
- Batch effects between studies (GTEx vs GEO) can create false tissue differences
- Read depth and normalization methods affect codon frequency estimates

**Test:**
- Compare multiple ALS studies (GSE124439 vs GSE67196) - if codon bias inconsistent, suspect batch effects
- Control for RNA integrity number (RIN) - if low RIN samples drive signal, suspect degradation artifact

---

## Honest Assessment: What is Likely True vs Speculative

### High Confidence (Supported by Current Data)

1. **P-adic embeddings encode some biophysical information**
   - Evidence: DDG ρ=0.61, force constant ρ=0.86, contact prediction AUC 0.67
   - Limitation: "Some" is vague - quantify what fraction of structural variance is captured

2. **RA PTM analysis shows Goldilocks zones exist for immune modulation**
   - Evidence: 47 citrullination sites analyzed, structural validation with AlphaFold3
   - Limitation: Only citrullination, only RA, needs expansion to other PTMs/diseases

3. **HIV immune escape shows distance-efficacy relationships**
   - Evidence: 77.8% of CTL escape mutations cross cluster boundaries, optimal zone 5.8-6.9
   - Limitation: HIV-specific, adaptive immune context, not generalizable without validation

4. **Contact prediction works for fast-folding small proteins**
   - Evidence: AUC 0.814 for lambda repressor, correlation with folding rate ρ=0.625
   - Limitation: Fails for slow folders, unknown mechanism for why p-adic geometry predicts contacts

### Medium Confidence (Plausible but Needs Validation)

1. **Tissue-specific codon bias exists in neurological tissues**
   - Supporting: Well-documented for other tissues (liver, muscle), tRNA pools vary by cell type
   - Opposing: Brain regions may have similar tRNA pools (all neuronal), bias may be subtle
   - Test: GTEx comparison (executable now)

2. **PTMs cluster by disease mechanism in p-adic space**
   - Supporting: RA citrullination shows functional clustering (Goldilocks zones)
   - Opposing: May cluster by modification chemistry, not disease (phospho vs glycosylation)
   - Test: Cross-disease PTM clustering (executable now with RA + Tau data)

3. **Dengue serotype distances correlate with DHF risk**
   - Supporting: Serotype-specific DHF rates are well-documented, NS1 is central to pathogenesis
   - Opposing: ADE is multifactorial (antibody titer, host genetics), NS1 structure is one factor
   - Test: Literature DHF rates vs computed distances (executable now)

### Low Confidence (Speculative, Requires Extensive Validation)

1. **"Handshake failure" is a universal mechanism across diseases**
   - Supporting: Metaphorically appealing, some PPI failures are disease-relevant
   - Opposing: Different biophysics (antibody-antigen vs chaperone-substrate vs autophagy receptor-cargo)
   - Test: Requires experimental PPI assays, not computable from sequence alone

2. **P-adic distance generalizes from HIV immune escape to neurological PPIs**
   - Supporting: HIV data shows distance-efficacy relationship exists for some systems
   - Opposing: Immune recognition (MHC-peptide-TCR) is very different from protein aggregation
   - Test: Validate with known ALS PPI data (if available), otherwise speculative

3. **Codon usage drives PTM accumulation**
   - Supporting: Co-translational folding affects PTM site accessibility (some evidence)
   - Opposing: Most PTMs are post-translational, occur in fully folded proteins
   - Test: Requires ribosome profiling + PTM stoichiometry measurements (beyond computational scope)

---

## Falsification Criteria

### What Would Disprove H1 (Tissue-Specific Codon Bias)?

1. GTEx analysis shows no enrichment of v=0 codons in motor neurons vs cerebellum for ALS genes (p > 0.05 for all 3 genes)
2. Any apparent bias disappears after controlling for gene expression level (high-expressing genes use "optimal" codons regardless of tissue)
3. Ribosome profiling shows codon occupancy times are identical across tissues despite mRNA-level bias

### What Would Disprove H2 (PTM Clustering)?

1. Silhouette score < 0.2 for cross-disease PTM clustering (no better than random)
2. PTMs cluster by modification chemistry (all phospho together, all glycosyl together) with no disease-specific subclustering
3. AlphaFold3 structures show PTMs cause large RMSD changes that do not correlate with p-adic radial shifts (r < 0.3)

### What Would Disprove H3 (Cross-Disease Validation)?

1. Dengue serotype distances show no correlation with DHF rates (ρ < 0.3, p > 0.05)
2. "Failed handshake" distances span wide range (std > 3.0), indicating no universal failure zone
3. ALS PPI failures cluster separately from Dengue immune failures in p-adic space (distinct mechanisms)

---

## Recommended Scientific Approach

### Phase 1: Null Hypothesis Testing (Computational Only)

**Goal:** Test whether p-adic embeddings distinguish signal from noise

**Experiments:**
1. RA + Tau PTM clustering vs random protein sites
2. Codon bias (ALS genes vs random genes, GTEx tissues)
3. Dengue distances vs DHF rates (literature)
4. HIV Goldilocks zone vs RA Goldilocks zone overlap

**Success Criteria:**
- At least 2 of 4 experiments reject null hypothesis (p < 0.05)
- Effect sizes are medium-to-large (Cohen's d > 0.5 for comparisons, ρ > 0.5 for correlations)

**If Successful:** Proceed to Phase 2 (expand to ALS/PD with new data)
**If Failed:** Conclude p-adic embeddings do not generalize beyond training domain (S669 DDG)

### Phase 2: Computational Expansion (Requires New Data)

**Goal:** Test disease-specific predictions with newly acquired datasets

**Experiments:**
1. ALS codon bias (GTEx motor cortex RNA-seq)
2. TDP-43 PTM clustering (PhosphoSitePlus)
3. Alpha-synuclein PTM clustering
4. Cross-disease PTM unified database analysis

**Success Criteria:**
- ALS/PD show patterns consistent with RA/Tau (supports generalization)
- Dengue predictions from Phase 1 replicate with patient-level data (if available)

**If Successful:** Proceed to Phase 3 (structural validation)
**If Failed:** Disease-specific tuning needed, or p-adic framework is limited to certain protein classes

### Phase 3: Structural Validation (AlphaFold3 + Experimental)

**Goal:** Validate computational predictions with structural ground truth

**Experiments:**
1. AlphaFold3 predict WT vs PTM variants for top candidates from Phase 2
2. Compute RMSD, contact map changes, correlate with p-adic predictions
3. If available, validate against experimental structures (PDB, cryo-EM)
4. For Dengue, validate against antibody binding assays (literature or collaboration)

**Success Criteria:**
- P-adic predictions correlate with structural changes (ρ > 0.5)
- Contact predictions validated by AlphaFold-Multimer for PPIs (AUC > 0.65)

**If Successful:** Publish with experimental validation, claim predictive framework
**If Failed:** P-adic embeddings are useful for some tasks (DDG, contacts) but not PTMs or PPIs

---

## Honest Limitations

### What This Framework Cannot Do

1. **Prove Causality**
   - Computational analysis is correlational
   - Directionality requires perturbation experiments (CRISPR, mutagenesis, etc.)

2. **Replace Experimental Validation**
   - AlphaFold3 is a model, not ground truth (pLDDT < 70 regions are unreliable)
   - PTM effects on dynamics (not just structure) require MD simulations or NMR
   - Immune assays (T-cell activation, antibody binding) require wet lab

3. **Generalize Beyond Training Data**
   - TrainableCodonEncoder trained on S669 (single-point mutations, stability)
   - May not capture multi-site PTMs, protein-protein interfaces, membrane interactions

4. **Account for Context-Dependence**
   - Tissue microenvironment (oxidative stress, pH, chaperone availability) affects PTMs
   - Genetic background (modifier genes) affects disease penetrance
   - Temporal dynamics (aging, disease progression) not captured in static embeddings

### What Would Strengthen the Conjecture

1. **Ribosome Profiling Data**
   - Actual translation rates, not just mRNA levels
   - Codon occupancy times in motor neurons vs other tissues

2. **PTM Stoichiometry Data**
   - What fraction of TDP-43 is phosphorylated at S409 in vivo?
   - If <1%, may be secondary marker, not driver

3. **Longitudinal Patient Data**
   - Track codon bias changes over disease progression
   - Distinguish cause from consequence

4. **Functional Validation**
   - Mutate "optimal" codons to "non-optimal" in ALS genes, test aggregation propensity
   - Mutate Dengue NS1 glycosylation sites, test ADE in vitro

5. **Cross-Species Validation**
   - Do mouse models of ALS show same codon bias patterns?
   - Do non-human Dengue strains show same distance-DHF relationships?

---

## Conclusion: Scientific Rigor Assessment

### Strengths of Current Approach

1. Falsifiable predictions (null hypotheses defined)
2. Multiple independent validation axes (codon bias, PTMs, immune escape)
3. Existing validated tools (TrainableCodonEncoder, PTM mapping, contact prediction)
4. Cross-disease comparison enables mechanistic insights

### Weaknesses Requiring Mitigation

1. Heavy reliance on p-adic framework (single model architecture)
2. Assumptions about codon optimality (v=0) not validated for all contexts
3. Limited experimental validation planned (mostly computational)
4. Risk of overfitting to training data (S669 DDG, RA citrullination)

### Recommended Path Forward

**Immediate (Weeks 1-4):**
1. Execute Phase 1 null hypothesis tests with existing data
2. If ≥2 of 4 reject null, conjecture has computational support
3. Document failures transparently (negative results are informative)

**Medium-Term (Months 2-3):**
1. Acquire GTEx, PhosphoSitePlus, GEO data
2. Execute Phase 2 computational expansion
3. Compare predictions to literature (DDG, aggregation assays, DHF rates)

**Long-Term (Months 4-6):**
1. AlphaFold3 structural validation
2. Seek experimental collaborations for wet-lab validation
3. Publish with appropriate caveats about generalization limits

**Critical Success Factor:** Willingness to reject conjecture if data does not support it. Negative results (e.g., "p-adic embeddings do not predict PTM effects") are publishable and scientifically valuable.

---

**Version:** 1.0 · **Status:** Raw assessment, requires peer review
**Next Steps:** Execute Phase 1 null hypothesis tests, document results regardless of outcome
