# Executive Summary: Neurological & Dengue Cross-Disease Research

**Doc-Type:** Executive Summary · Version 1.0 · Created 2026-01-03

---

## Core Conjecture

**"Neurology and the nervous system play a significant role in codon expression, which drives PTM accumulation patterns that can be exploited across diseases for validation and therapeutic discovery."**

---

## Key Findings from Infrastructure Assessment

### What We Have (Production-Ready)

| Asset | Status | Capability | Use Case |
|-------|:------:|------------|----------|
| **Alejandra Rojas Package** | ✅ PRODUCTION | Arbovirus surveillance, primer design, trajectory forecasting | Dengue baseline |
| **TrainableCodonEncoder** | ✅ VALIDATED | LOO ρ=0.61 DDG prediction, physics-informed embeddings | ALS/PD codon analysis |
| **PTM Mapping Pipeline** | ✅ COMPREHENSIVE | 30+ PTM types, RA (47 sites), Tau (47 sites) | Cross-disease PTM database |
| **HIV Immune Escape** | ✅ COMPREHENSIVE | CTL escape, drug resistance, tropism, neutralization | Dengue DHF ADE modeling |
| **Contact Prediction** | ✅ VALIDATED | AUC 0.586-0.814 for small proteins, folding rate correlation | Protein-protein "handshakes" |
| **HLA-Peptide Analysis** | ✅ VALIDATED | RA autoantigen binding, epitope clustering | Neuroinflammation epitopes |

### What We Need (Gaps)

| Missing Component | Impact | Effort | Priority |
|-------------------|--------|:------:|:--------:|
| **ALS PTM Database** | No TDP-43/SOD1 phosphorylation data | 1 week | **HIGH** |
| **Parkinson's PTM Database** | No alpha-synuclein S129 phospho data | 1 week | **HIGH** |
| **Dengue DHF Immune Model** | No ADE risk prediction | 2 weeks | **HIGH** |
| **Neurological Codon Bias** | No motor neuron/SN-specific data | 2 weeks | **HIGH** |
| **Cross-Disease Framework** | No unified validation pipeline | 4 weeks | MEDIUM |

**Total Time to Fill Gaps:** 8-12 weeks (parallelizable)

---

## Research Strategy

### Three-Way Cross-Validation

```
        ALS/Parkinson's (Neurological)
               ↕
        Shared Mechanisms
               ↕
        Dengue DHF (Immune)
```

**Validation Axes:**
1. **Codon Bias** → Tissue-specific expression patterns
2. **PTM Accumulation** → Disease-specific modifications cluster in p-adic space
3. **Immune Recognition** → HLA-DR epitope presentation (neuroinflammation + dengue)
4. **Protein "Handshakes"** → Failed interactions show universal p-adic distance

---

## Immediate High-Impact Projects (Weeks 1-4)

### Project 1: ALS Codon Bias Analysis
- **Question:** Do motor neurons optimize codons differently for ALS genes?
- **Data:** GTEx motor cortex RNA-seq + TARDBP/SOD1/FUS sequences
- **Tool:** TrainableCodonEncoder + `padic_valuation()`
- **Expected:** v=0 codons enriched >1.5x in motor neurons vs other tissues
- **Timeline:** 2 weeks
- **Impact:** First demonstration of neurological codon bias

### Project 2: Dengue NS1 PTM Sweep
- **Question:** Do DHF-associated serotypes have distinct PTM patterns?
- **Data:** NS1 sequences (DENV1-4), DHF patient metadata
- **Tool:** `ptm_mapping.py` + RA PTM database logic
- **Expected:** Glycosylation differences correlate with DHF severity
- **Timeline:** 3 weeks
- **Impact:** Mechanistic basis for ADE (antibody-dependent enhancement)

### Project 3: Cross-Disease PTM Clustering
- **Question:** Do ALS, Parkinson's, Dengue PTMs cluster in p-adic space?
- **Data:** RA (47 sites) + Tau (47 sites) + ALS (20+ sites) + PD (10+ sites) + Dengue (5+ sites)
- **Tool:** Unified PTM database + hierarchical clustering
- **Expected:** Immune-mediated diseases cluster together
- **Timeline:** 4 weeks (includes database curation)
- **Impact:** Universal PTM-disease map

---

## Expected Validation Metrics

| Hypothesis | Metric | Success Criterion | Clinical Impact |
|------------|--------|-------------------|-----------------|
| **H1:** Motor neurons optimize ALS gene codons | Fold enrichment (v=0) | > 1.5, p < 0.05 | Early detection biomarker |
| **H2:** DHF correlates with NS1 PTM patterns | Spearman ρ (PTM vs severity) | > 0.6 | ADE risk prediction |
| **H3:** PTMs cluster by disease mechanism | Silhouette score | > 0.3 | PTM-targeted therapies |
| **H4:** "Failed handshakes" share p-adic distance | Std across diseases | < 1.5 | Universal therapeutic target |
| **H5:** HLA epitopes cluster by disease/allele | Chi-squared p-value | < 0.05 | Vaccine design, immunotherapy |

---

## 6-Month Roadmap

### Phase 1: Foundation (Weeks 1-4) ✅ HIGH PRIORITY
- ✅ Data acquisition (GTEx, GEO, PhosphoSitePlus, NCBI Virus)
- ✅ ALS codon bias analysis
- ✅ Parkinson's codon bias analysis
- ✅ Dengue NS1 PTM sweep

**Milestone:** Codon bias validated + Dengue PTM baseline

### Phase 2: PTM Analysis (Weeks 5-8)
- ALS TDP-43 PTM sweep (20+ phospho-sites)
- Parkinson's alpha-synuclein PTM sweep (S129 focus)
- Unified PTM database integration (5 diseases)
- Cross-disease PTM clustering analysis

**Milestone:** Unified PTM database with clustering validation

### Phase 3: Immune Recognition (Weeks 9-12)
- ALS neuroinflammation epitopes (HLA-DR binding)
- Parkinson's microglial epitopes
- Dengue DHF HLA binding analysis
- Cross-disease immune recognition validation

**Milestone:** HLA-DR epitope maps for all 3 diseases

### Phase 4: Protein Interactions (Weeks 13-16)
- ALS PPI network (SMN, FUS, C9ORF72)
- Parkinson's LRRK2 substrate network
- Dengue ADE risk prediction (Goldilocks zone)
- Universal "handshake failure" analysis

**Milestone:** Protein interaction networks + universal failure mechanism

### Phase 5: Structural Validation (Weeks 17-20)
- AlphaFold3: ALS misfolded structures (TDP-43 phospho-variants)
- AlphaFold3: Parkinson's fibrils (alpha-synuclein S129)
- AlphaFold3: Dengue antibody complexes (NS1-antibody)
- Cross-disease structural comparison

**Milestone:** AlphaFold3 validation for all 3 diseases

### Phase 6: Integration & Publication (Weeks 21-24)
- Alejandra Rojas package extension (DHF risk module)
- Cross-disease validation dashboard
- Manuscript preparation
- Code release + supplementary materials

**Milestone:** Publication-ready research package

---

## Resource Requirements

### Personnel (Recommended)
- **Lead Researcher:** 1.0 FTE (6 months) - $60K
- **Bioinformatician:** 0.5 FTE (3 months) - $30K
- **Computational Biologist:** 0.5 FTE (3 months) - $30K
- **Domain Consultants:** 0.1 FTE each (Neurologist, Immunologist, Structural Biologist) - $15K total

**Total Personnel:** ~$135K

### Computational Resources
- **Training:** None (TrainableCodonEncoder already trained)
- **Storage:** ~100 GB (GTEx ~50 GB, GEO ~10 GB, structures ~10 GB, results ~30 GB)
- **Compute:** AlphaFold3 via free server (10 predictions/day) or Google Colab
- **Budget:** ~$2K cloud storage + compute

### Total Budget Estimate: ~$140K (6 months)

---

## Go/No-Go Decision Points

### End of Week 4 (Phase 1)
**Success:** ≥2 of 3 validated (ALS codon bias, PD codon bias, Dengue PTM)
- **GO:** Proceed to Phase 2 (PTM analysis)
- **NO-GO:** Refocus on strongest signal (likely Dengue DHF)

### End of Week 8 (Phase 2)
**Success:** PTM clustering silhouette > 0.3 OR clear PTM-phenotype correlation
- **GO:** Proceed to Phase 3 (immune recognition)
- **NO-GO:** Analyze diseases separately, skip cross-disease integration

### End of Week 12 (Phase 3)
**Success:** HLA-DR binding validated + correlation with clinical data
- **GO:** Proceed to Phase 4 (PPI networks)
- **NO-GO:** Focus on clinical translation (Dengue DHF risk tool)

### End of Week 20 (Phase 5)
**Minimum Publishable Unit:** ≥1 disease fully validated (all 5 phases)
- **High-Impact:** Nature Communications, Cell Systems, PNAS
- **Medium-Impact:** PLoS Computational Biology, Bioinformatics
- **Domain:** J Neuroinflammation, Antiviral Research

---

## Immediate Next Steps (Week 1 Action Items)

### Data Acquisition (Priority Order)

1. **GTEx Portal** → Motor cortex + Substantia nigra RNA-seq
   - URL: https://gtexportal.org/home/datasets
   - Deliverable: `motor_cortex_samples.csv`, `substantia_nigra_samples.csv`

2. **PhosphoSitePlus** → TDP-43, SOD1, Alpha-synuclein PTMs
   - URL: https://www.phosphosite.org/
   - Deliverable: `tdp43_ptms.csv`, `alpha_synuclein_ptms.csv`

3. **NCBI Virus** → Dengue genomes (DENV1-4, Paraguay)
   - URL: https://www.ncbi.nlm.nih.gov/labs/virus/
   - Deliverable: `dengue_1-4_Paraguay.fasta`, `dengue_NS1_sequences.fasta`

4. **GEO Database** → ALS patient transcriptomics
   - IDs: GSE124439, GSE67196
   - Deliverable: `als_patient_expression.csv`

### Environment Setup

```bash
# Create research directories
mkdir -p research/cross-disease-validation/{data,scripts,results,docs}
mkdir -p src/research/bioinformatics/codon_encoder_research/als/{data,scripts,results}
mkdir -p src/research/bioinformatics/codon_encoder_research/parkinsons/{data,scripts,results}
mkdir -p deliverables/partners/alejandra_rojas/dengue_dhf/{data,scripts,results}
```

### Script Templates (Week 1)

Create TODO-marked templates for:
- `01_als_codon_bias_analysis.py` (see DATA_ACQUISITION_GUIDE.md section 8.3)
- `02_tdp43_ptm_sweep.py`
- `01_parkinsons_codon_bias.py`
- `01_ns1_ptm_sweep.py`

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| Insufficient ALS/PD RNA-seq data | Low | High | Use published GEO datasets (>100 ALS studies available) |
| P-adic distances don't predict ADE | Medium | Medium | Fall back to ensemble ML predictor (p-adic as features) |
| Cross-disease PTMs don't cluster | Medium | Low | Still valuable per-disease (null hypothesis validated) |
| AlphaFold3 validation too slow | Low | Low | Parallelize predictions, use AlphaFold Server API |
| Timeline overruns | Medium | Medium | Prioritize Phases 1-3 (core validation), extend 4-6 if needed |

---

## Success Criteria Summary

### Minimum Viable Publication (16 weeks)
- ✅ ALS codon bias validated (motor neuron-specific)
- ✅ Dengue NS1 PTM-DHF correlation (ADE mechanism)
- ✅ Unified PTM database (5 diseases, 150+ sites)
- ✅ Cross-disease clustering analysis (even if negative result)

### High-Impact Publication (24 weeks)
- All above +
- ✅ HLA-DR epitope atlas (ALS/PD/Dengue)
- ✅ Universal "handshake failure" zone validated
- ✅ AlphaFold3 structural validation
- ✅ Clinical decision tool (Dengue DHF risk assessment)

---

## Long-Term Vision (Beyond 6 months)

### Clinical Translation
- **Dengue DHF:** Real-time ADE risk assessment app (input: patient serology + circulating serotypes)
- **ALS:** Biomarker-guided clinical trials (TDP-43 codon bias screening)
- **Parkinson's:** Precision medicine based on SNCA variants + S129 phosphorylation trajectory

### Technology Development
- Unified PTM-disease database (public release)
- Cross-disease epitope atlas (HLA-DR binding)
- Tissue-specific codon bias database (extend GTEx)

### Funding Opportunities
- **NIH R01:** $1.5-2M over 5 years (ALS/Parkinson's)
- **WHO/PAHO:** $500K-1M over 3 years (Dengue surveillance, Latin America)
- **Foundation Grants:** $250-500K over 2 years (Michael J. Fox Foundation, ALS Association)

**Estimated Total Funding Potential:** $2-4M over 5 years

---

## Key Contacts & Collaboration

### Internal
- **Alejandra Rojas** - Arbovirus surveillance expertise, Paraguay connections
- **Lead Researcher** - Cross-disease integration, manuscript writing
- **Bioinformatics Team** - Data processing, pipeline development

### External (Recommended Collaborations)
- **Neurologist** - ALS/Parkinson's clinical context, patient data access
- **Immunologist** - Dengue DHF pathogenesis, ADE validation
- **Structural Biologist** - AlphaFold3 validation, experimental structure comparison

---

## Deliverables Checklist

### Phase 1 (Week 4)
- [ ] ALS codon bias report (PDF + figures)
- [ ] Parkinson's codon bias report
- [ ] Dengue NS1 PTM analysis (CSV + visualizations)
- [ ] Data acquisition complete (all sources downloaded)

### Phase 2 (Week 8)
- [ ] Unified PTM database (JSON + CSV)
- [ ] Cross-disease PTM clustering results (dendrogram + UMAP)
- [ ] TDP-43 PTM sweep validation
- [ ] Alpha-synuclein PTM sweep validation

### Phase 3 (Week 12)
- [ ] HLA-DR epitope predictions (CSV for all 3 diseases)
- [ ] Cross-disease immune recognition analysis
- [ ] Correlation with clinical data (DHF severity, ALS progression)

### Phase 4 (Week 16)
- [ ] ALS PPI network results (network graphs + JSON)
- [ ] Parkinson's substrate network
- [ ] Dengue ADE risk matrix (12 serotype combinations)
- [ ] Universal handshake failure zone report

### Phase 5 (Week 20)
- [ ] AlphaFold3 structures (PDB files for 10+ key proteins/variants)
- [ ] Cross-disease structural validation report
- [ ] Contact prediction AUC results

### Phase 6 (Week 24)
- [ ] Manuscript (draft + final)
- [ ] Supplementary materials (tables, figures, methods)
- [ ] Code release (GitHub repository with documentation)
- [ ] Alejandra Rojas package extension (Dengue DHF module)
- [ ] Cross-disease validation dashboard (interactive)

---

## Questions to Address First

Before starting data acquisition, clarify:

1. **Geographic Focus:** Paraguay-only for Dengue, or expand to Brazil/Vietnam cohorts for DHF severity data?
2. **Patient Data Access:** Do we have IRB approval for ALS/PD patient transcriptomics, or use only public GEO datasets?
3. **Collaboration:** Should we reach out to neurologists/immunologists early, or wait for preliminary results?
4. **Publication Target:** Aim for high-impact general (Nature Comms, PNAS) or domain-specific (J Neuroinflammation)?
5. **Funding:** Apply for grants immediately (NIH deadlines), or self-fund initial validation?

---

## Conclusion

This research plan leverages **mature infrastructure** (TrainableCodonEncoder, PTM mapping, HLA analysis, contact prediction) to test a **bold hypothesis**: neurological and immune diseases share fundamental mechanisms visible in codon expression and PTM accumulation.

**Unique Advantages:**
- ✅ Production-ready tools (no development needed)
- ✅ Validated frameworks (HIV immune escape, RA PTM, contact prediction)
- ✅ Clear clinical applications (Dengue DHF risk, ALS biomarkers, PD precision medicine)
- ✅ Cross-validation strategy (three diseases validate each other)

**Immediate Action:** Begin Week 1 data acquisition (GTEx, PhosphoSitePlus, NCBI Virus, GEO)

**Go/No-Go Review:** End of Week 4 (Phase 1 milestone)

---

**Document Status:** Ready for execution
**Next Review:** Weekly progress updates, Go/No-Go decisions at Phase boundaries
**Contact:** AI Whisperers Research Team

---

## Quick Reference: Document Map

| Document | Purpose | Use Case |
|----------|---------|----------|
| **EXECUTIVE_SUMMARY.md** (this file) | High-level overview | Share with stakeholders, funding agencies |
| **RESEARCH_PLAN_NEUROLOGICAL_DISEASES.md** | Comprehensive technical plan | Day-to-day research guidance |
| **DATA_ACQUISITION_GUIDE.md** | Data sources & download instructions | Week 1 data collection |

**All documents located in:** `research/cross-disease-validation/`
