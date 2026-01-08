# Future Validations - Parking Lot

**Doc-Type:** Research Planning · Version 1.0 · 2026-01-03 · AI Whisperers

**Purpose:** Document validation studies requiring prospective data collection or significant new resources

**Status:** PARKED - Will revisit after completing current validations with existing data

---

## Future Validation 1: Bee Venom PTM-Dependent Response

**Hypothesis:** Individual PTM states (pNF-κB, ACPA, IgE, pSTAT3) predict therapeutic vs allergic response to bee venom acupuncture in RA patients.

**Requires:**
- Prospective patient cohort (n=100 RA patients)
- PTM biomarker measurements (pNF-κB, ACPA, IgE, tryptase, Th17/Treg, pSTAT3)
- 12-week BVA intervention
- Clinical outcome tracking (DAS28, adverse events)

**Timeline:** 2-3 years (IRB approval, recruitment, intervention, analysis)

**Resources Needed:**
- Clinical collaborator (rheumatologist)
- IRB approval
- Patient recruitment
- Biomarker assays (~$500/patient × 100 = $50,000)
- BVA intervention costs

**Documents:**
- `research/cross-disease-validation/bee-venom/BEE_VENOM_RA_INVESTIGATION.md`
- `research/cross-disease-validation/bee-venom/DATA_COLLECTION_PROTOCOL.md`

**Why Parked:** Requires prospective data collection, significant resources, clinical partnerships

---

## Future Validation 2: HybridPTMEncoder Training

**Hypothesis:** Extending TrainableCodonEncoder with PTM-specific features improves prediction of citrullination binding and phosphorylation stability.

**Requires:**
- ProTherm phosphorylation dataset (n=500 ΔΔG measurements) - need to curate
- IEDB citrullination dataset (n=200 ACPA binding measurements) - need to curate
- Acetylation/methylation stability data (n=100) - need to curate from literature
- 4-6 weeks implementation + training

**Timeline:** 8-12 weeks (data curation + implementation)

**Resources Needed:**
- ProTherm database access (free)
- IEDB database access (free)
- Literature mining for acetylation data (manual curation)
- GPU for training (available)

**Documents:**
- `research/cross-disease-validation/PHASE2_READINESS_ROADMAP.md` (Gap 1)

**Why Parked:** Requires significant data curation effort before implementation

---

## Future Validation 3: Curated PTM Database v2.0

**Hypothesis:** 73% of literature PTM positions can be corrected through systematic cross-referencing (UniProt, PhosphoSitePlus, dbPTM) and manual literature review.

**Requires:**
- Automated verification pipeline (UniProt API, PhosphoSitePlus API, dbPTM scraping)
- Manual literature review of 33 failed RA citrullination sites (download PDFs, contact authors)
- Expansion to Tau phosphorylation, MS citrullination, T1D PTMs
- 3-4 weeks curation work

**Timeline:** 6-8 weeks

**Resources Needed:**
- PhosphoSitePlus account (free academic)
- dbPTM web scraping
- PDF access (Sci-Hub or institutional)
- Author contact for unclear coordinates

**Documents:**
- `research/cross-disease-validation/PHASE2_READINESS_ROADMAP.md` (Gap 2)

**Why Parked:** Significant manual curation effort, not immediately executable

---

## Future Validation 4: AlphaFold Structural Contact Prediction

**Hypothesis:** Combining p-adic codon embeddings with AlphaFold structural features (pLDDT, RSA, contact number) improves contact prediction for medium/large proteins.

**Requires:**
- SwissProt CIF dataset processing (38GB, extract 1000 structures)
- Parse CIF files with BioPython (coordinates, pLDDT)
- Compute RSA, secondary structure, contact maps
- Implement StructuralContactPredictor architecture
- Train on 800 proteins, validate on 200
- 4-6 weeks implementation + training

**Timeline:** 8-10 weeks

**Resources Needed:**
- SwissProt CIF tar file (available: `research/big_data/swissprot_cif_v6.tar`)
- BioPython, DSSP (free)
- GPU for training (available)
- jose_colbes AlphaFold pipeline (available)

**Documents:**
- `research/cross-disease-validation/PHASE2_READINESS_ROADMAP.md` (Gap 4)

**Why Parked:** Requires significant data processing (38GB dataset) before implementation

---

## Future Validation 5: Optimized Melittin Variants (NSGA-II)

**Hypothesis:** carlos_brizuela NSGA-II framework can design melittin variants with preserved NF-κB inhibition but reduced IgE cross-reactivity and mast cell activation.

**Requires:**
- Implement objective functions (NF-κB inhibition, IgE epitope similarity, amphipathicity)
- Run NSGA-II optimization (200 population, 100 generations)
- In vitro validation (RA synoviocytes, mast cells, IgE binding assays)
- In vivo validation (collagen-induced arthritis mice)
- 6-12 months (in silico design + experimental validation)

**Timeline:** 1-2 years (design + synthesis + testing)

**Resources Needed:**
- carlos_brizuela VAE + NSGA-II framework (available)
- Peptide synthesis (~$500/peptide × 10 variants = $5,000)
- In vitro assays (~$2,000)
- Animal experiments (~$10,000)

**Documents:**
- `research/cross-disease-validation/BEE_VENOM_RA_INVESTIGATION.md`
- `deliverables/partners/carlos_brizuela/README.md`

**Why Parked:** Requires experimental validation (in vitro + in vivo), significant resources

---

## Future Validation 6: Prospective Test 2.1 (RA vs MS Citrullination)

**Hypothesis:** Citrullination sites in RA vs MS share hyperbolic distance patterns (within-PTM-type generalization).

**Requires:**
- Curated RA citrullination sites (n=20-25 verified, from Gap 2)
- Curated MS citrullination sites (n=15-20, need to collect from literature)
- HybridPTMEncoder trained on citrullination (from Gap 1)
- Compute hyperbolic distances, test overlap (>60% threshold)

**Timeline:** 4-6 weeks (after Gap 1 and Gap 2 complete)

**Resources Needed:**
- Curated PTM database v2.0 (from Gap 2)
- HybridPTMEncoder (from Gap 1)
- MS citrullination literature curation (~1 week)

**Documents:**
- `research/cross-disease-validation/PHASE2_READINESS_ROADMAP.md` (Gap 3, Test 2.1)

**Why Parked:** Depends on Gap 1 and Gap 2 completion first

---

## Future Validation 7: Prospective Test 2.2 (Tau vs TDP-43 Phosphorylation)

**Hypothesis:** Tau phosphorylation (Alzheimer's) vs TDP-43 phosphorylation (ALS) show correlated hyperbolic distances (within-phosphorylation generalization).

**Requires:**
- Curated Tau phosphorylation sites (n=40-45, available in existing data)
- Curated TDP-43 phosphorylation sites (n=20-25, need to collect from literature)
- HybridPTMEncoder trained on phosphorylation (from Gap 1)
- Compute correlation (Spearman ρ > 0.5 threshold)

**Timeline:** 3-4 weeks (after Gap 1 complete)

**Resources Needed:**
- HybridPTMEncoder (from Gap 1)
- TDP-43 phosphorylation literature curation (~1 week)

**Documents:**
- `research/cross-disease-validation/PHASE2_READINESS_ROADMAP.md` (Gap 3, Test 2.2)

**Why Parked:** Depends on Gap 1 completion first

---

## Future Validation 8: Large-Scale SwissProt Analysis

**Hypothesis:** P-adic valuation correlates with structural features (disorder, surface exposure, secondary structure) across 200,000+ proteins.

**Requires:**
- SwissProt CIF dataset (38GB, 200k+ structures)
- Process all structures (not just 1000 sample)
- Extract per-residue features: pLDDT, RSA, disorder, SS
- Correlate p-adic valuation with structural features
- 4-8 weeks data processing + analysis

**Timeline:** 2-3 months

**Resources Needed:**
- SwissProt CIF tar file (available)
- High-performance computing (process 200k structures)
- BioPython, DSSP, IUPred (disorder prediction)

**Documents:**
- `.claude/CLAUDE.md` (SwissProt Structure Dataset section)

**Why Parked:** Massive computational undertaking, requires HPC resources

---

## Prioritization Criteria

**Immediate (Current Validations):**
- Uses existing validated tools (TrainableCodonEncoder, v5_11_structural)
- Uses existing datasets (S669, contact prediction, HIV, GO)
- Executable within 1-4 weeks
- No new data collection required

**Near-Term (3-6 months):**
- Requires moderate data curation (<4 weeks)
- Builds on current validations
- Extends validated tools to new domains

**Long-Term (6-24 months):**
- Requires prospective data collection
- Requires experimental validation (in vitro, in vivo)
- Requires significant resources ($10k+)
- Requires clinical collaborations

---

## Revisit Triggers

**Bee Venom (Validation 1):** Revisit when:
- HybridPTMEncoder validated on phosphorylation (Gap 1 complete)
- Curated PTM database available (Gap 2 complete)
- Clinical collaborator identified
- Funding secured

**HybridPTMEncoder (Validation 2):** Revisit when:
- ProTherm phosphorylation subset curated (1 week effort)
- Ready to implement architecture (after current validations)

**Curated PTM Database (Validation 3):** Revisit when:
- Automated verification pipeline implemented (2 weeks)
- Ready for manual literature review (3 weeks)

**AlphaFold Contact Prediction (Validation 4):** Revisit when:
- Small protein conjecture fully validated (current validation)
- SwissProt processing pipeline ready (1 week setup)

**All others:** Revisit when dependencies (Gap 1, Gap 2) complete

---

## Summary

**Total Future Validations:** 8 major studies
**Timeline Range:** 3 months to 2+ years
**Resource Range:** Moderate (data curation) to High (clinical trials, $50k+)

**Decision:** Focus on CURRENT validations with existing data first. Revisit these when:
1. Current validations complete (4-8 weeks)
2. Phase 2 gaps addressed (8-12 weeks)
3. Resources/collaborations secured (ongoing)

---

**Version:** 1.0 · **Date:** 2026-01-03 · **Status:** PARKED - Documented for Future Reference
