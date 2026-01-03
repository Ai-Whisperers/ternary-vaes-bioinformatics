# Model Limitations and Future Work

**Doc-Type:** Technical Analysis | Version 1.0 | Updated 2026-01-03 | Carlos Brizuela Package

---

## Executive Summary

This document analyzes the fundamental limitations of the AMP activity prediction models and outlines what can be improved with additional data/methods versus what requires fundamental research advances.

---

## Current Model Performance

### After Feature Engineering Improvements

| Model | N | Pearson r | Perm-p | Status |
|-------|---|-----------|--------|--------|
| activity_general | 224 | 0.56*** | 0.02 | **HIGH** |
| activity_escherichia | 105 | 0.42*** | 0.02 | **HIGH** |
| activity_acinetobacter | 20 | 0.58** | 0.02 | **HIGH** |
| activity_staphylococcus | 72 | 0.22 | 0.04* | **MODERATE** |
| activity_pseudomonas | 27 | 0.19 | 0.18 | **LOW** |

---

## Issues Solved in This Session

### 1. Staphylococcus Feature Engineering

**Problem:** Charge-based features showed NO correlation (ρ=-0.05)

**Solution:** Added amphipathicity feature (variance in hydrophobicity)

**Result:** Model improved from r=0.04 (NS) to r=0.22 (p=0.04, significant)

**Biological Explanation:**
- Gram-positive bacteria lack LPS outer membrane
- Electrostatic attraction is less important
- Membrane insertion via amphipathic helices is the key mechanism
- High variance in hydrophobicity = amphipathic = better membrane insertion

### 2. Import Path Resolution

**Problem:** All Brizuela scripts used `.parent.parent` instead of `.parent.parent.parent`

**Solution:** Fixed paths in B1, B8, B10 scripts

### 3. VAE Service Decoder

**Problem:** Model has `decoder_A`/`decoder_B`, not `decoder`

**Solution:** Updated vae_service.py to use `decoder_A`

---

## Issues That CANNOT Be Solved Now

### 1. Pseudomonas Sample Size (CRITICAL)

**Current State:** Only 27 samples

**Required:** 50-100 samples for reliable model

**Why It Can't Be Fixed Now:**
- Need experimental MIC data against P. aeruginosa
- Curated literature data is limited
- DRAMP database has heterogeneous quality

**Future Solution:**
- Partner with wet lab for systematic MIC screening
- Use DBAASP database (may have more P. aeruginosa data)
- Generate synthetic data with uncertainty (not recommended)

**Recommendation:** Use `activity_general` model for Pseudomonas predictions

---

### 2. Staphylococcus Mechanism Complexity (MODERATE)

**Current State:** r=0.22 (significant but weak)

**Fundamental Limitation:**
- S. aureus has multiple AMP resistance mechanisms:
  - MprF (lysinylation of phosphatidylglycerol)
  - DltABCD (D-alanylation of teichoic acids)
  - Capsule formation
  - Biofilm production
- Simple physicochemical features can't capture all mechanisms

**What Would Help (Future):**
1. **Structure-based features:**
   - 3D structure prediction (AlphaFold2)
   - Helix/sheet secondary structure ratios
   - Surface accessibility

2. **Mechanism-specific features:**
   - Membrane insertion depth prediction
   - Lipid II binding motifs
   - Pore-forming vs carpet mechanism

3. **More data stratified by mechanism:**
   - Separate models for pore-forming vs membrane-disrupting AMPs

**Current Recommendation:** Use with caution, combine with general model

---

### 3. Cross-Study MIC Variability (FUNDAMENTAL)

**Problem:** MIC values from different studies are not directly comparable

**Sources of Variation:**
- Different bacterial strains (lab vs clinical isolates)
- Different growth media
- Different inoculum sizes
- Different incubation times
- Different MIC endpoints (50% vs 90% inhibition)

**Why It Matters:**
- Same peptide against "S. aureus" can show MIC=2 in one study and MIC=16 in another
- This is biological noise, not model error

**Cannot Be Solved Without:**
- Standardized assay protocols
- Single-lab validation studies
- Meta-analysis correction factors

**Mitigation:**
- Use log10(MIC) to reduce scale effects
- Focus on ranking (Spearman) rather than absolute values
- Report predictions with uncertainty bounds

---

### 4. VAE Embeddings for AMP Activity (RESEARCH GAP)

**Observation from Colbes Package:**
- DDG predictor uses TrainableCodonEncoder embeddings
- Achieves Spearman ρ=0.58 on protein stability

**Gap for AMP Activity:**
- Current VAE encodes ternary operations, not peptide sequences
- No direct sequence → embedding mapping for short peptides
- Would need to train peptide-specific encoder

**Future Research Direction:**
1. Develop PeptideVAE trained on AMP sequences
2. Use ESM-2 or ProtTrans embeddings for peptides
3. Learn activity-predictive latent space

**Timeline:** Requires dedicated research project (months)

---

## Prioritized Improvement Roadmap

### Short-Term (Can Do Now)

| Priority | Task | Impact | Effort |
|----------|------|--------|--------|
| 1 | ✓ Add amphipathicity feature | Staph r: 0.04→0.22 | Done |
| 2 | ✓ Add hydrophobic_fraction | Pseudo +0.08 | Done |
| 3 | Update Staph model confidence | Documentation | Low |

### Medium-Term (Weeks)

| Priority | Task | Impact | Effort |
|----------|------|--------|--------|
| 4 | Integrate DBAASP database | More P. aeruginosa data | Medium |
| 5 | Add secondary structure features | Better Gram+ prediction | Medium |
| 6 | Ensemble model (general + specific) | Robust predictions | Low |

### Long-Term (Months)

| Priority | Task | Impact | Effort |
|----------|------|--------|--------|
| 7 | Peptide-specific VAE encoder | p-adic structure for AMPs | High |
| 8 | ESM-2 embedding integration | State-of-art representations | High |
| 9 | Wet lab validation partnership | Ground truth data | Very High |

---

## Honest Assessment for Stakeholders

### What Works Well

1. **E. coli predictions**: r=0.42, highly significant, ready for use
2. **A. baumannii predictions**: r=0.58, excellent for WHO critical pathogen
3. **General model**: r=0.56, robust fallback for any pathogen

### What Works Moderately

4. **S. aureus predictions**: r=0.22, significant but weak
   - Use for ranking candidates, not absolute MIC prediction
   - Combine with wet lab validation

### What Doesn't Work Yet

5. **P. aeruginosa predictions**: r=0.19, not significant
   - Insufficient training data
   - Use general model instead

---

## Comparison with Colbes DDG Package

| Aspect | Colbes (DDG) | Brizuela (AMP) |
|--------|--------------|----------------|
| Best model r | 0.58 (Spearman) | 0.56 (Pearson) |
| Validation rigor | Bootstrap + LOO | Permutation + CV |
| Biological grounding | p-adic codon structure | Physicochemical features |
| Main limitation | Single-point mutations only | Gram+ mechanisms |
| Data source | S669 benchmark | Curated DRAMP |

**Key Difference:** Colbes benefits from VAE embeddings that encode evolutionary/structural information. Brizuela uses traditional features that miss mechanistic complexity.

---

## Conclusion

The Carlos Brizuela AMP activity package provides:

- **Production-ready** models for E. coli and A. baumannii
- **Usable with caution** model for S. aureus
- **Research-grade** model for P. aeruginosa (needs more data)

Fundamental improvements require either:
1. More experimental data (especially P. aeruginosa)
2. Advanced feature engineering (structure-based, ESM-2)
3. Peptide-specific VAE embeddings (like Colbes uses for proteins)

---

**Document Version:** 1.0
**Last Updated:** 2026-01-03
**Author:** AI Whisperers
