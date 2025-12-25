# SARS-CoV-2 Research

**Doc-Type:** Research Index · Version 1.0 · Updated 2025-12-25

---

## Overview

This directory contains research on SARS-CoV-2 spike protein using 3-adic hyperbolic codon geometry. The focus is on identifying asymmetric therapeutic targets that disrupt viral binding while preserving host protein function.

---

## Key Achievement

The framework validates that **N439/N440 glycan sites are critical handshake positions** for RBD-ACE2 binding. Phosphomimetic modifications at these sites disrupt viral binding without affecting host ACE2 function.

---

## Research Contents

### GLYCAN_SHIELD/

Analysis of spike protein glycan shield and RBD-ACE2 binding interface.

| Document | Description |
|:---------|:------------|
| `README.md` | Detailed project overview |
| `CONJECTURE_SPIKE_GLYCANS.md` | Initial hypothesis |
| `ANALYSIS_RESULTS.md` | Glycan shield findings |
| `HANDSHAKE_ANALYSIS_FINDINGS.md` | Interface mapping results |
| `ALPHAFOLD3_VALIDATION.md` | Structure validation with AF3 |

### Validated Findings

| Hypothesis | Prediction | Validation | Status |
|:-----------|:-----------|:-----------|:-------|
| N439/N440 critical | Tightest convergence | AlphaFold3 confirmed | VALIDATED |
| ACE2 unaffected | 0% host shift | pTM unchanged | VALIDATED |
| Y449 alternative target | High asymmetry | 19% PAE increase | VALIDATED |

### Therapeutic Candidates

1. **N439D + N440D double phosphomimic** - 12.7% interface disruption
2. **Y449D phosphomimic** - 19.0% interface disruption
3. **Peptide inhibitor**: `Ac-VIAWNDNLDDKVGG-NH2`

---

## Scripts

| Script | Purpose |
|:-------|:--------|
| `01_spike_sentinel_analysis.py` | Glycan site analysis |
| `02_handshake_interface_analysis.py` | Interface mapping |
| `03_deep_handshake_sweep.py` | 19 modification types |
| `04_alphafold3_validation_jobs.py` | Generate AF3 inputs |

---

## Related Documentation

- **Presentation Tier 1:** [Clinicians & Virologists](../../../../03_PRESENTATION_TIERS/01_TIER_1_PUBLIC/01_CLINICIANS_AND_VIROLOGISTS/)
- **Case Studies:** [Tier 2 Investor](../../../../03_PRESENTATION_TIERS/02_TIER_2_INVESTOR/05_CASE_STUDIES/)
- **Validation Suite:** [07_VALIDATION](../../02_THEORY_AND_FOUNDATIONS/07_VALIDATION/)

---

*Last updated: 2025-12-25*
