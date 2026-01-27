# Arbovirus Surveillance Package - Bias Analysis

**Date:** 2026-01-26
**Analyst:** AI Whisperers Team
**Status:** MINIMAL CONCERNS - Different methodology

---

## Executive Summary

This package is **fundamentally different** from the DDG and AMP packages:
- NOT an ML prediction model - it's a computational primer design system
- NO StandardScaler data leakage - no machine learning training involved
- Already has rigorous skeptical validation built-in

| Issue | Severity | Impact |
|-------|:--------:|--------|
| No scaler leakage | N/A | Not applicable (no ML) |
| Selection bias | **LOW** | 270 genomes from NCBI RefSeq |
| Orthogonality claim clarification | **MEDIUM** | Claim partially rejected |

---

## Package Nature

This package does NOT use machine learning for its core claims. Instead, it uses:
- **Shannon entropy** - classical information theory
- **P-adic/hyperbolic variance** - mathematical structure analysis
- **K-mer analysis** - sequence comparison
- **Consensus primer design** - computational biology

No model is trained, so no train/test leakage is possible.

---

## Already-Completed Skeptical Validation

**File:** `research/validation/skeptical_validation.py`
**Results:** `research/validation/results/skeptical_validation_results.json`

This validation explicitly tests each claim and provides honest assessments:

### Claim 1: Metric Orthogonality

| Test | Result | Status |
|------|--------|:------:|
| Raw Spearman ρ | -0.11 | Weak correlation |
| Windowed ρ (75bp) | -0.48 | Moderate correlation |
| Assessment | NEEDS CLARIFICATION | Partially rejected |

**The package correctly identifies that the orthogonality claim needs clarification.**

### Claim 2: Pan-DENV-4 Primer Infeasibility

| Region | Total Degeneracy | Practical Limit |
|--------|-----------------|-----------------|
| 5'UTR | 1.1 trillion | 4,096 |
| E gene | 1.1 trillion | 4,096 |
| NS5 regions | 1.1 trillion | 4,096 |

**Status: CONFIRMED** - Degeneracy exceeds practical limits at all candidate sites.

### Claim 3: K-mer Classification

| Metric | Value |
|--------|-------|
| Average Jaccard similarity | 0.803 |
| Status | MECHANISTICALLY EXPLAINED |

The high Jaccard similarity (clades share 80% of k-mers) means classification relies on rare differentiating k-mers, which is still valid but less dramatic than "completely distinct signatures."

---

## Sequence Selection

**Source:** 270 DENV-4 genomes from NCBI RefSeq
**Selection criteria:** Complete genomes only

This is a reasonable and reproducible selection - RefSeq contains curated reference sequences.

**Potential concern:** RefSeq may under-represent recent variants or specific geographic regions. This is documented in the package.

---

## Recommendations

### For Emails

The Arbovirus package has **already been skeptically validated** and claims are appropriately nuanced:

1. **DO NOT** claim "orthogonal metrics" (ρ=-0.11, not orthogonal)
2. **CAN claim** pan-DENV-4 primer infeasibility (confirmed)
3. **CAN claim** k-mer clade classification (mechanistically explained)
4. **CAN claim** DENV-4 cryptic diversity (documented 71.7% identity)

### No Code Fixes Needed

Unlike the DDG and AMP packages, the Arbovirus package:
- Has no ML training with scaler leakage
- Already has skeptical validation
- Honestly reports claim status

---

## Comparison with ML Packages

| Package | Type | Scaler Leakage | Skeptical Validation |
|---------|------|:--------------:|:--------------------:|
| protein_stability_ddg | ML (Ridge) | FIXED | Now has ablation |
| antimicrobial_peptides | ML (GBR/VAE) | FIXED | Bootstrap exists |
| **arbovirus_surveillance** | Computational | N/A | **Already complete** |

---

*This package is ready for outreach pending minor claim adjustments (orthogonality clarification).*
