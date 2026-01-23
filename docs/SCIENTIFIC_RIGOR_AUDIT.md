# Scientific Rigor Audit Report

**Generated:** 2026-01-23 (REVISED)
**Auditor:** Claude Opus 4.5
**Scope:** deliverables/partners/, research/codon-encoder/
**Method:** Code review + validation script execution

---

## Executive Summary

Comprehensive audit with **actual execution of validation scripts**. Found **11 confirmed issues** (5 CRITICAL, 3 HIGH, 3 MEDIUM) including major comparison biases and broken validations missed in initial review.

**Overall Assessment:** The codebase has significant scientific rigor problems, particularly around **misleading literature comparisons** and **broken validation pipelines**. Jose Colbes package has honest internal documentation but misleading comparison tables. Carlos Brizuela validation is completely non-functional.

---

## CRITICAL SEVERITY

### Issue C1: Misleading Literature Comparison (Jose Colbes)
**Files:** `jose_colbes/validation/bootstrap_test.py:209-226`, `scientific_validation_report.py:296-305`
**Status:** ❌ CRITICAL BIAS

The comparison tables claim:
```
| **Our Method (LOO)**      | **0.58**   | **Sequence** |
| ESM-1v                    | 0.51     | Sequence   |
✓ Our method OUTPERFORMS ESM-1v (0.51)
```

**PROBLEM:** Literature methods are benchmarked on **N=669** (full S669). Jose Colbes uses **N=52** (curated subset). This is **apples-to-oranges**.

**Honest Assessment (from ValidatedDDGPredictor.py lines 6-16):**
> "Literature methods (ESM-1v 0.51, FoldX 0.48, etc.) are benchmarked on N=669. Our N=52 result is NOT directly comparable. On full N=669, we achieve ρ=0.37-0.40, which does NOT outperform these methods."

**Impact:** Published claims would be scientifically invalid. The validation scripts contradict the honest documentation in the predictor class.

**Fix:** Remove or caveat comparison tables. State clearly that N=52 results cannot be compared to N=669 benchmarks.

---

### Issue C2: Carlos Brizuela Validation BROKEN
**File:** `carlos_brizuela/validation/bootstrap_test.py`
**Status:** ❌ SCRIPT FAILS

```
ValueError: X has 37 features, but StandardScaler is expecting 32 features as input.
```

**Verified by execution:**
```bash
$ python3 deliverables/partners/carlos_brizuela/validation/bootstrap_test.py --verbose
FAILED (feature dimension mismatch)
```

**Impact:** Cannot validate any claims about AMP activity prediction. Package is NOT production ready.

**Root cause:** Model was trained with different features than current data loader produces.

---

### Issue C3: Carlos Brizuela Comprehensive Validation BROKEN
**File:** `carlos_brizuela/validation/comprehensive_validation.py`
**Status:** ❌ SCRIPT FAILS

```bash
$ python3 comprehensive_validation.py
Validating general... FAILED
```

**Impact:** No functioning validation for Carlos Brizuela package.

---

### Issue C4: Inconsistent Metrics (Jose Colbes)
**Files:** `bootstrap_test.py` vs `scientific_validation_report.py`
**Status:** ⚠️ INCONSISTENT

Two scripts in same package report different Spearman correlations:

| Script | Spearman ρ | Method |
|--------|------------|--------|
| bootstrap_test.py | 0.581 | LOO CV with Ridge(α=100) on raw features |
| scientific_validation_report.py | 0.521 | ValidatedDDGPredictor with hardcoded coefficients |

**Problem:** The predictor class uses pre-trained coefficients, not actual LOO predictions. This creates confusion about what the "validated" performance actually is.

---

### Issue C5: Documented vs Claimed Performance
**File:** `jose_colbes/src/validated_ddg_predictor.py`
**Status:** ⚠️ HONEST BUT INCONSISTENT WITH CLAIMS

The ValidatedDDGPredictor.py (lines 6-16) honestly states:
- N=52 subset: ρ=0.60
- N=669 full: ρ=0.37-0.40

But `get_performance_metrics()` (lines 353-379) returns a comparison dict that still implies outperformance:
```python
"comparison": {
    "Rosetta ddg_monomer": 0.69,
    "TrainableCodonEncoder (this)": 0.60,  # N=52, not comparable!
    "ESM-1v": 0.51,  # N=669
}
```

---

## HIGH SEVERITY

### Issue H1: Missing Random Seeds (carlos_brizuela) - FIXED
**File:** `carlos_brizuela/validation/bootstrap_test.py:88`
**Status:** ✅ FIXED (seed added)

### Issue H2: Improper Random Baseline (carlos_brizuela) - FIXED
**File:** `carlos_brizuela/validation/bootstrap_test.py:166-178`
**Status:** ✅ FIXED (1000 perms, proper p-value)

### Issue H3: Missing Random Seeds (train_codon_encoder.py) - FIXED
**File:** `research/codon-encoder/training/train_codon_encoder.py:37-42`
**Status:** ✅ FIXED (numpy + torch seeds added)

---

## MEDIUM SEVERITY

### Issue M1: Nested CV Structure
**File:** `train_codon_encoder.py:138-150`
**Status:** ACCEPTABLE - LOO is most conservative method

### Issue M2: Multiple Testing Without Correction
**File:** `ddg_benchmark.py:516-542`
**Status:** ACCEPTABLE - Results labeled exploratory

### Issue M3: Bootstrap CI Method
**File:** `carlos_brizuela/validation/bootstrap_test.py:95-98`
**Status:** ACCEPTABLE - Percentile method widely used

---

## Package-by-Package Assessment (REVISED)

### Jose Colbes (DDG Prediction)
**Status:** ⚠️ PARTIALLY PASSES - Needs comparison table fixes

**GOOD:**
- Proper LOO CV implementation
- Random seeds set
- Bootstrap CIs and permutation tests
- ValidatedDDGPredictor.py has honest documentation

**BAD:**
- Comparison tables make misleading claims (N=52 vs N=669)
- Two validation scripts give inconsistent results (0.581 vs 0.521)
- get_performance_metrics() implies false outperformance

**HONEST PERFORMANCE:**
- N=52 curated subset: Spearman ρ = 0.58 ± 0.11
- N=669 full dataset: Spearman ρ = 0.37-0.40
- Does NOT outperform ESM-1v, FoldX on comparable data

---

### Carlos Brizuela (AMP Prediction)
**Status:** ❌ FAILS AUDIT - Validation broken

**CRITICAL:**
- bootstrap_test.py crashes with feature mismatch
- comprehensive_validation.py fails
- Cannot verify ANY claims

**Blocked until:**
1. Fix feature dimension mismatch (model expects 32, data has 37)
2. Retrain model or update data loader
3. Verify at least one validation script runs

---

### Alejandra Rojas (Arbovirus Primers)
**Status:** ✅ PASSES AUDIT

**GOOD:**
- Skeptical validation runs successfully
- Reports "PARTIALLY CONFIRMED" (appropriately cautious)
- Investigates mechanistic hypotheses
- Alternative hypotheses test shows scientific skepticism

**Verified by execution:**
```bash
$ python3 skeptical_validation.py
CLAIM 1: METRIC ORTHOGONALITY VALIDATION
  Raw (n=10353): Spearman ρ = -0.1121
INTERPRETATION: PARTIALLY CONFIRMED: Weak correlation exists
```

---

### Research/codon-encoder
**Status:** ✅ PASSES (after H3 fix)

- Random seeds now set
- LOO validation appropriate for n=52
- Train/test separation maintained

---

## Verification Evidence

### Jose Colbes - Runs but with misleading output
```bash
$ python3 jose_colbes/validation/bootstrap_test.py
Spearman rho: 0.5810 (p = 6.30e-06)
95% CI: [0.337, 0.768]
✓ Our method OUTPERFORMS ESM-1v (0.51)  # <-- MISLEADING
```

### Carlos Brizuela - FAILS
```bash
$ python3 carlos_brizuela/validation/bootstrap_test.py
ValueError: X has 37 features, but StandardScaler is expecting 32 features
```

### Alejandra Rojas - Runs with honest output
```bash
$ python3 alejandra_rojas/research/validation/skeptical_validation.py
INTERPRETATION: PARTIALLY CONFIRMED: Weak correlation exists
```

### train_codon_encoder.py - Runs
```bash
$ python3 train_codon_encoder.py --epochs 10
Leave-One-Out Metrics (HONEST):
  Spearman r: 0.5796
```

---

## Recommendations

### Immediate (Before Any Publication)

1. **C1: Fix comparison tables** in Jose Colbes validation scripts
   - Remove claims of outperformance over N=669 benchmarks
   - State clearly: "N=52 results not directly comparable to literature"

2. **C2-C3: Fix Carlos Brizuela validation**
   - Debug feature dimension mismatch
   - Ensure at least one validation script runs

3. **C4: Reconcile inconsistent metrics**
   - Clarify which Spearman value (0.52 or 0.58) is authoritative
   - Document the difference in methods

### Short-Term

4. Update `get_performance_metrics()` to not imply false comparisons
5. Add N=669 validation results to Jose Colbes for fair comparison
6. Create unified validation runner for all packages

---

## Conclusion (REVISED)

The codebase has **significant scientific rigor issues** that were missed in initial review because validations were not actually executed.

| Package | Status | Blocking Issues |
|---------|--------|-----------------|
| Jose Colbes | ⚠️ PARTIAL | Misleading comparisons |
| Carlos Brizuela | ❌ FAIL | Validation broken |
| Alejandra Rojas | ✅ PASS | None |
| Research/codon-encoder | ✅ PASS | None (after fixes) |

**Risk Assessment:** MEDIUM-HIGH. Jose Colbes makes scientifically invalid claims in validation outputs. Carlos Brizuela cannot be validated at all.

**Key Lesson:** Always run validation scripts, don't just read them.
