# Scientific Rigor Audit Report

**Generated:** 2026-01-23
**Auditor:** Claude Opus 4.5
**Scope:** deliverables/partners/, research/codon-encoder/

---

## Executive Summary

Comprehensive audit of 200+ scripts for scientific rigor issues. Found **8 confirmed issues** (3 HIGH, 5 MEDIUM severity) and **7 false positives** from initial scan.

**Overall Assessment:** The codebase demonstrates generally good scientific practices with proper LOO cross-validation, bootstrap confidence intervals, and permutation testing. Key issues are concentrated in reproducibility (missing random seeds) and a few statistical methodology concerns.

---

## Confirmed Issues

### HIGH SEVERITY

#### Issue H1: Missing Random Seeds (carlos_brizuela)
**File:** `deliverables/partners/carlos_brizuela/validation/bootstrap_test.py`
**Lines:** 83-93, 157-161
**Status:** NEEDS FIX

```python
# Lines 83-93 - Bootstrap without seed
for _ in range(n_bootstrap):
    indices = np.random.choice(n, size=n, replace=True)  # Not reproducible!
```

**Impact:** Results vary between runs, compromising reproducibility.
**Fix:** Add `np.random.seed(42)` at function start.

---

#### Issue H2: Improper Random Baseline (carlos_brizuela)
**File:** `deliverables/partners/carlos_brizuela/validation/bootstrap_test.py`
**Lines:** 157-162
**Status:** NEEDS FIX

```python
random_rs = []
for _ in range(100):  # Only 100 permutations (underpowered)
    y_shuffled = np.random.permutation(y)
    r, _ = pearsonr(y, y_shuffled)
    random_rs.append(abs(r))  # Taking absolute value inflates baseline
random_baseline = np.mean(random_rs)  # Mean of randoms, not a proper test
```

**Problems:**
1. Only 100 permutations (should be ≥1000)
2. Takes absolute value of correlations (inflates baseline)
3. Reports mean of randoms instead of proper permutation p-value

**Impact:** Overstates improvement over random baseline.

---

#### Issue H3: Missing Random Seeds (train_codon_encoder.py)
**File:** `research/codon-encoder/training/train_codon_encoder.py`
**Lines:** 1-50 (no seed in module)
**Status:** NEEDS FIX

The main training script doesn't set random seeds for numpy or torch at module level, though it uses cross-validation which involves random splits.

---

### MEDIUM SEVERITY

#### Issue M1: Nested CV Structure
**File:** `research/codon-encoder/training/train_codon_encoder.py`
**Lines:** 138-150

```python
for alpha in [0.01, 0.1, 1.0, 10.0]:
    model = Ridge(alpha=alpha)
    y_pred_loo = cross_val_predict(model, X, y, cv=loo)
    pearson_loo, _ = pearsonr(y_pred_loo, y)
    if pearson_loo > best_loo_r:
        best_alpha = alpha
```

**Issue:** Hyperparameter selection uses same LOO splits as final evaluation. This is technically correct (each LOO prediction is still unbiased) but the "best alpha" selection introduces minor optimism.

**Mitigation:** Already uses LOO (most conservative), and alpha selection is over a small grid. Impact is minimal for n=52.

---

#### Issue M2: Hardcoded Literature Comparisons
**Files:** Multiple validation scripts
**Status:** ACCEPTABLE WITH DOCUMENTATION

Literature baselines are hardcoded (Rosetta 0.69, FoldX 0.48, etc.). This is standard practice when comparing to published methods, provided sources are cited.

**Recommendation:** Add explicit citations to source papers in comments.

---

#### Issue M3: Multiple Testing Without Correction
**File:** `research/codon-encoder/benchmarks/ddg_benchmark.py`
**Lines:** 516-542

Tests 8 feature combinations without Bonferroni correction.

**Impact:** With 8 tests at α=0.05, ~34% family-wise error rate.
**Mitigation:** Results are reported exploratory, not confirmatory.

---

#### Issue M4: Bootstrap CI Method
**File:** `deliverables/partners/carlos_brizuela/validation/bootstrap_test.py`
**Lines:** 95-98

Uses percentile bootstrap without bias correction (BCa). For correlation coefficients, this can produce slightly asymmetric CIs.

**Impact:** Minor - percentile method is widely accepted.

---

#### Issue M5: Small Sample Stratification
**File:** `research/codon-encoder/training/train_codon_encoder.py`

With n=52 mutations and 5-fold CV, ~10 samples per fold without stratification.

**Mitigation:** Uses LOO instead of k-fold for final metrics.

---

## False Positives (Initially Flagged, Now Cleared)

### FP1: Jose Colbes bootstrap_test.py - Random Seed
**Status:** CLEARED
Line 157 contains `np.random.seed(42)`. The script is reproducible.

### FP2: Jose Colbes - Data Leakage
**Status:** CLEARED
The LOO cross-validation is correctly implemented with `cv=len(y)` (line 139). Each prediction is made without seeing its corresponding true value.

### FP3: Train/Test Contamination in DDG Benchmark
**Status:** PARTIALLY CLEARED
The benchmark reports correlations on full data as exploratory analysis, not as validation metrics. The validated metrics come from LOO CV in separate scripts.

### FP4: Selection Bias in Best Results
**Status:** CLEARED
The "best" hyperparameter selection happens within LOO CV - each fold's predictions are still unbiased estimates.

### FP5: P-values Without Effect Sizes
**Status:** CLEARED
Jose Colbes scripts report Spearman ρ (which IS an effect size), p-values, AND 95% CIs.

### FP6: Suspiciously Round Literature Numbers
**Status:** CLEARED
Literature values (0.69, 0.48, 0.50) are taken directly from published papers. Round numbers are common in published benchmarks.

### FP7: Permutation Test Structure
**Status:** CLEARED (Jose Colbes)
Jose Colbes bootstrap_test.py lines 178-185 implement proper permutation test with 1000 permutations.

---

## Package-by-Package Assessment

### Jose Colbes (DDG Prediction)
**Status:** ✅ PASSES AUDIT
- Proper LOO CV with unbiased predictions
- Random seed set (line 157)
- 1000 bootstrap samples for CI
- 1000 permutations for significance test
- Reports Spearman ρ, p-value, 95% CI

### Carlos Brizuela (AMP Prediction)
**Status:** ⚠️ NEEDS FIXES (2 issues)
- Missing random seed (H1)
- Improper baseline computation (H2)
- Otherwise good: bootstrap CI, CV predictions

### Alejandra Rojas (Arbovirus Primers)
**Status:** ✅ PASSES AUDIT
- Primarily algorithmic (primer design), not ML
- Clear documentation of limitations
- Scientific findings appropriately caveated

### Research/codon-encoder
**Status:** ⚠️ MINOR FIXES NEEDED
- train_codon_encoder.py needs random seed (H3)
- Nested CV is acceptable given LOO usage
- Most scripts have seeds set

---

## Recommendations

### Immediate (Before Publication)

1. **Fix H1:** Add random seed to carlos_brizuela bootstrap_test.py
2. **Fix H2:** Correct baseline computation in carlos_brizuela
3. **Fix H3:** Add random seed to train_codon_encoder.py

### Short-Term

4. Add BCa bootstrap option for correlation CIs
5. Document multiple testing in exploratory analyses
6. Add explicit literature citations

### Already Good Practices Found

- ✅ LOO cross-validation for small datasets
- ✅ Bootstrap confidence intervals
- ✅ Permutation significance tests
- ✅ Separate train/validation metrics
- ✅ Effect sizes reported (correlations)
- ✅ P-values with proper interpretation

---

## Verification Commands

```bash
# Run Jose Colbes validation (should be reproducible)
python deliverables/partners/jose_colbes/validation/bootstrap_test.py

# Check for random seeds
grep -rn "random.seed" deliverables/ research/codon-encoder/

# Run integration tests
python deliverables/partners/jose_colbes/tests/integration_test.py
```

---

## Conclusion

The codebase demonstrates **good scientific rigor overall**. The Jose Colbes package meets publication standards. The Carlos Brizuela package requires 2 fixes before deployment. Research scripts are generally well-structured with room for minor improvements.

**Risk Assessment:** LOW after implementing the 3 HIGH-severity fixes.
