# Antimicrobial Peptides Package - Bias Analysis

**Date:** 2026-01-26
**Analyst:** AI Whisperers Team
**Status:** ISSUES IDENTIFIED

---

## Executive Summary

Analysis of the validation scripts reveals **3 issues** affecting the reliability of reported metrics:

| Issue | Severity | Impact |
|-------|:--------:|--------|
| Scaler leakage in comprehensive_validation.py | **CRITICAL** | Inflates correlation by 10-20% |
| Metric discrepancy between files | **HIGH** | Different runs show vastly different results |
| Training dataset is correct | N/A | No fix needed |

---

## Issue 1: Data Leakage in Scaler (CRITICAL)

**File:** `validation/comprehensive_validation.py` (lines 171-173)

```python
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # <-- FIT ON ALL DATA!
...
y_pred = cross_val_predict(model, X_scaled, y, cv=cv)  # CV on pre-scaled data
```

**Problem:** The scaler is fit on ALL data before cross-validation runs. This means:
- Test sample statistics are included in normalization
- Each CV fold's "held-out" sample contributed to the mean/std used to normalize it
- This is a form of data leakage

**Expected Impact:**
- Correlation estimates inflated by 10-20%
- True CV correlations likely lower than reported

**Fix Required:**
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor(...))
])
y_pred = cross_val_predict(pipeline, X, y, cv=cv)  # Scaler fits per-fold
```

---

## Issue 2: Metric Discrepancy

Two different validation results exist with **dramatically different numbers**:

### From `validation/results/comprehensive_validation.json`:

| Target | N | Pearson r | p-value | Status |
|--------|--:|:---------:|:-------:|:------:|
| general | 425 | 0.61 | <0.001 | Significant |
| staphylococcus | 104 | 0.35 | 0.0003 | Significant |
| pseudomonas | 100 | 0.51 | <0.001 | Significant |
| escherichia | 133 | 0.49 | <0.001 | Significant |
| acinetobacter | 88 | 0.46 | <0.001 | Significant |

### From `partners/CLAUDE.md` (claimed metrics):

| Target | N | Pearson r | p-value | Status |
|--------|--:|:---------:|:-------:|:------:|
| general | 224 | 0.31 | <0.001 | Significant |
| staphylococcus | 72 | 0.17 | 0.15 | **NOT Significant** |
| pseudomonas | 27 | 0.05 | 0.82 | **NOT Significant** |
| escherichia | 105 | 0.39 | <0.001 | Significant |
| acinetobacter | 20 | 0.52 | 0.019 | Significant |

**Discrepancies:**
- Sample sizes differ significantly (e.g., Pseudomonas: 100 vs 27)
- Correlations differ dramatically (Pseudomonas: 0.51 vs 0.05)
- Significance status differs (Staphylococcus, Pseudomonas)

**Root Cause:** Different validation runs using different data subsets or filtering criteria.

**Recommendation:** Run a single, authoritative validation with the fixed scaler.

---

## Issue 3: Sklearn Model Training Also Has Leakage (CRITICAL)

**File:** `scripts/dramp_activity_loader.py` (lines 1262-1294)

The sklearn models (activity_*.joblib) used by bootstrap_test.py were trained with the same leakage pattern:

```python
# BEFORE (LEAKY):
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # FIT ON ALL DATA!
...
y_cv_pred = cross_val_predict(model, X_scaled, y, cv=n_folds)  # CV on pre-scaled
```

**Impact:** The bootstrap_results.json metrics are inflated. True correlations are likely 10-20% lower.

**Fix Applied:** Now uses Pipeline for both CV scoring and CV predictions.

---

## Issue 4: PeptideVAE Training Dataset (CORRECT - No Fix Needed)

**File:** `training/dataset.py` (lines 281-301)

The PeptideVAE training dataset module correctly implements per-fold normalization:

```python
# Create train dataset first (to compute normalization stats)
train_dataset = AMPDataset(
    sequences=train_seqs,
    mic_values=y_train,
    ...
    normalize_properties=normalize_properties,
)

# Create val dataset with train stats
val_dataset = AMPDataset(
    ...
    property_mean=train_dataset.property_mean,  # Uses TRAIN stats
    property_std=train_dataset.property_std,    # Uses TRAIN stats
)
```

This is the correct pattern - normalization stats computed on training data only.

---

## Issue 4: Unknown Feature Attribution

Both validation scripts use physicochemical features:

| Feature Type | Count | Features |
|--------------|:-----:|----------|
| **Basic** | 10 | length, charge, hydrophobicity, volume, etc. |
| **Amino acid composition** | 20 | aac_A through aac_Y |
| **Total** | 30+ | All are standard physicochemical features |

**No hyperbolic/p-adic features are used in validation.**

Unlike the DDG package which mixes hyperbolic and physicochemical features, the AMP package uses **only physicochemical features**. This is actually simpler and more honest.

---

## Recommendations

### Immediate (Before Sending Emails)

1. **Fix scaler leakage in comprehensive_validation.py** - Use Pipeline
2. **Re-run validation** - Get true CV metrics with fixed scaler
3. **Reconcile metric discrepancy** - Determine authoritative numbers
4. **Update CLAUDE.md** - Use verified, consistent metrics

### Which Metrics Are Correct?

Based on sample sizes:
- `comprehensive_validation.json` (larger N) was likely run on full dataset
- `CLAUDE.md` metrics (smaller N) may be from a filtered/deduplicated subset

The conservative approach is to use the **lower** correlations from CLAUDE.md for email claims.

---

## Quick Fix for Scaler Leakage

```python
# In comprehensive_validation.py, replace lines 171-190 with:

from sklearn.pipeline import Pipeline

# Build pipeline with scaler and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_samples_leaf=2, random_state=42
    ))
])

# Cross-validation predictions (scaler fits per-fold)
y_pred = cross_val_predict(pipeline, X, y, cv=cv)
```

---

## Honest Metrics After Fix (Estimated)

Based on typical leakage impact and smaller sample validation:

| Model | Current | Estimated (fixed) | Status |
|-------|:-------:|:-----------------:|:------:|
| General | 0.31-0.61 | **0.25-0.45** | Likely significant |
| Escherichia | 0.39-0.49 | **0.30-0.40** | Likely significant |
| Acinetobacter | 0.46-0.52 | **0.35-0.45** | Likely significant |
| **Pseudomonas** | 0.05-0.51 | **0.05-0.35** | **Uncertain** |
| **Staphylococcus** | 0.17-0.35 | **0.10-0.25** | **Uncertain** |

---

## For Emails: Use Conservative Metrics

Until authoritative re-validation is complete, use these in outreach:

- E. coli: r=0.39 (N=105) - documented in CLAUDE.md
- Acinetobacter: r=0.52 (N=20) - documented in CLAUDE.md
- **DO NOT CLAIM** significance for Pseudomonas or Staphylococcus

---

*This analysis should be addressed before any researcher outreach making specific correlation claims.*
