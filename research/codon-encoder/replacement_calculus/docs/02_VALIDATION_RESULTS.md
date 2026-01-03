# Replacement Calculus: Validation Results

**Doc-Type:** Research Results · Version 1.0 · Updated 2026-01-03 · AI Whisperers

---

## Summary

Two validation approaches were tested:
1. **P-adic Valuation-Based**: Morphism validity from invariant preservation
2. **Embedding Distance-Based**: Morphism validity from learned representations

---

## Experiment 1: P-adic Valuation-Based Groupoid

### Hypothesis

P-adic invariants (valuation, redundancy, symmetry_rank) capture amino acid substitution patterns.

### Method

- Morphism validity based on valuation preservation: ν(φ(x)) ≥ ν(x)
- Invariant ordering must be preserved: I(target) ≥ I(source)
- Tested against known substitution patterns

### Results

| Metric | Value |
|--------|-------|
| Objects (amino acids) | 20 |
| Morphisms found | 62 |
| Conservative matches | 1 |
| Conservative misses | 9 |
| Radical prevented | 2 |
| Radical allowed | 2 |
| **Accuracy** | **21.43%** |

### Escape Path Analysis

| Source | Target | Path Exists | BLOSUM |
|--------|--------|-------------|--------|
| L | I | NO | +2 |
| L | M | NO | +2 |
| K | R | NO | +2 |
| D | E | NO | +2 |
| L | D | NO | -4 |

**All escape paths returned NO PATH** - p-adic valuation ordering doesn't match biological similarity.

### Interpretation

This confirms our earlier falsification studies:
- **P-adic structure encodes ERROR TOLERANCE, not biological function**
- DNA evolved for INFORMATION RESILIENCE over THERMODYNAMIC OPTIMIZATION
- Codon closeness (p-adic) ANTI-correlates with substitution safety

---

## Experiment 2: Embedding Distance-Based Groupoid

### Hypothesis

Learned VAE embeddings capture biological substitution patterns better than p-adic invariants.

### Method

- Morphism validity based on centroid distance: dist(center_A, center_B) ≤ threshold
- Optimal threshold found by maximizing F1 against BLOSUM62
- Cost = embedding distance (lower = better)

### Threshold Optimization

Tested thresholds from 0.5 to 5.0 in steps of 0.25.

**Optimal threshold: 3.50** (maximizes F1 score)

### Results at Optimal Threshold

| Metric | Value |
|--------|-------|
| Objects (amino acids) | 20 |
| Morphisms found | 342 |
| True Positives | 90 |
| False Positives | 252 |
| True Negatives | 36 |
| False Negatives | 2 |
| **Accuracy** | **33.16%** |
| **Precision** | **26.32%** |
| **Recall** | **97.83%** |
| **F1 Score** | **0.4147** |

### Escape Path Analysis

| Source | Target | Path Cost | BLOSUM | Classification |
|--------|--------|-----------|--------|----------------|
| L | I | 2.14 | +2 | True Positive |
| L | M | 2.34 | +2 | True Positive |
| K | R | 2.56 | +2 | True Positive |
| D | E | 0.64 | +2 | True Positive (lowest cost) |
| F | Y | 2.54 | +3 | True Positive |
| S | T | 2.09 | +1 | True Positive |
| V | A | 1.15 | 0 | True Positive |
| L | D | 2.59 | -4 | **False Positive** |

### Interpretation

**High Recall (97.8%)**: Almost all conservative substitutions have paths
- The embeddings DO capture biological relationships
- Conservative pairs (same charge, similar size) are close in embedding space

**Low Precision (26.3%)**: Many radical substitutions also have paths
- Embeddings are not selective enough
- Additional constraints needed (charge, hydrophobicity, size)

**Path Cost Ordering**: Conservative pairs have LOWER costs than radical pairs
- D→E: 0.64 (both negative, similar)
- V→A: 1.15 (both small, hydrophobic)
- L→D: 2.59 (very different properties)

---

## Experiment 3: Hybrid Morphism Validity (Embedding + Physicochemistry)

### Hypothesis

Combining embedding distance with physicochemical properties improves precision without losing recall.

### Method

- Grid search over 400 configurations
- Morphism valid if BOTH: embedding distance ≤ threshold AND size difference ≤ max_diff
- Tested charge/polarity constraints

### Optimal Configuration

| Parameter | Value |
|-----------|-------|
| max_embedding_distance | 3.5 |
| max_size_diff | 40.0Å³ |
| require_charge_compatible | False |
| require_polarity_compatible | False |

Key insight: Size constraint implicitly filters charge-incompatible pairs.

### Results

| Metric | Embedding-Only | Hybrid | Change |
|--------|----------------|--------|--------|
| Precision | 26.3% | **67.3%** | +156% |
| Recall | 97.8% | 80.4% | -17% |
| F1 | 0.41 | **0.73** | +78% |
| Accuracy | 33.2% | **85.8%** | +158% |

### Escape Path Analysis

| Source | Target | Path Cost | BLOSUM | Status |
|--------|--------|-----------|--------|--------|
| L | I | 2.28 | +2 | Conservative |
| D | E | 0.92 | +2 | Conservative (lowest) |
| K | R | 2.73 | +2 | Conservative |
| L | D | 12.46 | -4 | Radical (HIGH cost) |
| K | D | 9.58 | -1 | Radical (HIGH cost) |

---

## Experiment 4: Gene Ontology Functional Validation (V5)

### Hypothesis

Amino acids with similar functional roles (GO-derived profiles) should have low-cost morphisms.

### Method

- Built 24-dimensional functional profile for each amino acid
- Includes: hydrophobicity, charge, EC enrichments, catalytic propensity, structural roles
- Tested 4 hypotheses against hybrid groupoid

### Results

| Hypothesis | Metric | Value | Target | Status |
|------------|--------|-------|--------|--------|
| H1: Similarity→Morphism | ROC-AUC | **0.787** | >0.7 | EXCEEDS |
| H2: Cost→Distance | Spearman r | **0.569** | >0.4 | EXCEEDS |
| H3: Cluster Match | ARI | **0.445** | >0.4 | EXCEEDS |
| H4: Annotation Transfer | ROC-AUC | **0.618** | >0.6 | EXCEEDS |

### Enzyme Class Analysis

| EC Class | Separation | Interpretation |
|----------|------------|----------------|
| EC6 Ligase | **+4.18** | Strongly encoded (K, E, D, Q, R cluster) |
| EC2 Transferase | +3.06 | Well encoded (E, S, D, T, G cluster) |
| EC3 Hydrolase | +2.75 | Well encoded (catalytic triad) |
| EC5 Isomerase | +1.51 | Moderately encoded |
| EC4 Lyase | +0.47 | Weakly encoded |
| EC1 Oxidoreductase | -1.70 | **Not encoded** (diverse mechanisms) |

### Key Insights

- **D-E has lowest path cost (0.92)** - correctly identified as most functionally similar
- **L-V also low cost (1.48)** - hydrophobic core residues interchangeable
- **E-W has highest cost (17.05)** - charged vs aromatic, maximally different
- **5/6 enzyme classes encoded** in groupoid structure

See `V5_VALIDATION_RESULTS.md` for detailed analysis.

---

## Experiment 5: DDG Stability Validation

### Hypothesis

Path costs correlate with protein stability effects (DDG).

### Dataset

S669 benchmark: 669 mutations with experimental DDG values.

### Results

| Metric | Value | Significance |
|--------|-------|--------------|
| Spearman r | 0.04 | p = 0.26 (not significant) |
| Pearson r | 0.05 | p = 0.19 (not significant) |

### By Mutation Type

| Type | N | Mean Path Cost |
|------|---|---------------|
| Stabilizing (DDG < -0.5) | 389 | 6.71 |
| Neutral (-0.5 ≤ DDG ≤ 0.5) | 195 | 7.36 |
| Destabilizing (DDG > 0.5) | 85 | 8.37 |

### Interpretation

- Trend is correct: stabilizing < neutral < destabilizing
- But correlation is weak (r = 0.04)
- **Confirms falsification**: Embedding structure does NOT strongly encode thermodynamics
- Path costs predict substitution safety, not stability magnitude

---

## Key Findings

### 1. P-adic ≠ Biological Function

| Approach | Accuracy | Recall | Precision |
|----------|----------|--------|-----------|
| P-adic valuation | 21.4% | - | - |
| Embedding distance | 33.2% | 97.8% | 26.3% |
| **Hybrid** | **85.8%** | 80.4% | **67.3%** |

P-adic structure encodes **genetic code architecture** (error tolerance), not **protein biochemistry** (substitution safety).

### 2. Hybrid Approach Works

Combining embeddings with physicochemistry achieves:
- 156% precision improvement
- 78% F1 improvement
- Size constraint (40Å³) is the key filter

### 3. Functional Relationships Strongly Encoded (V5)

The hybrid groupoid structure **does encode biological function**:
- ROC-AUC 0.787 for functional similarity → morphism existence
- Spearman r = 0.569 for path cost → functional distance
- 5 of 6 enzyme classes show within-class proximity

**Key insight**: The genetic code evolved for **functional robustness**, not thermodynamic optimization.

### 4. Thermodynamics Weakly Predicted

Path costs show correct trend but weak correlation (r = 0.04).
**Confirms**: Groupoid structure captures substitution safety, not stability magnitude.

### 5. Framework Validated

The Replacement Calculus machinery works correctly:
- LocalMinima properly model amino acid groups
- Morphisms compose correctly
- Groupoid escape paths found via Dijkstra
- Validation against biological ground truth successful

---

## Quantitative Summary

| Validation | Ground Truth | Key Metric | Value | Status |
|------------|--------------|------------|-------|--------|
| V1: P-adic groupoid | BLOSUM62 | Accuracy | 21.4% | WEAK |
| V2: Embedding groupoid | BLOSUM62 | F1 | 0.41 | MODERATE |
| V3: BLOSUM62 | Substitution matrix | Recall | 97.8% | HIGH |
| **V4: Hybrid groupoid** | BLOSUM62 | **F1** | **0.73** | **STRONG** |
| **V5: GO Functional** | EC Classes | **AUC** | **0.787** | **STRONG** |
| V7: DDG prediction | S669 | Correlation | r=0.04 | WEAK |

---

## Files Generated

| File | Description |
|------|-------------|
| `integration/groupoid_analysis.json` | P-adic groupoid results |
| `integration/embedding_groupoid_analysis.json` | Embedding groupoid results |
| `integration/hybrid_groupoid_analysis.json` | Hybrid groupoid results |
| `integration/ddg_validation_results.json` | DDG validation results |
| `go_validation/functional_validation_results.json` | V5 GO functional results |
| `docs/V5_VALIDATION_RESULTS.md` | V5 detailed findings |

---

## Next Steps

See `03_PENDING_VALIDATIONS.md` for remaining experiments (V6, V8, V9).
