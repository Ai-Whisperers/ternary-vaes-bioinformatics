# Regime-Conditional Inference

> **Biological signal lives in local submanifolds, not global averages.**

---

## Overview

Regime-Conditional Inference is the principle that predictive signal for biological outcomes often exists only within specific subpopulations defined by measurable features. Global models fail because they average over regimes with fundamentally different behavior.

This principle has been independently discovered across three partner investigations in the Ternary VAE project:
- **Brizuela (AMP):** Pathogen specificity exists within peptide clusters, not globally
- **Rojas (Arbovirus):** Cryptic diversity reveals subclade-specific conservation
- **Colbes (DDG):** Buried vs surface mutations require different prediction strategies

---

## Theoretical Foundation

### The Problem with Global Models

Global models assume a single function applies to all samples:

```
E[Y|X] = f(X)
```

This fails when the true relationship is regime-dependent:

```
E[Y|X, R=r₁] ≠ E[Y|X, R=r₂]
```

Where R is the regime indicator (cluster, clade, position type).

### Why Averaging Destroys Signal

When regimes have opposite effects, averaging produces zero signal:

```
Regime A: Gram- pathogens → high MIC for short peptides
Regime B: Gram+ pathogens → low MIC for short peptides
Global average: no correlation between length and MIC
```

The signal exists but cancels out. Only regime-conditional analysis reveals it.

---

## Evidence Across Domains

### AMP Pathogen Specificity (Brizuela) - C3 Theorem

**Finding:** Pathogen-specific MIC signal exists within peptide submanifolds.

| Regime (Cluster) | Length | Pathogens | Signal? | Effect Ratio |
|------------------|--------|:---------:|:-------:|:------------:|
| Clusters 1, 3, 4 | 13-14 AA | 5 | YES | 0.54-0.77 |
| Cluster 0 | 22.6 AA | 3 | NO | 0.18 |
| Cluster 2 | 18.3 AA | 5 | statistical only | 0.48 |

**Key Insight:** Short peptides (13-14 AA) show pathogen separation that long peptides average out.

**Verified:** Signal survives seed-artifact falsification (3/6 seeds show within-seed pathogen separation).

**Reference:** `deliverables/partners/carlos_brizuela/docs/P1_C3_THEOREM_FORMALIZATION.md`

### DENV-4 Primer Design (Rojas) - Cryptic Diversity

**Finding:** Global consensus sequences miss subclade-specific conservation.

| Approach | Coverage | Why |
|----------|:--------:|-----|
| Global consensus | 13.3% | Averages over divergent subclades |
| Clade-specific | 85-100% | Respects local conservation patterns |

**Key Insight:** Hyperbolic variance identifies conserved regions that entropy-based methods miss. These regions are "cryptically diverse" - appearing variable globally but conserved locally.

**Reference:** `deliverables/partners/alejandra_rojas/results/pan_arbovirus_primers/`

### DDG Prediction (Colbes) - Arrow Flip Validation

**Finding:** Buried vs surface mutations require different prediction strategies.

| Position Type | Simple Predictor | Hybrid Predictor | Advantage |
|---------------|:----------------:|:----------------:|:---------:|
| Buried | r = 0.249 | r = 0.689 | +0.440 hybrid |
| Surface | ~equal | ~equal | ~0 |

**Key Insight:** The "arrow flips" at hydrophobicity difference = 3.5 for buried positions. Below threshold, simple sequence-based prediction suffices. Above threshold, structure-aware hybrid required.

**Reference:** `research/codon-encoder/replacement_calculus/docs/V5_EXPERIMENTAL_VALIDATION.md`

---

## The Unified Pattern

| Domain | Global Failure | Local Success | Regime Feature |
|--------|----------------|---------------|----------------|
| AMP (Brizuela) | No pathogen separation | Cluster 1,3,4 signal | Peptide length |
| DENV-4 (Rojas) | Consensus misses subclades | Clade-specific works | Phylogenetic clade |
| DDG (Colbes) | Mixed prediction quality | Position-stratified wins | Solvent accessibility |

**Common Thread:** All three domains exhibit signal that:
1. Vanishes under global averaging
2. Emerges within properly defined regimes
3. Uses computable features for regime assignment

---

## Implications for Model Development

### 1. Don't Trust Global Metrics

High average performance may mask regime-specific failure:
- 80% overall accuracy could be 95% in regime A, 50% in regime B
- Report performance per regime, not just aggregate

### 2. Identify Regimes First

Before prediction, classify samples into regimes:
- Clustering (KMeans, hierarchical)
- Domain knowledge (position type, clade, etc.)
- Learned regime indicators

### 3. Build Regime-Conditional Models

Options:
- Separate models per regime
- Single model with regime-conditional heads
- Mixture-of-experts architecture

### 4. Validate Within Regimes

Hold-out tests must respect regime structure:
- Stratified sampling within regimes
- Report per-regime generalization
- Identify regime-specific failure modes

---

## Mathematical Framework

### Mixture Model Formulation

The regime-conditional approach can be formalized as:

```
p(y|x) = Σᵣ p(y|x,r) × p(r|x)
```

Where:
- `p(r|x)` = regime assignment probability
- `p(y|x,r)` = regime-specific prediction

When regimes are deterministically assigned (e.g., cluster membership):

```
p(y|x) = p(y|x, r=assign(x))
```

### Effect Ratio as Regime Quality Metric

For continuous outcomes, the effect ratio measures regime utility:

```
ER = std_between_groups / std_within_groups
```

- ER > 0.5: Regime captures meaningful variation
- ER < 0.5: Regime boundaries are arbitrary

---

## When NOT to Use Regime-Conditional Inference

1. **Truly homogeneous populations:** If biological mechanism is uniform, regime splitting adds noise
2. **Insufficient data:** Small N per regime loses statistical power
3. **Unclear regime boundaries:** Arbitrary clustering can create spurious signal
4. **Regime features unavailable at inference:** If you can't compute the regime, you can't use the model

---

## Related Concepts

- **Cryptic diversity:** Hidden variation below apparent consensus (Rojas)
- **Submanifold learning:** Identifying lower-dimensional structure in data
- **Conditional density estimation:** p(y|x,r) instead of p(y|x)
- **Mixture of experts:** Architecture for regime-conditional prediction
- **Simpson's paradox:** Aggregate trends opposite to within-group trends

---

## References

### Partner Results

| Partner | Document | Key Finding |
|---------|----------|-------------|
| Brizuela | `deliverables/partners/carlos_brizuela/results/validation_batch/P1_CONJECTURE_TESTS.md` | C3 theorem survives falsification |
| Rojas | `deliverables/partners/alejandra_rojas/results/pan_arbovirus_primers/` | Cryptic diversity in DENV-4 |
| Colbes | `research/codon-encoder/replacement_calculus/docs/V5_EXPERIMENTAL_VALIDATION.md` | Arrow flip at hydrophobicity threshold |

### Related Theory Documents

- `p-adic.md` - P-adic valuation and hierarchical structure
- `hyperbolic.md` - Poincare ball embeddings and radial organization
- `synthesis.md` - Historical synthesis of discoveries

---

_Last updated: 2026-01-07_
