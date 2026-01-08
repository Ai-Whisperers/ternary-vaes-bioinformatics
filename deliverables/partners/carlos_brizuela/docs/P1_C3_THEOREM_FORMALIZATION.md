# P1 Conjecture 3: Regime-Conditional Inference Theorem

**Doc-Type:** Theorem Formalization | Version 1.0 | 2026-01-07 | AI Whisperers

---

## Formal Statement

**C3 Theorem (Regime-Conditional Pathogen Specificity):**

For PeptideVAE MIC predictions generated via NSGA-II optimization with DRAMP-filtered candidates:

> Pathogen-specific signal exists within KMeans clusters (k=5) defined by peptide features (length, net_charge, hydrophobicity), specifically clusters 1, 3, 4 (short peptides, 13-14 AA), with effect ratios >0.5 and Kruskal-Wallis p<0.01.

**Scope Limitations:**
- Applies to THIS pipeline: PeptideVAE + NSGA-II + DRAMP filtering
- Applies to THESE pathogens: A. baumannii, S. aureus, P. aeruginosa, Enterobacteriaceae, H. pylori
- Verified against seed-artifact confound via within-seed pathogen separation test

---

## Reproducibility Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Dataset** | 399 candidates, 5 pathogens | From validation_batch/ |
| **Clustering** | KMeans, k=5 | sklearn, random_state=42 |
| **Features** | length, net_charge, hydrophobicity | z-score normalized |
| **Statistical Test** | Kruskal-Wallis H | scipy.stats.kruskal |
| **Significance** | p < 0.01 | Bonferroni-corrected across clusters |
| **Effect Size** | between_std / within_std > 0.5 | Practical significance threshold |
| **Falsification** | Within-seed pathogen separation | 3/6 testable seeds show signal |

---

## Signal Clusters (from P1_C3_results.json)

| Cluster | N | Length | Charge | Hydro | Effect Ratio | p-value | Verdict |
|:-------:|---|--------|--------|-------|:------------:|---------|:-------:|
| **1** | 87 | 13.1 +/- 1.8 | 1.0 +/- 0.6 | -0.22 +/- 0.18 | **0.54** | 0.0005 | **SIGNAL** |
| **3** | 64 | 13.2 +/- 2.0 | 0.2 +/- 0.7 | -0.86 +/- 0.23 | **0.77** | 0.002 | **SIGNAL** |
| **4** | 114 | 13.8 +/- 1.5 | 1.6 +/- 0.4 | 0.28 +/- 0.15 | **0.72** | <0.0001 | **SIGNAL** |
| 0 | 45 | 22.6 +/- 1.8 | 3.6 +/- 0.7 | 0.01 +/- 0.18 | 0.18 | 0.678 | no_signal |
| 2 | 89 | 18.3 +/- 1.6 | 2.3 +/- 0.6 | 0.09 +/- 0.25 | 0.48 | 0.012 | statistical_only |

**Pattern:** Short peptides (13-14 AA) show pathogen separation. Long peptides (18-23 AA) do not.

---

## Falsification Test: Seed-Artifact Elimination

**Hypothesis to Falsify:** C3 signal is an artifact of different seed sequences used per pathogen during NSGA-II optimization, NOT real biological pathogen specificity.

**Method:**
1. Assign each candidate to its closest seed sequence (8 unique seeds identified)
2. Test pathogen separation WITHIN each seed origin group
3. If signal exists within same-seed candidates, it cannot be explained by seed differences

**Results (from P1_C3_falsification.json):**

| Seed Sequence | N | Pathogens | Kruskal H | p-value | Effect Ratio | Verdict |
|---------------|---|:---------:|-----------|---------|:------------:|:-------:|
| GIGKFLHSAKKFGKAFVGEIMNS | 123 | 4 | 16.76 | 0.0008 | 0.56 | **SIGNAL** |
| RLKKTFFKIVKTVKW | 124 | 5 | 23.81 | <0.0001 | 0.63 | **SIGNAL** |
| RLKKTFFKIV | 76 | 4 | 17.31 | 0.0006 | 0.85 | **SIGNAL** |
| KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK | 23 | 4 | 2.22 | 0.528 | 1.27 | no_signal |
| KLAKLAKKLAKLAK | 19 | 3 | 0.75 | 0.385 | 0.88 | no_signal |
| FKCRRWQWRMKKLGAPS | 27 | 3 | 1.07 | 0.300 | 0.75 | no_signal |

**Conclusion:** 3/6 testable seeds show within-seed pathogen separation (p<0.01, ER>0.5). This is NOT possible if signal were purely seed-dependent.

**Verdict: SURVIVES falsification.**

---

## Connection to Rojas Framework (Cryptic Diversity)

The C3 theorem aligns with findings from the Rojas arbovirus investigation:

| Concept | Rojas (DENV-4 Primers) | Brizuela (AMP MIC) |
|---------|------------------------|-------------------|
| **Phenomenon** | Cryptic diversity | Regime-conditional signal |
| **Failure of global** | Consensus primers miss subclades | Global MIC models show no pathogen separation |
| **Success of local** | Clade-specific primers work | Cluster-specific models show separation |
| **Key insight** | Hyperbolic variance reveals conserved regions entropy misses | Effect ratio reveals signal statistical significance misses |
| **Implication** | Design primers per subclade | Train predictors per cluster |

**Unified Principle:** Biological signal lives in LOCAL structure, not global averages.

---

## Implementation: Cluster Assignment at Inference Time

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Trained cluster centers (from this analysis)
CLUSTER_CENTERS = {
    # (length_z, charge_z, hydro_z) - z-scored values
    0: [2.15, 1.89, 0.42],   # Long, charged - NO SIGNAL
    1: [-0.62, -0.71, -0.89], # Short, low charge - SIGNAL
    2: [0.78, 0.32, 0.31],   # Medium - statistical only
    3: [-0.58, -1.21, -2.61], # Short, hydrophobic - SIGNAL
    4: [-0.41, -0.08, 1.12],  # Short, moderate - SIGNAL
}

def assign_cluster(length: int, net_charge: float, hydrophobicity: float,
                   scaler_params: dict) -> int:
    """Assign a novel peptide to its cluster for regime-conditional prediction."""
    # Z-score normalize using training set params
    features = np.array([
        (length - scaler_params['length_mean']) / scaler_params['length_std'],
        (net_charge - scaler_params['charge_mean']) / scaler_params['charge_std'],
        (hydrophobicity - scaler_params['hydro_mean']) / scaler_params['hydro_std'],
    ])

    # Find nearest cluster center
    distances = [np.linalg.norm(features - np.array(center))
                 for center in CLUSTER_CENTERS.values()]
    return np.argmin(distances)

def predict_with_regime(peptide_features: dict, cluster_id: int) -> str:
    """Return prediction confidence based on cluster membership."""
    if cluster_id in [1, 3, 4]:
        return "HIGH_CONFIDENCE: Cluster has pathogen-specific signal"
    elif cluster_id == 2:
        return "MEDIUM_CONFIDENCE: Statistical signal only"
    else:
        return "LOW_CONFIDENCE: No pathogen separation in this regime"
```

---

## R3 Classification (Inference-Time Availability)

| Component | Available at Inference? | Classification |
|-----------|:-----------------------:|----------------|
| Peptide length | YES | Deployable |
| Peptide net charge | YES | Deployable |
| Peptide hydrophobicity | YES | Deployable |
| Cluster assignment | YES (computed) | Deployable |
| Pathogen separation signal | YES (within cluster) | Deployable |

**Final Classification: DEPLOYABLE**

Cluster membership is computable for any novel peptide using only sequence-derived features available at inference time.

---

## Cross-Domain Enhancement: Arrow-Flip Threshold Detection

Following the pattern from Colbes' DDG prediction (hydrophobicity_diff=3.5 arrow-flip) and Rojas' cryptic diversity analysis, we applied cross-domain methodology to enhance C3.

### Arrow-Flip Thresholds (Colbes Method)

Grid search over feature values to find thresholds where cluster-conditional outperforms global prediction:

| Feature | Threshold | Improvement | Significant? |
|---------|:---------:|:-----------:|:------------:|
| **hydrophobicity** | **0.107** | **+0.238** | **YES** |
| length | 12.0 | +0.225 | YES |
| net_charge | 0.50 | +0.161 | YES |
| charge_density | 0.115 | +0.110 | YES |

**Best arrow-flip:** hydrophobicity @ 0.107 with +0.238 improvement over global prediction.

### Regime Routing Rules

```python
def route_peptide(hydrophobicity: float) -> str:
    """Route peptide to prediction regime based on arrow-flip threshold."""
    if hydrophobicity > 0.107:
        # Pathogen separation = 0.284 (strong signal)
        return "CLUSTER_CONDITIONAL"
    else:
        # Pathogen separation = 0.150 (weaker signal)
        return "GLOBAL_MODEL"
```

### Dual-Metric Analysis (Rojas Method)

Clusters classified by hyperbolic variance vs feature entropy:

| Cluster | N | Type | Effect Ratio | Has Signal |
|:-------:|---|------|:------------:|:----------:|
| 1 | 87 | VARIABLE | 0.54 | YES |
| 3 | 64 | CRYPTIC | 0.78 | YES |
| 4 | 114 | CRYPTIC | 0.72 | YES |
| 0 | 45 | CRYPTIC | 0.18 | NO |
| 2 | 89 | CRYPTIC | 0.48 | NO |

**Key finding:** Signal clusters (1, 3, 4) validated by cross-domain methods. The arrow-flip analysis provides an ALTERNATIVE routing strategy to cluster assignment.

### Enhanced Prediction Strategy

Two deployment options:

1. **Cluster-based:** Assign to cluster by (length, charge, hydrophobicity), trust clusters 1, 3, 4
2. **Threshold-based:** Route by hydrophobicity > 0.107, use cluster-conditional for high-hydrophobicity peptides

Both strategies achieve pathogen-specific signal; threshold-based is simpler but cluster-based is more granular.

---

## Cross-References

| Resource | Path |
|----------|------|
| Raw cluster data | `../results/validation_batch/P1_C3_results.json` |
| Falsification data | `../results/validation_batch/P1_C3_falsification.json` |
| Enhanced analysis | `../results/validation_batch/P1_C3_enhanced_results.json` |
| All P1 conjectures | `../results/validation_batch/P1_CONJECTURE_TESTS.md` |
| Platform theory | `../../../../docs/content/theory/regime_conditional_inference.md` |
| Test script | `../scripts/P1_test_C3_sequence_conditional.py` |
| Falsification script | `../scripts/P1_falsify_C3_seed_artifact.py` |
| Enhanced script | `../scripts/P1_C3_enhanced_clustering.py` |

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-07 | 1.1 | Added cross-domain enhancement with arrow-flip thresholds (Colbes) and dual-metric analysis (Rojas) |
| 2026-01-07 | 1.0 | Initial theorem formalization with reproducibility parameters |
