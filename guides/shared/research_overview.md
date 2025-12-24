# Research Directory

**Doc-Type:** Project Index · Version 1.0 · Updated 2025-12-16

---

## Overview

This directory contains all research analysis derived from the Ternary VAE v1.1.0 model. The research is organized into three interconnected domains:

```
research/
├── spectral_analysis/     # Mathematical foundations (Riemann hypothesis connection)
├── genetic_code/          # Core discovery (codon→p-adic mapping)
└── bioinformatics/        # Clinical applications (RA, HIV)
```

---

## Domain Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    SPECTRAL ANALYSIS                            │
│         P-adic geometry, eigenvalue spectrum, zeta              │
│                    (Mathematical Foundation)                    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GENETIC CODE                                │
│         64 codons → 21 clusters (100% accuracy)                 │
│                    (The Discovery Bridge)                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BIOINFORMATICS                               │
│         RA: HLA prediction, citrullination, regeneration        │
│         HIV: CTL escape, drug resistance, fitness               │
│                    (Clinical Applications)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Spectral Analysis

**Path:** `spectral_analysis/`

Mathematical analysis connecting the VAE's learned embedding space to p-adic geometry and the Riemann hypothesis.

### Key Findings
- **Radial Exponent**: r^0.63 scaling in embedding distances
- **Prime Capacity**: 21 clusters match amino acid degeneracy
- **P-adic Structure**: Ultrametric inequality satisfied

### Scripts
| Script | Purpose |
|--------|---------|
| 01_extract_embeddings.py | Extract VAE latent space |
| 02_compute_spectrum.py | Compute eigenvalue spectrum |
| 03_compare_zeta.py | Compare to Riemann zeta zeros |
| 04_padic_spectral_analysis.py | P-adic spectral analysis |
| 05_exact_padic_analysis.py | Exact p-adic computations |

---

## 2. Genetic Code

**Path:** `genetic_code/`

The foundational discovery: the VAE's 64 natural positions map perfectly to the 64 codons, clustering into 21 amino acid groups with 100% accuracy.

### Key Findings
- **64→21 Mapping**: Codon encoder achieves 100% cluster accuracy
- **Wobble Pattern**: Position 3 (wobble) shows highest variance
- **Synonymous Clustering**: 193.5x separation ratio

### Scripts
| Script | Purpose |
|--------|---------|
| 01_bioinformatics_analysis.py | Initial biological analysis |
| 02_genetic_code_padic.py | P-adic genetic code mapping |
| 03_reverse_padic_search.py | Reverse search for natural positions |
| 04_fast_reverse_search.py | Optimized search |
| 05_analyze_natural_positions.py | Position analysis |
| 06_learn_codon_mapping.py | Train codon encoder |

### Key Artifacts
- `data/codon_encoder.pt` - Trained neural network
- `data/learned_codon_mapping.json` - Codon→position mapping

---

## 3. Bioinformatics

**Path:** `bioinformatics/`

Clinical applications using p-adic geometry for disease analysis.

### Rheumatoid Arthritis (`rheumatoid_arthritis/`)

| Analysis | Key Finding |
|----------|-------------|
| HLA Prediction | p < 0.0001, r = 0.751 with odds ratio |
| Citrullination | 14% cross boundaries (sentinel epitopes) |
| Regenerative Axis | Parasympathetic is geometrically central |
| Codon Optimizer | 100% citrullination safety achievable |

### HIV (`hiv/`)

| Analysis | Key Finding |
|----------|-------------|
| CTL Escape | Distance correlates with fitness cost |
| Drug Resistance | INSTIs most constrained (d=4.30) |
| Fitness Prediction | r = 0.24 distance-fitness correlation |

---

## Dependencies

All scripts require:
```python
torch           # Neural network
numpy           # Numerical computation
scipy           # Statistical tests
matplotlib      # Visualization
scikit-learn    # PCA, clustering
```

Plus the trained models from `src/` and `configs/`.

---

## Running the Research Pipeline

### 1. Spectral Analysis
```bash
cd research/spectral_analysis/scripts
python 01_extract_embeddings.py
python 02_compute_spectrum.py
python 03_compare_zeta.py
```

### 2. Genetic Code Discovery
```bash
cd research/genetic_code/scripts
python 02_genetic_code_padic.py
python 06_learn_codon_mapping.py
```

### 3. Bioinformatics Applications
```bash
# Rheumatoid Arthritis
cd research/bioinformatics/rheumatoid_arthritis/scripts
python 01_hla_functionomic_analysis.py
python 02_hla_expanded_analysis.py

# HIV
cd research/bioinformatics/hiv/scripts
python 01_hiv_escape_analysis.py
python 02_hiv_drug_resistance.py
```

---

## Connection to Main Project

| Component | Research Application |
|-----------|---------------------|
| **VAE v1.1.0** | Source of p-adic embedding geometry |
| **Codon Encoder** | Bridge from math to biology |
| **Engine** | High-performance ternary computation |

---

## Changelog

| Date | Version | Description |
|------|---------|-------------|
| 2025-12-16 | 1.0 | Reorganized from riemann_hypothesis_sandbox |

---

**Status:** Active research with experimental validation pending
