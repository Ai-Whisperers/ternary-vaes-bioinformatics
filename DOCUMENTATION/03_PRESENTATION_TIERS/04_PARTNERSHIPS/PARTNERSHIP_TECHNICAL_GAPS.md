# Partnership Technical Gap Analysis

> **Critical Gap Assessment for Expert MVP Implementation**

This document identifies the technical prerequisites, missing infrastructure, and architectural blockers that must be resolved to execute the `PARTNERSHIP_IMPLEMENTATION_PLAN.md`.

---

## 🛑 Critical Blockers

| Domain             | Gap Description                                                                                                                                                                                                                                                                     | Severity     | Remediation Strategy                                                                                                       |
| :----------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------- | :------------------------------------------------------------------------------------------------------------------------- |
| **Models**         | **Domain Specificity:** The current `TernaryVAEV5_11` uses a `FrozenEncoder` (likely optimized for HIV protease/RT). It cannot be applied zero-shot to **Antimicrobial Peptides** (different length/alphabet stats) or **Protein Rotamers** (geometric angles vs. discrete tokens). | **CRITICAL** | **Retraining Required:** We need to train a new "Base Encoder" for each domain or implement a "Transfer Learning" wrapper. |
| **Infrastructure** | **Missing Dependencies:** `biopython` and `ncbi-datasets-cli` are not confirmed in the environment.                                                                                                                                                                                 | **HIGH**     | Update `requirements.txt` and create a `setup_partnership_env.ps1` script.                                                 |
| **Data Ingest**    | **StarPep Ingest Limitations:** The existing `ingest_starpep.py` uses a hardcoded `3adic` encoder version. It needs to support a "Universal" or "AMP-Specific" encoder.                                                                                                             | **MEDIUM**   | Refactor `ingest_starpep.py` to accept dynamic encoder paths.                                                              |

---

## 🔍 Detailed Domain Gaps

### 1. Carlos Brizuela (AMPs)

- **Missing Regressors:** We plan to optimize for "Activity" vs "Toxicity", but we have no trained regressors (`models/predictors/toxicity_regressor.pt` does not exist).
- **Variable Length Handling:** AMPs vary in length (10-50 AA). The current VAE might expect fixed-length inputs (e.g., HIV protease 99 AA). **Gap:** Need a `Padding/Pooling` strategy in `differentiable_controller.py`.

### 2. Dr. José Colbes (Protein Optimization)

- **No Geometric Parser:** We lack code to convert PDB `.cif` files into the "Side Chain Angle" format required for VAE input.
- **Metric Mismatch:** The current `PAdicGeodesicLoss` operates on _sequence_ hierarchy. We need to adapt it to operate on _rotamer angle_ hierarchy (e.g., $0^\circ, 120^\circ, 240^\circ$ rotamer wells).

### 3. Alejandra Rojas (Arboviruses)

- **Genome Size Scaling:** Dengue genomes (11kb) are much larger than HIV proteins. The current `TernaryVAE` cannot embed a whole genome in one pass.
- **Gap:** We need a "Sliding Window" embedder (`scripts/analysis/sliding_window_embedder.py`) to break the genome into VAE-digestible chunks (e.g., 300nt windows).

---

## 🛠 Recommended "Integration Phase" (Week 0)

Before starting the expert-specific phases, we must execute a **Week 0 Integration Sprint**:

1.  **Universal Encoder:** Train a small, general-purpose VAE on Uniprot50 (or a diverse subset) to serve as a better starting point than the HIV-specific FrozenEncoder.
2.  **Env Setup:** Create `requirements_partners.txt` with `biopython`, `pandas`, `scipy`.
3.  **Sliding Window Utility:** Implement a generic sequence chunker for Arbovirus support.

---
