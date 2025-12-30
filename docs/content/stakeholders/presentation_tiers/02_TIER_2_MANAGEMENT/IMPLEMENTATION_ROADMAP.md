# Implementation Roadmap: Q1 2025

**Tier 2: Management & Operations**

---

## 1. Program Status Overview

| Project                | Lead     | Phase       | Status      | Next Milestone          |
| :--------------------- | :------- | :---------- | :---------- | :---------------------- |
| **Peptide Foundry**    | Brizuela | Scaffolding | 🟢 On Track | VAE Model Integration   |
| **Phylogenetic Radar** | Rojas    | Scaffolding | 🟢 On Track | NCBI Data Pipeline Live |
| **Geometric QC**       | Colbes   | Scaffolding | 🟢 On Track | AlphaFold Batch Scanner |

---

## 2. Technical Roadmap

### Sprint 1: The "Wire-Up" (Weeks 1-2)

- **Goal:** Connect the scaffolding to real data and models.
- **Tasks:**
  - [Brizuela] Implement `TernaryVAEInterface` with PyTorch.
  - [Rojas] Connect `NCBIFetcher` to Entrez API.
  - [Colbes] Test `PDBScanner` on a sample AlphaFold dataset.

### Sprint 2: The "Logic" (Weeks 3-4)

- **Goal:** Implement the core mathematical engines.
- **Tasks:**
  - [Brizuela] Code the `NSGA-II` loop with Toxicity/Activity predictors.
  - [Rojas] Implement `RiemannianParallelTransport` for forecasting.
  - [Colbes] Finalize the `p-adic` valuation metric.

### Sprint 3: Validation (Weeks 5-6)

- **Goal:** Run the pilots and generate results.
- **Tasks:**
  - Generate 100 candidate AMPs (Brizuela).
  - Forecast 2026 Dengue serotypes (Rojas).
  - Filter a library of 1000 de novo proteins (Colbes).

---

## 3. Resource Allocation

- **Compute:** Requires GPU access for VAE inference (Brizuela) and Hyperbolic embedding (Rojas).
- **Data:**
  - Need `checkpoints/ternary_vae.pt` (Available).
  - Need NCBI API Key (Pending).
  - Need AlphaFold/RFdiffusion outputs (User provided).

---

## 4. Risks & Mitigations

- **Risk:** VAE Latent Space Collapse.
  - _Mitigation:_ Use `p-adic` regularization (Colbes's tech) to stabilize Brizuela's optimizer.
- **Risk:** NCBI Data Gaps.
  - _Mitigation:_ Supplement with GISAID data if necessary.
- **Risk:** Hyperbolic computations are slow.
  - _Mitigation:_ Use `geomstats` with GPU backend.
