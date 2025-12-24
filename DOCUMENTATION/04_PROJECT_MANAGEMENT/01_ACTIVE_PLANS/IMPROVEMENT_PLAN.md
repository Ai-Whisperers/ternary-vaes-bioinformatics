# Research-Driven Code Improvement Plan

This document outlines structural and functional improvements to the codebase, derived directly from the project's research documentation (`DEPENDENCIES.md`, `RELEVANT_REPOS.md`) and a fresh code analysis.

## 1. Core Geometry & Stability (The "Manifold-Native" Upgrade)

**Source:** `DEPENDENCIES.md` (Sec 1), `RELEVANT_REPOS.md` (Sec 1)
**Current State:** `src/geometry/poincare.py` uses `geoopt` optionally, with manual fallbacks that may be numerically unstable.
**Goal:** Guarantee numerical stability for Poincare ball operations.

- [ ] **Enforce `geoopt` Requirement**: Remove the `try-except` block in `poincare.py` and make `geoopt` a hard dependency.
  - _Why_: Documentation states manual implementation is "numerically unstable".
- [ ] **Adopt `hgcn` Mobius Addition**: Review the manual fallback of `mobius_add` against `HazyResearch/hgcn/manifolds/poincare.py` if manual fallback must remain.
- [ ] **Integrate Riemannian Optimizers**: Ensure ALL training scripts use `get_riemannian_optimizer` instead of vanilla Adam when training hyperbolic parameters.

## 2. Biological Data Interoperability

**Source:** `DEPENDENCIES.md` (Sec 2), `RELEVANT_REPOS.md` (Sec 4)
**Current State:** Custom data loaders.
**Goal:** Make the tool a "Plugin" for standard biotech pipelines.

- [ ] **Implement `AnnData` Support**: Create a new data loader `src/data/bio_loader.py` that uses `scanpy` to read/write `.h5ad` files.
  - _Feature_: Allow saving latent embeddings directly into `adata.obsm['X_ternary_vae']`.
- [ ] **Evolutionary Velocity**: Prototype a "Fitness Gradient" metric based on `scvelo` logic, calculating the direction of higher fitness in hyperbolic space.

## 3. Advanced Metrics: Disentanglement

**Source:** `RELEVANT_REPOS.md` (Sec 3)
**Current State:** Basic coverage and reconstruction metrics. No independence checks.
**Goal:** Prove features are independent (e.g., "Infectivity" vs "Replication").

- [ ] **Implement MIG (Mutual Information Gap)**: Port `compute_mig` from `google-research/disentanglement_lib` to `src/metrics/disentanglement.py`.
  - _Action_: Measure if specific dimensions of the 16D Poincare ball correlate with known biological factors (if labeled data exists).

## 4. Hyperparameter Visualization

**Source:** `DEPENDENCIES.md` (Sec 3)
**Current State:** TensorBoard scalars.
**Goal:** visualize the "Pareto Frontier" of Hyperparams vs Quality.

- [ ] **Integrate `hiplot`**: Add a script `scripts/analysis/analyze_hyperparams.py` that loads training run logs (YAML/JSON) and launches a HiPlot server to visualize `lr` vs `rho` vs `coverage`.

## 5. Architectural Refactoring (Maintenance)

**Source:** `DUPLICATION_REPORT.md` (Codebase Analysis)
**Current State:** Forked logic in Trainers (`trainer.py` vs `appetitive_trainer.py`).
**Goal:** Reduce technical debt.

- [ ] **Base Trainer Class**: Create `src/training/base.py` containing the `__init__`, checkpointing, and loop boilerplate.
- [ ] **Inheritance Refactor**: Refactor `TernaryVAETrainer` and `AppetitiveVAETrainer` to inherit from the base class.

## 6. Science & Experiments (Knowledge Translation)

**Source:** `01_PROJECT_KNOWLEDGE_BASE` (Experiments Index)
**Goal:** Translate theoretical findings into reproducible lab results.

- [ ] **Rheumatoid Arthritis**: Verify "Goldilocks Zone" hypothesis (`P1_RA_AUTOIMMUNITY`).
- [ ] **Viral Evolution**: Analyze Glycan Shield geometry (`P1_VIRAL_EVOLUTION`).
- [ ] **Spectral Analysis**: Compute Adelic/Spectral fingerprints (`P2_SPECTRAL_ANALYSIS`).
- [ ] **Validation Pillars**: Execute the full benchmark suite (`P1_VALIDATION_SUITE`).

## Summary of Priority Actions

| Priority | Area     | Task File                                                                                | Action                   |
| :------- | :------- | :--------------------------------------------------------------------------------------- | :----------------------- |
| **P0**   | Geometry | **[P0_GEOMETRY.md](../00_TASKS/02_MODEL_ARCHITECTURE/P0_GEOMETRY.md)**                   | Make `geoopt` mandatory  |
| **P0**   | Science  | **[P1_RA_AUTOIMMUNITY.md](../00_TASKS/01_BIOINFORMATICS/P1_RA_AUTOIMMUNITY.md)**         | Validate RA Hypothesis   |
| **P1**   | Security | **[P1_SECURITY_FIXES.md](../00_TASKS/03_INFRASTRUCTURE/P1_SECURITY_FIXES.md)**           | Fix Syntax/Security bugs |
| **P1**   | Refactor | **[P2_COMPLEXITY_REFACTOR.md](../00_TASKS/03_INFRASTRUCTURE/P2_COMPLEXITY_REFACTOR.md)** | Fix Monster Functions    |
| **P1**   | Bio      | **[P1_VIRAL_EVOLUTION.md](../00_TASKS/01_BIOINFORMATICS/P1_VIRAL_EVOLUTION.md)**         | Glycan Shield Analysis   |
| **P2**   | Metrics  | **[P2_METRICS.md](../00_TASKS/05_VALIDATION/P2_METRICS.md)**                             | Implement MIG            |
| **P3**   | Viz      | **[P3_VISUALIZATION.md](../00_TASKS/03_INFRASTRUCTURE/P3_VISUALIZATION.md)**             | Add `hiplot` script      |
