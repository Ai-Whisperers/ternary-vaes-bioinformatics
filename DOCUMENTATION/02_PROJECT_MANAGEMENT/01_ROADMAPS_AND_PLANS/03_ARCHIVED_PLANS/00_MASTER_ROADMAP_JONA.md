# The Jona Research Roadmap

> **Strategic Direction for v6.0+**

This folder contains the "Blue Ocean" research plan to take the project beyond its current scope.

---

## 🏗️ Phase 1: The Hyperbolic Manifold Pivot (Q1 2025)

> **Core Directive**: Transition from "Generalization" to "Exhaustive Manifold Mapping". See [`2025_HYPERBOLIC_MANIFOLD_STRATEGY.md`](./2025_HYPERBOLIC_MANIFOLD_STRATEGY.md).

**Goal:** Train on 100% of the finite 3-adic space (19,683 ops) to ensure perfect structural preservation.

1.  **[Tooling Upgrade (Immediate)]**

    - **Action:** Refactor `models/layers.py` to use `geoopt`.
    - **Action:** Switch to `scanpy` for data loading.

2.  **[Manifold Implementation]**
    - **Action:** Update `src/data/loaders.py` to handle `val_split=0` (Full Manifold Mode).
    - **Action:** Implement "Coverage-Based" Early Stopping in `monitor.py`.
    - **Action:** Establish `verify_mathematical_proofs.py` as the daily regression test.

## 🔬 Phase 2: Scientific Expansion (The "Next")

_Goal: Apply the model to new scientific domains._

3.  **[Scientific Impact Domains](SCIENTIFIC_DOMAINS.md)**

    - **Engineering:** Supply Chain Optimization (Resilience).
    - **Code:** Bug detection via AST (Abstract Syntax Tree) embedding.
    - **Materials:** Predicting polymer strength via branching factors.

4.  **[Medical Frontiers](MEDICAL_FRONTIERS.md)**
    - **Pharma:** "Generative Immunology" (Universal Antibodies).
    - **Clinical:** Autoimmune "Severity Scoring" for personalized chemo.
    - **Future:** "Bio-Digital Twins" on wearables.

---

**Summary:**
We are moving from a "Bioinformatics Tool" -> "General Purpose Hierarchical AI".
