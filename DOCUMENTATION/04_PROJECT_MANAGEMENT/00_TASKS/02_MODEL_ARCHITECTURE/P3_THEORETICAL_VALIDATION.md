# Task: Theoretical Conjectures Validation

**Objective**: Empirically validate the core theoretical claims of the Ternary VAE (Hyperbolic Compression, Adelic Capacity).
**Source**: `01_PROJECT_KNOWLEDGE_BASE/02_THEORY_AND_FOUNDATIONS/foundations/CONJECTURES_INFORMATIONAL_GEOMETRY.md`

## High-Level Goals

- [ ] **Theory Scorecard**: A specialized test suite that outputs "Confirmed/Refuted" for each major conjecture.
- [ ] **Compression Bounds**: Verify the exponential compression ratio ($b^D / D \log b$).

## Detailed Tasks (Implementation)

### 1. The Conjectures Script

- [ ] **Create Script**: `research/theory/validate_conjectures.py`.
- [ ] **Test Conjecture 1 (Ultrametric Compression)**:
  - Embed full ternary tree ($3^4$ to $3^9$).
  - Measure separation ratio vs Euclidean baseline.
  - Confirm compression > 1000x.
- [ ] **Test Conjecture 5 (Arithmetic Geometry)**:
  - Train a regressor to predict $a \times b$ results from embeddings $z_a, z_b$.
  - Verify accuracy > 75% (reproduce the 78.7% claim).

### 2. Adelic Holography (Conjecture 6)

- [ ] **Capacity Check**: Verify if model performance drops sharply when `latent_dim < n_trits + 1 + 6`.
  - Run "Capacity Sweep": Train models with dim 8, 10, 12, 14, 16.
  - identifying the "Phase Transition" point.

### 3. Radial Hierarchy (Conjecture 2)

- [ ] **Radius-Depth Correlation**: Plot embedding radius $r(x)$ vs tree depth $D(x)$.
- [ ] **Fit Curve**: Verify $r(x) \approx a \cdot p^{-c \cdot D(x)}$.

## Deliverables

- [ ] `THEORY_SCORECARD.md`: Generated report of the findings.
- [ ] `capacity_sweep_results.json`.
