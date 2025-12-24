# Losses: The Geometry of Error

This directory contains the custom loss functions that enforce the Project's core hypothesis.

## 1. 3-Adic Losses (`padic_losses.py`)

This is the heart of the project.

- **Goal:** Force the Neural Net to think in Base-3.
- **Mechanism:** It penalizes Euclidean distance but rewards **Ultrametric Distance**.
  - _Euclidean:_ dist(A, C) = 1
  - _3-Adic:_ dist(A, C) = 1/3 (if they share a parent).

## 2. Hyperbolic Reconstruction (`hyperbolic_recon.py`)

- **Goal:** Reconstruct the input sequence from a curved space.
- **Contrast:** Normal VAEs assume a flat Gaussian. We assume a **Wrapped Normal** distribution on the Poincaré ball.

## 3. Homeostatic Loss (`homeostasis.py` - imported)

- **Goal:** Biological Plausibility.
- **Mechanism:** Applies a penalty if the "Energy" of the model drifts too far from the "Regenerative Axis".

---

**Summary:** We do not minimize MSE. We minimize "Topological Distortion".
