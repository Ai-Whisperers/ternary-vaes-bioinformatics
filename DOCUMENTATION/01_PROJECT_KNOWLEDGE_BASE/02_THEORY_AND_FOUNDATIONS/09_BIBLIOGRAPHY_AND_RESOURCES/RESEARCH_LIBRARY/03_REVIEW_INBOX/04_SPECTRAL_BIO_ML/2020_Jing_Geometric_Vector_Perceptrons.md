<!-- SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

---
title: "Learning from Protein Structure with Geometric Vector Perceptrons"
date: 2025-12-24
authors:
  - AI Whisperers
version: "0.1"
license: PolyForm-Noncommercial-1.0.0
---

# Learning from Protein Structure with Geometric Vector Perceptrons

**Author:** Jing, B., et al. (MIT)
**Year:** 2020 (ICLR 2021)
**Journal:** ICLR
**Link:** [arXiv:2009.01411](https://arxiv.org/abs/2009.01411)
**Tags:** #geometric-deep-learning #protein-structure #equivariance #gvp #seminal

## Abstract

A foundational paper introducing **Geometric Vector Perceptrons (GVPs)**, a neural network architecture designed to process 3D protein structures directly. Unlike standard MLPs which lose geometric information, GVPs maintain **Rotation Equivariance** and Translation Invariance, ensuring that the model understands the protein's physics regardless of its orientation in space.

## Key Architecture: The GVP Layer

The GVP replaces the standard dense layer ($x \to \sigma(Wx+b)$) with a tuple processing unit:

- **Input:** Tuple $(s, V)$ where:
  - $s$: Scalar features (e.g., atom charge, sequence). _Invariant_.
  - $V$: Vector features (e.g., relative positions, bond directions). _Equivariant_.
- **Mechanism:**
  1.  **Vector Update:** $V' = W_h V$ (Linear transform, preserves direction).
  2.  **Norm Extraction:** $s_{norm} = ||V'||_2$ (Converts vector information to invariant scalar).
  3.  **Scalar Update:** $s' = \sigma(W_m [s, s_{norm}])$.

## Key Findings

- **Equivariance is Crucial:** Models that respect physical symmetries (SE(3)) vastly outperform those that don't (e.g., 3D-CNNs) on protein design and quality assessment tasks.
- **Efficiency:** GVPs are computationally efficient compared to spherical harmonic networks (e.g., SE(3)-Transformers).

## Relevance to Project

**The Computational Engine.**

- We must adapt the GVP to work with **P-adic Vectors**.
- **Challenge:** P-adic space has no "angles" (orthogonality is different).
- **Hypothesis:** A "P-adic GVP" would extract "Hierarchical Features" rather than "Geometric Features."
