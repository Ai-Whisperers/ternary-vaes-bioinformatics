<!-- SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

---
title: "Deciphering interaction fingerprints from protein molecular surfaces using geometric deep learning"
date: 2025-12-24
authors:
  - AI Whisperers
version: "0.1"
license: PolyForm-Noncommercial-1.0.0
---

# Deciphering interaction fingerprints from protein molecular surfaces using geometric deep learning

**Author:** Gainza, P., et al. (EPFL)
**Year:** 2020
**Journal:** Nature Methods
**Link:** [PubMed Record](https://pubmed.ncbi.nlm.nih.gov/31819266/)
**Tags:** #masif #geometric-deep-learning #protein-surface #fingerprinting

## Abstract

Gainza et al. introduced **MaSIF** (Molecular Surface Interaction Fingerprinting), a geometric deep learning pipeline that analyzes protein _surfaces_ rather than sequences or 3D voxel grids. It treats the protein surface as a 2D Riemann manifold and computes "fingerprints" of interaction potential based on geometric and chemical features.

## Key Methodologies

### 1. Geodesic Polar Coordinates

Instead of 3D Euclidean distance (which cuts through the protein volume), MaSIF uses **Geodesic Distance** (along the surface).

- **Patch Extraction:** For every point, it extracts a "patch" of radius $r$ (geodesic).
- **Grid:** It maps this curved patch to a 2D polar grid $(\rho, \theta)$.

### 2. The Fingerprint

The network learns a vector descriptor (fingerprint) for each patch such that:

- Patches that _bind_ to each other have high dot-product similarity (or complementary geometry).
- It creates a continuous "Interaction Landscape."

## Key Findings

- **Homology Agnostic:** MaSIF can predict binders even if the proteins have <20% sequence identity, identifying "mimics" purely by surface shape/charge.
- **Speed:** It scans surfaces thousands of times faster than docking simulations.

## Relevance to Project

**Surface-Based Autoimmunity.**

- Autoantibodies bind to _surfaces_ (epitopes).
- **Molecular Mimicry** (Lanz 2022) is a surface phenomenon.
- **Action:** We should use MaSIF to compute the "Geodesic Fingerprint" of EBNA1 and GlialCAM. Even if their sequences differ (except for the mimicry region), their _surface fingerprints_ should be identical in the mimicry zone.
