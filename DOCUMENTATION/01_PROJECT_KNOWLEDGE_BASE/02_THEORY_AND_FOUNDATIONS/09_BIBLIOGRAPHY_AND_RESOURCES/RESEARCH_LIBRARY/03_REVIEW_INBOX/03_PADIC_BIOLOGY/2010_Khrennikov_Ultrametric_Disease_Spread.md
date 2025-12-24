<!-- SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

---
title: "Modeling of the spread of viral infection on hierarchical social clusters"
date: 2025-12-24
authors:
  - AI Whisperers
version: "0.1"
license: PolyForm-Noncommercial-1.0.0
---

# Modeling of the spread of viral infection on hierarchical social clusters

**Author:** Khrennikov, A., et al.
**Year:** 2010 (Refined 2020-21)
**Journal:** Physica A / Various
**Link:** [arXiv](https://arxiv.org/)
**Tags:** #padic #epidemiology #hierarchy #diffusion

## Abstract

Khrennikov applies p-adic diffusion equations to model disease spread. The core insight is that human/biological populations are not "flat" (Euclidean grids) but "hierarchical" (Social Clusters: Family $\subset$ City $\subset$ Nation). A virus spreads rapidly _within_ a cluster but faces "Barriers" to jump _between_ clusters.

## Key Formalism: Ultrametric Diffusion

- **Space:** An Ultrametric Space (Tree of Clusters).
- **Barriers:** The barrier height depends on the hierarchical distance. Jumping from Family A to Family B is easier than Family A to Foreign Nation C.
- **Equation:** P-adic Diffusion Equation (Pseudo-differential operator).
  $$ \frac{\partial f}{\partial t} + \mathcal{D}\_p f = 0 $$
where $\mathcal{D}_p$ is the Vladmir-operator (p-adic derivative).

## Key Findings

- **Herd Immunity Power Law:** Unlike standard SIR models (exponential decay), ultrametric diffusion predicts a **Power Law** approach to herd immunity ($1 - t^{-\alpha}$).
- **Spreading Entropy:** A new metric characterizing how infection distributes across hierarchy levels.

## Relevance to Project

**Viral Latency & Reservoirs.**

- The body is also a hierarchy (Cell $\subset$ Tissue $\subset$ Organ).
- **Hypothesis:** HIV Latency is an "Ultrametric Trap." The virus is stuck in a profound "deep branch" (Reservoir Cell) and cannot overcome the barrier to the bloodstream.
- **Reactivation:** Corresponds to a "P-adic Jump" (Quantum Tunneling) out of the trap.
