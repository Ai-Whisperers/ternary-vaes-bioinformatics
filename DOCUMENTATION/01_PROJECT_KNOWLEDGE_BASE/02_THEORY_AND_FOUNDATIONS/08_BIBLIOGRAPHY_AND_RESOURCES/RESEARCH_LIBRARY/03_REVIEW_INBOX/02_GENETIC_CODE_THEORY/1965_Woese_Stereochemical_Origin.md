<!-- SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

---
title: "On the evolution of the genetic code"
date: 2025-12-24
authors:
  - AI Whisperers
version: "0.1"
license: PolyForm-Noncommercial-1.0.0
---

# On the evolution of the genetic code

**Author:** Woese, C.R.
**Year:** 1965
**Journal:** PNAS
**Link:** [PubMed Record](https://pubmed.ncbi.nlm.nih.gov/5323569/)
**Tags:** #genetic-code #stereochemical-theory #origins #classic

## Abstract

The seminal paper proposing the **Stereochemical Hypothesis** (or "Affinity Hypothesis"). Woese challenged Crick's "Frozen Accident" theory, arguing that the genetic code was not a random assignment but determined by direct physicochemical affinity between amino acids and their corresponding codons (or anticodons) on early RNA templates.

## Key Theoretical Claims

### 1. Direct Binding Affinity

Woese postulated that in the prebiotic era, amino acids physically bound to specific RNA sequences (aptamers) based on shape and charge complementarity.

- **Evidence:** Hydrophobic amino acids tend to be encoded by central U (Uracil) codons, while hydrophilic ones are encoded by A/G?
- **Implication:** The code is "deterministic" based on atomic physics.

### 2. Evolution of the Code

The code started as a "rough" classification based on affinity (e.g., "Any Pyrimidine at 2nd position = Hydrophobic") and was later refined by natural selection for error minimization (as shown by Haig/Freeland).

## Relevance to Project

**The Physical Prior.**

- In our VAE, the "Latent Space" represents the physicochemical properties of amino acids.
- **Initialization:** We should initialize the VAE's embedding layer using **Woese's affinities** (or modern variants like Table of Codon-AA Binding Energies) rather than random noise. This gives the model a "physics-informed" start.
