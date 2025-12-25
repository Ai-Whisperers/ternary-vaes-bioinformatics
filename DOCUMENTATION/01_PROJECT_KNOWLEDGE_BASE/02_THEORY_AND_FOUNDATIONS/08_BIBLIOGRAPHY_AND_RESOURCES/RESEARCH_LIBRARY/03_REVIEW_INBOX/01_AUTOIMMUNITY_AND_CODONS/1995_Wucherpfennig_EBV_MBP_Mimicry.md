<!-- SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

---
title: "Molecular mimicry in T cell-mediated autoimmunity: Viral peptides activate human T cell clones specific for myelin basic protein"
date: 2025-12-24
authors:
  - AI Whisperers
version: "0.1"
license: PolyForm-Noncommercial-1.0.0
---

# Molecular mimicry in T cell-mediated autoimmunity: Viral peptides activate human T cell clones specific for myelin basic protein

**Author:** Wucherpfennig, K.W., & Strominger, J.L.
**Year:** 1995
**Journal:** Cell
**Link:** [PubMed Record](https://pubmed.ncbi.nlm.nih.gov/7548050/)
**Tags:** #molecular-mimicry #ms #mbp #t-cell #hla-dr2 #classic

## Abstract

A foundational paper that established the structural basis of T-cell molecular mimicry. It identified that viral peptides (from EBV, Influenza, HPV) could activate T-cell clones originally raised against Myelin Basic Protein (MBP), specifically the immunodominant epitope MBP(85-99).

## Key Biological Sequences

### 1. The Target Self-Epitope

- **MBP(85-99):** `ENPVVHFFKNIVTPR`
- **MHC Restriction:** HLA-DR2 (DRB1\*1501), the primary genetic risk factor for MS.

### 2. Identified Mimics

The study screened a library of viral peptides and found activators that did _not_ necessarily share high sequence identity but shared **structural/binding motifs**.

- **EBV DNA Polymerase (BALF5) 627-641:**
  - _Sequence:_ `TGGVYHFVKKHVHES` (Note the conserved Hydrophobic-Hydrophobic-Positive motif).
- **Influenza Type A Hemagglutinin:**
  - Activates the same clones.

## Key Findings

- **Degeneracy of TCR Recognition:** A single T-cell receptor (TCR) can recognize multiple distinct peptides if they present the same key residues (anchors) to the MHC groove.
- **Structural Mimicry:** Mimicry is not just about string matching (Edit Distance); it is about 3D surface compatibility (P-adic proximity?).

## Relevance to Project

**Training Data for "Fuzzy" Matching.**

- Our VAE should _not_ just learn sequence identity.
- It must learn that `ENPVVHFFKNIVTPR` $\approx$ `TGGVYHFVKKHVHES` in Latent Space, despite low Hamming similarity.
- This confirms the need for our **Genetic Code Error Minimization** loss—these peptides are "synonymous" in immune space.
