<!-- SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

---
title: "p-Adic numbers in bioinformatics: from genetic code to PAM-matrix"
date: 2025-12-24
authors:
  - AI Whisperers
version: "0.1"
license: PolyForm-Noncommercial-1.0.0
---

# p-Adic numbers in bioinformatics: from genetic code to PAM-matrix

**Author:** Khrennikov, A., & Kozyrev, S.V.
**Year:** 2010
**Journal:** Theoretical Biology and Medical Modelling
**Link:** [PubMed Record](https://pubmed.ncbi.nlm.nih.gov/20459738/)
**Tags:** #padic #pam-matrix #substitution-rates #theory

## Abstract

Khrennikov and Kozyrev demonstrate that the empirically derived PAM (Point Accepted Mutation) matrix, used for protein alignment, possesses a hidden hierarchical structure best described by 2-adic numbers. They decompose the PAM matrix into a "regular" component rooted in the genetic code's structure and a "sparse" component reflecting amino acid physicochemical properties.

## Key Theoretical Formalisms

### 1. The PAM Decomposition Equation

The authors proposed that the PAM matrix $A$ is a sum of two components:
$$ A = A^{(2)} + A^{(\infty)} $$

- **$A^{(2)}$ (The 2-Adic Component):** The "regular" part. Matrix elements are locally constant with respect to the 2-adic parametrization. This component accounts for substitutions driven by the genetic code's structure (e.g., single point mutations, transition bias).
- **$A^{(\infty)}$ (The Sparse Component):** The "irregular" part. Captures rare or specific substitutions driven by side-chain geometry and chemistry that defy the simple 2-adic logic.

### 2. Parametric Alignment Family

They introduced a family of substitution matrices tunable by parameters $\alpha, \beta$:
$$ A(\alpha, \beta) = \alpha A^{(2)} + \beta A^{(\infty)} $$
This allows for "tuning" sequence alignment algorithms to prioritize either evolutionary history ($\alpha$) or structural function ($\beta$).

## Key Findings

- **Hierarchical Substitution:** Amino acid substitution rates are not random; they follow a hierarchy defined by the 2-adic distance of their codons.
- **Code vs. Chemistry:** The paper provides a mathematical way to separate the "signal" of the genetic code (nucleotide level) from the "signal" of protein chemistry (phenotype level).

## Relevance to Project

**High Priority:** This decomposition ($A^{(2)} + A^{(\infty)}$) is a perfect analogue for a VAE Loss Function.

- **Reconstruction Loss** $\approx A^{(2)}$ (Code fidelity).
- **Regularization (KL) Loss** $\approx A^{(\infty)}$ (Structural constraint).
  We can use this formalism to design a custom "Khrennikov Loss" for the Ternary VAE.
