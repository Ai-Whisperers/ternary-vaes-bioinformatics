<!-- SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

---
title: "p-Adic Modelling of the Genome and the Genetic Code"
date: 2025-12-24
authors:
  - AI Whisperers
version: "0.1"
license: PolyForm-Noncommercial-1.0.0
---

# p-Adic Modelling of the Genome and the Genetic Code

**Author:** Dragovich, B., & Dragovich, A.
**Year:** 2009
**Journal:** P-adic Numbers, Ultrametric Analysis and Applications
**Link:** [arXiv:0707.3043](https://arxiv.org/abs/0707.3043)
**Tags:** #padic #genomics #ultrametric #theory

## Abstract

This paper presents a rigorous mathematical model of the genome and genetic code using p-adic numbers. It posits that the "space" of DNA sequences is not Euclidean but ultrametric, governed by the properties of p-adic distance. The authors propose a 5-adic model for DNA ($A, G, C, T + \text{gap}$) and demonstrate that the degeneracy of the genetic code is an inherent property of this p-adic information space.

## Key Theoretical Formalisms

### 1. The P-adic Metric

The core axiom is that genetic information distance ($d_p$) satisfies the **Strong Triangle Inequality**:
$$ d_p(x, y) \le \max \{ d_p(x, z), d_p(z, y) \} $$
This implies a hierarchical, tree-like structure where "balls" (clusters of sequences) are either disjoint or nested.

### 2. P-adic Norm Definition

For a rational number $x = p^\nu \frac{r}{s}$, the p-adic norm is:
$$ |x|\_p = p^{-\nu} $$
This norm measures "divisibility by p" rather than magnitude. In the context of sequences, it measures the depth of the common branch in the phylogenetic/structural tree.

### 3. The 5-adic Genome Model

The authors map nucleotides to digits in a 5-adic number system ($Z_5$):

- 0: Gap / Null
- 1, 2, 3, 4: A, C, G, T (Specific mapping depends on purine/pyrimidine distance).

### 4. 2-adic Distance for Nucleotides

To capture physicochemical similarities, they use a 2-adic distance:

- $d_2(Pyr, Pur) = 1$ (Distance between C/U and A/G is maximal).
- $d_2(C, U) = 1/2$ (Distance within pyrimidines is smaller).
- $d_2(A, G) = 1/2$ (Distance within purines is smaller).
  This mathematically encodes the transition/transversion bias.

## Key Findings

- **Ultrametric Degeneracy:** The degeneracy of the genetic code (synonymous codons) maps to p-adic balls. Codons coding for the same amino acid are "close" in p-adic space.
- **Information Space:** The set of all possible DNA sequences forms an ultrametric informational space, suggesting that evolution moves along branches of a p-adic tree rather than a continuous Euclidean landscape.

## Relevance to Project

**High Priority:** This paper provides the exact mathematical definition ($d_p$ and $|x|_p$) needed for our VAE's latent space metric. We should implement the 5-adic mapping for the input layer and the 2-adic distance for the loss function.
