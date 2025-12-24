<!-- SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

---
title: "A Co-Evolution Theory of the Genetic Code"
date: 2025-12-24
authors:
  - AI Whisperers
version: "0.1"
license: PolyForm-Noncommercial-1.0.0
---

# A Co-Evolution Theory of the Genetic Code

**Author:** Wong, J.T.
**Year:** 1975
**Journal:** PNAS
**Link:** [PubMed Record](https://pubmed.ncbi.nlm.nih.gov/1093156/)
**Tags:** #genetic-code #co-evolution #metabolism #biosynthesis #seminal

## Abstract

Wong proposed the **Co-Evolution Theory** (or "Biosynthetic Theory"). He argued that the code evolved alongside amino acid biosynthetic pathways. The code originally encoded only a few "Phase 1" amino acids (prebiotic). As metabolism evolved "Phase 2" amino acids (from Phase 1 precursors), the codons of the precursors were _conceded_ or _shared_ with the products.

## Key Theoretical Formalisms

### 1. Precursor-Product Relationships

- **Phase 1 (Prebiotic):** Gly, Ala, Val, Asp, Glu, Ser...
- **Phase 2 (Biosynthetic):** Phe, Tyr (from Prephenate?), Lys, Arg, Trp...
- **Rule:** Contiguous codons encode metabolically related amino acids. e.g., Glu $\to$ Gln share similar codons.

### 2. Code Expansion

The code expanded from a simple (low degeneracy) doublet code to the current triplet code to accommodate the new "metabolic inventions."

## Relevance to Project

**Generative Roadmap.**

- This provides a "Curriculum Learning" strategy for our VAE.
- **Step 1:** Train VAE only on Phase 1 amino acids/codons (creating a "Primitive Code").
- **Step 2:** Introduce Phase 2 acids as "perturbed" versions of Phase 1 acids in the latent space.
- The VAE should naturally "invent" coding assignments for Phase 2 acids that are close to their metabolic parents.
