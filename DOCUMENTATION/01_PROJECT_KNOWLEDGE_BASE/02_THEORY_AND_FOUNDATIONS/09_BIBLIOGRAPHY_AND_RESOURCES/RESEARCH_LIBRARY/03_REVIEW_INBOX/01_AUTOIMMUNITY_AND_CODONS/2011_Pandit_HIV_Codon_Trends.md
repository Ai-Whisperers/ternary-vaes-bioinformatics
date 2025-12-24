# Differential evolution of codon usage bias in HIV-1 genes

**Author:** Pandit, A., & Sinha, S.
**Year:** 2011
**Journal:** Virology Journal
**Link:** [PubMed Record](https://pubmed.ncbi.nlm.nih.gov/22151786/)
**Tags:** #hiv #codon-bias #evolution #genomics

## Abstract

This study analyzed 23 years of HIV-1 genome data to track how the virus adapts its codon usage to the human host. It revealed that HIV-1 genes do not evolve uniformly; rather, specific genes (gag, env, pol) show differential adaptation trajectories towards the host's preferred codons, driven by "Translation Selection."

## Key Quantitative Findings

### 1. Gene-Specific Trends

- **Early Infection:** HIV genomes show distinct codon biases that differ significantly from the human host (Goldilocks mismatch).
- **Late Infection/Evolution:** Over decades, the viral genome drifts towards simpler, host-compatible codon usage to maximize replication efficiency.
- **Correlation:** Strong temporal correlation ($R^2 > 0.8$) between time and codon adaptation index (CAI) for specific genes.

### 2. The "Goldilocks" Drift

- The virus begins in a "Frustrated State" (low CAI, low efficiency but high immune evasion/latency potential).
- It evolves towards a "Relaxed State" (high CAI, high efficiency, high visibility).

## Relevance to Project

**The Latency Clock.**

- We hypothesize that this "Codon Drift" is the clock that wakes the virus from latency.
- **VAE Application:** We can feed the VAE a time-series of HIV sequences. The VAE latent vector magnitude should correlate with the "Time since infection" or "Latency Depth."
- If we can predict the specific codons that "break" latency, we can design inhibitors for them (e.g., tRNAs).
