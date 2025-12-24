# Diffusion maps

**Author:** Coifman, R.R., & Lafon, S.
**Year:** 2006
**Journal:** Applied and Computational Harmonic Analysis
**Link:** [ScienceDirect](https://doi.org/10.1016/j.acha.2006.04.006)
**Tags:** #manifold-learning #diffusion-maps #spectral-theory #metric-learning #classic

## Abstract

The seminal paper defining **Diffusion Maps**, a non-linear dimensionality reduction technique. It interprets data points as nodes in a graph and defines a "Diffusion Distance" based on the probability of a random walker transitioning between points in $t$ steps. This metric captures the intrinsic geometry of the data manifold better than Euclidean distance.

## Key Mathematical Formalisms

### 1. Diffusion Distance ($D_t$)

The squared diffusion distance between points $x$ and $y$ at time $t$ is:
$$ D*t(x, y)^2 = \sum*{j=1}^{k} \lambda_j^{2t} (\psi_j(x) - \psi_j(y))^2 $$

- $\lambda_j$: Eigenvalues of the diffusion operator (Markov matrix).
- $\psi_j$: Eigenvectors (Diffusion coordinates).
- **Interpretation:** $D_t(x,y)$ is small if there are _many high-probability paths_ connecting $x$ and $y$. It is robust to noise (short-circuiting).

### 2. Spectral Embedding

The map $\Psi_t(x) = (\lambda_1^t \psi_1(x), ..., \lambda_k^t \psi_k(x))$ embeds the data into a Euclidean space where:
$$ || \Psi*t(x) - \Psi_t(y) ||*{Euclidean} \approx D*t(x, y)*{Diffusion} $$

## Relevance to Project

**The Latent Space Regularizer.**

- We are building a VAE. The Latent Space is usually Gaussian (KL Divergence).
- **Proposal:** Force the VAE latent space to respect the **Diffusion Metric** of the input data (or the P-adic metric).
- If biological data lies on a sub-manifold, standard Gaussian VAEs destroy this topology. A "Diffusion VAE" (or Geometric VAE) preserves it.
