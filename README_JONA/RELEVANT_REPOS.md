# Relevant Repositories for Inspiration

These projects solve similar problems. We can learn from their architecture (or steal their code).

## 1. Hyperbolic Deep Learning

- **[HazyResearch/hgcn](https://github.com/HazyResearch/hgcn)** (Stanford)

  - **What:** Hyperbolic Graph Convolutional Networks.
  - **Lesson:** How to handle graph data in hyperbolic space. Perfect for your "StateNet" controller.

- **[facebookresearch/poincare-embeddings](https://github.com/facebookresearch/poincare-embeddings)**
  - **What:** The original paper code.
  - **Lesson:** C++ optimized sampling for massive datasets.

## 2. VAEs & Disentanglement

- **[google-research/disentanglement_lib](https://github.com/google-research/disentanglement_lib)**
  - **What:** The gold standard for VAE metrics.
  - **Lesson:** Implement their "FactorVAE" metric to prove your dimensions are truly independent.

## 3. Bioinformatics

- **[theislab/scvelo](https://github.com/theislab/scvelo)**
  - **What:** RNA Velocity (predicting cell future state).
  - **Lesson:** They use dynamical systems to predict cell trajectories. Your "Ternary Operator" is a discrete version of this.
