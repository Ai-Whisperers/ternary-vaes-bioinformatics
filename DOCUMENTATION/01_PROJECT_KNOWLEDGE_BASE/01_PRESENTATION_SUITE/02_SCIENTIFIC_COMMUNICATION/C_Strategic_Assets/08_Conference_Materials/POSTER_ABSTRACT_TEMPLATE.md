# Conference Abstract Template

**Target**: ISMB / NeurIPS (AI for Science)
**Type**: Poster / Lightning Talk

## Title

**Hyperbolic Geometry as a Universal Prior for Generative Biology**

## Abstract

Generative models in biology often struggle with the discrete, hierarchical nature of genetic code, leading to "hallucinations" of non-functional proteins. We introduce a novel **Ternary Variational Autoencoder** that enforces a hyperbolic (p-adic) prior on the latent space. Unlike Euclidean models (e.g., VAEs with Gaussian priors), our model respects the fractal branching structure of evolution. We demonstrate that the **Hyperbolic Radius** of an embedding serves as an unsupervised metric for evolutionary fitness, successfully predicting viral escape mutants in HIV-1 and auto-antigen risks in Rheumatoid Arthritis. This "Geometric Prior" reduces the data requirement for training by 10x compared to baseline transformers, suggesting a path toward more data-efficient generative biology.

## Key Figures

1.  **The Manifold**: Visual of the latent space separation.
2.  **The Benchmark**: Bar chart showing superior "Dead Code" rejection vs ESM-1b.
3.  **The Application**: Heatmap of predicted escape mutations.
