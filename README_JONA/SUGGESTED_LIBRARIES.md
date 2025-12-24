# Suggested Libraries for Termary VAEs

Based on the v5.11 codebase analysis, these libraries could significantly accelerate development and improve performance.

## 1. Hyperbolic Geometry & Optimization

Currently, the project implements hyperbolic projections manually in `src/models/layers.py`.
**Recommendation:** Switch to specialized libraries for numerical stability and speed.

- **[Geoopt](https://github.com/geoopt/geoopt)**

  - **Why:** It's "PyTorch on Manifolds". It creates tensors that _know_ they are in hyperbolic space.
  - **Feature:** Riemannian Adam (RADAM) optimizer included.
  - **Impact:** Likely 10-20% faster training and fewer "NaN values" in the Poincaré ball.

- **[HypTorch](https://github.com/leymir/hyperbolic-image-embeddings)**
  - **Why:** optimized specifically for Hyperbolic VAEs.
  - **Feature:** Implements the "Wrapped Normal Distribution" (crucial for VAE sampling in hyperbolic space).

## 2. Biological Data Handling

Currently, you load sequences as text.
**Recommendation:** Use standard bio-formats to integrate with public datasets.

- **[Biopython](https://biopython.org/)**

  - **Why:** The industry standard.
  - **Use Case:** Parsing `.fasta` files from GISAID (COVID) or NCBI (HIV) directly.

- **[Scanpy](https://scanpy.readthedocs.io/)**
  - **Why:** If you move to Single-Cell RNA seq.
  - **Use Case:** Visualizing 10,000+ cells. Your hyperbolic embedding could be a plugin for Scanpy.

## 3. Visualization

- **[HiPlot](https://github.com/facebookresearch/hiplot)**

  - **Why:** High-dimensional parallel coordinates.
  - **Use Case:** Visualizing the 16-dimensional latent vectors better than PCA.

- **[Manifold-SNE](https://github.com/.../manifold-sne)**
  - **Why:** t-SNE is Euclidean. You need Hyperbolic t-SNE to see the "Tree Structure" correctly.
