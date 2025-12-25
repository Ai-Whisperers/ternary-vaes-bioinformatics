# Glossary

Definitions of key terms used in Ternary VAE.

---

## A

### Autoencoder
A neural network that learns to compress data into a lower-dimensional representation (encoding) and then reconstruct it (decoding). The compressed representation is called the **latent space**.

### AdS/CFT Correspondence
A theoretical physics concept relating gravity in Anti-de Sitter space to conformal field theory on its boundary. Used in holographic extensions of the Poincare ball.

---

## B

### β-VAE
A variant of VAE where the KL divergence is scaled by β > 1, encouraging more disentangled representations. Often combined with annealing (starting β=0, increasing over time).

### Batch Size
Number of samples processed together in one forward/backward pass. Larger batches = more stable gradients but more memory.

---

## C

### Codon
A sequence of three nucleotides (e.g., AUG, GCA) that encodes a single amino acid or stop signal. There are 64 possible codons encoding 20 amino acids.

### Codon Optimization
Selecting synonymous codons (different codons encoding the same amino acid) to improve expression, stability, or other properties.

### Conformal Factor
In the Poincare ball, the factor λ(x) = 2/(1-κ||x||²) that scales the metric. Grows unboundedly as points approach the boundary.

### Curvature (κ)
A parameter controlling how "curved" hyperbolic space is. κ=0 gives Euclidean space; κ>0 gives hyperbolic space. Higher values = more hierarchical capacity.

---

## D

### Decoder
The neural network component that reconstructs input from latent representations. In TernaryVAE, maps from Poincare ball to 19,683-way softmax.

### Degeneracy (Genetic)
The property that multiple codons can encode the same amino acid. For example, GCU, GCC, GCA, GCG all encode Alanine.

---

## E

### ELBO (Evidence Lower Bound)
The objective function for VAEs: ELBO = E[log p(x|z)] - KL(q(z|x) || p(z)). Maximizing ELBO approximately maximizes log p(x).

### Encoder
The neural network component that maps input to latent space parameters (μ and σ in VAEs).

### Exponential Map (exp_map)
In differential geometry, maps from tangent space to the manifold. `exp_map_zero` maps from Euclidean R^n to the Poincare ball.

---

## F

### Free Bits
A technique to prevent posterior collapse by giving the model "free" KL capacity: KL_effective = max(KL - free_bits, 0).

---

## G

### Geodesic
The shortest path between two points on a curved surface. In the Poincare ball, geodesics are circular arcs perpendicular to the boundary.

### geoopt
A PyTorch library for optimization on Riemannian manifolds. Provides `PoincareBall` manifold and Riemannian optimizers.

### Glycan Shield
A layer of sugar molecules (glycans) on viral surfaces that can block antibody access. Important for vaccine design.

---

## H

### Homeostasis Controller
An adaptive mechanism that adjusts the β parameter (KL weight) to maintain target KL divergence during training.

### Hyperbolic Geometry
A non-Euclidean geometry with constant negative curvature. Trees embed with less distortion than in Euclidean space.

### Hyperbolic Space
A space with hyperbolic geometry. The Poincare ball is one model of 2D or higher-dimensional hyperbolic space.

---

## K

### KL Divergence
Kullback-Leibler divergence measures the difference between two probability distributions. In VAEs, measures how far the learned posterior q(z|x) is from the prior p(z).

---

## L

### Latent Dimension
The number of dimensions in the latent space. Higher dimensions = more capacity but harder to train.

### Latent Space
The compressed representation learned by an autoencoder. In TernaryVAE, this is the Poincare ball.

### Logarithmic Map (log_map)
Inverse of the exponential map. Maps from the manifold back to tangent space.

### Loss Registry
A design pattern for composing multiple loss functions dynamically without subclassing.

---

## M

### Manifold
A curved space that locally looks like Euclidean space. The Poincare ball is a manifold.

### max_radius
The maximum allowed norm for points in the Poincare ball (default 0.95). Prevents numerical instability near the boundary.

### Mobius Addition (⊕)
The addition operation in hyperbolic space. Unlike Euclidean addition, respects the curved geometry.

### mRNA
Messenger RNA, a molecule that carries genetic information from DNA to ribosomes for protein synthesis.

---

## N

### Nucleotide
The building blocks of DNA/RNA: Adenine (A), Guanine (G), Cytosine (C), Thymine/Uracil (T/U).

---

## P

### Parallel Transport
Moving a vector along a curve while keeping it "parallel" in curved space. Important for comparing gradients at different points.

### Phylogenetic Tree
A branching diagram showing evolutionary relationships between species or sequences.

### Poincare Ball
A model of hyperbolic geometry: the open unit ball with a metric that makes distances grow near the boundary.

### Poincare Distance
The distance function in the Poincare ball: d(x,y) = (2/√κ) * arctanh(√κ * ||(-x)⊕y||).

### Posterior Collapse
A failure mode where the VAE ignores the latent space, making q(z|x) = p(z) for all x. Prevented by free bits, β-annealing, etc.

### Prior (p(z))
The assumed distribution of latent variables before seeing data. Usually standard Gaussian N(0,I) or wrapped Gaussian in hyperbolic space.

### p-adic Numbers
An alternative number system where "closeness" is measured by divisibility by prime p. 3-adic numbers are used for ternary codon structure.

---

## R

### Radial Stratification
Organizing latent representations by distance from origin, with higher p-adic valuations (more divisible by 3) closer to center.

### Reconstruction Loss
The loss measuring how well the decoder reconstructs the input. Usually cross-entropy for classification.

### Reparameterization Trick
A technique enabling backpropagation through stochastic sampling: z = μ + σ * ε, where ε ~ N(0,1).

### Riemannian Gradient
The gradient on a curved manifold, computed by projecting Euclidean gradient onto tangent space.

### Riemannian Optimizer
An optimizer that respects manifold geometry, using Riemannian gradients and exponential maps for updates.

---

## S

### Softmax
A function that converts logits to probabilities summing to 1. Output layer for 19,683-way classification.

### SwarmVAE
A multi-agent VAE architecture where multiple "agents" explore latent space collaboratively.

---

## T

### Tangent Space
The flat space of possible directions at a point on a manifold. Euclidean operations happen in tangent space, then project to manifold.

### Ternary Operations
The 3^9 = 19,683 possible combinations in our codon encoding scheme.

### tRNA
Transfer RNA, molecules that bring amino acids to ribosomes during protein synthesis. Codon optimization considers tRNA abundance.

---

## V

### Valuation (p-adic)
The function v_p(n) counting how many times prime p divides n. For 3-adic: v_3(9) = 2 because 9 = 3².

### VAE (Variational Autoencoder)
An autoencoder with a probabilistic latent space, trained to maximize the ELBO. Enables generation and structured representations.

### Variational Inference
Approximating intractable probability distributions by optimizing over a family of tractable distributions.

---

## W

### Warmup
Gradually increasing a parameter (learning rate, β) at the start of training to improve stability.

### Weight Decay
A regularization technique adding λ||θ||² to the loss, preventing weights from growing too large.

---

## See Also

- [[FAQ]] - Common questions
- [[Geometry]] - Hyperbolic geometry details
- [[Architecture]] - System overview

---

*Missing a term? [Open an issue](https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics/issues) to suggest additions.*
