# Frequently Asked Questions

Common questions about Ternary VAE and bioinformatics applications.

---

## General Questions

### What is Ternary VAE?

Ternary VAE is a variational autoencoder that uses **hyperbolic geometry** (Poincare ball) and **3-adic number theory** to learn representations of biological sequences, particularly codon operations.

**Key innovations**:
- Latent space lives in hyperbolic space (better for hierarchical data)
- 3-adic structure encodes ternary codon operations (3^9 = 19,683 total)
- Designed for vaccine design, drug discovery, and codon optimization

### Why "Ternary"?

The genetic code uses **codons** (3-nucleotide sequences). Each nucleotide position has 3 choices in our encoding, leading to:
- 3 positions × 3 choices = 3^3 per codon
- Extended to 3^9 = 19,683 ternary operations

This ternary structure naturally maps to **3-adic numbers**, a mathematical system where divisibility by 3 creates hierarchical structure.

### Why hyperbolic geometry?

Biological data is inherently **hierarchical**:
- Phylogenetic trees (evolutionary relationships)
- Protein domain hierarchies
- Taxonomic classifications

**Hyperbolic space** can embed trees with **exponentially less distortion** than Euclidean space. This makes learned representations more meaningful for biological applications.

### What can I use Ternary VAE for?

| Application | Description |
|-------------|-------------|
| **Vaccine Design** | Optimize antigen sequences for immune response |
| **Codon Optimization** | Improve mRNA stability and expression |
| **Drug Discovery** | Model molecular interactions |
| **Phylogenetics** | Learn evolutionary embeddings |
| **Synthetic Biology** | Design novel sequences |

---

## Technical Questions

### What is the Poincare ball?

The Poincare ball is a model of **hyperbolic geometry**. It's the open unit ball (all points with norm < 1) equipped with a special metric that makes distances grow exponentially near the boundary.

```
Euclidean disk:     Poincare ball:
    ___                 ___
   /   \               /   \
  |  o  |             |  o  |  <- Center = "root"
  |     |             | . . |
   \___/               \.../  <- Boundary = "leaves" (infinitely far)
```

**Key property**: A tree with n leaves needs O(n) space in hyperbolic geometry but O(n²) in Euclidean space.

### What are p-adic numbers?

**p-adic numbers** are an alternative number system where closeness is measured by divisibility by prime p.

For **3-adic** (p=3):
- 9 is "closer" to 0 than 1 (because 9 = 3²)
- 27 is even "closer" to 0 (because 27 = 3³)

**In Ternary VAE**: Operations divisible by higher powers of 3 should be closer to the center of the Poincare ball, creating natural hierarchy.

### What is the "valuation"?

The **3-adic valuation** v₃(n) counts how many times 3 divides n:
- v₃(1) = 0 (3⁰ divides 1)
- v₃(3) = 1 (3¹ divides 3)
- v₃(9) = 2 (3² divides 9)
- v₃(27) = 3 (3³ divides 27)

This valuation determines where points should lie radially in the Poincare ball.

### What is "curvature" in the config?

**Curvature** (κ) controls how "hyperbolic" the space is:
- κ = 0: Flat Euclidean space
- κ > 0: Hyperbolic space
- Higher κ = more curvature = more hierarchical capacity

**Default**: κ = 1.0

**Tuning**: Use higher curvature (1.5-2.0) for deeply hierarchical data, lower (0.5-1.0) for shallower structures.

### What is "max_radius"?

Points in the Poincare ball must have norm < 1. **max_radius** (default 0.95) prevents points from getting too close to the boundary where:
- Gradients explode
- Numerical precision fails
- Training becomes unstable

### What are "free bits"?

**Free bits** is a technique to prevent **posterior collapse** (when the VAE ignores the latent space).

```python
KL = max(KL_raw - free_bits, 0)
```

This gives the model "free" capacity in the latent space before the KL penalty kicks in.

**Default**: 0.5 bits per dimension

---

## Training Questions

### Why is my loss NaN?

Common causes:

1. **Learning rate too high**
   ```python
   # Fix: Reduce learning rate
   optimizer = RiemannianAdam(model.parameters(), lr=1e-4)  # Was 1e-3
   ```

2. **Points escaping Poincare ball**
   ```python
   # Fix: Reduce max_radius
   config = TrainingConfig(geometry={"max_radius": 0.9})  # Was 0.95
   ```

3. **Gradient explosion**
   ```python
   # Fix: Add gradient clipping
   torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   ```

See [[Troubleshooting]] for more solutions.

### Why is reconstruction accuracy low?

1. **Posterior collapse**: The model ignores latent space
   - Check if KL divergence is near 0
   - Solution: Use free bits, reduce KL weight, or use β-VAE warmup

2. **Latent dimension too small**
   - Try increasing `latent_dim` (16 → 32 → 64)

3. **Not enough training**
   - VAEs often need 100+ epochs to converge

### How do I prevent posterior collapse?

1. **Free bits**: `KLDivergenceLossComponent(free_bits=0.5)`
2. **β-VAE warmup**: Start with β=0, increase gradually
3. **Higher latent dim**: More capacity prevents collapse
4. **Homeostasis controller**: Adaptive β based on KL

### What batch size should I use?

| GPU Memory | Recommended Batch Size |
|------------|----------------------|
| 4 GB | 16-32 |
| 8 GB | 32-64 |
| 16 GB | 64-128 |
| 24 GB+ | 128-256 |

**Note**: Larger batches need lower learning rates.

### How many epochs do I need?

| Task | Typical Epochs |
|------|---------------|
| Overfitting test | 10-50 |
| Quick experiment | 100-200 |
| Full training | 500-1000 |
| Publication-quality | 1000+ |

Use early stopping to avoid wasting compute.

---

## Biology Questions

### How does this help vaccine design?

1. **Antigen optimization**: Learn which sequence variations improve immune response
2. **Escape prediction**: Model how viruses might mutate to evade vaccines
3. **Conserved region identification**: Find stable targets across variants
4. **mRNA optimization**: Improve codon usage for expression

### What is codon optimization?

**Codons** are 3-nucleotide sequences encoding amino acids. Multiple codons can encode the same amino acid (degeneracy).

**Codon optimization** selects synonymous codons to:
- Match host tRNA abundance
- Avoid rare codons that slow translation
- Improve mRNA stability
- Reduce immunogenicity

Ternary VAE learns which codon choices work best.

### Can I use this for protein structure prediction?

Ternary VAE focuses on **sequence-level** representations. For structure prediction, consider:
- AlphaFold
- ESMFold
- Our `GeometricVectorPerceptron` encoder (for 3D-aware encoding)

### What organisms are supported?

The framework is organism-agnostic. Codon usage tables are available for:
- Human
- E. coli
- Yeast
- Many others

See `src/losses/autoimmunity.py` for human codon bias tables.

---

## Licensing Questions

### Can I use this commercially?

**Code**: PolyForm Non-Commercial 1.0.0
- Academic/research: Yes
- Non-profit: Yes
- Commercial: Requires separate license

**Research outputs** (models, data, figures): CC-BY-4.0
- Any use with attribution: Yes

Contact: `support@aiwhisperers.com` for commercial licensing.

### How do I cite this project?

```bibtex
@software{ternary_vae,
  author = {{AI Whisperers}},
  title = {Ternary VAE Bioinformatics},
  year = {2025},
  url = {https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics}
}
```

### Can I contribute?

Yes! See [[Contributing-Guide]] for:
- Code style requirements
- CLA (Contributor License Agreement)
- Pull request process

---

## Still Have Questions?

- **GitHub Issues**: [Open an issue](https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics/discussions)
- **Email**: `support@aiwhisperers.com`

---

*See also: [[Troubleshooting]], [[Glossary]]*
