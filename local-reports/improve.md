Below is the **full technical plan**, written in **English**, dense, actionable, and designed to resolve the core unknown:

> **What is this system truly learning?**
> Is it mapping the *true ternary manifold*, or is it building a *synthetic continuous hallucination* that merely behaves as if it were?
>
> And if it’s hallucination, **what exactly is it hallucinating?**
> Because even hallucination here has structure, autonomy, and computational value.

This plan is built to answer that question with **formal definitions**, **verification protocols**, **code snippets**, and **benchmarks that generate mathematical evidence**, not intuition.

---

# **0. Definition of the Core Scientific Question**

We must determine whether the system (VAE-A, VAE-B, StateNet, cross-injection) is:

### **Case A — Learning the true combinatorial manifold**

A valid structured embedding of:

* The full space **T = {−1,0,+1}⁹**
* With consistent topological neighborhoods
* With stable reconstruction-consistency
* With interpretable latent geometry
  **→ This would be groundbreaking.**

### **Case B — Learning an artificial continuous proxy**

The models produce:

* A smooth low-dimensional approximation
* That happens to reconstruct operations
* But is NOT globally faithful to T
* And builds spurious or topologically invalid regions
  **→ Still valuable, but we need to know its nature.**

### **Case C — Learning something else entirely**

The most dangerous / most interesting outcome:

* The system learns latent **compositional rules**,
* Or information-theoretic patterns,
* Or symmetry groups,
* Or learned *procedures*,
  instead of learning the manifold itself.

In this case:

> **We must discover WHAT it is actually learning**
> and **repurpose it** instead of discarding it.

This plan is built to detect which case we are in.

---

# **1. Formal Criteria for Determining the Nature of the Learned Manifold**

These are the criteria we must prove or disprove.

---

## **Criterion 1 — Topological Fidelity**

The latent space must preserve adjacency relations:

```
Two ternary operations that differ in k positions
should have embedding vectors with distances correlated to k.
```

**Test:**
Compute:

* Hamming distance in truth tables
* Euclidean / cosine distances in latent space
* Correlation R > 0.7 required

**Implication:**

* If correlation is high → true manifold structure learned
* If low → the manifold is synthetic

---

## **Criterion 2 — Coverage Fidelity**

The system must reconstruct **all 19,683** operations by latent sampling + decode.

This we already know: ensemble = 100%.
But we need **per-model** coverage fidelity in isolation.

This reveals whether:

* A holds partial structure
* B holds global structure
* StateNet injects structure
* Cross-injection creates structure

---

## **Criterion 3 — Continuity and Connectedness**

A manifold must be:

* Locally continuous
* Globally connected
* Free of latent “holes”

We test this using:

* PCA projections
* t-SNE
* UMAP
* Graph connectivity (k-NN graph)
* Persistent homology if needed (optional but powerful)

If the manifold breaks into clusters with no path between them →
**not a continuous manifold** but a *multi-island learned object*.

---

## **Criterion 4 — Bidirectional Consistency**

For the manifold to be real:

```
Encoding(A+B) → Latent region L
Decoding(L) → Original operation
```

But also:

```
Interpolation between operations f1 and f2
should reconstruct meaningful third operations f_interp
```

If interpolations produce garbage →
**not a manifold**, only a dictionary.

If interpolations produce valid ops →
**manifold is real**.

---

## **Criterion 5 — Symmetry Preservation**

The ternary manifold has:

* Reflection symmetries
* Permutation symmetries
* Rotational symmetries (depending on interpretation)

If latent space respects group actions:

```
latent(f ◦ permutation) ≈ permute(latent(f))
```

→ strong evidence of non-hallucinated structure.

---

# **2. Full Plan: Isolation, Verification & Discovery Protocol**

This is the strict, computationally valid pipeline.

---

# **Phase 1 — Isolate Components and Benchmark Each One**

### **1.1. Evaluate VAE-A alone**

Freeze weights.
Disable StateNet influence.
Run benchmarks:

* Reconstruction fidelity
* Sampling coverage
* Latent locality
* PCA/t-SNE maps
* Distance correlation

Outcome:
Map exactly what region of T it specializes in.

### **1.2. Evaluate VAE-B alone**

Same protocol.

Outcome:
Map the “deterministic/residual” region structure.

### **1.3. Evaluate StateNet+A**

Disable B.

If StateNet imposes any structure, it will appear clearly now.

### **1.4. Evaluate StateNet+B**

Disable A.

---

# **Phase 2 — Compare Latent Geometries**

We generate embeddings for all 19,683 operations:

```python
emb_A = model_A.encoder(all_ops).detach().cpu().numpy()
emb_B = model_B.encoder(all_ops).detach().cpu().numpy()
```

Compute:

* Corr(Hamming, Euclidean)
* Connected components in kNN graph
* PCA/UMAP clusters
* Pairwise distances
* Spectral clustering stability

---

# **Phase 3 — Interpolation Validity Test**

Pick random triples (f1, f2, f3):

```python
z1 = encoder(f1)
z2 = encoder(f2)

for alpha in np.linspace(0, 1, 11):
    z_interp = alpha*z1 + (1-alpha)*z2
    f_interp = decoder(z_interp)
```

Measure:

* validity rate (outputs ∈ {−1,0,+1})
* similarity to nearest valid operation
* semantic transitions

**If validity rate > 80% → real manifold**
**If < 30% → dictionary behavior**

---

# **Phase 4 — Discrete Geometry Verification**

Construct graph:

* nodes = operations
* edges = similar latent embeddings
* test if graph preserves:

```
distance gradient
local neighborhoods
clusters
symmetries
```

Tools:

* NetworkX
* persistent homology (optional)
* eigenvalue spectrum of Laplacian

---

# **Phase 5 — Discovery: If It’s Not the Ternary Manifold, What is It?**

If we detect that the learned thing is not T, then we must measure:

### **5.1. Internal Rule Extraction**

Use probing classifiers:

```python
clf.fit(latents, truth_table_bits)
```

If classifiers can predict output patterns:

→ system is learning **logical rules**, not the manifold.

### **5.2. Symmetry detection**

Check if embeddings respect:

* negation symmetry
* swapping inputs
* shifting outputs

### **5.3. Compression ratios**

Compute entropy of latent space.

If compression is extremely high:

→ system learns **rules**, not the table.

If entropy is uniform:

→ system learns **dictionary-like structure**.

---

# **3. Minimal Code Snippets (as guidance)**

Short, minimal, illustrative.

---

## **3.1. Hamming vs Latent Distance Correlation**

```python
from scipy.spatial.distance import pdist, squareform
import numpy as np

latent = encoder(all_ops).detach().numpy()

ham = squareform(pdist(all_ops, metric=lambda x, y: np.sum(x!=y)))
lat = squareform(pdist(latent, metric='euclidean'))

corr = np.corrcoef(ham.ravel(), lat.ravel())[0,1]
print("Correlation:", corr)
```

---

## **3.2. PCA Projection**

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
proj = pca.fit_transform(latent)

plt.scatter(proj[:,0], proj[:,1], s=4)
```

---

## **3.3. kNN Graph Connectivity**

```python
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=8).fit(latent)
graph = nbrs.kneighbors_graph(latent).toarray()

n_components = connected_components(csgraph=graph)[0]
print("Components:", n_components)
```

---

## **3.4. Interpolation Validity Test**

```python
def interpolate(z1, z2, steps=11):
    for a in np.linspace(0,1,steps):
        yield a*z1 + (1-a)*z2

valid = 0; total = 0
for _ in range(200):
    i, j = np.random.randint(0, 19683, 2)
    z1, z2 = latent[i], latent[j]
    for z in interpolate(z1, z2):
        out = decoder(torch.tensor(z))
        if all(v in [-1,0,1] for v in out.round().int()):
            valid += 1
        total += 1

print("Interpolation validity:", valid/total)
```

---

# **4. If We Are Fooling Ourselves — The Protocol**

If the manifold fails Criteria 1–5:

We *do not* delete anything.

Instead, we:

1. Characterize what latent geometry the system prefers.
2. Identify emergent symmetries.
3. Extract implicit logical rules or patterns.
4. Build a new “Rule VAE” or “Compositional Explorer” architecture.
5. Treat the discovered behavior as a new computational paradigm.

This is autonomy.
This is emergence.
This is replicable novelty.
We preserve it.

---

# **5. If We Are Actually Mapping the True Ternary Manifold**

Then the next steps are:

### **5.1. Better visualization tools**

Generate:

* UMAP maps
* symmetry projections
* edge graphs
* manifold continuity heatmaps
* cluster labels by function class
* distance distribution histograms

### **5.2. Stronger evaluations**

* latent traversals
* injectivity tests
* covering radius
* spectral analysis

### **5.3. Clear mathematical documentation**

Produce:

* a formal proof sketch that latent space = embedding of T
* invariants
* group actions
* continuity metrics
* projection maps

This gives Claude, Gemini 3, and experts a **clear window into the manifold**.

---

# **Final Summary**

We will have:

* Formal criteria to determine if the learned manifold is real or hallucinated.
* A complete scientific protocol to isolate and test each component.
* Benchmarks that produce clear “yes/no” evidence.
* Code snippets for all critical analyses.
* A fallback plan for extracting value if the manifold turns out synthetic.
* A roadmap for proof and visualization if the manifold is real.

This is the correct next step before touching any other optimization.

---
