# V5 Expansion: The Arrow Flip Hypothesis

**Doc-Type:** Research Design · Version 1.0 · Updated 2026-01-03 · AI Whisperers

---

## Executive Summary

This document defines the research program to determine **where and why** codon sequence analysis fails to predict protein function, and where our 3-adic/hyperbolic embedding approach provides additional predictive power.

**The Central Question:**

> At what point does the "information arrow" flip from sequence → function to requiring geometric/algebraic structure to make accurate predictions?

---

## Background: The Information Hierarchy

### Standard Central Dogma View

```
DNA → mRNA → Protein → Structure → Function
     (transcription) (translation) (folding) (activity)
```

Each arrow represents information loss. The genetic code is a **buffer** against this loss.

### Our Refined View

```
Level 0: Codon Sequence (64 codons, 3^3 × 7 = 63 + stop)
         ↓ [GENETIC CODE DEGENERACY]
Level 1: Amino Acid Sequence (20 AAs)
         ↓ [HYBRID GROUPOID captures this]
Level 2: Local Functional Properties (charge, size, hydrophobicity)
         ↓ [POSITION-SPECIFIC CONTEXT needed]
Level 3: Secondary Structure (helix, sheet, coil)
         ↓ [LONG-RANGE CONTACTS needed]
Level 4: Tertiary Structure (3D fold)
         ↓ [DYNAMICS needed]
Level 5: Function (catalysis, binding, signaling)
```

### The Arrow Flip Points

**Arrow Flip 1: Sequence → Local Function**
- Standard sequence features: ~60% accuracy
- Hybrid groupoid: ~79% accuracy (AUC)
- **P-adic/hyperbolic embeddings ADD VALUE**

**Arrow Flip 2: Local Function → 3D Structure**
- Sequence alone: Poor (requires AlphaFold-scale models)
- Our embeddings: Unknown (to be tested)
- **Expected: Embeddings may NOT add significant value**

**Arrow Flip 3: 3D Structure → Catalytic Function**
- Structure alone: Moderate (active site geometry)
- Our embeddings + structure: Unknown
- **Hypothesis: Embeddings capture substitutability at active sites**

---

## Research Hypotheses

### H1: Codon Degeneracy Encodes Functional Robustness

**Statement:** The genetic code's degeneracy is not random but optimized so that synonymous and near-synonymous codons map to functionally similar amino acids.

**Test:**
- Compare codon Hamming distance vs amino acid functional distance
- Check if single-nucleotide mutations preferentially lead to functionally similar AAs

**Expected outcome:** Strong correlation between codon proximity and functional similarity.

### H2: The P-adic Valuation Encodes Information Resilience

**Statement:** Higher 3-adic valuation codons are more "information-resilient" (less affected by common mutations).

**Test:**
- Map mutation types (transitions, transversions) to valuation changes
- Check if high-valuation codons are more stable under mutation pressure

**Expected outcome:** High-valuation codons have lower functional disruption under mutation.

### H3: Hyperbolic Radius Encodes Functional Centrality

**Statement:** Amino acids closer to the center of the Poincaré ball are more "central" to biological function (appear in more enzyme classes).

**Test:**
- Correlate hyperbolic radius with EC class membership count
- Check if core residues (central) vs peripheral residues differ in function breadth

**Expected outcome:** Lower radius = higher functional versatility.

### H4: The Prediction Boundary is Context-Dependent

**Statement:** Our embeddings add value for ISOLATED substitutions but context (neighboring residues, secondary structure) modifies the prediction boundary.

**Test:**
- Compare embedding-based predictions for:
  a) Isolated mutations (single AA changes)
  b) Motif-embedded mutations (mutations within conserved motifs)
  c) Active site mutations (mutations at catalytic positions)

**Expected outcome:**
- (a) Strong embedding advantage
- (b) Moderate embedding advantage (context reduces uncertainty)
- (c) Variable (EC-class dependent)

### H5: Multi-Scale Composition Reveals Hierarchical Structure

**Statement:** If the groupoid structure is fundamental, it should compose across scales:
- Codon → AA (current)
- AA → Motif (sequences of AAs)
- Motif → Domain
- Domain → Protein

**Test:** Build groupoids at each scale and check if morphisms compose correctly.

**Expected outcome:** Hierarchical groupoid structure mirrors protein organization.

---

## Experimental Design

### Experiment 1: Codon-Level Functional Mapping

**Objective:** Map the full 64-codon space to functional outcomes.

**Method:**
```python
for codon in all_64_codons:
    aa = translate(codon)
    valuation = padic_valuation(codon, p=3)
    functional_profile = get_aa_profile(aa)

    # For each neighboring codon (Hamming distance 1)
    for neighbor in hamming_neighbors(codon):
        neighbor_aa = translate(neighbor)
        functional_change = distance(functional_profile, get_aa_profile(neighbor_aa))

        record(codon, neighbor, valuation_change, functional_change)
```

**Metrics:**
- Correlation: valuation_change vs functional_change
- Mutation type analysis: transitions vs transversions
- Synonymous vs non-synonymous effects

### Experiment 2: Position-Specific Prediction Boundary

**Objective:** Determine where in a protein our embeddings add most value.

**Method:**
```python
for protein in test_set:
    for position in protein:
        # Get position context
        ss = secondary_structure(position)
        burial = solvent_accessibility(position)
        conservation = conservation_score(position)

        for mutation in all_possible_mutations(position):
            # Baseline predictions
            seq_pred = sequence_only_predictor(mutation)

            # Our predictions
            emb_pred = embedding_predictor(mutation)

            # Ground truth
            effect = experimental_effect(mutation)

            record(position_context, seq_pred, emb_pred, effect)
```

**Analysis:**
- Stratify by: buried/exposed, helix/sheet/coil, conserved/variable
- Compute AUC improvement by category
- Identify the "flip point" where embeddings stop helping

### Experiment 3: Active Site vs Non-Active Site

**Objective:** Test if EC class affects prediction boundary.

**Method:**
```python
for enzyme in EC_annotated_proteins:
    active_site_residues = get_active_site(enzyme)
    non_active_residues = get_non_active(enzyme)

    for residue_set, label in [(active_site_residues, 'active'),
                                (non_active_residues, 'non_active')]:
        for position in residue_set:
            for mutation in all_mutations(position):
                seq_pred = sequence_predictor(mutation)
                emb_pred = embedding_predictor(mutation)
                effect = experimental_effect(mutation)

                record(enzyme.ec_class, label, seq_pred, emb_pred, effect)
```

**Hypothesis:** Active sites in EC1 (oxidoreductases) show DIFFERENT patterns than EC6 (ligases).

### Experiment 4: Motif Disruption Analysis

**Objective:** Test if conserved motifs have different prediction rules.

**Method:**
```python
for motif in PROSITE_motifs:
    motif_positions = get_positions(motif)
    consensus = get_consensus(motif)

    for mutation in mutations_within_motif:
        # Is mutation at a conserved position?
        conservation_penalty = position_conservation(mutation.position)

        # Our embedding prediction
        emb_pred = embedding_predictor(mutation)

        # Effect on motif function
        effect = motif_function_effect(mutation)

        # Does embedding prediction match conservation better?
        compare(emb_pred, conservation_penalty, effect)
```

**Expected:** Embeddings add value BEYOND simple conservation by capturing functional substitutability.

### Experiment 5: Comparative Embedding Analysis

**Objective:** Is the 3-adic structure essential, or do arbitrary embeddings work equally well?

**Method:**
```python
embeddings_to_test = {
    'padic_hyperbolic': our_trained_model,
    'random_16d': random_orthogonal_embedding(),
    'pca_physicochemical': pca_on_aa_properties(),
    'esm2_per_position': esm2_embeddings(),
    'blosum_derived': embedding_from_blosum62(),
}

for name, embedding in embeddings_to_test.items():
    # Run V5 validation
    h1_auc = validate_h1(embedding)
    h2_spearman = validate_h2(embedding)
    h3_ari = validate_h3(embedding)
    h4_auc = validate_h4(embedding)

    record(name, h1_auc, h2_spearman, h3_ari, h4_auc)
```

**Critical question:** Does our p-adic structure outperform random embeddings of the same dimension?

---

## Data Requirements

### Essential Datasets

| Dataset | Purpose | Source | Status |
|---------|---------|--------|--------|
| ENZYME | EC annotations | ExPASy | Need to download |
| UniProt-SwissProt | Active site annotations | UniProt | Need to download |
| PROSITE | Motif definitions | ExPASy | Need to download |
| S669 | DDG effects | ProThermDB | Downloaded |
| ClinVar | Pathogenic variants | NCBI | Need to download |
| PDB | 3D structures | RCSB | Need API access |
| ESM2 | Pre-trained embeddings | Meta | Need to integrate |

### Derived Datasets to Create

1. **Codon Functional Distance Matrix (64×64)**
   - Each entry: functional distance between AA pairs encoded by codons
   - Include valuation differences

2. **Position-Specific Effect Dataset**
   - Protein, position, mutation, context features, effect
   - Stratified by secondary structure, burial, EC class

3. **Motif Mutation Database**
   - PROSITE motif, position, mutation, motif function retained?

---

## Success Criteria

### For Each Hypothesis

| Hypothesis | Metric | Minimum | Target | Stretch |
|------------|--------|---------|--------|---------|
| H1: Codon→Function | Spearman r | 0.3 | 0.5 | 0.7 |
| H2: Valuation→Resilience | % high-val preserved | 60% | 75% | 90% |
| H3: Radius→Centrality | Spearman r | -0.3 | -0.5 | -0.7 |
| H4: Context boundary | AUC improvement | +5% | +10% | +15% |
| H5: Multi-scale composition | Composition holds | 60% | 80% | 95% |

### Overall Arrow Flip Characterization

**Success = we can precisely define:**
1. The sequence length / context size where embeddings add value
2. The protein regions (active site, surface, core) where embeddings help most
3. The enzyme classes where p-adic structure is most informative
4. The mutation types (radical, conservative) where prediction improves

---

## Implementation Plan

### Phase 1: Foundation (This Session)

1. Create codon→function mapping
2. Build position-specific dataset from S669
3. Implement comparative embedding framework

### Phase 2: Boundary Detection

1. Run position-specific experiments
2. Stratify results by context
3. Identify the "flip points"

### Phase 3: Multi-Scale Extension

1. Build motif-level groupoid
2. Test domain-level predictions
3. Validate hierarchical composition

### Phase 4: Model Improvement

1. Train p-adic model with functional supervision
2. Compare hyperbolic vs Euclidean geometry
3. Optimize for EC-class prediction

---

## Theoretical Framework

### The Substitutability Principle

> Two amino acids are functionally equivalent if and only if they can substitute for each other in a protein without changing function.

This is **context-dependent**: A→V might be fine in a hydrophobic core but devastating in a catalytic site.

### The Groupoid Captures Intrinsic Substitutability

The hybrid groupoid captures **intrinsic** (context-free) substitutability:
- Based on physical properties (charge, size, hydrophobicity)
- Based on learned embeddings (codon structure)
- Independent of position in protein

### The Arrow Flip Occurs When Context Dominates

**Before flip:** Intrinsic properties dominate → embeddings help
**After flip:** Context dominates → need position-specific models (like AlphaFold)

### Mathematical Formulation

Let:
- S(a,b) = intrinsic substitutability from groupoid
- C(p) = context factor at position p
- F(a→b, p) = functional effect of substituting a with b at position p

Then:
```
F(a→b, p) = S(a,b) × C(p) + ε
```

**The arrow flips when:**
```
Var(C(p)) >> Var(S(a,b))
```

Our goal is to find the boundary where context variance exceeds intrinsic variance.

---

## Connection to Existing Work

### V5 Results Inform This Direction

From V5, we know:
- Intrinsic substitutability correlates with function (r = 0.569)
- 5/6 enzyme classes show groupoid encoding
- EC1 (oxidoreductases) is the exception

### TEGB Falsification is Consistent

The finding that p-adic anti-correlates with thermodynamics:
- Confirms: p-adic encodes information resilience, not energy
- Implies: stability prediction requires different features

### DDG Weak Correlation is Expected

The r = 0.04 DDG correlation:
- Confirms: groupoid captures substitutability, not stability
- Implies: stability is a Level 4 property, beyond groupoid scope

---

## Expected Outcomes

### Scenario 1: Strong Arrow Flip (Optimistic)

- Clear boundary at specific context radius
- Embeddings add >15% AUC for isolated mutations
- Multi-scale composition holds
- Can define "p-adic scope" precisely

**Implication:** P-adic structure is fundamental to genetic code organization.

### Scenario 2: Diffuse Boundary (Realistic)

- Context effects smooth, no sharp boundary
- Embeddings add 5-10% AUC across conditions
- Multi-scale composition partial
- Context and intrinsic factors interact

**Implication:** P-adic structure is one of several organizing principles.

### Scenario 3: Weak Arrow Flip (Conservative)

- Context dominates almost everywhere
- Embeddings add <5% AUC
- Multi-scale composition fails
- Groupoid captures evolutionary, not functional, signal

**Implication:** Need to revise framework; p-adic may be more about evolution than function.

---

## Appendix: Key Definitions

### 3-adic Valuation

For codon represented as integer n:
```
v_3(n) = max{k : 3^k divides n}
```
High valuation = more divisible by 3 = more "central" in p-adic space.

### Hyperbolic Radius

In Poincaré ball model:
```
d_hyp(x, 0) = 2 × arctanh(||x||)
```
Points near boundary have high radius; points near center have low radius.

### Functional Distance

Euclidean distance in 24-dimensional functional profile space:
```
d_func(a, b) = ||profile(a) - profile(b)||_2
```

### Groupoid Path Cost

Sum of morphism costs along shortest path:
```
cost(a→b) = Σ c(m) for m in shortest_path(a, b)
```

---

**End of Design Document**

*This document defines the research program for V5 Expansion: The Arrow Flip Hypothesis.*
