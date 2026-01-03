# Multimodal Integration: Expectation Matrix

**Doc-Type:** Research Planning · Version 1.0 · Updated 2026-01-03 · AI Whisperers

---

## Purpose

This document establishes baseline expectations for multimodal integration of:
1. **P-adic Codon Embeddings** (TrainableCodonEncoder)
2. **ESM-2 Contextual Embeddings** (Protein Language Model)
3. **Structural Features** (AlphaFold, Contact Maps, DSSP)

The expectations serve as:
- **Reproducibility anchor** for future comparisons
- **Success criteria** for integration efforts
- **Audit trail** documenting pre-integration state

---

## Current Baseline (Pre-Integration)

### TrainableCodonEncoder Performance

| Task | Metric | Baseline Value | Dataset | Date |
|------|--------|----------------|---------|------|
| DDG Prediction | LOO Spearman | **0.61** | S669 (n=52) | 2026-01-03 |
| DDG Prediction | LOO Pearson | 0.64 | S669 (n=52) | 2026-01-03 |
| DDG Prediction | LOO MAE | 0.81 | S669 (n=52) | 2026-01-03 |
| DDG Prediction | LOO RMSE | 1.11 | S669 (n=52) | 2026-01-03 |
| P-adic Structure | Distance Correlation | 0.74 | 64 codons | 2026-01-03 |
| Contact Prediction | AUC-ROC | 0.67 | Insulin B-chain | 2026-01-03 |

### Architecture Baseline

```
TrainableCodonEncoder (current):
  Input:  12-dim one-hot (4 bases × 3 positions)
  Hidden: 64-dim MLP (2 layers, LayerNorm, SiLU, Dropout=0.1)
  Output: 16-dim Poincaré ball embedding
  Params: ~6K trainable parameters
```

### Checkpoint Reference

| Checkpoint | Purpose | Location |
|------------|---------|----------|
| trained_codon_encoder.pt | DDG prediction | research/codon-encoder/training/results/ |
| v5_11_structural | Contact prediction | sandbox-training/checkpoints/ |
| homeostatic_rich | High richness | sandbox-training/checkpoints/ |

---

## Integration Components

### Component 1: ESM-2 Embeddings

**Source:** Meta AI ESM-2 (esm2_t33_650M_UR50D recommended)

| Property | Value |
|----------|-------|
| Embedding dim | 1280 |
| Context window | 1024 tokens |
| Training data | UniRef50 (65M sequences) |
| Pre-training | Masked language modeling |

**Expected Contribution:**
- Evolutionary context (conservation, co-evolution)
- Secondary structure prediction
- Disorder prediction
- Binding site identification

### Component 2: Structural Features

**Sources:**
- AlphaFold2/3 predicted structures
- Contact maps (8Å threshold)
- DSSP secondary structure
- Solvent accessibility (RSA)
- B-factors (pLDDT from AlphaFold)

| Feature | Dimension | Source |
|---------|-----------|--------|
| Contact map | N×N binary | AlphaFold + 8Å |
| DSSP (3-state) | N×3 one-hot | DSSP |
| RSA | N×1 continuous | DSSP |
| pLDDT | N×1 continuous | AlphaFold |
| Distance matrix | N×N continuous | AlphaFold |

---

## Expectation Matrix: DDG Prediction

### Expected Performance Improvements

| Integration Level | Components | Expected LOO Spearman | Confidence | Rationale |
|-------------------|------------|----------------------|------------|-----------|
| **Baseline** | Codon only | 0.61 | Measured | Current TrainableCodonEncoder |
| **+ESM** | Codon + ESM-2 | 0.65-0.70 | Medium | ESM captures evolutionary conservation |
| **+Structure** | Codon + Contacts | 0.63-0.68 | Medium | Contacts add spatial context |
| **+Full** | Codon + ESM + Structure | 0.70-0.75 | High | Multi-source synergy |

### Literature Comparison Targets

| Method | Spearman | Type | Notes |
|--------|----------|------|-------|
| Rosetta ddg_monomer | 0.69 | Structure | Gold standard |
| ESM-1v | 0.51 | Sequence | Zero-shot |
| ThermoMPNN | 0.72 | Structure | State-of-art 2024 |
| **Target (ours)** | **0.70+** | **Hybrid** | **Codon + ESM + Structure** |

### Risk Factors

| Risk | Impact | Mitigation |
|------|--------|------------|
| Overfitting (small dataset) | High | LOO CV, regularization |
| ESM dominates codon signal | Medium | Gated fusion, attention weights |
| Structural features noisy | Medium | pLDDT filtering |
| Compute requirements | Low | ESM embeddings pre-computed |

---

## Expectation Matrix: Contact Prediction

### Expected Performance Improvements

| Integration Level | Components | Expected AUC-ROC | Confidence | Rationale |
|-------------------|------------|------------------|------------|-----------|
| **Baseline** | Codon pairwise | 0.67 | Measured | v5_11_structural checkpoint |
| **+ESM** | Codon + ESM coevolution | 0.72-0.78 | High | ESM attention = coevolution proxy |
| **+AlphaFold** | Codon + pLDDT | 0.75-0.80 | High | pLDDT predicts contacts |
| **+Full** | Codon + ESM + AlphaFold | 0.80-0.85 | Medium | Multi-source, but diminishing returns |

### Literature Comparison Targets

| Method | Precision@L | AUC | Notes |
|--------|-------------|-----|-------|
| AlphaFold2 contacts | 0.85+ | - | Structure-derived |
| ESM-2 attention | 0.70 | - | Attention heads |
| trRosetta | 0.65 | - | Co-evolution |
| **Target (ours)** | **0.75+** | **0.80+** | **Hyperbolic + ESM** |

---

## Expectation Matrix: Physics Invariants

### Force Constant Prediction (Existing)

| Integration Level | Components | Expected ρ | Confidence | Notes |
|-------------------|------------|------------|------------|-------|
| **Baseline** | Codon radial | 0.86 | Measured | k = r × m / 100 |
| **+ESM** | Codon + ESM | 0.86-0.88 | Low | Minimal expected gain |
| **+Structure** | Codon + B-factors | 0.88-0.92 | Medium | B-factors ∝ dynamics |

**Note:** P-adic structure already captures force constants well. ESM/structure may add marginal gains.

### Folding Kinetics

| Integration Level | Components | Expected ρ | Confidence | Notes |
|-------------------|------------|------------|------------|-------|
| **Baseline** | Property-based | 0.94 | Measured | Existing benchmark |
| **+ESM** | Codon + ESM disorder | 0.95-0.97 | Medium | ESM predicts disorder |
| **+Contact Order** | Codon + CO | 0.96-0.98 | High | Contact order ∝ folding rate |

---

## Integration Architecture Options

### Option A: Late Fusion (Recommended for Start)

```
Codon Encoder  →  16-dim  ─┐
                           ├→ Concat → MLP → Output
ESM-2 Encoder  → 128-dim  ─┤
                           │
Structure Enc  →  32-dim  ─┘

Total: 176-dim → MLP(176, 64, 1)
```

**Pros:** Simple, interpretable, preserves modality separation
**Cons:** May miss cross-modal interactions

### Option B: Cross-Attention Fusion

```
Codon Encoder  →  16-dim  ─┐
                           ├→ CrossAttention → Output
ESM-2 Encoder  → 128-dim  ─┘
                    ↑
Structure Enc  →  32-dim (as keys/values)
```

**Pros:** Learns cross-modal interactions
**Cons:** More complex, needs more data

### Option C: Hierarchical Fusion

```
Stage 1: Codon + ESM → Sequence Embedding (64-dim)
Stage 2: Sequence + Structure → Final Embedding (32-dim)
Stage 3: Final → Task-specific head
```

**Pros:** Respects information hierarchy
**Cons:** Multi-stage training complexity

---

## Success Criteria

### Minimum Viable Improvement

| Task | Baseline | Minimum Target | Stretch Target |
|------|----------|----------------|----------------|
| DDG (Spearman) | 0.61 | **0.65** | 0.70+ |
| Contact (AUC) | 0.67 | **0.72** | 0.80+ |
| Force k (ρ) | 0.86 | 0.87 | 0.90+ |

### Integration NOT Successful If:

1. DDG Spearman drops below 0.55 (regression)
2. Any component ablation shows no contribution
3. Overfitting ratio exceeds 1.5×
4. Training becomes unstable

---

## Experimental Plan

### Phase 1: ESM Integration (Priority)

1. Extract ESM-2 embeddings for S669 sequences
2. Implement late fusion with TrainableCodonEncoder
3. Evaluate on DDG with LOO CV
4. Ablation: Codon-only vs ESM-only vs Combined

### Phase 2: Structural Features

1. Obtain AlphaFold structures for S669 proteins
2. Extract contact maps, DSSP, pLDDT
3. Add structural encoder branch
4. Evaluate combined model

### Phase 3: Full Multimodal

1. Implement cross-attention fusion
2. Joint training with multi-task loss
3. Hyperparameter optimization
4. Final benchmarking

---

## Reproducibility Checklist

### Pre-Integration State (Frozen)

- [x] TrainableCodonEncoder implemented
- [x] trained_codon_encoder.pt saved
- [x] LOO Spearman 0.61 documented
- [x] Git commit: `89b3f27`
- [ ] Git tag: `v0.1.0-codon-encoder-baseline` (to be created)

### Data Requirements

| Dataset | Size | Status | Location |
|---------|------|--------|----------|
| S669 | 52 mutations | Available | deliverables/partners/jose_colbes/ |
| ESM-2 embeddings | ~67MB | To extract | research/codon-encoder/multimodal/data/ |
| AlphaFold structures | ~100MB | To download | research/codon-encoder/multimodal/structures/ |

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-03 | 1.0 | Initial expectation matrix, baseline measurements |

---

## Notes

This document will be updated with a **RESULTS_MATRIX.md** after integration experiments are complete, allowing direct comparison of expectations vs. actual outcomes.
