# Model Checkpoint Index

**Generated:** 2026-01-23
**Total Checkpoints:** 108 files (119 MB)
**Repository:** ternary-vaes-bioinformatics

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Directory Overview](#directory-overview)
3. [Complete Checkpoint Index](#complete-checkpoint-index)
   - [Production Models](#production-models)
   - [Specialized Models](#specialized-models)
   - [Experimental & Sweep](#experimental--sweep)
4. [Appendix A: Production-Ready Models](#appendix-a-production-ready-models)
5. [Appendix B: Model Selection Guide](#appendix-b-model-selection-guide)

---

## Executive Summary

This repository contains **108 PyTorch checkpoint files** across three main locations, representing various training runs of the Ternary VAE architecture. The models embed 19,683 ternary operations (3^9) into a hyperbolic Poincare ball where radial position encodes 3-adic valuation.

### Key Metrics Explained

| Metric | Target | Description |
|--------|--------|-------------|
| **Coverage** | 100% | Percentage of 19,683 operations correctly reconstructed |
| **Hierarchy (hier_B)** | -0.83 to -1.0 | Spearman correlation between 3-adic valuation and radius (negative = correct ordering) |
| **Richness** | >0.005 | Average within-valuation-level variance of radii |
| **Q Score** | >1.5 | Composite metric: Q = dist_corr + 1.5 x \|hierarchy\| |

### Mathematical Limit

**Hierarchy ceiling: -0.8321** - This is the mathematical maximum achievable correlation when any within-level variance exists, due to v=0 containing 66.7% of all samples.

---

## Directory Overview

| Location | Count | Size | Purpose |
|----------|-------|------|---------|
| `sandbox-training/checkpoints/` | 102 | ~116 MB | Main training runs across versions v5.5-v5.12 |
| `research/codon-encoder/training/results/` | 1 | 51 KB | TrainableCodonEncoder for DDG prediction |
| `deliverables/sandbox-training/checkpoints/` | 5 | 6 MB | Partner deliverable copies (PeptideVAE) |

---

## Complete Checkpoint Index

### Production Models

These checkpoints have achieved 100% coverage and stable hierarchy metrics.

| Directory | Best Checkpoint | Size | Coverage | Hier_B | Richness | Q | Status |
|-----------|-----------------|------|----------|--------|----------|---|--------|
| **v5_12_4_fixed** | `best_Q.pt` | 931 KB | 100% | -0.82 | - | 1.96 | **CURRENT** |
| **homeostatic_rich** | `best.pt` | 421 KB | 100% | -0.8321 | 0.00662 | - | **RECOMMENDED** |
| **v5_11** | `best.pt` | 845 KB | 100% | -0.8302 | - | - | Stable |
| **v5_11_homeostasis** | `best.pt` | 845 KB | 99.9% | -0.8318 | - | - | Controller-based |
| **v5_11_structural** | `best.pt` | 1.4 MB | 100% | -0.8320 | ~0.003 | - | Contact prediction |
| **v5_11_progressive** | `best.pt` | 1.2 MB | 99.9% | -0.8299 | - | - | Frequency-optimal |
| **v5_12** | `best.pt` | - | 100% | -0.8288 | - | - | Intermediate |
| **v5_12_4** | `best_Q.pt` | 981 KB | 100% | -0.8165 | - | - | Pre-fix baseline |
| **v5_5** | `best.pt` | 2.0 MB | 97.1% | -0.30 | 0.94 | ~1.5 | **FOUNDATION** - Continuum mesh |

#### v5_5 (TOPOLOGICAL FOUNDATION)

**Path:** `sandbox-training/checkpoints/v5_5/best.pt`

- **Role:** Provides the geometric substrate ("continuum mesh") for the entire Ternary VAE system
- **Architecture:** 9→256→128→64→16 (pure Euclidean, no hyperbolic components)
- **Key Discovery:** Despite Euclidean training, exhibits p-adic-like geometry:
  - Perfect monotonic radial ordering (all 10 valuation levels)
  - 82.8% ultrametric compliance (p-adic metric signature)
  - Hamming-Euclidean correlation ρ=0.55 (structure preservation)
  - 89% neighbor valuation consistency (continuum mesh)
- **Usage:** Frozen in later versions (v5.11+, v5.12.4) to preserve topology while training hierarchy
- **Documentation:** `sandbox-training/checkpoints/v5_5/V5_5_ANALYSIS.md`

#### v5_12_4_fixed (CURRENT PRODUCTION)

**Path:** `sandbox-training/checkpoints/v5_12_4_fixed/best_Q.pt`

- **Architecture:** TernaryVAEV5_11_PartialFreeze with ImprovedEncoder/Decoder
- **Features:** SiLU activation, LayerNorm, Dropout (0.1), logvar clamping [-10, 2]
- **Training:** Mixed precision (FP16), Homeostasis controller, Option-C per-parameter LR
- **Frozen from:** v5.5 encoder for coverage preservation
- **Config:** `configs/v5_12_4_fixed_checkpoint.yaml`

#### homeostatic_rich (RECOMMENDED FOR MOST TASKS)

**Path:** `sandbox-training/checkpoints/homeostatic_rich/best.pt`

- **Unique achievement:** Reaches hierarchy ceiling (-0.8321) WITH high richness (0.00662)
- **Significance:** Proves hierarchy and richness are NOT mutually exclusive
- **Richness:** 5.8x more than v5_11.8, 28x more than max_hierarchy
- **Training:** `scripts/epsilon_vae/train_homeostatic_rich.py`
- **Loss weights:** hierarchy=5.0, coverage=1.0, richness=2.0, separation=3.0

#### v5_11_structural (CONTACT PREDICTION)

**Path:** `sandbox-training/checkpoints/v5_11_structural/best.pt`

- **Special property:** Low richness (~0.003) enables consistent AA-level distances
- **Validated:** AUC-ROC = 0.6737 on Insulin B-chain contact prediction
- **Cohen's d:** -0.474 (medium effect size)
- **Use case:** Pairwise residue-residue 3D contact prediction

---

### Specialized Models

#### TrainableCodonEncoder

**Path:** `research/codon-encoder/training/results/trained_codon_encoder.pt`

| Metric | Value |
|--------|-------|
| Size | 51 KB |
| Architecture | MLP (12→64→64→16) with LayerNorm, SiLU, Dropout |
| LOO Spearman (DDG) | **0.614** |
| LOO Pearson (DDG) | 0.636 |
| LOO MAE | 0.81 kcal/mol |

**Comparison to baselines:**

| Method | Spearman | Type |
|--------|----------|------|
| Rosetta ddg_monomer | 0.69 | Structure-based |
| **TrainableCodonEncoder** | **0.61** | **Sequence-only** |
| ELASPIC-2 (2024) | 0.50 | Sequence |
| FoldX | 0.48 | Structure |
| Baseline p-adic | 0.30 | Sequence |

**Key discoveries:**
- Dim 13 ("Physics dimension") encodes mass, volume, force constants (ρ = -0.695)
- Force constant formula: k = radius × mass / 100 (ρ = 0.860)

#### PeptideVAE (AMP Prediction)

**Path:** `sandbox-training/checkpoints/peptide_vae_v1/`

| File | Size | Fold |
|------|------|------|
| `best_production.pt` | 1.2 MB | Best overall |
| `fold_0_definitive.pt` | 1.2 MB | CV fold 0 |
| `fold_1_definitive.pt` | 1.2 MB | CV fold 1 |
| `fold_2_definitive.pt` | 1.2 MB | CV fold 2 |
| `fold_3_definitive.pt` | 1.2 MB | CV fold 3 |
| `fold_4_definitive.pt` | 1.2 MB | CV fold 4 |

**5-Fold Cross-Validation Results:**

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Spearman | 0.633 | 0.094 | 0.481 | 0.760 |
| Pearson | 0.618 | 0.142 | 0.347 | 0.749 |
| MIC MAE | 0.32 | 0.05 | 0.25 | 0.39 |

**Architecture:**
- Latent dim: 16, Hidden dim: 64
- Attention: 4 heads, 2 layers
- Poincare ball: curvature=1.0, max_radius=0.95

---

### Experimental & Sweep

#### Loss Experiments (11 directories)

| Directory | Coverage | Hier_B | Richness | Notes |
|-----------|----------|--------|----------|-------|
| balanced_radial | 100% | -0.8321 | 0.000048 | High hierarchy, collapsed richness |
| hierarchy_extreme | 100% | -0.8321 | - | Maximum hierarchy push |
| hierarchy_focused | 100% | -0.8320 | - | Hierarchy-first approach |
| max_hierarchy | 100% | -0.8298 | 0.000265 | Near-ceiling hierarchy |
| radial_collapse | 100% | -0.8321 | - | Intentional collapse test |
| radial_target | 100% | -0.8321 | - | Target radii loss |
| final_rich_lr1e4 | 100% | -0.6840 | 0.00683 | High richness variant |
| final_rich_lr3e4 | 100% | -0.6691 | 0.00821 | High richness variant |
| final_rich_lr5e5 | 100% | -0.6932 | 0.00858 | High richness variant |

#### Sweep Test Results (45 directories)

**Key findings from 50+ experiments:**

| Parameter | Best Value | Finding |
|-----------|------------|---------|
| Curvature | 2.0 | ~2% improvement over 1.0 |
| Learning rate | 3e-4 | Best balance of speed and stability |
| Early stopping | patience=5 | Prevents overtraining degradation |
| Best hierarchy | -0.792 | Achieved at epoch 3 (sweep3_early_stop_p5) |

**Top sweep checkpoints:**

| Checkpoint | Hierarchy | Notes |
|------------|-----------|-------|
| sweep4_lr_1e3 | -0.790 | Best overall from LR sweep |
| sweep3_early_stop_p5 | -0.792 | Best Phase 3 |
| sweep4_lr_3e4 | -0.789 | Most reliable |
| sweep3_step_decay | -0.767 | Step LR decay |

#### Training Experiments (6 directories)

| Directory | Coverage | Hier_B | Notes |
|-----------|----------|--------|-------|
| progressive_conservative | 0.1% | -0.8320 | Too conservative |
| progressive_tiny_lr | 60.4% | -0.8320 | LR too small |
| v5_11_annealing_long | 98.2% | -0.8320 | Long annealing |
| v5_11_learnable_qreg | 100% | -0.8303 | Q-regularized |
| v5_11_progressive_50ep | 1.3% | -0.8320 | Underfitting |
| v5_11_progressive_non_fixed | 0.3% | -0.8320 | Non-fixed baseline |

---

## Appendix A: Production-Ready Models

### Tier 1: Validated & Recommended

These models have been thoroughly validated with documented metrics and are ready for production use.

---

#### 1. homeostatic_rich/best.pt

**Status:** PRODUCTION-READY

**Path:** `sandbox-training/checkpoints/homeostatic_rich/best.pt`
**Size:** 421 KB

**Metrics:**
- Coverage: 100.0%
- Hierarchy: -0.8321 (ceiling)
- Richness: 0.00662 (5.8x baseline)

**Applications:**
| Application | Suitability | Notes |
|-------------|-------------|-------|
| ΔΔG/Stability prediction | Excellent | High richness preserves geometric diversity |
| General codon embedding | Excellent | Best balance of hierarchy and richness |
| Research baseline | Excellent | Proves hierarchy-richness tradeoff is false |

**Usage:**
```python
from src.models import TernaryVAEV5_11_PartialFreeze
import torch

model = TernaryVAEV5_11_PartialFreeze(
    latent_dim=16, hidden_dim=64, max_radius=0.95,
    curvature=1.0, use_controller=True, use_dual_projection=True
)
ckpt = torch.load('sandbox-training/checkpoints/homeostatic_rich/best.pt')
model.load_state_dict(ckpt['model_state_dict'])
```

---

#### 2. v5_12_4_fixed/best_Q.pt

**Status:** PRODUCTION-READY (CURRENT)

**Path:** `sandbox-training/checkpoints/v5_12_4_fixed/best_Q.pt`
**Size:** 931 KB

**Metrics:**
- Coverage: 100%
- Hierarchy: -0.82
- Q Score: 1.96

**Architecture:** ImprovedEncoder/Decoder with:
- SiLU activation (smoother gradients)
- LayerNorm (stable training)
- Dropout 0.1 (regularization)
- Logvar clamping [-10, 2] (prevents KL collapse)

**Applications:**
| Application | Suitability | Notes |
|-------------|-------------|-------|
| Production inference | Excellent | Latest architecture with optimizations |
| Downstream fine-tuning | Excellent | Clean initialization for task-specific training |
| Mixed precision deployment | Excellent | Validated with FP16 |

**Usage:**
```python
from src.models import TernaryVAEV5_11_PartialFreeze

model = TernaryVAEV5_11_PartialFreeze(
    latent_dim=16, hidden_dim=64, max_radius=0.95,
    curvature=1.0, use_controller=True, use_dual_projection=True,
    encoder_type='improved', decoder_type='improved'
)
ckpt = torch.load('sandbox-training/checkpoints/v5_12_4_fixed/best_Q.pt')
model.load_state_dict(ckpt['model_state_dict'])
```

---

#### 3. v5_11_structural/best.pt (Contact Prediction)

**Status:** PRODUCTION-READY

**Path:** `sandbox-training/checkpoints/v5_11_structural/best.pt`
**Size:** 1.4 MB

**Metrics:**
- Coverage: 100%
- Hierarchy: -0.8320
- Richness: ~0.003 (intentionally low)
- Contact AUC-ROC: 0.6737

**Validated on:** Insulin B-chain (30 residues)

**Applications:**
| Application | Suitability | Notes |
|-------------|-------------|-------|
| Contact prediction | Best | Low richness = consistent AA distances |
| Structure inference | Good | Pairwise distances predict 3D contacts |
| NOT for ΔΔG | Poor | Low richness hurts stability prediction |

**Critical tradeoff:** This checkpoint optimizes for contact prediction at the expense of richness. Use homeostatic_rich for ΔΔG tasks.

---

#### 4. trained_codon_encoder.pt (DDG Prediction)

**Status:** PRODUCTION-READY

**Path:** `research/codon-encoder/training/results/trained_codon_encoder.pt`
**Size:** 51 KB

**Metrics:**
- LOO Spearman: 0.614
- LOO Pearson: 0.636
- LOO MAE: 0.81 kcal/mol

**Applications:**
| Application | Suitability | Notes |
|-------------|-------------|-------|
| Protein stability (ΔΔG) | Excellent | +105% over baseline, approaches Rosetta |
| Sequence-only analysis | Excellent | No structure required |
| Codon-level features | Excellent | 12-dim one-hot input, no information loss |

**Usage:**
```python
from src.encoders import TrainableCodonEncoder
import torch

encoder = TrainableCodonEncoder(latent_dim=16, hidden_dim=64)
ckpt = torch.load('research/codon-encoder/training/results/trained_codon_encoder.pt')
encoder.load_state_dict(ckpt['model_state_dict'])
encoder.eval()

# Get all codon embeddings
z_hyp = encoder.encode_all()  # (64, 16)

# Get amino acid embeddings (averaged over synonymous codons)
aa_embs = encoder.get_all_amino_acid_embeddings()

# Compute hyperbolic distance between amino acids
dist = encoder.compute_aa_distance('A', 'V')
```

---

#### 5. peptide_vae_v1/best_production.pt (AMP Prediction)

**Status:** PRODUCTION-READY

**Path:** `sandbox-training/checkpoints/peptide_vae_v1/best_production.pt`
**Size:** 1.2 MB

**5-Fold CV Metrics:**
- Mean Spearman: 0.633 ± 0.094
- Mean Pearson: 0.618 ± 0.142
- Mean MIC MAE: 0.32

**Applications:**
| Application | Suitability | Notes |
|-------------|-------------|-------|
| MIC prediction | Excellent | 5-fold validated, r=0.74 |
| AMP optimization | Good | Radial structure encodes activity |
| Pathogen clustering | Good | Learned pathogen-specific embeddings |

---

---

#### 6. v5_5/best.pt (Topological Foundation)

**Status:** PRODUCTION-READY (Foundation Model)

**Path:** `sandbox-training/checkpoints/v5_5/best.pt`
**Size:** 2.0 MB

**Metrics:**
- Coverage A: 97.1%
- Hierarchy A: -0.30 (weak but monotonic)
- Ultrametric compliance: 82.8%
- Neighbor consistency: 89.3%

**Applications:**
| Application | Suitability | Notes |
|-------------|-------------|-------|
| Transfer learning base | Excellent | Freeze encoder, train projection layers |
| Topology research | Excellent | Study p-adic emergence from Euclidean training |
| Custom VAE development | Good | Stable foundation for new architectures |

**Key Properties:**
- Pure Euclidean training produces p-adic-like geometry
- Perfect monotonic radial ordering across all 10 valuation levels
- Intrinsic dimension: 8 (effective: 4)
- Forms the frozen base for v5.11+ and v5.12.4

**Documentation:** `sandbox-training/checkpoints/v5_5/V5_5_ANALYSIS.md`

---

### Tier 2: Stable Production

These checkpoints are production-stable but may be superseded by Tier 1 models.

| Checkpoint | Path | Size | Coverage | Hier_B | Use Case |
|------------|------|------|----------|--------|----------|
| v5_11/best.pt | `sandbox-training/checkpoints/v5_11/` | 845 KB | 100% | -0.8302 | Legacy baseline |
| v5_11_homeostasis/best.pt | `sandbox-training/checkpoints/v5_11_homeostasis/` | 845 KB | 99.9% | -0.8318 | Dynamic controller |
| v5_11_progressive/best.pt | `sandbox-training/checkpoints/v5_11_progressive/` | 1.2 MB | 99.9% | +0.78* | Frequency-optimal |

*Note: v5_11_progressive has positive hierarchy, representing a valid frequency-optimal (Shannon) manifold organization rather than p-adic ordering.

---

## Appendix B: Model Selection Guide

### Decision Tree

```
What is your task?
│
├─► Protein stability (ΔΔG) prediction
│   └─► Use: trained_codon_encoder.pt OR homeostatic_rich/best.pt
│
├─► Residue-residue contact prediction
│   └─► Use: v5_11_structural/best.pt (AUC=0.67)
│
├─► Antimicrobial peptide (AMP) analysis
│   └─► Use: peptide_vae_v1/best_production.pt
│
├─► General codon/amino acid embeddings
│   └─► Use: homeostatic_rich/best.pt (best balance)
│
├─► Latest architecture with optimizations
│   └─► Use: v5_12_4_fixed/best_Q.pt
│
└─► Downstream fine-tuning
    └─► Start with: v5_12_4_fixed/best_Q.pt
```

### Use Case Matrix

| Task | Best Checkpoint | Alternative | Avoid |
|------|-----------------|-------------|-------|
| ΔΔG prediction | trained_codon_encoder.pt | homeostatic_rich | v5_11_structural |
| Contact prediction | v5_11_structural | - | homeostatic_rich |
| AMP MIC prediction | peptide_vae_v1 | - | - |
| General embedding | homeostatic_rich | v5_12_4_fixed | max_hierarchy |
| Research baseline | homeostatic_rich | v5_11 | overnight* |
| Production deployment | v5_12_4_fixed | homeostatic_rich | any sweep_* |

*v5_11_overnight is problematic (training collapsed); do not use.

### Hardware Requirements

| Checkpoint | VRAM (Inference) | VRAM (Training) | Notes |
|------------|------------------|-----------------|-------|
| trained_codon_encoder.pt | <1 GB | 2 GB | Smallest, fastest |
| homeostatic_rich | 2 GB | 4 GB | RTX 3050 compatible |
| v5_12_4_fixed | 2 GB | 4-6 GB | Mixed precision recommended |
| peptide_vae_v1 | 2 GB | 4 GB | Attention adds overhead |
| v5_11_structural | 2 GB | 4 GB | Larger due to architecture |

---

## Appendix C: Checkpoint Loading Best Practices

### Critical Warning: Architecture Mismatch

Loading with `strict=False` can silently fail if architecture doesn't match:

```python
# WRONG - Silent failure with random projection layers
model = TernaryVAEV5_11_PartialFreeze(curvature=2.0, use_controller=False)
model.load_state_dict(ckpt['model_state_dict'], strict=False)  # 10+ keys missing!

# CORRECT - Match checkpoint architecture exactly
model = TernaryVAEV5_11_PartialFreeze(
    curvature=1.0,        # Match checkpoint
    max_radius=0.95,      # Match checkpoint
    use_controller=True,  # Match checkpoint
)
model.load_state_dict(ckpt['model_state_dict'], strict=True)  # Catches mismatches
```

### Recommended Loading Pattern

```python
import torch
from src.models import TernaryVAEV5_11_PartialFreeze

def load_checkpoint(path: str) -> TernaryVAEV5_11_PartialFreeze:
    """Load checkpoint with architecture verification."""
    ckpt = torch.load(path, map_location='cpu')

    # Extract config if available
    config = ckpt.get('config', {})
    model_config = config.get('model', {})

    # Create model with matching architecture
    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=model_config.get('latent_dim', 16),
        hidden_dim=model_config.get('hidden_dim', 64),
        max_radius=model_config.get('max_radius', 0.95),
        curvature=model_config.get('curvature', 1.0),
        use_controller=model_config.get('use_controller', True),
        use_dual_projection=model_config.get('use_dual_projection', True),
    )

    # Load with strict=True to catch mismatches
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    return model
```

---

## Version History

| Date | Changes |
|------|---------|
| 2026-01-23 | Initial comprehensive index generated |
| - | Catalogued 108 checkpoints across 119 directories |
| - | Documented 5 production-ready models with applications |
