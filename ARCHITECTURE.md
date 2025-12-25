# Architecture Overview

This document provides a high-level overview of the Ternary VAE codebase architecture.

---

## Directory Structure

```
ternary-vaes-bioinformatics/
├── src/                          # Core library
│   ├── models/                   # VAE architectures
│   │   ├── ternary_vae_v5_6.py  # Current production model
│   │   └── ternary_vae.py       # Base model
│   ├── losses/                   # Loss functions
│   │   ├── dual_vae_loss.py     # Complete loss system
│   │   ├── padic_losses.py      # 3-adic geometry losses
│   │   └── ...
│   ├── training/                 # Training infrastructure
│   │   ├── trainer.py           # Training loop orchestration
│   │   ├── schedulers.py        # Parameter scheduling
│   │   └── monitor.py           # Logging and metrics
│   ├── data/                     # Data handling
│   │   ├── generation.py        # Ternary operation generation
│   │   └── dataset.py           # PyTorch datasets
│   ├── artifacts/                # Checkpoint management
│   └── utils/                    # Metrics and utilities
├── scripts/                      # Entry points
│   ├── train/                    # Training scripts
│   ├── benchmark/                # Benchmarking
│   └── visualization/            # Visualization tools
├── configs/                      # YAML configurations
├── tests/                        # Test suite
├── research/                     # Research experiments
│   └── alphafold3/              # AlphaFold3 integration
├── results/                      # Training outputs
└── DOCUMENTATION/                # Comprehensive docs
```

---

## Core Components

### 1. Dual-VAE Architecture

The model consists of two VAEs that work together:

```
Input → [VAE-A Encoder] → z_A (exploration)
      → [VAE-B Encoder] → z_B (refinement)
      → [StateNet] → control signals (ρ, λ adjustments)
      → [Decoder] → reconstruction
```

- **VAE-A**: Explores the ternary operation space (chaotic)
- **VAE-B**: Refines and stabilizes (anchor)
- **StateNet**: Meta-controller that balances both VAEs

### 2. Hyperbolic Geometry

The latent space uses Poincare ball geometry:
- Points near origin = high-valuation (simple operations)
- Points near boundary = low-valuation (complex operations)
- Geodesic distance encodes 3-adic relationships

### 3. Loss System

```python
total_loss = (
    reconstruction_loss +        # Cross-entropy
    beta_A * kl_divergence_A +   # VAE-A regularization
    beta_B * kl_divergence_B +   # VAE-B regularization
    lambda_3 * padic_loss +      # 3-adic structure
    entropy_regularization       # Output diversity
)
```

---

## Training Flow

```
1. Load config (YAML)
2. Initialize model, optimizer, schedulers
3. For each epoch:
   a. Phase-based scheduling (beta, temperature)
   b. Train step with gradient balancing
   c. StateNet corrections
   d. Logging to TensorBoard
   e. Checkpoint if best
4. Final evaluation and reporting
```

---

## Key Design Decisions

### Single Responsibility Principle (SRP)
Each module has one job:
- `trainer.py` → training loop only
- `schedulers.py` → parameter scheduling only
- `monitor.py` → logging only
- `dual_vae_loss.py` → loss computation only

### Dependency Injection
Components are injected, not hard-coded:
```python
trainer = TernaryVAETrainer(
    model,
    config,
    device,
    monitor=custom_monitor  # Optional injection
)
```

### Phase-Scheduled Training
Training proceeds in phases:
1. **Phase 1 (0-40)**: VAE-A exploration, β-warmup
2. **Phase 2 (40-49)**: Consolidation
3. **Phase 3 (50)**: Disruption (β-B warmup)
4. **Phase 4 (50+)**: Convergence

---

## Bioinformatics Applications

### Codon Encoder Research
Located in `DOCUMENTATION/.../bioinformatics/codon_encoder_research/`:
- **HIV**: Glycan shield analysis, drug resistance
- **SARS-CoV-2**: Spike protein analysis
- **Rheumatoid Arthritis**: HLA-autoantigen relationships
- **Neurodegeneration**: Tau phosphorylation

### AlphaFold3 Integration
Located in `research/alphafold3/`:
- 6300x storage reduction via hybrid approach
- Structural validation pipeline

---

## Configuration

Example config structure (`configs/ternary_v5_6.yaml`):

```yaml
model:
  input_dim: 9
  latent_dim: 16
  rho_min: 0.1
  rho_max: 0.7

vae_a:
  beta_start: 0.3
  beta_end: 0.8
  beta_warmup_epochs: 50

vae_b:
  beta_start: 0.0
  beta_end: 0.5
  beta_warmup_epochs: 50

training:
  batch_size: 256
  total_epochs: 300

torch_compile:
  enabled: true
  backend: inductor
```

---

## Performance

| Metric | Value |
|:-------|:------|
| Model Parameters | 168,770 |
| Coverage (hash-validated) | 86-87% |
| Inference Speed (VAE-A) | 4.4M samples/sec |
| Inference Speed (VAE-B) | 6.1M samples/sec |
| Training Speedup (torch.compile) | 1.4-2x |

---

*For detailed API documentation, see `DOCUMENTATION/03_PRESENTATION_TIERS/03_TIER_3_TECHNICAL/`*
