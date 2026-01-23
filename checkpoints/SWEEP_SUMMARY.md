# Training Sweep Summary - 2025-12-29

**Duration**: ~45 minutes total
**Experiments**: 50+ runs across 6 phases

---

## Key Findings

### 1. Curvature (Phase 1)
| Curvature | Hierarchy | Richness |
|-----------|-----------|----------|
| 0.5 | -0.676 | 0.00499 |
| 1.0 | -0.688 | 0.00452 |
| **2.0** | **-0.702** | 0.00493 |

**Finding**: Curvature 2.0 is marginally better (~2% improvement).

### 2. Loss Weights (Phase 2)
All configs peaked at epochs 10-20, then **degraded**.

| Config | Peak Hier | Final Hier | Delta |
|--------|-----------|------------|-------|
| balanced (h=5,r=5) | -0.758 | -0.598 | +0.16 worse |
| baseline (h=5,r=2) | -0.752 | -0.559 | +0.19 worse |

**Finding**: Overtraining hurts hierarchy significantly.

### 3. Early Stopping & LR Scheduling (Phase 3)
| Strategy | Best Hier | @ Epoch | Final |
|----------|-----------|---------|-------|
| **early_stop_p5** | **-0.792** | 3 | -0.715 |
| step_decay | -0.767 | 3 | -0.677 |
| cosine | -0.695 | 42 | -0.692 |

**Finding**: Aggressive early stopping (patience=5) achieves best hierarchy at epoch 3.

### 4. Learning Rate (Phase 4)
| LR | Best Hier | @ Epoch |
|----|-----------|---------|
| **1e-3** | **-0.790** | 0 |
| 3e-4 | -0.789 | 5 |
| 5e-4 | -0.783 | 2 |
| 1e-4 | -0.735 | 29 |

**Finding**: Higher LR (1e-3) achieves best hierarchy immediately but degrades fast.

---

## Best Configurations

### For Maximum Hierarchy
```python
curvature = 2.0
lr = 3e-4  # or 1e-3 for single-epoch
early_stopping_patience = 5
epochs = 10  # Don't overtrain!
hierarchy_weight = 5.0
```
**Best achieved**: -0.792 (Phase 3, early_stop_p5)

### For Stable Training
```python
curvature = 2.0
lr = 3e-4
scheduler = CosineAnnealing
epochs = 60
```
**Preserves quality** without early stopping.

### Variance Analysis (LR=5e-4, 3 runs)
- Mean: -0.762
- Std: 0.020 (~2.6%)
- Range: [-0.783, -0.735]

---

## Actionable Recommendations

1. **Use curvature=2.0** instead of 1.0
2. **Train for only 3-10 epochs** with high LR (3e-4 to 1e-3)
3. **Enable early stopping** with patience=5
4. **Do NOT overtrain** - hierarchy degrades significantly after peak
5. **Use cosine annealing** if you want longer training without early stopping

---

## Checkpoints Generated

| Checkpoint | Hierarchy | Notes |
|------------|-----------|-------|
| sweep4_lr_1e3/best.pt | -0.790 | Best overall |
| sweep3_early_stop_p5/best.pt | -0.792 | Best Phase 3 |
| sweep4_lr_3e4/best.pt | -0.789 | Most reliable |
| sweep3_step_decay/best.pt | -0.767 | Step LR decay |
| sweep_curv_2.0/best.pt | -0.702 | Curvature test |

---

---

## Critical Bug Found: Checkpoint Loading

### The Problem
Loading `v5_11_homeostasis/best.pt` gave **wildly different hierarchies** each time:
- Load 1: -0.70
- Load 2: +0.76 (INVERTED!)
- Load 3: -0.34
- Load 4: +0.70 (INVERTED!)
- Load 5: -0.67

**Root Cause**: Architecture mismatch between checkpoint and model:
- Checkpoint saved with: `use_controller=True`, `curvature=1.0`, `max_radius=0.95`
- Model created with: `use_controller=False`, `curvature=2.0`, `max_radius=0.99`

The `strict=False` loading **silently fails** - 10+ projection layer keys are missing and randomly initialized!

### The Fix
Match architecture exactly:
```python
model = TernaryVAEV5_11_PartialFreeze(
    curvature=1.0,        # Match checkpoint!
    max_radius=0.95,      # Match checkpoint!
    use_controller=True,  # Match checkpoint!
    ...
)
```

### Impact on Results
| Metric | Wrong Arch | Correct Arch |
|--------|------------|--------------|
| Init Std | 0.53 | 0.46 |
| Best Std | 0.038 | **0.019** (50% better) |

---

## Stability Verification (25 runs total)

### Summary Statistics
| Metric | Value |
|--------|-------|
| Best Hier Mean | **-0.740** |
| Best Hier Std | **0.030** |
| Best Achieved | **-0.783** |
| Worst Achieved | -0.651 |

### Init→Best Correlation
**ρ = 0.92** - Initial state almost completely determines final quality!
- Good init (-0.6 to -0.7) → Best -0.77 to -0.78
- Inverted init (+0.3 to +0.7) → Best -0.67 to -0.72

---

## Recommendations for Future Training

### Before Training
1. **Verify checkpoint architecture** - Check stored config matches model
2. **Use strict=True first** - Catch mismatches early
3. **Validate init hierarchy** - Should be negative, if positive restart

### During Training
1. Use curvature=2.0 (marginally better)
2. Use lr=3e-4 with early stopping (patience=5)
3. Stop at epoch 3-10 (hierarchy peaks early!)
4. Monitor for degradation - don't overtrain

### Architecture Settings
```python
# Optimal config
curvature = 2.0
lr = 3e-4
early_stopping_patience = 5
max_epochs = 10
hierarchy_weight = 5.0
richness_weight = 2.0
```

---

## Next Steps

1. ~~**Fix checkpoint loading** - Create utility function that validates architecture match~~ ✓ DONE
2. ~~**Investigate projection layer architecture** - Why are there missing keys?~~ ✓ DONE (use_controller mismatch)
3. **Create "clean" checkpoint** - Save with consistent architecture
4. ~~**Test from-scratch training** with correct architecture~~ ✓ DONE (see below)

---

## From-Scratch Training Results

### Key Finding: From-scratch training is significantly harder

| Config | Epochs | Max Coverage | Notes |
|--------|--------|--------------|-------|
| hidden_dim=64 | 50 | 20-30% | Plateau, NaN issues |
| hidden_dim=256 | 100 | 26% | Still plateauing |

### Analysis
- Coverage plateaus around 20-30% even with 100 epochs
- Pretrained checkpoints have massive advantage (start at 100% coverage)
- From-scratch uses only cross-entropy loss; original training used full VAE loss with KL divergence
- Two-phase approach (coverage → hierarchy) works but coverage never reaches target

### Recommendation
**Continue fine-tuning pretrained checkpoints rather than training from scratch.**
The existing checkpoints provide a strong initialization that from-scratch training cannot match in reasonable time.

### Scripts Created
- `train_from_scratch.py` - Two-phase from-scratch training
  - Phase 1: Train for coverage with cross-entropy
  - Phase 2: Train for hierarchy with target radii loss
  - Uses new `ArchitectureConfig` and `save_checkpoint` utilities

---

## Timing

| Phase | Experiments | Duration |
|-------|-------------|----------|
| 1: Curvature | 6 | 5.5 min |
| 2: Loss Weights | 5 | 11.8 min |
| 3: Scheduling | 6 | 5.9 min |
| 4: Learning Rate | 8 | 2.0 min |
| 5: Stability (10 runs) | 10 | 3.4 min |
| 6: Extended (15 runs) | 15 | 4.1 min |
| Investigation | 5 | 2.0 min |
| **Total** | **55** | **~35 min** |
