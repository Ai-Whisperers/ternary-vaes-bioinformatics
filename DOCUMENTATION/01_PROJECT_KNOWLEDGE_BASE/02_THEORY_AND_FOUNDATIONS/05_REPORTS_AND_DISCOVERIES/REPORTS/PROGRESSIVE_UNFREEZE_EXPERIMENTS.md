# Progressive Unfreeze Training Experiments

**Date:** December 26, 2025
**Version:** V5.11.3
**Status:** Experimental Results

---

## Executive Summary

Progressive unfreezing of the frozen encoder_A in V5.11 architecture dramatically improves geodesic distance correlation (from 0.58 to 0.93+) but trades off coverage. Finding the optimal balance requires careful learning rate tuning.

---

## Experiment 1: Conservative Unfreeze (Aggressive LR)

**Configuration:**
```
--progressive_unfreeze
--encoder_a_lr_scale 0.05
--unfreeze_start_epoch 10
--unfreeze_warmup_epochs 20
--epochs 100
```

**Results:**

| Metric | Start | Final | Target |
|--------|-------|-------|--------|
| Coverage | 100% | **0.1%** | 100% |
| Distance Corr | 0.11 | **0.965** | 0.80 |
| Radial Hierarchy | -0.35 | **-0.832** | -0.70 |

**Observation:** Coverage collapsed completely. Encoder_A learning rate too high - destroyed the learned representations that gave 100% coverage.

---

## Experiment 2: Tiny LR Unfreeze

**Configuration:**
```
--progressive_unfreeze
--encoder_a_lr_scale 0.005  # 10x smaller
--unfreeze_start_epoch 20   # Later start
--unfreeze_warmup_epochs 50 # Much slower warmup
--epochs 150
```

**Results:**

| Metric | Start | Final | Target |
|--------|-------|-------|--------|
| Coverage | 100% | **59.5%** | 100% |
| Distance Corr | 0.19 | **0.935** | 0.80 |
| Radial Hierarchy | -0.48 | **-0.831** | -0.70 |

**Coverage Trajectory:**
```
Epoch 70:  100.0%  (full unfreeze begins)
Epoch 75:   99.9%
Epoch 80:   99.8%
Epoch 85:   99.4%
Epoch 90:   98.9%
Epoch 95:   98.0%
Epoch 100:  96.4%  <-- SWEET SPOT
Epoch 110:  91.6%
Epoch 120:  85.1%
Epoch 130:  76.8%
Epoch 149:  59.5%
```

**Key Finding:** Epoch 100 checkpoint achieves **96.4% coverage + 0.895 distance correlation** - best tradeoff point.

---

## Comparison: Frozen vs Progressive Unfreeze

| Approach | Coverage | Distance Corr | Radial Hier | Notes |
|----------|----------|---------------|-------------|-------|
| Frozen (v5_11_11_production) | 100% | 0.582 | -0.718 | Safe but limited structure |
| Aggressive Unfreeze (0.05 LR) | 0.1% | 0.965 | -0.832 | Destroyed coverage |
| Tiny LR Unfreeze (0.005 LR) | 59.5% | 0.935 | -0.831 | Good tradeoff |
| **Tiny LR @ Epoch 100** | **96.4%** | **0.895** | **-0.827** | **Optimal balance** |

---

## Theoretical Insights

### Why Coverage Degrades

The frozen encoder_A was trained (v5.5) to maximize coverage of ternary operations. When we unfreeze and train with geodesic/radial losses:

1. **Competing objectives**: Coverage loss vs structure loss push in different directions
2. **Representation drift**: Small encoder changes accumulate, shifting embeddings away from decodable regions
3. **No coverage regularization**: Current losses don't penalize coverage degradation

### Why Structure Improves

Progressive unfreezing allows encoder_A to:

1. **Adapt to hyperbolic geometry**: Learn representations that naturally embed in Poincare ball
2. **Align with p-adic structure**: Co-optimize with the geodesic distance targets
3. **Escape local minima**: Projection-only training may be stuck in suboptimal configurations

---

## Recommendations

### Immediate: Use Epoch 100 Checkpoint

The `progressive_tiny_lr/epoch_100.pt` checkpoint offers best balance:
- 96.4% coverage (acceptable)
- 0.895 distance correlation (exceeds 0.80 target)
- -0.827 radial hierarchy (exceeds -0.70 target)

### Future: Coverage-Aware Training

1. **Early stopping on coverage**: Stop if coverage < 95%
2. **Coverage regularization**: Add coverage loss term during unfreezing
3. **Alternating training**: Freeze/unfreeze cycles to balance objectives

### Novel Direction: Sideways VAE

Design a secondary VAE that:
- Analyzes checkpoint trajectories
- Identifies optimal encoder configurations
- Guides training restarts with better initial conditions

---

## Checkpoints

| Path | Epoch | Coverage | Dist Corr | Notes |
|------|-------|----------|-----------|-------|
| `progressive_conservative/best.pt` | 99 | 0.1% | 0.965 | DO NOT USE |
| `progressive_tiny_lr/epoch_100.pt` | 100 | 96.4% | 0.895 | RECOMMENDED |
| `progressive_tiny_lr/best.pt` | 148 | ~60% | 0.935 | High structure |
| `v5_11_11_production/best.pt` | 101 | 100% | 0.582 | Safe baseline |

---

## Next Steps

1. Extract and validate epoch 100 checkpoint
2. Design sideways VAE for checkpoint exploration
3. Implement coverage-aware early stopping
4. Explore even smaller LR (0.002) with coverage gate

---

*Generated: December 26, 2025*
