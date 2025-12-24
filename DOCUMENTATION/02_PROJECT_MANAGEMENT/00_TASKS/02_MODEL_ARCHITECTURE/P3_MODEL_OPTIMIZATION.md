# Task: Model Optimization (Riemannian & JIT)

**Objective**: Optimize training speed and numerical stability using Riemannian techniques and JIT.
**Source**: `IMPROVEMENT_PLAN.md` (Sec 1 & Ideas)

## High-Level Goals

- [ ] **Stability**: Eliminate NaNs in hyperbolic training.
- [ ] **Speed**: 2x speedup in training loop.

## Detailed Tasks (Implementation)

### 1. Riemannian Optimization

- [ ] **Audit Optimizers**: Scan all training scripts (`src/training/*.py`) to ensure `geoopt.optim.RiemannianAdam` is used for Poincare parameters.
- [ ] **Update Configs**: Add `optimizer_type: "RiemannianAdam"` to experimental configs.

### 2. JIT Compilation

- [ ] **Numba Kernels**: Identify tight loops in `src/geometry/` that are not vectorized and decorate with `@jit`.
- [ ] **Torch Compile**: Test `torch.compile()` on the VAE Decoder for inference speedup.

### 3. Precision

- [ ] **Mixed Precision**: Implement `torch.amp` (Automatic Mixed Precision) correctly for Hyperbolic logic (careful with float16 stability in Poincare ball).

## Deliverables

- [ ] Benchmarking script comparing "Before" vs "After".
- [ ] Stable training loss curves.
