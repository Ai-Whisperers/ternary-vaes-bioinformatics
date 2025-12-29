# Debug / RCA Protocol

Systematic Root Cause Analysis for Bioinformatics Pipelines.

## When to Use

- **Trigger**: `NaN` loss, CUDA OOM, Shape Mismatch, or Silent Failures.

## Debugging Workflow

### 1. Isolate the Failure

- [ ] Can you reproduce it with `pytest -k test_name`?
- [ ] Can you reproduce it with a minimal script?
- [ ] Does it happen on CPU as well as GPU?

### 2. Inspect Tensor State

- **Shapes**: Print `x.shape` at every layer boundary.
- **Stats**: Print `x.min()`, `x.max()`, `x.mean()` to check for explosions.
- **Gradients**: `print(param.grad.norm())` to check for vanishing/exploding grads.

### 3. Common Bio-ML Culprits

- **Numerical Stability**: `exp(large_val)` -> Inf. Use `log_softmax` instead of `log(softmax)`.
- **Hyperbolic Geometry**: Points approaching the boundary (norm -> 1). Use `c` clipping.
- **Data Loader**: Malformed sequences or empty batches.

## Resolution Template

```markdown
## Incident Report

**Error**: [Error Name]
**Root Cause**: [What actually broke]
**Fix**: [Code Change]

## Prevention

- Added assert for [Condition]
- Added unit test [Test Name]
```
