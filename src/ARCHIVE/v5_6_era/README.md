# V5.6 Era Legacy Source Code

This folder contains source code from the V5.6 era that references the now-removed
`DualNeuralVAEV5` model architecture.

## Contents

### evaluation/
- `benchmark_utils.py` - V5.6 benchmark utilities with `create_v5_6_model()` function

## Why Archived

- References `src.models.ternary_vae_v5_6.DualNeuralVAEV5` which doesn't exist
- The `create_v5_6_model()` function constructs V5.6-specific architecture
- Current architecture is V5.11

## Current Alternatives

For V5.11 evaluation, use:
- `src/core/metrics.py:compute_comprehensive_metrics()` - Full metrics matching checkpoints
- `src/core/metrics.py:ComprehensiveMetrics` - Dataclass for metric storage
