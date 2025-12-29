# V5.6 Era Legacy Scripts

These scripts were created during the V5.6 era of the Ternary VAE project and reference
the now-removed `DualNeuralVAEV5` model from `src.models.ternary_vae_v5_6`.

## Why Archived

- The V5.6 model file (`src/models/ternary_vae_v5_6.py`) no longer exists
- Current architecture is V5.11 (`TernaryVAEV5_11`, `TernaryVAEV5_11_PartialFreeze`)
- These scripts would crash if invoked due to missing imports

## Contents

### benchmarks/
- `run_benchmark.py` - Multi-version benchmark runner (V5.6, V5.10)
- `measure_coupled_resolution.py` - Coupled resolution metrics
- `measure_manifold_resolution.py` - Manifold resolution metrics

### visualization/
- `calabi_yau_surface_mesh.py` - Calabi-Yau surface visualization
- `calabi_yau_projection.py` - Calabi-Yau projection analysis
- `calabi_yau_fibration.py` - Calabi-Yau fibration analysis
- `analyze_advanced_manifold.py` - Advanced manifold analysis
- `analyze_3adic_structure.py` - 3-adic structure analysis

## If Needed

To revive these scripts for V5.11:
1. Replace `from src.models.ternary_vae_v5_6 import DualNeuralVAEV5` with:
   `from src.models import TernaryVAEV5_11`
2. Update model instantiation to match V5.11 API
3. Update output key names (`z_A_hyp` instead of `z_A`, etc.)

## Current Alternatives

- Use `scripts/epsilon_vae/` training scripts for V5.11 experiments
- Use `src/core/metrics.py:compute_comprehensive_metrics()` for evaluation
