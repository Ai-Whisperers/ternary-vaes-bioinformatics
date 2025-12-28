# P-adic Codon Encoder Research

Research scripts for validating p-adic codon embeddings against physical ground truth.

## Key Discoveries

| Analysis | Finding | Correlation |
|----------|---------|-------------|
| Dimension 13 | "Physics dimension" - encodes mass, volume, force constants | ρ = -0.695 |
| Radial structure | Encodes amino acid mass | ρ = +0.760 |
| Force constant formula | `k = radius × mass / 100` | **ρ = 0.860** |
| Vibrational frequency | `ω = √(k/m)` - derivable from embeddings | ρ = 1.000 |

## Directory Structure

```
codon-encoder/
├── config.py           # Shared configuration and paths
├── benchmarks/         # Validation benchmarks
│   ├── mass_vs_property_benchmark.py   # Mass vs property feature comparison
│   ├── kinetics_benchmark.py           # Folding/aggregation kinetics
│   ├── deep_physics_benchmark.py       # Multi-level physics validation
│   └── ddg_benchmark.py                # ΔΔG stability benchmark
├── training/           # Model training scripts
│   ├── ddg_predictor_training.py       # sklearn-based training
│   └── ddg_pytorch_training.py         # PyTorch hyperparameter search
├── analysis/           # Embedding space analysis
│   ├── proteingym_pipeline.py          # Dimension-property correlations
│   └── padic_dynamics_predictor.py     # Force constant/frequency prediction
├── pipelines/          # Integration pipelines
│   ├── padic_3d_dynamics.py            # 3D structure + dynamics
│   ├── af3_pipeline.py                 # AlphaFold3 integration
│   └── ptm_mapping.py                  # PTM site analysis
└── results/            # Output directories
    ├── benchmarks/
    ├── training/
    ├── analysis/
    └── pipelines/
```

## Usage

### Run a benchmark
```bash
python benchmarks/deep_physics_benchmark.py
```

### Train ΔΔG predictor
```bash
python training/ddg_pytorch_training.py
```

### Analyze embedding space
```bash
python analysis/proteingym_pipeline.py
```

## Dependencies

- PyTorch
- NumPy
- SciPy
- scikit-learn

## Key Results Summary

### Thermodynamics vs Kinetics Split

| Task Type | Winner | Best Feature |
|-----------|--------|--------------|
| Stability (ΔΔG) | Mass-based | padic_mass ρ=0.94 |
| Folding rates | Property-based | property ρ=0.94 |
| Aggregation | Property-based | property |

### Physics Level Correlations

| Level | Physics | P-adic Correlation |
|-------|---------|-------------------|
| 0 | Biochemistry | ρ = 0.760 (mass) |
| 1 | Classical mechanics | ρ = 0.665 (moment) |
| 2 | Statistical mechanics | ρ = 0.517 (entropy) |
| 3 | Vibrational (force constants) | **ρ = 0.608** |
| 4 | Experimental dynamics | ρ = -0.01 (B-factor) |
| 5 | Quantum corrections | ρ = 0.466 |

The p-adic encoder captures thermodynamic invariants (force constants) but NOT local experimental dynamics (B-factors).
