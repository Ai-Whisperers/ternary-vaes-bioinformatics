# Scripts

CLI entry points and executable scripts for the Ternary VAE project.

> **Note**: Reusable library code lives in `src/`. This directory contains only CLI entry points and executable scripts.

---

## Directory Structure

```
scripts/
├── train.py                    # Main training entry point (v5.11)
├── download_with_custom_dns.py # DNS-aware dataset downloader
│
├── analysis/                   # Code quality and validation
│   ├── audit_repo.py          # Linting, security, complexity checks
│   ├── code_stats.py          # Code duplication analysis
│   ├── run_metrics.py         # VAE quality metrics
│   ├── validate_codon_classification.py
│   └── verify_mathematical_proofs.py
│
├── benchmark/                  # Performance benchmarking
│   ├── run_benchmark.py       # Comprehensive VAE benchmark
│   ├── measure_manifold_resolution.py
│   └── measure_coupled_resolution.py
│
├── docs/                       # Documentation utilities
│   └── add_spdx_frontmatter.py
│
├── epsilon_vae/               # Epsilon-VAE research experiments
│   ├── train_*.py             # Training variants (11 scripts)
│   ├── analyze_*.py           # Analysis scripts
│   ├── collect_checkpoints.py
│   └── extract_embeddings.py
│
├── eval/                       # Model evaluation
│   └── downstream_validation.py
│
├── hiv/                        # HIV analysis entry points
│   ├── run_hiv_analysis.py    # Unified analysis router
│   ├── validate_hiv_setup.py  # Environment validation
│   ├── download_hiv_datasets.py
│   └── train_codon_vae_hiv.py
│
├── ingest/                     # Data ingestion
│   ├── ingest_arboviruses.py
│   ├── ingest_pdb_rotamers.py
│   └── ingest_starpep.py
│
├── legal/                      # License management
│   └── add_copyright_headers.py
│
├── maintenance/               # Codebase maintenance
│   ├── maintain_codebase.py   # Format, lint, typecheck
│   ├── doc_auditor.py
│   ├── doc_builder.py
│   ├── validate_all_implementations.py
│   ├── project_diagrams_generator.py
│   ├── comprehensive_vocab_scan.py
│   └── migrate_paths.py
│
├── optimization/              # Latent space optimization
│   └── latent_nsga2.py        # NSGA-II multi-objective
│
├── setup/                     # Environment setup
│   └── setup_hiv_analysis.py
│
├── training/                  # Training variants
│   ├── train_universal_vae.py
│   ├── train_toxicity_regressor.py
│   └── train_v5_11_11_homeostatic.py
│
└── visualization/             # Plotting and visualization
    ├── visualize_ternary_manifold.py
    ├── calabi_yau_*.py        # Calabi-Yau projections (7 scripts)
    ├── analyze_3adic_*.py     # 3-adic structure (2 scripts)
    └── plot_training_artifacts.py
```

---

## Quick Start

### Training

```bash
# Main training (canonical v5.11)
python scripts/train.py --config configs/ternary.yaml

# Universal VAE training
python scripts/training/train_universal_vae.py --config configs/universal.yaml

# HIV-specific training
python scripts/hiv/train_codon_vae_hiv.py
```

### Benchmarking

```bash
python scripts/benchmark/run_benchmark.py --config configs/ternary.yaml
```

### Visualization

```bash
python scripts/visualization/visualize_ternary_manifold.py
python scripts/visualization/plot_training_artifacts.py --run outputs/runs/latest
```

### Code Quality

```bash
python scripts/analysis/audit_repo.py
python scripts/maintenance/maintain_codebase.py
```

### HIV Analysis

```bash
python scripts/hiv/run_hiv_analysis.py --analysis all
python scripts/hiv/validate_hiv_setup.py
python scripts/hiv/download_hiv_datasets.py
```

---

## Script Categories

| Category | Purpose | Location |
|----------|---------|----------|
| **Training** | Model training | `train.py`, `training/`, `epsilon_vae/` |
| **Evaluation** | Model evaluation | `eval/`, `benchmark/` |
| **Analysis** | Code quality | `analysis/` |
| **Visualization** | Plotting | `visualization/` |
| **HIV Research** | HIV analysis | `hiv/` |
| **Maintenance** | Codebase health | `maintenance/` |
| **Data** | Data ingestion | `ingest/` |
| **Setup** | Environment | `setup/` |

---

## Library Code (in src/)

The following modules contain reusable library code that was moved from scripts/:

| src/ Module | Description |
|-------------|-------------|
| `src/implementations/literature/` | Literature implementations (p-adic encoders, VAE) |
| `src/clinical/` | Clinical decision support systems |
| `src/clinical/hiv/` | HIV clinical applications |
| `src/analysis/hiv/` | HIV analysis algorithms |
| `src/analysis/` | Primer stability, rotamer stability, sliding window |
| `src/reporting/` | Report generation |
| `src/research/hiv/` | HIV research pipelines |

---

## Guidelines

1. **Scripts vs Library Code**: Scripts should be thin wrappers that import from `src/`
2. **CLI Arguments**: Use `argparse` for command-line arguments
3. **Logging**: Use `src.observability.get_logger()` for logging
4. **Paths**: Use `src.config.paths` for all path resolution
5. **Configuration**: Load configs from `configs/` using `src.config.load_config()`

---

## Related

- `src/` - Reusable library code
- `configs/` - Configuration files
- `tests/` - Unit and integration tests
- `outputs/` - Training outputs and results
