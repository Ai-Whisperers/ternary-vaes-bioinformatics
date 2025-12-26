# Research Module

Bridge between core library (src/) and research experiments (research/).

## Purpose

This module provides utilities for:
1. Path resolution between src/ and research/ directories
2. Consistent access to research data and results
3. Experiment configuration management

## Path Resolution

### Project Root

```python
from src.research import get_project_root

root = get_project_root()
print(root)  # Path to repository root (parent of src/)
```

### Research Experiments

```python
from src.research import get_research_path

# Get research directory
research_dir = get_research_path()

# Get specific experiment
hiv_path = get_research_path("bioinformatics/codon_encoder_research/hiv")
```

### Data Directory

```python
from src.research import get_data_path

# Get data directory
data_dir = get_data_path()

# Get external data
hiv_data = get_data_path("external/github/HIV-data")
```

### Results Directory

```python
from src.research import get_results_path

# Get results directory
results_dir = get_results_path()

# Get specific results
experiment_results = get_results_path("hiv_analysis_2024")
```

### Config Files

```python
from src.research import get_config_path

# Get config path (returns None if doesn't exist)
config = get_config_path("ternary.yaml")
if config:
    print(f"Found config at: {config}")
```

## Listing Functions

### List Experiments

```python
from src.research import list_research_experiments

experiments = list_research_experiments()
print(experiments)  # ['bioinformatics', 'immunology', ...]
```

### List Datasets

```python
from src.research import list_datasets

datasets = list_datasets()
print(datasets)  # ['external', 'processed', ...]
```

## Directory Structure

```
project_root/
├── src/                    # Core library
│   └── research/           # This module (bridge)
├── research/               # Research experiments
│   ├── bioinformatics/     # Bioinformatics experiments
│   └── immunology/         # Immunology experiments
├── data/                   # Data directory
│   ├── external/           # External datasets
│   └── processed/          # Processed data
├── results/                # Experiment results
└── configs/                # Configuration files
```

## Usage in Research Notebooks

```python
# In a Jupyter notebook under research/
import sys
sys.path.insert(0, str(Path("../..").resolve()))

from src.research import get_data_path, get_results_path

# Load data
data_path = get_data_path("external/github/HIV-data")
df = pd.read_csv(data_path / "sequences.csv")

# Save results
results_path = get_results_path("my_experiment")
results_path.mkdir(exist_ok=True)
df.to_csv(results_path / "analysis.csv")
```

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Path utilities and listing functions |

## Best Practices

1. **Use path functions**: Don't hardcode paths in research scripts
2. **Relative to project root**: All paths are relative to project root
3. **Check existence**: Use `get_config_path()` pattern for optional files
4. **Create directories**: Results directories are created on demand
