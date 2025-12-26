# Benchmark Module

Benchmarking utilities for Ternary VAE evaluation.

## Purpose

This module provides utilities for:
- Standardized benchmark execution
- Model loading and configuration
- Result collection and export

## BenchmarkBase

Base class for creating benchmarks:

```python
from src.benchmark import BenchmarkBase

class MyBenchmark(BenchmarkBase):
    def setup(self):
        # Load model, data, etc.
        self.model = self.load_model()
        self.data = self.load_data()

    def run(self):
        # Execute benchmark
        results = {}
        for batch in self.data:
            output = self.model(batch)
            results["metric"] = compute_metric(output)
        return results

# Run benchmark
benchmark = MyBenchmark(config_path="configs/benchmark.yaml")
results = benchmark.execute()
```

## Utility Functions

### Device Selection

```python
from src.benchmark import get_device

# Auto-select best available device
device = get_device()  # Returns "cuda" if available, else "cpu"
```

### Model Creation

```python
from src.benchmark import create_v5_6_model, load_checkpoint_safe

# Create model from config
model = create_v5_6_model(config)

# Load checkpoint with error handling
checkpoint = load_checkpoint_safe("checkpoints/model.pt")
if checkpoint:
    model.load_state_dict(checkpoint["model"])
```

### Configuration Loading

```python
from src.benchmark import load_config

# Load benchmark configuration
config = load_config("configs/benchmark.yaml")
```

### Result Export

```python
from src.benchmark import save_results, convert_to_python_types

# Convert tensors to Python types for JSON serialization
results = convert_to_python_types({
    "accuracy": torch.tensor(0.95),
    "loss": torch.tensor([0.1, 0.2, 0.3])
})

# Save results
save_results(results, "results/benchmark.json")
```

## Standard Benchmarks

### Coverage Benchmark

Measure model coverage of ternary operations:

```bash
python scripts/benchmark/coverage.py --checkpoint model.pt
```

### Latency Benchmark

Measure inference latency:

```bash
python scripts/benchmark/latency.py --checkpoint model.pt --batch-size 256
```

### Ranking Correlation Benchmark

Measure p-adic structure preservation:

```bash
python scripts/benchmark/ranking.py --checkpoint model.pt
```

## Files

| File | Description |
|------|-------------|
| `utils.py` | Benchmark utilities and base class |

## Benchmark Reports

Results are saved in JSON format:

```json
{
    "benchmark": "coverage",
    "timestamp": "2024-01-15T10:30:00",
    "config": {"batch_size": 256, ...},
    "results": {
        "coverage": 0.98,
        "unique_operations": 19289,
        "latency_ms": 12.5
    },
    "metadata": {
        "model_version": "v5.11",
        "checkpoint": "epoch_500.pt"
    }
}
```
