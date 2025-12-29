# Outputs Directory

This directory contains generated visualization outputs.

## Structure

```
outputs/
└── viz/          # Visualization files (PNG, SVG, HTML)
```

## Related Output Directories

| Directory | Purpose | Location |
|-----------|---------|----------|
| **results/** | Training results, analysis outputs, research findings | `./results/` |
| **runs/** | Active training run checkpoints | `./runs/` |
| **reports/** | Generated reports and audits | `./reports/` |
| **sandbox-training/** | Development/testing artifacts | `./sandbox-training/` |

## Usage

Visualization scripts typically save outputs here:

```python
from pathlib import Path

OUTPUT_DIR = Path("outputs/viz")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save visualization
fig.savefig(OUTPUT_DIR / "my_plot.png")
```

## See Also

- [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) - Full directory structure guide
- [results/README.md](../results/README.md) - Results directory documentation
