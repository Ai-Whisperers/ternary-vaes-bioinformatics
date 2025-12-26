# Artifacts Module

Checkpoint and artifact lifecycle management.

## Purpose

This module handles the complete lifecycle of training artifacts:
- Checkpoint saving and loading
- Metadata management
- Artifact promotion (raw → validated → production)

## CheckpointManager

Manage model checkpoints:

```python
from src.artifacts import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir="checkpoints",
    max_checkpoints=5  # Keep only last 5
)

# Save checkpoint
manager.save(
    model=model,
    optimizer=optimizer,
    epoch=100,
    metrics={"loss": 0.5, "accuracy": 0.95}
)

# Load latest checkpoint
checkpoint = manager.load_latest()
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])

# Load specific checkpoint
checkpoint = manager.load(epoch=50)

# Load best checkpoint by metric
best = manager.load_best(metric="loss", mode="min")
```

## Checkpoint Structure

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": 100,
    "metrics": {"loss": 0.5, "accuracy": 0.95},
    "config": config_dict,
    "metadata": {
        "timestamp": "2024-01-15T10:30:00",
        "git_hash": "abc123",
        "python_version": "3.11.0"
    }
}
```

## Automatic Cleanup

The manager automatically removes old checkpoints:

```python
manager = CheckpointManager(
    checkpoint_dir="checkpoints",
    max_checkpoints=5,      # Keep last 5 by epoch
    keep_best=True,         # Always keep best checkpoint
    metric_for_best="loss"  # Metric to determine best
)
```

## Artifact Promotion Pipeline

```
Raw Checkpoint          Validated              Production
(training output)  →  (passed tests)  →  (ready for inference)
      │                    │                    │
      └── checkpoints/     └── validated/       └── production/
          epoch_100.pt         v1.0.0.pt            model.pt
```

## Files

| File | Description |
|------|-------------|
| `checkpoint_manager.py` | CheckpointManager implementation |

## Best Practices

1. **Regular saves**: Save checkpoints every N epochs
2. **Keep best**: Always keep the best-performing checkpoint
3. **Include metadata**: Add git hash, config for reproducibility
4. **Version production**: Use semantic versioning for production artifacts
