# Observability Module

Decoupled observability layer for training monitoring.

## Purpose

This module provides observability components that are **decoupled from the training loop**, ensuring training is not blocked by I/O operations:

- **MetricsBuffer**: In-memory buffer (zero I/O during training)
- **AsyncTensorBoardWriter**: Async I/O in background thread
- **CoverageEvaluator**: Vectorized coverage evaluation

## Architecture

```
Training Loop                    Observability Layer
─────────────                    ───────────────────
train_epoch()
     │
     ├──> buffer.record()  ──────> MetricsBuffer (in-memory)
     │                                   │
     └──> evaluator.evaluate()           │ (drain periodically)
               │                         v
               │              AsyncTensorBoardWriter
               │                         │
               v                         v (background thread)
          CoverageStats            TensorBoard files
```

## Metrics Buffer

Record metrics without blocking:

```python
from src.observability import MetricsBuffer, ScopedMetrics

buffer = MetricsBuffer(max_size=1000)

# Record individual metrics
buffer.record("loss", 0.5, step=100)
buffer.record("accuracy", 0.95, step=100)

# Use scoped context for batch recording
with ScopedMetrics(buffer, step=100) as metrics:
    metrics.record("train/loss", train_loss)
    metrics.record("train/accuracy", train_acc)
```

## Async TensorBoard Writer

Write to TensorBoard without blocking training:

```python
from src.observability import create_writer, AsyncTensorBoardWriter

# Create async writer (or NullWriter if disabled)
writer = create_writer("runs/experiment1", enabled=True)

# Write scalars (non-blocking)
writer.add_scalar("loss", 0.5, step=100)
writer.add_histogram("embeddings", embeddings, step=100)

# Flush when ready (typically end of epoch)
writer.flush()

# Close when done
writer.close()
```

### Null Writer for Benchmarks

```python
from src.observability import NullWriter

# Use null writer to disable all observability
writer = NullWriter()  # All operations are no-ops
```

## Coverage Evaluation

Evaluate model coverage efficiently:

```python
from src.observability import CoverageEvaluator, evaluate_model_coverage

# Create evaluator
evaluator = CoverageEvaluator(model, dataloader)

# Evaluate coverage
stats = evaluator.evaluate()
print(f"Unique operations: {stats.unique_count}")
print(f"Coverage: {stats.coverage_pct:.2f}%")

# Or use convenience function
stats = evaluate_model_coverage(model, dataloader)
```

## Logging

Structured logging with colors:

```python
from src.observability import setup_logging, get_logger, LogContext

# Setup logging once at start
setup_logging(level="INFO", structured=True)

# Get logger for module
logger = get_logger(__name__)

# Log with context
with LogContext(experiment="exp1", epoch=10):
    logger.info("Training started")
    logger.info("Loss: 0.5", extra={"loss": 0.5})
```

## Training History

Track training progress across runs:

```python
from src.observability import TrainingHistory, TrainingState

history = TrainingHistory()

# Record epoch results
history.add_epoch(
    epoch=10,
    train_loss=0.5,
    val_loss=0.6,
    metrics={"accuracy": 0.95}
)

# Get state for checkpointing
state = history.get_state()

# Check for early stopping
if history.should_stop(patience=10):
    print("Early stopping triggered")
```

## Files

| File | Description |
|------|-------------|
| `metrics_buffer.py` | In-memory metrics buffer |
| `async_writer.py` | Async TensorBoard writer |
| `coverage.py` | Coverage evaluation |
| `logging.py` | Structured logging utilities |
| `training_history.py` | Training progress tracking |

## Performance Benefits

1. **Zero I/O during training**: Metrics buffered in memory
2. **Single flush per epoch**: Batched writes instead of per-step
3. **Vectorized coverage**: GPU-accelerated evaluation
4. **Easy disable**: NullWriter for benchmarks
