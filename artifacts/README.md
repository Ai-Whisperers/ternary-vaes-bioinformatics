# Artifacts Directory

This directory contains model artifacts organized by lifecycle stage.

## Directory Structure

```
artifacts/
├── raw/           # Direct training outputs (checkpoints, metrics, logs)
├── validated/     # Models that passed validation tests
└── production/    # Deployment-ready models
```

## Artifact Lifecycle

### 1. Raw Artifacts

**Purpose:** Store direct outputs from training sessions

**Contents:**
- Checkpoints (every N epochs + best + latest)
- Training metrics (JSON format)
- Training logs
- Configuration files used
- Manifest with metadata

**Example:**
```
raw/v5_5_20251123_103epochs/
├── checkpoints/
│   ├── manifest.json
│   ├── latest.pt
│   ├── best.pt
│   └── epoch_*.pt
├── metrics/
│   ├── training_metrics.json
│   └── coverage_history.json
├── config.yaml
├── training.log
└── README.md
```

### 2. Validated Artifacts

**Purpose:** Models that passed comprehensive validation tests

**Requirements:**
- Coverage > target threshold
- Generalization tests passed
- Entropy utilization validated
- No posterior collapse detected

**Contents:**
- Clean model weights (no optimizer state)
- Validation test results
- Original configuration
- Model card (metadata)

**Example:**
```
validated/v5_5_epoch70_validated/
├── model.pt
├── config.yaml
├── validation/
│   ├── test_results.json
│   ├── coverage_report.json
│   └── generalization_report.json
├── metadata.json
└── README.md
```

### 3. Production Artifacts

**Purpose:** Deployment-ready models approved for production use

**Requirements:**
- Passed validation stage
- Performance benchmarks completed
- Security review (if applicable)
- Deployment configuration ready

**Contents:**
- Optimized model weights
- Performance benchmarks
- Deployment configurations
- Complete documentation

**Example:**
```
production/v5_5_prod_v1.0/
├── model.pt
├── config.yaml
├── performance/
│   ├── benchmarks.json
│   └── resource_usage.json
├── deployment/
│   ├── docker/
│   └── kubernetes/
├── metadata.json
├── CHANGELOG.md
└── README.md
```

## Promotion Workflow

```
Training → raw/          (automatic)
         ↓
         validation tests
         ↓
         validated/      (manual promotion after tests pass)
         ↓
         approval process
         ↓
         production/     (manual promotion after approval)
```

## Usage

**Register Training Session:**
```python
from src.artifacts import ArtifactRepository

repo = ArtifactRepository('artifacts')
repo.register_training_session(
    session_id='v5_5_20251124',
    checkpoint_dir=Path('sandbox-training/checkpoints/v5_5'),
    config=config,
    metrics={'best_val_loss': 0.38, 'coverage': 0.99}
)
```

**Promote to Validated:**
```python
repo.promote_to_validated(
    session_id='v5_5_20251124',
    checkpoint_name='epoch_70',
    validation_results={...}
)
```

**Promote to Production:**
```python
repo.promote_to_production(
    validated_id='v5_5_20251124_epoch_70_validated',
    version='1.0.0',
    deployment_config={...}
)
```

## Notes

- Raw artifacts are automatically created during training
- Validation promotion requires passing all tests
- Production promotion requires manual approval
- Each stage maintains complete traceability
- Artifacts are immutable once promoted
