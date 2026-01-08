# DVC (Data Version Control) Guide

**Doc-Type:** Infrastructure Guide · Version 1.0 · Created 2026-01-05 · AI Whisperers

---

## Overview

This project uses **DVC (Data Version Control)** to manage large datasets without bloating the git repository.

**Key Benefits:**
- ✅ **Automatic checksums** - MD5 hashes computed automatically
- ✅ **Zero-copy** - efficient C libraries, not Python loops
- ✅ **Git integration** - `.dvc` files tracked in git, data stored separately
- ✅ **Version control** - for datasets, checkpoints, models
- ✅ **Remote storage** - local, S3, GCS, Azure, SSH, etc.

---

## Why DVC Instead of Custom Checksum Scripts?

**DVC provides:**
- Industry-standard tool (actively maintained)
- Automatic integrity verification
- Efficient hashing (zero-copy, optimized C libraries)
- Seamless git integration
- Built-in remote storage support
- No manual maintenance needed

**Custom scripts are:**
- ❌ Bloat and overhead
- ❌ Manual maintenance required
- ❌ Not zero-copy (inefficient)
- ❌ Not future-proof
- ❌ Reinventing the wheel

---

## Tracked Datasets

DVC currently tracks these large files:

| File | Size | Purpose | Status |
|------|------|---------|--------|
| `research/big_data/swissprot_cif_v6.tar` | 37.3 GB | AlphaFold3 structures | ✅ Tracked |
| `deliverables/.../S669.zip` | 42 MB | DDG benchmark | ✅ Tracked |

**Total tracked:** ~37.4 GB

---

## Quick Start

### Pull All Datasets

```bash
# Pull all DVC-tracked files
dvc pull
```

### Pull Specific Dataset

```bash
# Pull only SwissProt CIF
dvc pull research/big_data/swissprot_cif_v6.tar.dvc
```

### Check DVC Status

```bash
# See which files need updating
dvc status

# See what's tracked
dvc list . --dvc-only
```

---

## How DVC Works

### File Tracking

When you add a file to DVC:

```bash
dvc add path/to/large_file.dat
```

DVC:
1. Computes MD5 hash of the file
2. Moves file to `.dvc/cache/`
3. Creates `large_file.dat.dvc` (small metadata file)
4. Adds `large_file.dat` to `.gitignore`

**You commit `.dvc` file to git, not the actual data.**

### Storage Structure

```
project/
├── .dvc/
│   ├── cache/          # Local DVC cache (not in git)
│   └── config          # DVC configuration
├── path/to/
│   ├── large_file.dat.dvc    # Tracked in git (tiny)
│   └── large_file.dat        # In .gitignore (not in git)
└── .gitignore                # Contains DVC-tracked files
```

---

## Adding New Datasets to DVC

### Single File

```bash
# Add file to DVC
dvc add data/new_dataset.tar

# Commit the .dvc file to git
git add data/new_dataset.tar.dvc data/.gitignore
git commit -m "Add new_dataset to DVC"

# Push data to DVC remote
dvc push
```

### Directory

```bash
# Add entire directory
dvc add data/large_experiments/

# Commit
git add data/large_experiments.dvc data/.gitignore
git commit -m "Add large_experiments to DVC"

# Push
dvc push
```

---

## Updating Tracked Datasets

After modifying a DVC-tracked file:

```bash
# Update DVC tracking
dvc add data/dataset.tar

# This updates dataset.tar.dvc with new hash

# Commit the change
git add data/dataset.tar.dvc
git commit -m "Update dataset (describe changes)"

# Push new version to remote
dvc push
```

---

## Remote Storage

### Current Configuration

**Remote:** Local filesystem
**Location:** `C:/Users/Gestalt/dvc-storage`

```bash
# View remotes
dvc remote list

# Show remote details
cat .dvc/config
```

### Upgrading to Cloud Storage (Future)

**AWS S3:**
```bash
dvc remote add -d s3remote s3://mybucket/dvc-storage
dvc remote modify s3remote region us-west-2
```

**Google Cloud Storage:**
```bash
dvc remote add -d gcs gs://mybucket/dvc-storage
```

**Azure Blob:**
```bash
dvc remote add -d azure azure://mycontainer/dvc-storage
dvc remote modify azure connection_string 'your-connection-string'
```

**SSH/SFTP:**
```bash
dvc remote add -d ssh ssh://user@server/path/to/dvc-storage
```

---

## Collaboration Workflow

### Clone Repository (First Time)

```bash
# Clone git repo
git clone <repo-url>
cd ternary-vaes

# Pull DVC-tracked files
dvc pull
```

### Daily Workflow

```bash
# Pull latest code + data
git pull
dvc pull

# ... work on project ...

# Push changes
git push        # Code changes
dvc push        # Data changes (if any)
```

### Handling Conflicts

If `.dvc` files conflict:

```bash
# Resolve git conflict first
git merge ...

# Then sync data
dvc checkout
```

---

## Best Practices

### What to Track with DVC

✅ **Do track:**
- Large datasets (>10 MB)
- Model checkpoints
- Pre-trained embeddings
- Binary data files
- Large archives (.tar, .zip)

❌ **Don't track:**
- Source code (use git)
- Small config files (use git)
- Temporary files
- Generated outputs (regenerate instead)

### File Size Guidelines

| Size | Tool |
|------|------|
| <1 MB | Git |
| 1-100 MB | Git LFS or DVC |
| >100 MB | DVC |
| >1 GB | DVC (required) |

### Commit Messages

**For .dvc file updates:**
```bash
git commit -m "Update S669 dataset (added 200 new mutations)"
git commit -m "Add SwissProt CIF v7 (AlphaFold3 2026 release)"
```

---

## Troubleshooting

### File Not Found After dvc pull

```bash
# Check DVC status
dvc status

# Re-checkout files
dvc checkout

# Force pull
dvc pull --force
```

### Cache Issues

```bash
# Verify cache integrity
dvc cache dir

# Clean unused cache
dvc gc --workspace
```

### Large File Taking Forever

DVC shows progress bars. For very large files (>10 GB):
- First `dvc add` computes hash (one-time, slow)
- Subsequent `dvc push/pull` uses hash (fast)
- Be patient with first-time adds

### Wrong Remote

```bash
# Check current remote
dvc remote list

# Change default remote
dvc remote default <remote-name>
```

---

## Advanced Usage

### Pipelines (Future)

DVC can track data pipelines:

```yaml
# dvc.yaml
stages:
  preprocess:
    cmd: python preprocess.py
    deps:
      - raw_data.tar
    outs:
      - processed_data.pkl
```

### Metrics Tracking

```bash
# Track metrics file
dvc metrics show

# Compare across branches
dvc metrics diff main my-branch
```

### Experiments

```bash
# Run experiment
dvc exp run

# Compare experiments
dvc exp show
```

---

## Integration with Git

### .gitignore

DVC automatically updates `.gitignore`:

```gitignore
# DVC-tracked files
/research/big_data/swissprot_cif_v6.tar
/deliverables/partners/jose_colbes/reproducibility/data/S669.zip
```

### Git LFS vs DVC

| Feature | Git LFS | DVC |
|---------|---------|-----|
| Storage | GitHub LFS, custom | Any remote (S3, GCS, etc.) |
| Versioning | Yes | Yes |
| Pipelines | No | Yes |
| Metrics | No | Yes |
| Free tier | 1 GB | Unlimited (self-hosted) |

**Recommendation:** Use DVC for large datasets, Git LFS for medium files if needed.

---

## CI/CD Integration

### GitHub Actions

```yaml
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: iterative/setup-dvc@v1

      - name: Pull DVC data
        run: dvc pull

      - name: Run tests
        run: pytest
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/iterative/dvc
    rev: main
    hooks:
      - id: dvc-pre-commit
      - id: dvc-post-checkout
```

---

## Monitoring

### Check DVC Cache Size

```bash
# Linux/Mac
du -sh .dvc/cache

# Windows
dir .dvc\cache /s
```

### List Tracked Files

```bash
# All DVC files
find . -name "*.dvc"

# With sizes
dvc list . --dvc-only --show-json | jq
```

---

## Migration from Custom Checksums

**Old approach (removed):**
- ❌ Custom Python scripts (`generate_checksums.py`, `verify_checksums.py`)
- ❌ Manual JSON file (`checksums.json`)
- ❌ Per-directory `checksums.txt` files

**New approach (DVC):**
- ✅ `dvc add` automatically computes checksums
- ✅ `dvc pull` automatically verifies integrity
- ✅ `.dvc` files in git (small, automatic)
- ✅ No manual maintenance

---

## Resources

**Official Documentation:**
- Main docs: https://dvc.org/doc
- Get started: https://dvc.org/doc/start
- User guide: https://dvc.org/doc/user-guide

**Community:**
- GitHub: https://github.com/iterative/dvc
- Forum: https://discuss.dvc.org
- Discord: https://dvc.org/chat

**Tutorials:**
- Data versioning: https://dvc.org/doc/use-cases/versioning-data-and-models
- Pipelines: https://dvc.org/doc/start/data-pipelines

---

## Summary

**DVC provides:**
- ✅ Zero-copy, efficient checksumming
- ✅ Automatic integrity verification
- ✅ Git integration
- ✅ Remote storage flexibility
- ✅ Industry-standard tool

**No need for:**
- ❌ Custom checksum scripts
- ❌ Manual verification
- ❌ Reinventing the wheel

**Simple workflow:**
```bash
dvc pull   # Get data
# ... work ...
dvc push   # Share data
```

---

**Last Updated:** 2026-01-05
**Maintainer:** AI Whisperers
**Status:** Active (local remote, can upgrade to cloud later)
