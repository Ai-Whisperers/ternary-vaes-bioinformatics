# Immediate Tasks Implementation Plan

**Doc-Type:** Implementation Plan · Version 1.0 · Created 2026-01-03 · AI Whisperers

---

## Overview

This document outlines the implementation plan for the three immediate priority tasks identified in the Training Infrastructure and Dataset audits.

---

## Task 1: Complete V5.12.2 Research Script Fixes

### Status Summary

**Total norm() calls analyzed:** 278
**Core files (COMPLETE):** 11 files (HIGH priority)
**Research scripts (PENDING):** ~40 files needing fixes

### Issue

Many research scripts use Euclidean `.norm()` on hyperbolic Poincaré ball embeddings instead of `poincare_distance()`.

**Impact:**
- Incorrect radial hierarchy computation
- Metric correlations computed in wrong geometry
- Training scripts producing misleading results

### Correct Pattern

```python
# WRONG - Euclidean norm on hyperbolic embeddings
radius = torch.norm(z_hyp, dim=-1)

# CORRECT - Hyperbolic distance from origin
from src.geometry import poincare_distance
origin = torch.zeros_like(z_hyp)
radius = poincare_distance(z_hyp, origin, c=curvature)
```

### Files Needing Fixes (from V5.12.2 audit)

**Priority 6: Research Scripts - HIV (4 files, ~12 usages)**
- `hiv/scripts/03_hiv_handshake_analysis.py` (lines 274, 283, 295)
- `hiv/src/03_hiv_handshake_analysis.py` (lines 267, 276, 288) - duplicate
- `hiv/scripts/analyze_tropism_switching.py` (lines 273, 340)
- `hiv/scripts/esm2_integration.py` (lines 279, 283, 460, 615, 626, 720)

**Priority 7: Research Scripts - RA (3 files, ~8 usages)**
- `rheumatoid_arthritis/scripts/03_citrullination_analysis.py` (lines 273, 279)
- `rheumatoid_arthritis/scripts/04_codon_optimizer.py` (lines 228, 273, 323, 324)
- `rheumatoid_arthritis/scripts/cross_validate_encoder.py` (line 185)

**Priority 8: Research Scripts - Genetic Code (5 files, ~12 usages)**
- `genetic_code/scripts/04_fast_reverse_search.py` (lines 138, 184)
- `genetic_code/scripts/07_extract_v5_11_3_embeddings.py` (lines 136, 137)
- `genetic_code/scripts/09_train_codon_encoder_3adic.py` (line 412)
- `genetic_code/scripts/10_extract_fused_embeddings.py` (lines 151, 152)
- `genetic_code/scripts/11_train_codon_encoder_fused.py` (lines 166, 302, 395, 459)

**Priority 9: Research Scripts - Spectral Analysis (6 files, ~18 usages)**
- `spectral_analysis/scripts/01_extract_embeddings.py` (lines 253, 254)
- `spectral_analysis/scripts/04_padic_spectral_analysis.py` (line 94)
- `spectral_analysis/scripts/05_exact_padic_analysis.py` (line 242)
- `spectral_analysis/scripts/07_adelic_analysis.py` (lines 65, 134, 194, 275, 280, 296, 341)
- `spectral_analysis/scripts/08_alternative_spectral_operators.py` (lines 106, 196)
- `spectral_analysis/scripts/09_binary_ternary_decomposition.py` (lines 115, 181, 289)
- `spectral_analysis/scripts/10_semantic_amplification_benchmark.py` (line 195)

**Total:** ~18 files, ~50 usages

### Implementation Approach

1. **Batch Fix Script:** Create automated fix script for common patterns
2. **Manual Review:** Cases requiring context-specific fixes
3. **Testing:** Verify fixes don't break functionality
4. **Documentation:** Update comments to explain hyperbolic geometry

### Estimated Time

- Automated fixes: 1-2 hours
- Manual review: 2-3 hours
- Testing: 1-2 hours
- **Total:** 4-7 hours

---

## Task 2: Clarify ProteInGym Status

### Current Situation

**Directory:** `research/codon-encoder/data/proteingym/` (EMPTY)
**Script:** `research/codon-encoder/analysis/proteingym_pipeline.py` (EXISTS)
**Actual Usage:** S669 dataset (deliverables/partners/jose_colbes/reproducibility/data/S669.zip)

### Analysis

**References to ProteinGym:**
- `proteingym_pipeline.py` - designed to download ProteinGym substitution benchmark
- `README.md` - mentions "Larger Datasets: ProteinGym, Mega-scale mutational scans"
- `EXPECTATION_MATRIX.md` - recommends larger datasets

**References to S669:**
- `train_codon_encoder.py` - uses S669 for DDG validation
- `ddg_hyperbolic_training.py` - loads S669 dataset
- `ddg_benchmark.py` - validates against S669
- `multimodal_ddg_predictor.py` - loads S669 (n=52)

**Current Performance (S669):**
- TrainableCodonEncoder: LOO Spearman **0.61**
- Rosetta ddg_monomer: LOO Spearman 0.69 (structure-based)
- **Conclusion:** S669 is sufficient for current validation

### Decision Matrix

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| **Download ProteinGym** | Larger dataset (~500 MB), more proteins | Overkill for current needs, adds complexity | Low priority |
| **Keep S669 only** | Sufficient for validation, already working | Smaller sample size (n=52) | **RECOMMENDED** |
| **Document status** | Clarifies intent, prevents confusion | - | **DO THIS** |

### Recommended Actions

1. **Document S669 sufficiency** in README.md
2. **Mark ProteInGym as optional** (future enhancement)
3. **Update proteingym_pipeline.py docstring** to clarify status
4. **Add note** in empty directory: "ProteinGym download optional, S669 sufficient for current work"

### Implementation

Create `research/codon-encoder/data/proteingym/README.md`:

```markdown
# ProteinGym Dataset (Optional)

**Status:** Not Downloaded - S669 is sufficient for current DDG validation

## Current Approach

The project uses the **S669 dataset** for DDG (ΔΔG) prediction validation:
- Location: `deliverables/partners/jose_colbes/reproducibility/data/S669.zip`
- Size: 43 MB (2,800 mutations across 17 proteins)
- Performance: TrainableCodonEncoder achieves LOO Spearman **0.61**

## ProteinGym (Optional Future Enhancement)

ProteinGym provides a larger benchmark dataset (~500 MB):
- URL: https://marks.hms.harvard.edu/proteingym/ProteinGym_substitutions.zip
- Size: ~500 MB uncompressed
- Coverage: 200k+ proteins with DMS (Deep Mutational Scanning) data

### When to Download

Download ProteinGym if:
- Extending validation beyond S669's 17 proteins
- Benchmarking across broader protein families
- Research requires mega-scale mutational scans

### How to Download

```bash
python research/codon-encoder/analysis/proteingym_pipeline.py --download
```

## Recommendation

**For most use cases, S669 is sufficient.** ProteinGym is a future enhancement for large-scale validation.
```

### Estimated Time

- Documentation: 30 minutes
- Code updates: 30 minutes
- **Total:** 1 hour

---

## Task 3: Generate Checksums for Datasets >1MB

### Purpose

- Data integrity verification
- Detect corruption during transfer
- Enable reproducibility

### Datasets Requiring Checksums

**Large Datasets (>10 MB):**
1. `research/big_data/swissprot_cif_v6.tar` (38 GB)
2. `deliverables/partners/jose_colbes/reproducibility/data/S669.zip` (43 MB)
3. `data/research/catnap_assay.txt` (15 MB)
4. `research/codon-encoder/data/v5_11_3_embeddings.pt` (6.0 MB)
5. `research/codon-encoder/data/fused_embeddings.pt` (6.0 MB)

**Medium Datasets (1-10 MB):**
6. `data/external/github/HIV-data/` (~53 MB total)
7. `data/external/github/HIV-1_Paper/` (~4.1 MB total)
8. `data/research/stanford_hivdb_nnrti.txt` (1.7 MB)
9. `data/research/stanford_hivdb_nrti.txt` (1.2 MB)
10. `data/external/huggingface/human_hiv_ppi/` (~3.3 MB total)
11. `research/contact-prediction/checkpoints/v5_11_structural_best.pt` (1.4 MB)

### Checksum Algorithm

Use **SHA256** (more secure than MD5, standard for data integrity)

### Implementation

**Script:** `scripts/generate_checksums.py`

```python
#!/usr/bin/env python3
"""Generate SHA256 checksums for all datasets >1MB."""

import hashlib
from pathlib import Path
import json

def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def find_large_files(root: Path, min_size_mb: float = 1.0) -> list[Path]:
    """Find all files larger than min_size_mb."""
    min_bytes = int(min_size_mb * 1024 * 1024)
    large_files = []

    for file in root.rglob('*'):
        if file.is_file() and file.stat().st_size >= min_bytes:
            # Skip .git, __pycache__, etc.
            if not any(part.startswith('.') or part == '__pycache__'
                      for part in file.parts):
                large_files.append(file)

    return sorted(large_files, key=lambda f: f.stat().st_size, reverse=True)

def main():
    project_root = Path(__file__).parent.parent

    # Directories to scan
    scan_dirs = [
        project_root / 'data',
        project_root / 'research',
        project_root / 'deliverables',
    ]

    checksums = {}

    print("Scanning for files >1MB...")
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue

        large_files = find_large_files(scan_dir, min_size_mb=1.0)

        for file in large_files:
            rel_path = file.relative_to(project_root)
            size_mb = file.stat().st_size / (1024 * 1024)

            print(f"Computing checksum: {rel_path} ({size_mb:.1f} MB)")
            sha256 = compute_sha256(file)

            checksums[str(rel_path)] = {
                'sha256': sha256,
                'size_mb': round(size_mb, 2)
            }

    # Save to JSON
    output_file = project_root / 'data' / 'checksums.json'
    with open(output_file, 'w') as f:
        json.dump(checksums, f, indent=2, sort_keys=True)

    print(f"\nGenerated checksums for {len(checksums)} files")
    print(f"Saved to: {output_file}")

    # Also create per-directory checksums.txt files
    by_dir = {}
    for filepath, info in checksums.items():
        dir_path = str(Path(filepath).parent)
        if dir_path not in by_dir:
            by_dir[dir_path] = []
        by_dir[dir_path].append((Path(filepath).name, info['sha256']))

    for dir_path, files in by_dir.items():
        checksum_file = project_root / dir_path / 'checksums.txt'
        checksum_file.parent.mkdir(parents=True, exist_ok=True)

        with open(checksum_file, 'w') as f:
            f.write(f"# SHA256 checksums for {dir_path}\n")
            f.write(f"# Generated: 2026-01-03\n\n")
            for filename, sha256 in sorted(files):
                f.write(f"{sha256}  {filename}\n")

        print(f"Created: {checksum_file.relative_to(project_root)}")

if __name__ == '__main__':
    main()
```

**Verification Script:** `scripts/verify_checksums.py`

```python
#!/usr/bin/env python3
"""Verify dataset integrity using stored checksums."""

import hashlib
import json
from pathlib import Path
import sys

def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def main():
    project_root = Path(__file__).parent.parent
    checksums_file = project_root / 'data' / 'checksums.json'

    if not checksums_file.exists():
        print(f"ERROR: {checksums_file} not found")
        print("Run scripts/generate_checksums.py first")
        sys.exit(1)

    with open(checksums_file) as f:
        checksums = json.load(f)

    print(f"Verifying {len(checksums)} files...")
    errors = []

    for filepath, info in checksums.items():
        full_path = project_root / filepath

        if not full_path.exists():
            errors.append(f"MISSING: {filepath}")
            continue

        actual_sha256 = compute_sha256(full_path)
        expected_sha256 = info['sha256']

        if actual_sha256 == expected_sha256:
            print(f"✓ {filepath}")
        else:
            errors.append(f"CORRUPT: {filepath}")
            print(f"✗ {filepath}")
            print(f"  Expected: {expected_sha256}")
            print(f"  Actual:   {actual_sha256}")

    if errors:
        print(f"\n{len(errors)} ERRORS:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print(f"\n✓ All {len(checksums)} files verified successfully")

if __name__ == '__main__':
    main()
```

### Estimated Time

- Script creation: 1 hour
- Checksum generation: 30 minutes (38 GB SwissProt takes time)
- Testing: 30 minutes
- **Total:** 2 hours

---

## Task 4: Implement DVC for Large Datasets

### Purpose

- Version control for large datasets (>10 MB)
- Track dataset changes without bloating git
- Enable collaboration with shared remote storage

### DVC Setup

**Install DVC:**
```bash
pip install dvc
```

**Initialize DVC:**
```bash
dvc init
```

**Configure Remote (Local for now, can upgrade to cloud later):**
```bash
dvc remote add -d local-storage ~/dvc-cache
```

### Datasets to Track with DVC

1. **SwissProt CIF v6** (38 GB)
   - `research/big_data/swissprot_cif_v6.tar`
   - Justification: Massive file, would break git

2. **S669 Dataset** (43 MB)
   - `deliverables/partners/jose_colbes/reproducibility/data/S669.zip`
   - Justification: Large binary, versioning important

3. **CATNAP Assay** (15 MB)
   - `data/research/catnap_assay.txt`
   - Justification: Large text file, periodic updates

4. **Embeddings** (6 MB each)
   - `research/codon-encoder/data/v5_11_3_embeddings.pt`
   - `research/codon-encoder/data/fused_embeddings.pt`
   - Justification: Large binaries, version-specific

### Implementation Steps

1. **Add datasets to DVC:**
   ```bash
   dvc add research/big_data/swissprot_cif_v6.tar
   dvc add deliverables/partners/jose_colbes/reproducibility/data/S669.zip
   dvc add data/research/catnap_assay.txt
   dvc add research/codon-encoder/data/v5_11_3_embeddings.pt
   dvc add research/codon-encoder/data/fused_embeddings.pt
   ```

2. **Commit .dvc files to git:**
   ```bash
   git add *.dvc .dvc/.gitignore
   git commit -m "feat: Add DVC tracking for large datasets"
   ```

3. **Push to DVC remote:**
   ```bash
   dvc push
   ```

4. **Update .gitignore:**
   ```
   # DVC-tracked files
   /research/big_data/swissprot_cif_v6.tar
   /deliverables/partners/jose_colbes/reproducibility/data/S669.zip
   /data/research/catnap_assay.txt
   /research/codon-encoder/data/v5_11_3_embeddings.pt
   /research/codon-encoder/data/fused_embeddings.pt
   ```

### Documentation

Create `docs/infrastructure/DVC_GUIDE.md`:

```markdown
# DVC (Data Version Control) Guide

## Overview

This project uses DVC to version control large datasets (>10 MB) without bloating the git repository.

## Tracked Datasets

- SwissProt CIF v6 (38 GB) - AlphaFold3 structures
- S669 DDG benchmark (43 MB) - Protein stability data
- CATNAP assay (15 MB) - HIV neutralization data
- VAE embeddings (6 MB each) - Codon encoder outputs

## Quick Start

**Pull all datasets:**
```bash
dvc pull
```

**Pull specific dataset:**
```bash
dvc pull research/big_data/swissprot_cif_v6.tar.dvc
```

**Check status:**
```bash
dvc status
```

## Updating Datasets

After modifying a tracked dataset:

```bash
dvc add research/codon-encoder/data/v5_11_3_embeddings.pt
git add research/codon-encoder/data/v5_11_3_embeddings.pt.dvc
git commit -m "Update v5_11_3 embeddings (new training)"
dvc push
```

## Remote Storage

**Current:** Local cache (`~/dvc-cache`)
**Future:** Can upgrade to S3, GCS, Azure, etc.

## Troubleshooting

**File not found:**
```bash
dvc pull  # Download from remote
```

**Conflict:**
```bash
dvc checkout  # Reset to committed version
```

See: https://dvc.org/doc
```

### Estimated Time

- DVC setup: 1 hour
- Dataset tracking: 2 hours (including 38 GB upload)
- Documentation: 1 hour
- **Total:** 4 hours

---

## Task 5: Document SwissProt CIF Extraction Plan

### Purpose

Create a comprehensive plan for utilizing the 38 GB SwissProt CIF dataset for research applications.

### Implementation

Create `docs/research/SWISSPROT_CIF_EXTRACTION_PLAN.md` (see separate document)

### Key Sections

1. **Dataset Overview** - Contents, format, size
2. **Potential Applications** - Contact prediction, DDG enhancement, codon-structure mining
3. **Technical Requirements** - CIF parser, batch processing, memory management
4. **Extraction Pipeline** - Sample extraction, feature extraction, validation
5. **Feature Extraction** - RSA, secondary structure, pLDDT, contact number
6. **Performance Considerations** - Parallelization, chunking, caching
7. **Integration Points** - Contact prediction, DDG predictor, codon encoder
8. **Testing Strategy** - Sample proteins, known structures, benchmarking
9. **Scalability Plan** - 100 proteins → 1,000 → 10,000 → 200,000
10. **Timeline & Resources** - Phased approach

### Estimated Time

- Research & planning: 2 hours
- Documentation writing: 2 hours
- **Total:** 4 hours

---

## Summary Timeline

| Task | Est. Time | Priority | Dependencies |
|------|-----------|----------|--------------|
| **V5.12.2 Research Fixes** | 4-7 hours | HIGH | None |
| **Clarify ProteInGym** | 1 hour | HIGH | None |
| **Generate Checksums** | 2 hours | HIGH | None |
| **Implement DVC** | 4 hours | MEDIUM | Checksums complete |
| **SwissProt CIF Plan** | 4 hours | MEDIUM | None |
| **TOTAL** | **15-18 hours** | - | - |

## Recommended Order

1. **ProteInGym clarification** (1 hour) - Quick win, removes confusion
2. **Generate checksums** (2 hours) - Data integrity foundation
3. **V5.12.2 fixes** (4-7 hours) - Core functionality correctness
4. **Implement DVC** (4 hours) - Builds on checksums
5. **SwissProt CIF plan** (4 hours) - Future research direction

**Total Execution Time:** 15-18 hours over 2-3 days

---

**Next Steps:**

1. Review and approve this plan
2. Begin execution in recommended order
3. Track progress using TodoWrite
4. Document any deviations or issues
5. Update audits upon completion
