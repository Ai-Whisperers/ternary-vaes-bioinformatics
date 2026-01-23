# Full Extraction Audit - Deliverables Folder

**Doc-Type:** Extraction Planning · Version 2.0 · Updated 2026-01-23 · AI Whisperers

---

## Purpose

This document provides a comprehensive audit for extracting the `deliverables/` folder (or individual partner packages) to an independent repository. It includes:
- Complete file inventory with sizes and line counts
- Full transitive dependency chains
- Checkpoint file manifest
- Import statement locations with line numbers
- Checklist format for tracking extraction progress
- **NEW v2.0:** Data file inventory, environment variables, external APIs, sys.path analysis, security concerns

**Policy:** This document should be updated whenever dependencies change.

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Deliverables Size** | 28 MB |
| **Total Python Files** | 134 files |
| **Total Data Files (JSON/CSV/FASTA)** | 147+ files |
| **Total Source Files to Copy** | 14 files (~7,500 LOC) |
| **Total Checkpoint Size** | ~9.5 MB (external) + 7.2 MB (local) |
| **External pip Dependencies** | 8 packages |
| **External APIs** | 2 (Stanford HIVdb, NCBI Entrez) |
| **Environment Variables** | 3 (all optional) |
| **sys.path.insert() Calls** | 125+ locations |
| **Estimated Extraction Effort** | High (2-3 days) |

### Partner Package Sizes

| Package | Size | Python Files | Data Files |
|---------|------|--------------|------------|
| alejandra_rojas | 7.0 MB | ~35 | 50+ |
| carlos_brizuela | 8.8 MB | ~30 | 35+ |
| jose_colbes | 2.1 MB | ~25 | 8+ |
| hiv_research_package | 360 KB | ~15 | 5+ |
| shared/ | ~500 KB | 12 | 0 |

### Quick Decision Matrix

| Extraction Scenario | Effort | Code Duplication | Maintenance Burden |
|---------------------|--------|------------------|-------------------|
| Keep in main repo | None | None | None |
| Extract shared/ only | Low | ~2,700 LOC | Low |
| Extract partner packages (partial) | Medium | ~4,700 LOC | Medium |
| Full extraction with ML | High | ~7,500 LOC | High |

---

## 1. Complete Dependency Tree

### 1.1 Visual Dependency Graph

```
deliverables/                        [28 MB total, 134 Python files]
├── shared/                          [2,719 LOC - SELF-CONTAINED]
│   ├── config.py (153)             → NCBI_API_KEY env var
│   │                                → stanford_hivdb_url constant
│   ├── constants.py (167)
│   ├── hemolysis_predictor.py (399)
│   ├── logging_utils.py (304)
│   ├── peptide_utils.py (319)
│   ├── primer_design.py (557)
│   ├── uncertainty.py (372)
│   └── vae_service.py (348)        → src.models.TernaryVAEV5_11_PartialFreeze
│                                    → checkpoints/homeostatic_rich/best.pt
│
├── partners/jose_colbes/            [2.1 MB, ~70% SELF-CONTAINED]
│   ├── core/ (LOCAL)               ✓ padic_math.py, constants.py
│   ├── models/                     ✓ ddg_predictor.joblib (532 KB)
│   ├── reproducibility/data/       ✓ s669.csv (52 mutations)
│   ├── src/validated_ddg_predictor.py
│   │   └── EXTERNAL: TrainableCodonEncoder, poincare_distance
│   └── validation/bootstrap_test.py
│       └── EXTERNAL: TrainableCodonEncoder, poincare_distance
│
├── partners/alejandra_rojas/        [7.0 MB, ~85% SELF-CONTAINED]
│   ├── src/ (LOCAL)                ✓ padic_math.py, codons.py, constants.py
│   ├── results/                    ✓ 3.9 MB (50+ JSON/CSV/FASTA)
│   ├── scripts/denv4_*.py
│   │   └── EXTERNAL: TrainableCodonEncoder, poincare_distance
│   └── src/ncbi_client.py          → NCBI_API_KEY, NCBI_EMAIL env vars
│
├── partners/carlos_brizuela/        [8.8 MB, ~60% SELF-CONTAINED]
│   ├── src/ (LOCAL)                ✓ constants.py, peptide_utils.py, uncertainty.py
│   ├── models/                     ✓ 5 joblib files (587 KB total)
│   ├── checkpoints_definitive/     ✓ 6 PyTorch files (7.0 MB)
│   ├── results/                    ✓ 596 KB (35+ CSV)
│   ├── scripts/predict_mic.py
│   │   └── EXTERNAL: PeptideVAE
│   └── training/train_*.py
│       └── EXTERNAL: PeptideVAE, PeptideLossManager
│
└── partners/hiv_research_package/   [360 KB, ~90% SELF-CONTAINED]
    ├── results/                    ✓ 24 KB
    └── scripts/stanford_hivdb_client.py
        → https://hivdb.stanford.edu/graphql (external API)
```

### 1.2 Transitive Dependency Chain (Full)

```
TrainableCodonEncoder (586 LOC)
├── src.biology.codons (250 LOC)           [LEAF]
├── src.geometry (413 LOC total)
│   ├── poincare.py (356 LOC)              [geoopt]
│   └── __init__.py (57 LOC)               [LEAF]
└── src.encoders.codon_encoder (451 LOC)
    ├── src.biology.codons                 [ALREADY COUNTED]
    ├── src.core.padic_math (489 LOC)      [LEAF]
    └── src.geometry                       [ALREADY COUNTED]

PeptideVAE (1,059 LOC)
├── src.encoders.padic_amino_acid_encoder (832 LOC)
│   └── src.core.padic_math                [ALREADY COUNTED]
├── src.geometry                           [ALREADY COUNTED]
└── src.models.hyperbolic_projection (327 LOC)
    └── src.geometry.ManifoldParameter     [ALREADY COUNTED]

PeptideLossManager (862 LOC)
├── src.geometry.poincare_distance         [ALREADY COUNTED]
└── src.losses.base (216 LOC)              [LEAF]
```

---

## 2. Complete File Inventory

### 2.1 Main Project Source Files Required

| File | Lines | Size | Direct Dependencies | Extraction Status |
|------|-------|------|---------------------|:-----------------:|
| **TIER 1: Core Encoders** |||||
| `src/encoders/trainable_codon_encoder.py` | 586 | 22K | biology.codons, geometry, codon_encoder | [ ] |
| `src/encoders/peptide_encoder.py` | 1,059 | 38K | padic_aa_encoder, geometry, hyperbolic_projection | [ ] |
| `src/encoders/codon_encoder.py` | 451 | 16K | biology.codons, core.padic_math, geometry | [ ] |
| `src/encoders/padic_amino_acid_encoder.py` | 832 | 30K | core.padic_math | [ ] |
| **TIER 2: Geometry/Math** |||||
| `src/geometry/__init__.py` | 57 | 2K | poincare.py | [ ] |
| `src/geometry/poincare.py` | 356 | 13K | geoopt (external) | [ ] |
| `src/core/padic_math.py` | 489 | 17K | None (pure Python) | [ ] |
| `src/biology/codons.py` | 250 | 9K | None (pure Python) | [ ] |
| **TIER 3: Models/Losses** |||||
| `src/models/hyperbolic_projection.py` | 327 | 12K | geometry.ManifoldParameter | [ ] |
| `src/losses/peptide_losses.py` | 862 | 31K | geometry.poincare_distance, losses.base | [ ] |
| `src/losses/base.py` | 216 | 8K | None (pure Python) | [ ] |
| **TIER 4: TernaryVAE (for vae_service)** |||||
| `src/models/ternary_vae.py` | ~1,500 | ~55K | geometry, core, improved_components | [ ] |
| `src/models/improved_components.py` | ~400 | ~15K | None | [ ] |
| **TOTAL** | **~7,385** | **~268K** |||

### 2.2 Shared Infrastructure (Within Deliverables)

| File | Lines | Dependencies | Extraction Status |
|------|-------|--------------|:-----------------:|
| `shared/__init__.py` | 100 | Internal only | [x] MOVES WITH |
| `shared/config.py` | 153 | None | [x] MOVES WITH |
| `shared/constants.py` | 167 | None | [x] MOVES WITH |
| `shared/hemolysis_predictor.py` | 399 | sklearn | [x] MOVES WITH |
| `shared/logging_utils.py` | 304 | None | [x] MOVES WITH |
| `shared/peptide_utils.py` | 319 | numpy | [x] MOVES WITH |
| `shared/primer_design.py` | 557 | biopython | [x] MOVES WITH |
| `shared/uncertainty.py` | 372 | sklearn | [x] MOVES WITH |
| `shared/vae_service.py` | 348 | **TernaryVAEV5_11** | [ ] NEEDS MAIN |
| **TOTAL** | **2,719** |||

### 2.3 Checkpoint Files Required (External)

| Checkpoint | Size | Used By | Required For |
|------------|------|---------|--------------|
| `checkpoints/homeostatic_rich/best.pt` | 421 KB | vae_service | TernaryVAE inference |
| `checkpoints/v5_11_homeostasis/best.pt` | 845 KB | vae_service (fallback) | TernaryVAE inference |
| `checkpoints/peptide_vae_v1/best_production.pt` | 1.2 MB | predict_mic.py | PeptideVAE inference |
| `research/codon-encoder/training/results/trained_codon_encoder.pt` | 51 KB | jose_colbes DDG, alejandra_rojas | TrainableCodonEncoder |
| **TOTAL** | **~2.5 MB** |||

### 2.4 Partner-Local Checkpoints (Already Self-Contained)

| Package | Path | Size | Status |
|---------|------|------|:------:|
| carlos_brizuela | `checkpoints_definitive/best_production.pt` | 1.2 MB | [x] LOCAL |
| carlos_brizuela | `checkpoints_definitive/fold_*_definitive.pt` (5) | 5.8 MB | [x] LOCAL |
| jose_colbes | `models/ddg_predictor.joblib` | 532 KB | [x] LOCAL |
| **TOTAL LOCAL** | | **~7.5 MB** ||

---

## 3. NEW: Data File Inventory

### 3.1 Joblib/Pickle Model Files

| File | Size | Package | Purpose |
|------|------|---------|---------|
| `carlos_brizuela/models/activity_general.joblib` | 129 KB | carlos_brizuela | General AMP activity |
| `carlos_brizuela/models/activity_escherichia.joblib` | 122 KB | carlos_brizuela | E. coli specific |
| `carlos_brizuela/models/activity_staphylococcus.joblib` | 117 KB | carlos_brizuela | S. aureus specific |
| `carlos_brizuela/models/activity_pseudomonas.joblib` | 119 KB | carlos_brizuela | P. aeruginosa specific |
| `carlos_brizuela/models/activity_acinetobacter.joblib` | 100 KB | carlos_brizuela | A. baumannii specific |
| `jose_colbes/models/ddg_predictor.joblib` | 532 KB | jose_colbes | DDG prediction (legacy) |
| **TOTAL** | **~1.1 MB** |||

### 3.2 Results Directory Sizes

| Package | Path | Size | File Count |
|---------|------|------|------------|
| alejandra_rojas | `results/` | 3.9 MB | 50+ files |
| carlos_brizuela | `results/` | 596 KB | 35+ files |
| jose_colbes | `results/` | 780 KB | 8+ files |
| hiv_research_package | `results/` | 24 KB | 5+ files |
| **TOTAL** | | **~5.3 MB** | **~100 files** |

### 3.3 Critical Data Files

| File | Package | Size | Purpose | Runtime Required? |
|------|---------|------|---------|:-----------------:|
| `reproducibility/data/s669.csv` | jose_colbes | 2 KB | Benchmark dataset (N=52) | Yes |
| `results/ml_ready/denv4_genome_sequences.json` | alejandra_rojas | ~1 MB | DENV-4 sequences | Yes (validation) |
| `results/ml_ready/padic_integration_results.json` | alejandra_rojas | ~500 KB | P-adic embeddings | Yes (validation) |
| `results/primers/*.json` | alejandra_rojas | ~100 KB | Primer candidates | No (output) |
| `results/pathogen_specific/*.csv` | carlos_brizuela | ~200 KB | AMP candidates | No (output) |

---

## 4. NEW: Environment Variables

### 4.1 Required Environment Variables

| Variable | File | Line | Default | Impact if Missing |
|----------|------|------|---------|-------------------|
| `NCBI_API_KEY` | `shared/config.py` | 69 | `None` | NCBI rate limiting (3 req/sec vs 10 req/sec) |
| `NCBI_API_KEY` | `alejandra_rojas/src/data_pipeline.py` | 10 | `None` | Same as above |
| `NCBI_EMAIL` | `alejandra_rojas/src/ncbi_client.py` | 279 | `"user@example.com"` | Required for NCBI Entrez |

**Note:** All environment variables are optional. Scripts degrade gracefully with demo/mock modes.

### 4.2 Setting Environment Variables

```bash
# For NCBI access (optional, improves rate limits)
export NCBI_API_KEY="your-api-key-here"
export NCBI_EMAIL="your-email@example.com"
```

---

## 5. NEW: External APIs

### 5.1 Stanford HIVdb GraphQL API

| Property | Value |
|----------|-------|
| **URL** | `https://hivdb.stanford.edu/graphql` |
| **Used By** | `hiv_research_package/scripts/stanford_hivdb_client.py` |
| **Line** | 180 |
| **Method** | GraphQL mutation `AnalyzeSequences` |
| **Fallback** | Local heuristics in H6/H7 scripts |
| **Rate Limit** | Unknown |

**GraphQL Query Structure:**
```graphql
mutation AnalyzeSequences($sequences: [UnalignedSequenceInput]!) {
  viewer {
    sequenceAnalysis(sequences: $sequences) {
      inputSequence
      bestMatchingSubtype
      drugResistance { ... }
      mutationsByTypes { ... }
    }
  }
}
```

### 5.2 NCBI Entrez API

| Property | Value |
|----------|-------|
| **Used By** | `alejandra_rojas/src/ncbi_client.py` |
| **Requires** | `NCBI_EMAIL` (required), `NCBI_API_KEY` (optional) |
| **Rate Limit** | 3 req/sec without key, 10 req/sec with key |
| **Fallback** | Demo mode with mock sequences |

---

## 6. NEW: sys.path Manipulation Analysis

### 6.1 Summary

| Metric | Count |
|--------|-------|
| **Total `sys.path.insert()` calls** | 125+ |
| **Files with path manipulation** | 80+ |
| **Unique path patterns** | 5 |

### 6.2 Common Patterns

**Pattern 1: Project root + deliverables**
```python
# Most common (60+ instances)
project_root = Path(__file__).resolve().parents[N]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "deliverables"))
```

**Pattern 2: Package root**
```python
# Partner scripts (30+ instances)
_package_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_package_root))
```

**Pattern 3: Script directory**
```python
# Tests (20+ instances)
sys.path.insert(0, str(Path(__file__).parent))
```

### 6.3 Files with Path Manipulation (by package)

| Package | Count | Example File |
|---------|-------|--------------|
| alejandra_rojas | 25+ | `scripts/denv4_padic_integration.py:40` |
| carlos_brizuela | 20+ | `scripts/predict_mic.py:67` |
| jose_colbes | 15+ | `scripts/C4_mutation_effect_predictor.py:65` |
| hiv_research_package | 8+ | `scripts/stanford_hivdb_client.py:29` |
| shared/ | 5+ | `tests/test_integration.py:23` |
| tests/ | 10+ | `conftest.py:18` |

### 6.4 Extraction Impact

**Problem:** These path manipulations assume the deliverables folder is within the main project structure.

**Solution for extraction:**
1. Create proper Python package structure with `__init__.py` files
2. Replace `sys.path.insert()` with relative imports
3. Use `pip install -e .` for development

---

## 7. NEW: Security Concerns

### 7.1 Dynamic Code Execution

| File | Line | Code | Risk Level |
|------|------|------|:----------:|
| `scripts/biotools.py` | 610 | `exec(f"from {package} import {module}")` | **HIGH** |
| `scripts/biotools.py` | 611 | `mod = eval(module)` | **HIGH** |
| `scripts/biotools.py` | 613 | `exec(f"import {module_path}")` | **HIGH** |
| `scripts/biotools.py` | 614 | `mod = eval(module_path)` | **HIGH** |

**Risk:** `exec()` and `eval()` with string interpolation can execute arbitrary code if inputs are not sanitized.

**Context:** Used in biotools package manager for importing biological tool packages dynamically.

**Recommendation:** Replace with `importlib.import_module()` or whitelist allowed modules.

### 7.2 Subprocess Execution in Tests

| File | Line | Code | Risk Level |
|------|------|------|:----------:|
| `jose_colbes/tests/integration_test.py` | 234 | `exec(open('{script}').read()...)` | **MEDIUM** |
| `carlos_brizuela/tests/integration_test.py` | 79 | `exec(open('{script}').read()...)` | **MEDIUM** |

**Context:** Test execution of scripts. Paths are controlled by test code, not user input.

---

## 8. NEW: Test Fixtures

### 8.1 Pytest Configuration

| File | Lines | Fixtures | Markers |
|------|-------|----------|---------|
| `tests/conftest.py` | 171 | 9 | 3 |

### 8.2 Fixtures Defined

| Fixture | Purpose | Package Dependency |
|---------|---------|-------------------|
| `random_seed` | Reproducible randomness | numpy |
| `demo_sequence` | 500 AA sequence | numpy |
| `demo_nucleotide_sequence` | 1000 bp DNA | numpy |
| `sample_patient_data` | HIV patient mock | hiv_research_package.src.models |
| `sample_tdr_result` | TDR result mock | hiv_research_package.src.models |
| `sample_la_result` | LA selection mock | hiv_research_package.src.models |
| `temp_output_dir` | Temporary directory | pytest |
| `mock_rotamer_data` | 20 rotamers | numpy |
| `mock_arbovirus_sequences` | 4 virus sequences | numpy |

### 8.3 Custom Markers

```python
@pytest.mark.slow       # Long-running tests
@pytest.mark.network    # Requires network access
@pytest.mark.optional   # Optional feature tests
```

---

## 9. Import Statement Index

### 9.1 All External `from src.*` Imports

#### Jose Colbes Package (11 imports)

| File | Line | Import Statement | Status |
|------|------|------------------|:------:|
| `src/validated_ddg_predictor.py` | 58 | `from src.encoders.trainable_codon_encoder import TrainableCodonEncoder` | [ ] |
| `src/validated_ddg_predictor.py` | 59 | `from src.geometry import poincare_distance` | [ ] |
| `validation/bootstrap_test.py` | 19 | `from src.encoders.trainable_codon_encoder import TrainableCodonEncoder` | [ ] |
| `validation/bootstrap_test.py` | 20 | `from src.geometry import poincare_distance` | [ ] |
| `reproducibility/extract_aa_embeddings_v2.py` | 31 | `from src.biology.codons import CODON_TO_INDEX, ...` | [ ] |
| `reproducibility/extract_aa_embeddings_v2.py` | 37 | `from src.encoders.trainable_codon_encoder import TrainableCodonEncoder` | [ ] |
| `reproducibility/train_padic_ddg_predictor_v2.py` | 43 | `from src.biology.codons import ...` | [ ] |
| `reproducibility/analyze_padic_ddg_full.py` | 43 | `from src.biology.codons import ...` | [ ] |
| `reproducibility/archive/extract_aa_embeddings.py` | 102 | `from src.models import TernaryVAEV5_11_PartialFreeze` | [ ] |
| `reproducibility/archive/extract_aa_embeddings.py` | 204-205 | `from src.core import TERNARY`, `from src.data.generation import ...` | [ ] |
| `reproducibility/archive/extract_embeddings_simple.py` | 154 | `from src.models import TernaryVAEV5_11_PartialFreeze` | [ ] |

#### Alejandra Rojas Package (11 imports)

| File | Line | Import Statement | Status |
|------|------|------------------|:------:|
| `scripts/denv4_padic_integration.py` | 50 | `from src.encoders.trainable_codon_encoder import TrainableCodonEncoder` | [ ] |
| `scripts/denv4_padic_integration.py` | 51 | `from src.geometry import poincare_distance` | [ ] |
| `scripts/denv4_padic_integration.py` | 52 | `from src.biology.codons import ...` | [ ] |
| `scripts/denv4_synonymous_conjecture.py` | 48-50 | Same as above | [ ] |
| `scripts/denv4_codon_bias_conjecture.py` | 57 | `from src.biology.codons import ...` | [ ] |
| `scripts/denv4_revised_conjecture.py` | 55 | `from src.biology.codons import ...` | [ ] |
| `scripts/denv4_codon_pair_conjecture.py` | 49 | `from src.biology.codons import ...` | [ ] |
| `src/geometry.py` | 31 | `from src.geometry import (exp_map_zero, log_map_zero, ...)` | [ ] OPTIONAL |
| `research/clade_classification/train_clade_classifier.py` | 49-50 | `TrainableCodonEncoder`, `src.biology.codons` | [ ] |
| `research/functional_convergence/find_convergence_points.py` | varies | `TrainableCodonEncoder` | [ ] |
| `validation/test_padic_conservation_correlation.py` | 44 | `TrainableCodonEncoder`, `poincare_distance` | [ ] |

#### Carlos Brizuela Package (7 imports)

| File | Line | Import Statement | Status |
|------|------|------------------|:------:|
| `scripts/predict_mic.py` | 80 | `from src.encoders.peptide_encoder import PeptideVAE` | [ ] |
| `training/train_definitive.py` | 41 | `from src.encoders.peptide_encoder import PeptideVAE` | [ ] |
| `training/train_definitive.py` | 42 | `from src.losses.peptide_losses import PeptideLossManager, CurriculumSchedule` | [ ] |
| `training/train_peptide_encoder.py` | 46-47 | Same as above | [ ] |
| `training/train_improved.py` | 39-40 | Same as above | [ ] |
| `verify_paths.py` | varies | `PeptideVAE`, `PeptideLossManager` | [ ] |
| `training/dataset.py` | 14 | `from deliverables.shared.peptide_utils import ...` | [ ] |

#### Shared Infrastructure (1 import)

| File | Line | Import Statement | Status |
|------|------|------------------|:------:|
| `shared/vae_service.py` | 97 | `from src.models import TernaryVAEV5_11_PartialFreeze` | [ ] |

### 9.2 All `from deliverables.*` Absolute Imports

| File | Line | Import Statement | Fix Required |
|------|------|------------------|:------------:|
| `partners/alejandra_rojas/tests/test_notebook_integration.py` | 248,256,268 | `from deliverables.shared.primer_design import ...` | Convert to relative |
| `partners/jose_colbes/validation/scientific_validation_report.py` | 193 | `from deliverables.shared...` | Convert to relative |
| `partners/jose_colbes/validation/alphafold_validation_pipeline.py` | 392 | `from deliverables.shared...` | Convert to relative |
| `partners/carlos_brizuela/training/dataset.py` | 14 | `from deliverables.shared.peptide_utils import ...` | Convert to relative |

---

## 10. External pip Dependencies

### 10.1 Required for Full Extraction

| Package | Version | Required By | Installation |
|---------|---------|-------------|--------------|
| `geoopt` | >=0.5.0 | src.geometry.poincare | `pip install geoopt` |
| `torch` | >=2.0.0 | All ML components | See PyTorch install guide |
| `numpy` | >=1.20.0 | All packages | `pip install numpy` |
| `scipy` | >=1.7.0 | Statistics, optimization | `pip install scipy` |
| `scikit-learn` | >=1.0.0 | ML predictors | `pip install scikit-learn` |
| `deap` | >=1.4.0 | carlos_brizuela NSGA-II | `pip install deap` |
| `biopython` | >=1.80 | alejandra_rojas primers | `pip install biopython` |
| `joblib` | >=1.0.0 | Model serialization | `pip install joblib` |

### 10.2 Optional Dependencies

| Package | Required For | Impact if Missing |
|---------|--------------|-------------------|
| `matplotlib` | Visualization | No plots generated |
| `seaborn` | Visualization | No styled plots |
| `requests` | Data downloading, APIs | Manual download required |
| `tensorboard` | Training monitoring | No live metrics |

---

## 11. Extraction Scenarios

### Scenario A: Minimal (Keep ML in Main Repo)

**Goal:** Partner packages work for basic operations, ML prediction requires main repo.

**What to Extract:**
- [x] `deliverables/shared/` (except vae_service.py ML parts)
- [x] Partner `core/` or `src/` local modules (already done)
- [x] Partner `requirements.txt` files (already done)
- [x] Partner-local checkpoints and models

**What Stays:**
- [ ] All `src.*` imports remain as external dependencies
- [ ] ML inference requires main project installation

**Effort:** Already complete (current state)

### Scenario B: Partial (Copy Core Encoders)

**Goal:** ML prediction works for DDG and primers, training stays in main repo.

**Files to Copy (~4,700 LOC):**
- [ ] `src/encoders/trainable_codon_encoder.py` (586)
- [ ] `src/encoders/codon_encoder.py` (451)
- [ ] `src/geometry/` (413)
- [ ] `src/biology/codons.py` (250)
- [ ] `src/core/padic_math.py` (489)
- [ ] Checkpoints: `trained_codon_encoder.pt` (51 KB)

**Updates Required:**
- [ ] Update all `from src.*` imports to local paths
- [ ] Add `geoopt` to requirements
- [ ] Restructure as `deliverables/lib/encoders/`, etc.
- [ ] Replace 125+ `sys.path.insert()` calls with proper imports

**Effort:** Medium (1-2 days)

### Scenario C: Full (Complete Independence)

**Goal:** All features work without main repo.

**Additional Files (~2,800 LOC):**
- [ ] `src/encoders/peptide_encoder.py` (1,059)
- [ ] `src/encoders/padic_amino_acid_encoder.py` (832)
- [ ] `src/models/hyperbolic_projection.py` (327)
- [ ] `src/losses/peptide_losses.py` (862)
- [ ] `src/losses/base.py` (216)
- [ ] `src/models/ternary_vae.py` (~1,500) - for vae_service
- [ ] `src/models/improved_components.py` (~400)
- [ ] Additional checkpoints (~2.5 MB)

**Total:** ~7,500 LOC + ~9.5 MB external checkpoints + ~7.2 MB local checkpoints

**Effort:** High (2-3 days)

---

## 12. Extraction Checklist

### Phase 1: Preparation
- [ ] Create target repository structure
- [ ] Set up CI/CD for new repo
- [ ] Create comprehensive requirements.txt
- [ ] Document API keys and environment variables

### Phase 2: Shared Infrastructure
- [ ] Copy `deliverables/shared/` to new repo
- [ ] Update internal imports to relative
- [ ] Test shared module imports
- [ ] Verify vae_service.py placeholder/stub
- [ ] Update `config.py` checkpoint paths

### Phase 3: Partner Packages
- [ ] Copy jose_colbes with local core/
- [ ] Copy alejandra_rojas with local src/
- [ ] Copy carlos_brizuela with local src/ and checkpoints
- [ ] Copy hiv_research_package
- [ ] Copy all results/ directories

### Phase 4: ML Components (if Scenario B/C)
- [ ] Create `lib/` directory structure
- [ ] Copy encoder files
- [ ] Copy geometry files
- [ ] Copy biology files
- [ ] Copy loss files (if full extraction)
- [ ] Copy model files (if full extraction)

### Phase 5: Checkpoint Migration
- [ ] Copy required checkpoints
- [ ] Update checkpoint path references
- [ ] Test model loading
- [ ] Verify Git LFS handling

### Phase 6: Import Updates
- [ ] Update all `from src.*` to `from lib.*`
- [ ] Update all `from deliverables.*` to relative
- [ ] Replace 125+ `sys.path.insert()` calls
- [ ] Run import validation script

### Phase 7: Security Review
- [ ] Replace `exec()`/`eval()` in biotools.py
- [ ] Review subprocess execution in tests
- [ ] Validate API endpoint URLs

### Phase 8: Testing
- [ ] Run all partner package tests
- [ ] Verify ML inference (if extracted)
- [ ] Verify training (if full extraction)
- [ ] Test with/without environment variables

### Phase 9: Documentation
- [ ] Update README files
- [ ] Document external dependencies
- [ ] Create installation guide
- [ ] Document environment variables
- [ ] Document external API requirements

---

## 13. Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Circular imports after restructure | HIGH | Medium | Careful dependency ordering |
| sys.path manipulation breaks | HIGH | High | Convert to proper package structure |
| Checkpoint version mismatch | MEDIUM | Low | Include checkpoint metadata |
| geoopt API changes | LOW | Low | Pin version in requirements |
| Training reproducibility | MEDIUM | Low | Document exact versions |
| Missing transitive dependencies | MEDIUM | Medium | Comprehensive testing |
| External API unavailability | MEDIUM | Low | Verify fallback modes work |
| Security issues (exec/eval) | HIGH | Low | Replace with safe alternatives |
| Environment variable confusion | LOW | Medium | Clear documentation |

---

## 14. Recommended Approach

**Short-term:** Keep deliverables in main repo. Current partial self-containment (70-90%) is sufficient for most use cases.

**Medium-term (if extraction needed):**
1. Start with Scenario B (partial extraction)
2. Focus on inference capabilities only
3. Keep training in main repo
4. Create `pip install ternary-vae-lite` package
5. Prioritize fixing `sys.path` manipulation

**Long-term:**
1. Publish core components as separate pip packages:
   - `ternary-vae-geometry` (poincare, hyperbolic)
   - `ternary-vae-encoders` (codon, peptide)
   - `ternary-vae-bio` (biology utilities)
2. Deliverables `pip install` these packages
3. Zero code duplication
4. Replace all `exec()`/`eval()` with safe alternatives

---

## Update Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-23 | 2.0 | Deep iteration: Added data files, env vars, APIs, sys.path analysis, security review |
| 2026-01-23 | 1.0 | Initial comprehensive audit |

---

*Audit performed: 2026-01-23*
*Auditor: Claude Code*
