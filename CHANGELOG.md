# Changelog

All notable changes to the Ternary VAE Bioinformatics project.

---

## [Unreleased]

### Added
- Hierarchical PTM mapping with 14-level ultrametric tree validation
- Rigorous disruption prediction framework
- Triple PTM combinatorics analysis
- RA research documentation with dual-trigger mechanism

---

## [5.11] - 2025-12-14

### Added
- Unified hyperbolic geometry architecture
- Frozen v5.5 encoder as coverage base
- Three-Body system with position-dependent control
- Unified PAdicGeodesicLoss

### Changed
- StateNet gradient flow now fully differentiable
- Replaced competing losses with unified approach

---

## [5.10] - 2025-12-12

### Added
- Hyperbolic priors (Poincare geometry)
- StateNet with curvature awareness
- Pure hyperbolic training mode
- Config validation and environment checks

### Fixed
- Monitor injection for consistent logging
- Training observability improvements

---

## [5.6] - 2025-12-10

### Added
- TensorBoard integration (local, IP-safe)
- TorchInductor compilation (1.4-2x speedup)
- Weight histograms logging

### Changed
- Renamed v5.5 files to v5.6
- Version bump to 5.6.0

---

## [5.5-srp] - 2025-11-24

### Added
- SRP refactoring complete
- Modular architecture: `src/training/`, `src/losses/`, `src/data/`, `src/artifacts/`
- Comprehensive documentation (4,200+ lines)
- Artifact lifecycle management

### Changed
- Model reduced from 632 to 499 lines (-21%)
- Trainer streamlined from 398 to 350 lines (-12%)

---

## [5.5] - 2025-10-29

### Fixed
- Categorical sampling bug (expectation vs sampling)
- Benchmark script (required checkpoint)
- Test suite (meaningful assertions)
- Coverage metrics (hash-validated 86% vs inflated 99%)

### Added
- Checkpoint certification with SHA256
- JSON benchmark output with traceability

---

## [5.5-initial] - 2025-10-24

### Added
- Dual-VAE architecture (VAE-A/VAE-B)
- StateNet meta-controller
- Phase-scheduled training
- Beta-warmup and free bits

### Achieved
- 100% holdout accuracy
- 86% hash-validated coverage
- 16/16 active latent dimensions

---

*For detailed release notes, see `DOCUMENTATION/.../ARCHIVE/release_notes_archive.md`*
