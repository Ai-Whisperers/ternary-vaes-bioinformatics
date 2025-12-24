<!-- SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

---
title: "Theory Documentation Changelog"
date: 2025-12-24
authors:
  - AI Whisperers
version: "0.1"
license: PolyForm-Noncommercial-1.0.0
---

# Theory Documentation Changelog

All notable changes to the Theory and Foundations documentation.

This changelog follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

---

## [0.2.0] - 2025-12-24

### Added
- **SPDX License Headers**: Added `<!-- SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->` to all 299 markdown files
- **YAML Front-matter**: Added structured metadata to all files:
  - `title`: Extracted from first H1 heading
  - `date`: Creation/update date
  - `authors`: AI Whisperers
  - `version`: 0.1 initial version
  - `license`: PolyForm-Noncommercial-1.0.0
- **Theory README**: Enhanced with licensing section, cross-references, and navigation
- **Cross-links**: Added math ↔ biology reference table in README
- **CHANGELOG.md**: This file

### Changed
- **README.md**: Updated to version 0.2 with comprehensive structure
- **Documentation structure**: Standardized all theory documents with consistent headers

### Infrastructure
- **Script**: `scripts/docs/add_spdx_frontmatter.py` for automated header/frontmatter management
- **CI preparation**: Groundwork for SPDX header linting

---

## [0.1.0] - 2025-12-23

### Added
- Initial theory documentation structure
- Mathematical foundations documents
- Biological context documents
- Validation suite documentation
- Research library with HIV papers

### Structure
```
02_THEORY_AND_FOUNDATIONS/
├── 01_PROJECT_CONTEXT/
├── 02_MATHEMATICAL_FOUNDATIONS/
├── 03_BIOLOGY_CONTEXT/
├── 04_EMBEDDINGS_ANALYSIS/
├── 06_REPORTS_AND_DISCOVERIES/
├── 07_VALIDATION/
├── 08_METRICS_DOCS/
└── 09_BIBLIOGRAPHY_AND_RESOURCES/
```

---

## File Statistics

| Metric | Count |
|--------|-------|
| Total markdown files | 299 |
| Files with SPDX headers | 299 |
| Files with front-matter | 299 |
| Subdirectories | 35+ |

---

## Versioning Convention

- **Major (X.0.0)**: Breaking changes to document structure
- **Minor (0.X.0)**: New documents or significant updates
- **Patch (0.0.X)**: Typo fixes, minor clarifications

---

## Related

- [Open Medicine Policy](../../../LEGAL_AND_IP/OPEN_MEDICINE_POLICY.md)
- [Results License](../../../LEGAL_AND_IP/RESULTS_LICENSE.md)
- [Main Project CHANGELOG](../../../results/run_history/IMPROVEMENTS_SUMMARY.md)
