# Tasks Directory

**Doc-Type:** Task Index | Version 1.0 | Updated 2025-12-25

---

## Overview

This directory contains all task tracking, analysis, and implementation planning documents for the project.

---

## Top-Level Documents

| Document | Description | Status |
|:---------|:------------|:-------|
| [CODEBASE_DEEP_DIVE_ANALYSIS.md](./CODEBASE_DEEP_DIVE_ANALYSIS.md) | Complete analysis of 219 issues across 113 files | Complete |
| [RESEARCH_PROPOSAL_IMPLEMENTATION_ANALYSIS.md](./RESEARCH_PROPOSAL_IMPLEMENTATION_ANALYSIS.md) | Exhaustive analysis of 20 research proposals | Complete |

---

## Key Findings Summary

### Codebase Health

| Metric | Value |
|:-------|:------|
| Total Issues | 219 |
| Critical Bugs | 23 |
| Broken Scripts | 3 |
| Broken Tests | 5 |
| Untested Modules | 37 |
| Overall Rating | 3.4/5 |

### Top Research Proposals (Implementation-Ready)

| Proposal | Completion | Effort | Priority |
|:---------|:-----------|:-------|:---------|
| PTM Goldilocks Encoder | 70% | 11 days | P1 |
| Geometric Vaccine Design | 60% | 10 days | P1 |
| Autoimmunity Codon Adaptation | 50% | 9 days | P1 |

---

## Task Subdirectories

| Directory | Purpose |
|:----------|:--------|
| [01_BIOINFORMATICS/](./01_BIOINFORMATICS/) | Bioinformatics research tasks |
| [02_MODEL_ARCHITECTURE/](./02_MODEL_ARCHITECTURE/) | VAE and model improvement tasks |
| [03_INFRASTRUCTURE/](./03_INFRASTRUCTURE/) | Bug fixes, refactoring, testing |
| [04_DOCUMENTATION/](./04_DOCUMENTATION/) | Documentation improvement tasks |
| [05_VALIDATION/](./05_VALIDATION/) | Experimental validation tasks |
| [99_IDEAS/](./99_IDEAS/) | Future ideas and explorations |

---

## Quick Actions

### Critical Bug Fixes (16 hours)

1. **Valuation Clamping** - `src/core/ternary.py:158`
2. **Unused Poincare Distance** - `src/losses/dual_vae_loss.py:154`
3. **KL Math Error** - `src/losses/dual_vae_loss.py:449`
4. **Hyperbolic Gradients** - `src/training/hyperbolic_trainer.py:534`
5. **Shuffle Mutation** - `src/data/gpu_resident.py:127`

### Broken Script Fixes (2 hours)

1. `scripts/generate_hiv_papers.py` - Hardcoded Windows path
2. `scripts/benchmark/run_benchmark.py` - Non-existent imports
3. `scripts/ingest/ingest_starpep.py` - Non-existent path

---

## Related Documentation

- [ROADMAP.md](../01_ROADMAPS_AND_PLANS/ROADMAP.md) - Project roadmap
- [RISK_REGISTER.md](../RISK_REGISTER.md) - Risk tracking
- [CODE_HEALTH_METRICS/](../02_CODE_HEALTH_METRICS/) - Ongoing metrics

---

*Last updated: 2025-12-25*
