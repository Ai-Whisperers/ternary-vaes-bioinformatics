# Documentation Refactorization & Optimization Plan

**Objective**: To implement a disciplined, "Static vs Active" separation of concerns, reduce nesting depth, and abstract raw data from human-readable documentation.

---

## 🚀 Phase 1: Deduplication & Centralization

**Goal**: Establish Single Sources of Truth (SSOT).

### 1.1 The Validation Merger

- **Duplicate**: `KB/00_STRATEGY/VALIDATION_STRATEGY.md` vs `PM/01_PLANS/VALIDATION_AND_BENCHMARKING_PLAN.md`.
- **Action**:
  1.  Copy any unique "Theory" pillars from `VALIDATION_STRATEGY` into `KB/02_THEORY/metrics_documentation/validation_theory.md`.
  2.  Check `VALIDATION_AND_BENCHMARKING_PLAN` covers all execution steps.
  3.  **DELETE** `VALIDATION_STRATEGY.md`.

### 1.2 The Strategy Alignment

- **Collision**: `KB/00_STRATEGY/PITCH.md` vs `PM/01_PLANS/00_MASTER_ROADMAP.md`.
- **Action**:
  1.  Ensure `PITCH.md` focuses _only_ on the Vision/Market (Immutable 10-year goal).
  2.  Ensure `MASTER_ROADMAP` concerns _only_ the Execution (1-year plan).
  3.  Fix broken links in `PITCH.md`.

---

## 📦 Phase 2: Abstraction (Data Hygiene)

**Goal**: Remove machine-generated noise from the human documentation tree.

### 2.1 Dashboard Containment

- **Issue**: `CODE_HEALTH_DASHBOARD.md` is 129KB of raw text.
- **Action**:
  1.  Create `DOCUMENTATION/02_PROJECT_MANAGEMENT/02_CODE_HEALTH_METRICS/_raw_data/`.
  2.  Move `CODE_HEALTH_DASHBOARD.md` into `_raw_data/`.
  3.  Start `SUMMARY_CODE_HEALTH.md` (formerly `COMPREHENSIVE_CODE_HEALTH.md`) with a link to the raw data.

---

## 🔧 Phase 3: Refactorization (Flattening)

**Goal**: Reduce click-depth for high-traffic assets.

### 3.1 Scientific Communication Flattening

- **Issue**: `01_PRESENTATION_SUITE/02_SCIENTIFIC_COMMUNICATION/A_Research_Domains/Virology/...` is too deep.
- **Action**: Flatten to `02_SCIENTIFIC_COMMUNICATION/`.
  - Rename `A_Research_Domains/Virology/01_VIRAL_ESCAPE.md` -> `DOMAIN_Virology_Viral_Escape.md`.
  - Rename `B_Foundations/00_Core_Manuscripts/TECHNICAL_WHITEPAPER.md` -> `THEORY_Technical_Whitepaper.md`.
  - (Etc for Grants, Visuals).
  - Delete folders `A_Research_Domains`, `B_Foundations`, `C_Strategic_Assets`.

---

## 🏗️ Phase 4: Reorganization (Migration)

**Goal**: Enforce "Active Code belongs in Repo, not Docs".

### 4.1 Experiments Migration

- **Issue**: `KB/03_EXPERIMENTS_AND_LABS/` contains active notebooks.
- **Action**:
  1.  Propose creating a root-level `research/` directory in the repo (outside `DOCUMENTATION`).
  2.  Move "Active" notebooks there.
  3.  Leave "Completed" summary content in `KB/03_EXPERIMENTS_AND_LABS` (or rename to `LAB_REPORTS`).

---

## 📋 Execution Checklist

| Phase  | Task                             | Status |
| :----- | :------------------------------- | :----- |
| **P1** | Merge Validation Docs            | [ ]    |
| **P2** | Move Dashboard to `_raw_data`    | [ ]    |
| **P2** | Rename Comprehensive -> Summary  | [ ]    |
| **P3** | Flatten Scientific Comm          | [ ]    |
| **P4** | Migrate Notebooks to `research/` | [ ]    |
