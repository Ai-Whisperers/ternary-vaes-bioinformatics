# Documentation Reorganization Plan (v4)

**Objective**: Eliminate ambiguity between "Scientific Knowledge" and "Project Management", and fix the broken directory numbering.

## 1. Top-Level Renumbering

- **Current Issue**: `04_PROJECT_MANAGEMENT` implies missing folders `02` and `03`.
- **Change**: Rename `04_PROJECT_MANAGEMENT` -> `02_PROJECT_MANAGEMENT`.

## 2. Disambiguating "Reports"

We currently have "Reports" in two places with very different contents.

- **Knowledge Base**: `04_REPORTS_AND_HISTORY` contains _experiments_ and _discoveries_.
  - **Rename to**: `04_SCIENTIFIC_HISTORY`.
- **Project Management**: `02_REPORTS` contains _code audits_ and _linting_.
  - **Rename to**: `02_CODE_HEALTH_METRICS`.

## 3. Renaming "Active Plans"

- `01_ACTIVE_PLANS` contains long-term roadmaps.
- **Rename to**: `01_ROADMAPS_AND_PLANS` for clarity.

## 4. Final Proposed Structure

```text
DOCUMENTATION/
├── 01_PROJECT_KNOWLEDGE_BASE/
│   ├── 00_STRATEGY_AND_VISION/     (Pitch, Vision)
│   ├── 01_PRESENTATION_SUITE/      (Communication, Graphs)
│   ├── 02_THEORY_AND_FOUNDATIONS/  (Math, Deep Dives)
│   ├── 03_EXPERIMENTS_AND_LABS/    (Notebooks, Experiment Logs)
│   ├── 04_SCIENTIFIC_HISTORY/      (Past Discoveries, Academic Output)
│   └── 05_LEGAL_AND_IP/            (Patents, Licenses)
│
└── 02_PROJECT_MANAGEMENT/
    ├── 00_TASKS/                   (Active Task Lists)
    ├── 01_ROADMAPS_AND_PLANS/      (Q1/Q2/Q3 Strategies)
    ├── 02_CODE_HEALTH_METRICS/     (Linting, Debt Audits)
    └── 03_ARCHIVE/                 (Old Plans)
```

## 5. Execution Steps

1.  Rename `04_PROJECT_MANAGEMENT` -> `02_PROJECT_MANAGEMENT`.
2.  Rename subfolders inside Knowledge Base.
3.  Rename subfolders inside Project Management.
4.  Update root `README.md` to reflect the changes.
