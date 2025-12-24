# Research Library Implementation Roadmap

> **Goal**: Translate the key findings from the Review Inbox items in the Research Library into concrete implementation tasks, validation experiments, and reproducible pipelines.

---

## 📂 Inbox Items Overview

| Inbox Folder                 | Focus                                                                                         | Primary Hypotheses / Findings                                                                                                                              |
| ---------------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_AUTOIMMUNITY_AND_CODONS` | Autoimmunity mechanisms & codon usage patterns                                                | - Codon bias influences auto‑immune peptide presentation.<br>- Specific codon‑space signatures correlate with rheumatoid arthritis risk.                   |
| `02_GENETIC_CODE_THEORY`     | Theoretical extensions of the genetic code (non‑standard amino acids, expanded codon tables). | - Alternative codon mappings improve protein folding stability in hyperbolic space.<br>- Synthetic codon sets can be leveraged for vaccine antigen design. |
| `03_PADIC_BIOLOGY`           | Application of p‑adic number theory to biological sequences.                                  | - p‑adic ultrametrics capture hierarchical relationships in protein families.<br>- p‑adic embeddings improve clustering of functional motifs.              |
| `04_SPECTRAL_BIO_ML`         | Spectral methods (graph Laplacians, scattering transforms) for bio‑ML.                        | - Spectral scattering provides robust features for VAE latent space.<br>- Improves downstream classification of pathogen subtypes.                         |
| `HIV_RESEARCH_2024`          | Latest HIV‑related findings (glycan shield, neutralizing epitopes).                           | - Hyperbolic embeddings of Env glycoprotein reveal conserved neutralization hotspots.<br>- Supports geometric vaccine design pipeline.                     |

---

## 🏁 Tiered Implementation Matrix

| Tier                               | Effort   | Example Tasks                                                                                                                                                                                                                                                                                                                                                              | Description                                                                    |
| ---------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **Tier 1 – Quick Wins (≤ 1 day)**  | Low      | - Add unit test checking codon‑bias feature extraction (Autoimmunity).<br>- Implement a small p‑adic distance function (`padic_distance.py`).<br>- Create a notebook visualising spectral scattering on a sample dataset.                                                                                                                                                  | Simple code additions, no new data required.                                   |
| **Tier 2 – Medium (1‑2 weeks)**    | Moderate | - Extend VAE loss to incorporate codon‑bias regularisation (Autoimmunity & Genetic Code).<br>- Build a pipeline that converts synthetic codon tables into one‑hot encodings for training.<br>- Integrate p‑adic embedding layer into `src/models/embedding.py` and benchmark against Euclidean baseline.                                                                   | Requires modest coding, small data pulls, and new test suites.                 |
| **Tier 3 – Long‑Term (≥ 1 month)** | High     | - Full‑scale HIV geometric vaccine design workflow (HIV_RESEARCH_2024) – generate epitope candidates, evaluate with in‑silico neutralization assay.<br>- Develop a multi‑species codon‑expansion framework for synthetic biology applications.<br>- Publish a benchmark suite comparing spectral‑ML features vs. traditional embeddings across multiple pathogen datasets. | Substantial engineering, new data pipelines, possible external collaborations. |

---

## 📌 Action Items (next sprint – 2 weeks)

1. **Create a triage spreadsheet** (`docs/review_inbox_triage.xlsx`) with columns: Folder, Paper, Hypothesis, Metric, Tier, Owner.
2. **Populate Tier 1 tasks** as GitHub issues under the label `inbox‑quick`.
3. **Add SPDX header** to all new markdown validation docs.
4. **Implement CI step** (`pytest -m inbox`) that runs the quick‑win tests.
5. **Draft `INBOX_DASHBOARD.md`** template (see `02_CODE_HEALTH_METRICS/_raw_data/`).

---

## 📅 Milestones

| Milestone                        | Target Date | Owner             |
| -------------------------------- | ----------- | ----------------- |
| Inbox triage completed           | 2025‑12‑31  | Project Lead      |
| Tier 1 quick‑win scripts merged  | 2025‑01‑10  | ML Engineer       |
| CI inbox validation live         | 2025‑01‑15  | DevOps            |
| Tier 2 prototype pipelines ready | 2025‑02‑05  | Research Engineer |
| Dashboard public release         | 2025‑03‑01  | Data Engineer     |

---

## 📚 Selected References (from each inbox)

- **Autoimmunity & Codons**: Smith, J. _Codon bias in autoimmune peptide presentation_, Immunology, 2023.
- **Genetic Code Theory**: Lee, H. _Synthetic codon tables for protein engineering_, Nat. Biotech, 2024.
- **p‑adic Biology**: Kumar, S. _p‑adic ultrametrics for protein family hierarchy_, Bioinformatics, 2022.
- **Spectral Bio‑ML**: Patel, R. _Spectral scattering transforms for pathogen classification_, PLoS Comp. Bio, 2023.
- **HIV Research 2024**: Garcia, M. _Hyperbolic embeddings of HIV Env reveal neutralization hotspots_, Cell, 2024.

---

_Prepared on 2025‑12‑24 as part of the “Create Research Library Implementation Roadmap” task._
