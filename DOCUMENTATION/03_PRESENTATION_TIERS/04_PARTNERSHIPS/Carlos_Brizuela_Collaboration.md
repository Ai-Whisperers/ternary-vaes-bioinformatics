# 🎯 Tier 1 Presentation – Executive Overview

> **Speaker Note:** Open with a brief story about why interdisciplinary expertise matters for the Ternary VAE project.

---

## 1️⃣ Who is **Carlos A. Brizuela**?

- **Affiliation:** Department of Computer Science, CICESE, Ensenada, Mexico
- **Research Themes:**
  - Antimicrobial peptides & protein design
  - Bio‑informatics algorithms & multi‑objective optimisation
  - Evolutionary computation & ML for drug discovery
- **Credentials:** Ph.D. (Kyoto Institute of Technology, 2001) – >150 peer‑reviewed papers, creator of the _StarPep Toolbox_ and pioneering optimisation frameworks.

> **Speaker Note:** Emphasise his unique blend of computational biology and AI – the exact skill set we need.

---

## 2️⃣ Repository at a Glance (High‑Level)

| Folder                                     | Purpose                                                                                       |
| ------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `src/`                                     | Core VAE implementation, codon‑encoder, custom loss functions                                 |
| `data/`                                    | Curated datasets (geometric vaccine structures, drug‑interaction matrices, peptide libraries) |
| `DOCUMENTATION/01_PROJECT_KNOWLEDGE_BASE/` | Theory, bibliography, and the research library that underpins the science                     |
| `DOCUMENTATION/03_PRESENTATION_TIERS/`     | This folder – the slide decks you are reading now                                             |

> **Speaker Note:** Highlight the clean‑architecture layout; it makes extensions trivial.

---

## 3️⃣ Consolidated Knowledge Base

- **File:** `CARLOS_BRIZUELA_FULL_PROFILE_AND_RESEARCH_REPORT.md`
  - Merges Carlos’ full academic profile, a curated publication list, and a **comprehensive research report** on HIV‑related topics.
- **Validated Findings (HIV focus):**
  1. **Geometric HIV Vaccines** – nanoparticle scaffolds (ferritin, mi3, VLPs) that preserve native‑like antigen orientation.
  2. **Lenacapavir ↔ Sertraline Interaction** – CYP3A4 inhibition creates a pharmacokinetic penalty.
  3. **FIV + Tenofovir DNA Vaccines** – proof‑of‑concept for nucleoside‑analogue vaccine strategies.
  4. **Antibiotics ↔ Retrovirals** – synergistic immune‑modulatory effects useful as auxiliary features.
  5. **Codon‑Space Exploration** – a p‑adic metric for codon similarity that informs our codon‑encoder.

> **Speaker Note:** Use a single slide per bullet; keep visuals simple (icons + one‑line caption).

---

## 4️⃣ Why This Drives the **Ternary VAE**

- **Loss‑Function Design:** Directly embed geometric‑design terms and drug‑interaction penalties derived from the literature.
- **Feature Engineering:** Encode antibiotic‑retroviral interaction flags and immune‑biomarker ratios (CD4/CD8) as time‑varying covariates.
- **Dataset Construction:** Leverage the 30‑paper table to build **multi‑task training sets** covering HIV, FIV, and related infectious diseases.
- **Future Extensions:** The same latent‑space framework can be expanded to **pan‑infectious‑disease modelling** (Syphilis, Hepatitis, TB).

> **Speaker Note:** Connect each bullet to a concrete deliverable (e.g., _loss‑function prototype_, _feature matrix_).

---

## 5️⃣ Executive Next Steps

1. **Prototype the loss‑function** incorporating the five validated insights.
2. **Populate `data/`** with the curated datasets (geometric structures, interaction matrices, peptide libraries).
3. **Schedule a Tier 2 technical deep‑dive** with the development team to map implementation details.

> **Speaker Note:** End with a clear call‑to‑action; invite questions.

---

_Prepared for internal review – all sources are documented in the combined profile markdown._
