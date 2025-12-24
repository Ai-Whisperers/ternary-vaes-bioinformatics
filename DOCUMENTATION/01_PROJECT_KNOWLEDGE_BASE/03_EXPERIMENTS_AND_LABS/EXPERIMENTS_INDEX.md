# Experiments & Labs Index

> **Purpose**: This interactive index maps our **theoretical research** (from `ACADEMIC_DATABASE.md`) to our **active code labs**. Use this to find where to test specific hypotheses.

## 🧬 Domain: Bioinformatics

| Research Area         | Key Theory (Papers)                                     | Active Lab Directory                                                                                   | Status              |
| :-------------------- | :------------------------------------------------------ | :----------------------------------------------------------------------------------------------------- | :------------------ |
| **Genetic Code**      | _Dragovich et al. (2010)_ "p-Adic Modelling"            | [`scripts/analysis/`](../../../../scripts/analysis/)                                                   | 🟢 **Stable**       |
| **Viral Evolution**   | _Frazer et al. (2021)_ "EVE"; _Obermeyer (2022)_ "PyR0" | [`scripts/train/`](../../../../scripts/train/)                                                         | 🟡 **Active**       |
| **Glycan Shielding**  | _PeSTo-Carbs_; _Torres et al._                          | [`scripts/benchmark/`](../../../../scripts/benchmark/)                                                 | 🟡 **Active**       |
| **Autoimmunity**      | _MHC-II HLA interactions_                               | [`scripts/analysis/analyze_zero_structure.py`](../../../../scripts/analysis/analyze_zero_structure.py) | 🔴 **Experimental** |
| **Neurodegeneration** | _Ternary Logic in Cognition_                            | [Planned]                                                                                              | ⚪ **Planned**      |

## 📐 Domain: Mathematics & Geometry

| Research Area             | Key Theory (Papers)                           | Active Lab Directory                                                                                           | Status        |
| :------------------------ | :-------------------------------------------- | :------------------------------------------------------------------------------------------------------------- | :------------ |
| **Hyperbolic Embeddings** | _Nickel & Kiela (2017)_ "Poincaré Embeddings" | [`tests/suites/unit/test_geometry.py`](../../../../tests/suites/unit/test_geometry.py)                         | 🟢 **Stable** |
| **3-adic Numbers**        | _Khrennikov (2004)_ "p-adic Information"      | [`scripts/analysis/verify_mathematical_proofs.py`](../../../../scripts/analysis/verify_mathematical_proofs.py) | 🟢 **Stable** |
| **Spectral Analysis**     | _Smita Krishnaswamy_ "Geometric Scattering"   | [`scripts/visualization/`](../../../../scripts/visualization/)                                                 | 🟡 **Active** |

## 🏗️ Repository Health & Analysis

| Analysis Type         | Purpose                                    | Active Script                                                                                        |
| :-------------------- | :----------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| **Code Audits**       | Track technical debt and file stats.       | [`scripts/analysis/audit_repo.py`](../../../../scripts/analysis/audit_repo.py)                       |
| **Metric Generation** | Generate complexity and coverage metrics.  | [`scripts/analysis/run_metrics.py`](../../../../scripts/analysis/run_metrics.py)                     |
| **Reports**           | Synthesize all findings into final report. | [`scripts/analysis/generate_final_report.py`](../../../../scripts/analysis/generate_final_report.py) |

## 🧪 Workflow Guides

- **To run a new biological experiment**: Copy the template from [`bioinformatics/genetic_code/scripts/`](./bioinformatics/genetic_code/scripts/) and adapt the `PROJECT_ROOT` path.
- **To test a new theory**: Create a subfolder in `mathematics/` if it's pure theory, or add a domain folder in `codon_encoder_research/` if it requires biological data.

## 🔗 Connection to Academic Database

- **Hub A (Geometric DL)** -> Testing Grounds: `mathematics/` & `spectral_analysis_over_models/`
- **Hub B (Algebraic Bio)** -> Testing Grounds: `genetic_code/` (3-adic logic)
- **Hub C (P-adic Physics)** -> Testing Grounds: `genetic_code/`
- **Hub D (Viral Evolution)** -> Testing Grounds: `sars_cov_2/` & `hiv/`
