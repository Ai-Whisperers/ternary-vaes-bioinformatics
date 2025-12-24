# Task: Security & Code Safety Fixes

**Objective**: Address security vulnerabilities and safety issues identified by Bandit.

## 1. Syntax Errors (Immediate Action)

- [ ] `DOCUMENTATION/01_PROJECT_KNOWLEDGE_BASE/03_EXPERIMENTS_AND_LABS/experiments_and_labs/bioinformatics/codon_encoder_research/rheumatoid_arthritis/scripts/19_alphafold_structure_mapping.py`: Fix "syntax error while parsing AST".

## 2. Medium Severity Issues

Review and fix the following potential security risks:

- [ ] `embeddings_analysis/01_extract_compare_embeddings.py`
- [ ] `rheumatoid_arthritis/scripts/12_download_human_proteome.py`
- [ ] `rheumatoid_arthritis/scripts/18_extract_acpa_proteins.py`
- [ ] `rheumatoid_arthritis/scripts/hyperbolic_utils.py`

**Common Checks:**

- Remove hardcoded absolute paths (Use `pathlib` relative paths).
- Ensure `subprocess` calls are sanitized (if any).
- Check for unsafe YAML/Pickle loading (use `safe_load`).
- Verify SSL certificate validation is not disabled in download scripts.

## 3. Low Severity / Best Practices

- [ ] `rheumatoid_arthritis/scripts/15_predict_immunogenicity.py`
- [ ] Check for use of `assert` in production code (Bandit B101).
