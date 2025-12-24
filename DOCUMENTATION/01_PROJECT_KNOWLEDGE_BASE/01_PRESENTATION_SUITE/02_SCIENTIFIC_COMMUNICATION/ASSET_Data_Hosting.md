# Supplementary Data Hosting Strategy

**Objective**: Provide transparency and reproducibility without bloating the git repo.

## 1. Hosting Provider

- **Primary**: [Zenodo](https://zenodo.org) (CERN).
  - _Why_: Issues a permanent DOI (citable), free for open science.
- **Secondary**: HuggingFace Datasets.
  - _Why_: Easy integration with Python scripts.

## 2. Data Organization (To Upload)

Structure the ZIP file as follows:

```text
/supplementary_data
    /raw_predictions
        hiv_escape_scores_v5.11.csv       # (Mutant, Score, P-value)
        ra_patient_embeddings.npy         # (Patient_ID, Vector_64D)
    /model_weights
        ternary_vae_v5.11_encoder.pt      # PyTorch weights
        ternary_vae_v5.11_decoder.pt
    /benchmarks
        kill_sheet_results_2025.json      # Validation logs
```

## 3. Linking

- Update `TECHNICAL_WHITEPAPER.md` with the Zenodo DOI.
- Add a badge to the README: `[DOI: 10.5281/zenodo.xxxxxx]`
