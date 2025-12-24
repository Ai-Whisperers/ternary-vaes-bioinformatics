# Research Inbox Triage & Tracking

> **Purpose**: Track the implementation of recent "Review Inbox" items (Autoimmunity, Genetic Code, p-adic, Spectral, HIV-2024).

## 📊 Status Summary

- **Total Topics**: 5
- **Triage Complete**: 100% (Roadmap defined)
- **Implemented**: 0%

## 📝 Triage Table

| Inbox Item    | Topic                 | Tier | Key Hypothesis                                   | Implementation Task                        | Validation Metric                     | Status  |
| ------------- | --------------------- | ---- | ------------------------------------------------ | ------------------------------------------ | ------------------------------------- | ------- |
| `01_AUTO...`  | Autoimmunity & Codons | T1   | Codon bias correlates with peptide presentation. | `tests/validation/test_codon_bias.py`      | Pearson correlation > 0.6             | 🔴 Open |
| `02_GENET...` | Genetic Code Theory   | T2   | Synthetic codons stabilize hyperbolic folding.   | `src/simulations/synthetic_codons.py`      | Embedding stability score             | 🔴 Open |
| `03_PADIC...` | p-adic Biology        | T1   | p-adic distance captures family hierarchy.       | `src/utils/padic_metrics.py`               | Tree edit distance vs Euclidean       | 🔴 Open |
| `04_SPECT...` | Spectral Bio-ML       | T1   | Spectral scattering features improve VAE.        | `notebooks/spectral_scattering_demo.ipynb` | Reconstruction error (MSE)            | 🔴 Open |
| `HIV_2024`    | HIV Neutralization    | T3   | Hyperbolic embeddings reveal Env hotspots.       | `scripts/hiv/neutralization_hotspots.py`   | Accuracy vs experimental binding data | 🔴 Open |

## 🔑 Legend

- **Tier**: T1 (Quick), T2 (Medium), T3 (Long)
- **Status**: 🔴 Open, 🟡 In Progress, 🟢 Validated
