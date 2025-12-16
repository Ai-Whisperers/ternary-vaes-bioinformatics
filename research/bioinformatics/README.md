# Bioinformatics Applications

**Doc-Type:** Research Index · Version 1.0 · Updated 2025-12-16

---

## Overview

Clinical applications of p-adic geometry for disease analysis. Using the codon encoder from `../genetic_code/`, we analyze how mutations traverse the embedding space.

---

## Applications

### 1. Rheumatoid Arthritis (`rheumatoid_arthritis/`)

Analysis of RA using p-adic methods:

| Discovery | Metric | Significance |
|-----------|--------|--------------|
| HLA-RA Prediction | p < 0.0001, r = 0.751 | P-adic geometry predicts RA risk |
| Citrullination | 14% cross boundaries | Sentinel epitopes identified |
| Codon Optimization | 100% safety | Immunologically silent constructs |
| Regenerative Axis | Para closer to regen | Autonomic control of healing |

### 2. HIV (`hiv/`)

Analysis of HIV-1 mutations:

| Discovery | Metric | Significance |
|-----------|--------|--------------|
| Distance-Fitness | r = 0.24 | Larger jumps cost more fitness |
| INSTI Constraint | d = 4.30 | Integrase most constrained |
| NNRTI Flexibility | d = 3.59 | Allosteric pocket most flexible |
| HLA-B27 Protection | d = 4.40 | High escape cost explains protection |

---

## Shared Methodology

Both analyses use the same approach:

1. **Encode sequences** using codon encoder
2. **Compute p-adic distances** in embedding space
3. **Identify boundary crossings** between clusters
4. **Correlate with phenotype** (disease risk, fitness cost)

---

## Dependencies

All scripts require the codon encoder from `../genetic_code/data/`:
- `codon_encoder.pt` - Trained neural network
- `learned_codon_mapping.json` - Position→cluster mapping

---

## Running

```bash
# Rheumatoid Arthritis
cd rheumatoid_arthritis/scripts
python 01_hla_functionomic_analysis.py
python 02_hla_expanded_analysis.py
python 03_citrullination_analysis.py
python 04_codon_optimizer.py
python 05_regenerative_axis_analysis.py

# HIV
cd ../hiv/scripts
python 01_hiv_escape_analysis.py
python 02_hiv_drug_resistance.py
```

---

## Future Directions

1. **Expand to other diseases** - Cancer, neurodegenerative
2. **Validate experimentally** - Wet lab testing of predictions
3. **Clinical trials** - Multimodal regeneration protocol for RA

---

**Status:** Analysis complete, experimental validation pending
