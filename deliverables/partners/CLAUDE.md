# Partner Packages - Validation Status

**Doc-Type:** Validation Tracking · Version 1.0 · Updated 2026-01-08 · AI Whisperers

**Purpose:** This document tracks the ACTUAL validation status of each partner package based on reproducible model inference, not exploration scripts.

**Update Policy:** Must be updated at every commit before pushing. Status must reflect verified inference results.

---

## Package Status Summary

| Package | Delivery Status | Model Validated | Inference Tested | Last Verified |
|---------|:---------------:|:---------------:|:----------------:|---------------|
| jose_colbes | 95% | PENDING | PENDING | - |
| alejandra_rojas | 85% | PENDING | PENDING | - |
| carlos_brizuela | 70% | PENDING | PENDING | - |
| hiv_research_package | Complete | N/A (API) | PENDING | - |

**Legend:**
- PENDING: Not yet verified this session
- PASS: Inference produces expected outputs
- FAIL: Issues found (see details)
- N/A: Not applicable (e.g., API-only packages)

---

## Jose Colbes - Protein Stability (DDG)

### Claimed Status: 95% Ready

### Validation Evidence (TO VERIFY)
| Metric | Claimed | Verified | Source |
|--------|---------|----------|--------|
| LOO CV Spearman | 0.585 | PENDING | validation/results/ |
| p-value | <0.001 | PENDING | validation/results/ |
| 95% CI | [0.341, 0.770] | PENDING | bootstrap |
| Rosetta-blind detection | Working | PENDING | C1 script |

### Model Checkpoint
- **Path:** `jose_colbes/models/` or `research/codon-encoder/`
- **Type:** TrainableCodonEncoder or sklearn ensemble
- **Inference command:** PENDING

### Last Inference Test
- **Date:** PENDING
- **Command:** PENDING
- **Output:** PENDING
- **Status:** PENDING

---

## Alejandra Rojas - Arbovirus Primers

### Claimed Status: 85% Ready

### Validation Evidence (TO VERIFY)
| Metric | Claimed | Verified | Source |
|--------|---------|----------|--------|
| Pan-arbovirus primers | 7 viruses | PENDING | pan_arbovirus_primers/ |
| DENV-4 primers | CSV exists | PENDING | DENV-4_primers.csv |
| Clade-specific | 5 pairs | PENDING | clade_specific_primers.json |
| In-silico PCR | Complete | PENDING | insilico_pcr_results.json |

### Model Checkpoint
- **Path:** Uses src.core.padic_math (no ML model)
- **Type:** Algorithmic primer design
- **Inference command:** `python scripts/A2_pan_arbovirus_primers.py`

### Last Inference Test
- **Date:** PENDING
- **Command:** PENDING
- **Output:** PENDING
- **Status:** PENDING

---

## Carlos Brizuela - AMP Optimization

### Claimed Status: 70% Ready

### Validation Evidence (TO VERIFY)
| Metric | Claimed | Verified | Source |
|--------|---------|----------|--------|
| MIC prediction | r=0.74 | PENDING | PeptideVAE |
| NSGA-II working | Fixed | PENDING | B1 output |
| Toxicity | Heuristic | PENDING | physicochemical rules |
| S_aureus candidates | Generated | PENDING | results/pathogen_specific/ |

### Model Checkpoint
- **Path:** `checkpoints_definitive/best_production.pt`
- **Type:** PeptideVAE (PeptideMICPredictor)
- **Inference command:** `python scripts/predict_mic.py --sequence "KLWKKLKKALK"`

### Last Inference Test
- **Date:** PENDING
- **Command:** PENDING
- **Output:** PENDING
- **Status:** PENDING

---

## HIV Research Package - Drug Resistance

### Claimed Status: Complete

### Validation Evidence (TO VERIFY)
| Metric | Claimed | Verified | Source |
|--------|---------|----------|--------|
| TDR screening | H6 working | PENDING | H6 script |
| LA selection | H7 working | PENDING | H7 script |
| Stanford HIVdb | Integrated | PENDING | API calls |

### Model Checkpoint
- **Path:** N/A (uses Stanford HIVdb API)
- **Type:** API integration
- **Inference command:** `python scripts/H6_tdr_screening.py`

### Last Inference Test
- **Date:** PENDING
- **Command:** PENDING
- **Output:** PENDING
- **Status:** PENDING

---

## Verification Protocol

For each package, verification requires:

1. **Read package structure** - Understand what scripts/models exist
2. **Identify inference command** - What command produces model output
3. **Run inference** - Execute and capture output
4. **Verify output format** - Does it match expected structure
5. **Update this document** - Record results with timestamp

### Verification Commands Template

```bash
# Colbes - DDG prediction
python deliverables/partners/jose_colbes/scripts/C1_rosetta_blind_detection.py --test

# Rojas - Primer design
python deliverables/partners/alejandra_rojas/scripts/A2_pan_arbovirus_primers.py --demo

# Brizuela - MIC prediction
python deliverables/partners/carlos_brizuela/scripts/predict_mic.py --sequence "KLWKKLKKALK"

# HIV - TDR screening
python deliverables/partners/hiv_research_package/scripts/H6_tdr_screening.py --test
```

---

## Known Issues & Gaps

### Colbes
- PENDING: Need to verify checkpoint loads correctly

### Rojas
- PENDING: Need to verify primer output format

### Brizuela
- Toxicity/stability are heuristics, NOT ML models
- Need to verify PeptideVAE checkpoint loads

### HIV
- PENDING: Need to verify Stanford API connectivity

---

## Update Log

| Date | Package | Action | Result |
|------|---------|--------|--------|
| 2026-01-08 | ALL | Initial draft created | PENDING verification |

---

*This document must be updated before every push to main.*
