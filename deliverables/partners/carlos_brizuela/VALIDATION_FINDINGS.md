# Validation Findings: Carlos Brizuela Package

**Doc-Type:** Validation Report · Version 1.0 · Updated 2026-01-05 · AI Whisperers

---

## Executive Summary

**Package Status: NOT PRODUCTION READY**

Critical issues discovered during validation:
1. NSGA-II tools generate 3-character sequences instead of real AMPs (10-50 AA)
2. Wrong VAE being used (ternary VAE instead of PeptideVAE)
3. PeptideVAE (just trained, r=0.74) is not integrated into optimization pipeline

---

## What Works

| Component | Status | Evidence |
|-----------|:------:|----------|
| **PeptideVAE Training** | PASS | r=0.656 mean, 0.737 best, 0% collapse |
| **sklearn Baseline** | PASS | r=0.56, validated on DRAMP data |
| **DRAMP Data Loader** | PASS | 425 curated records load correctly |
| **NSGA-II Core** | PASS | Algorithm runs, produces Pareto fronts |
| **B1/B8/B10 Scripts** | RUN | Scripts execute without errors |

---

## Critical Issues

### Issue 1: Wrong VAE Model

**Symptom:** Sequences generated are 3 characters long (KPS, KLL, KLS)

**Root Cause:**
```
VAE Service: Loaded model from sandbox-training/checkpoints/homeostatic_rich/best.pt
```
This is the **Ternary VAE** (for 3-adic encoding), NOT the PeptideVAE.

**Evidence:**
```json
{
  "sequence": "KPS",
  "length": 3,
  "activity": 7.95
}
```

**Fix Required:** Update `shared/vae_service.py` to load PeptideVAE from:
```
checkpoints_definitive/best_production.pt
```

### Issue 2: No Real Activity Prediction

**Symptom:** Activity scores are heuristic-based, not ML predictions

**Root Cause:** The NSGA-II objectives use heuristic formulas:
```python
activity = 10 - (abs(charge - 4) + abs(hydro - 0.5) * 3)  # Heuristic!
```

**Fix Required:** Integrate PeptideVAE or sklearn models:
```python
# Using PeptideVAE
mic_pred = peptide_vae(sequence)['mic_pred']
activity = -mic_pred  # Lower MIC = higher activity

# Using sklearn
features = compute_ml_features(sequence)
activity = sklearn_model.predict(features)
```

### Issue 3: Decoder Not Producing Real Peptides

**Symptom:** Even with correct model, decoder produces short sequences

**Root Cause:** PeptideVAE is an **encoder** (sequence → latent → MIC), not a **decoder** (latent → sequence)

**Architecture Mismatch:**
```
Current Flow (Broken):
  Latent (16D) → Ternary VAE Decoder → 3-char sequence

Required Flow:
  Option A: Latent (16D) → PeptideVAE Decoder → 20-char sequence
  Option B: Mutation/Evolution on real seed sequences
```

---

## What Carlos Actually Needs

### Use Case 1: MIC Prediction for Candidate Peptides

**Need:** "Given a peptide sequence, predict its MIC against pathogens"

**Solution:** PeptideVAE (r=0.74) - READY

```python
from src.encoders.peptide_encoder import PeptideVAE

model = PeptideVAE.load('checkpoints_definitive/best_production.pt')
mic = model.predict("KLWKKLKKALK")  # Returns predicted MIC
```

### Use Case 2: Design Novel AMPs via Optimization

**Need:** "Find peptide sequences with optimal activity/toxicity tradeoff"

**Solution:** NOT READY - Requires one of:

1. **Sequence-space evolution** (recommended)
   - Start with known AMPs from DRAMP
   - Mutate sequences, evaluate with PeptideVAE
   - Select Pareto-optimal mutations

2. **Latent-space optimization** (requires work)
   - Train proper sequence decoder
   - Optimize in latent space
   - Decode to sequences

### Use Case 3: Pathogen-Specific Design

**Need:** "Design AMPs targeting S. aureus specifically"

**Solution:** PARTIAL - sklearn models ready, integration needed

```python
from scripts.dramp_activity_loader import DRAMPLoader
loader = DRAMPLoader()
mic = loader.predict_activity(
    sequence="KLWKKLKKALK",
    pathogen="saureus"
)
```

---

## Recommended Fix Plan

### Phase 1: Quick Win (1-2 hours)

1. **Create `predict_mic.py` script** - Direct inference with PeptideVAE
2. **Create `evaluate_candidates.py`** - Score a list of peptides
3. **Document MIC prediction API** - What works today

### Phase 2: Sequence Evolution (4-6 hours)

1. **Implement `sequence_nsga2.py`** - NSGA-II in sequence space
2. **Use real seed sequences** from DRAMP database
3. **Integrate PeptideVAE** as activity objective
4. **Add sklearn ensemble** for stability

### Phase 3: Proper VAE Decoder (Future)

1. **Train autoregressive decoder** on DRAMP sequences
2. **Enable latent → sequence** generation
3. **Integrate with existing NSGA-II**

---

## What Can Be Delivered Today

| Deliverable | Status | Value to Carlos |
|-------------|:------:|-----------------|
| MIC prediction API | READY | Score any peptide sequence |
| sklearn baselines | READY | Pathogen-specific predictions |
| PeptideVAE checkpoint | READY | Best-in-class activity predictor |
| NSGA-II optimizer | BROKEN | Needs sequence-space rewrite |
| B1/B8/B10 tools | BROKEN | Need complete integration |

---

## Business Recommendation

**Do NOT deliver B1/B8/B10 tools** in current state - they produce unusable results.

**DO deliver:**
1. PeptideVAE checkpoint with prediction script
2. sklearn baselines with pathogen-specific models
3. Clear documentation of prediction API
4. Honest assessment of optimization capabilities

**Promise for future:**
- Sequence-space optimization (Phase 2)
- Full latent-space optimization (Phase 3)

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-05 | 1.0 | Initial validation findings |
