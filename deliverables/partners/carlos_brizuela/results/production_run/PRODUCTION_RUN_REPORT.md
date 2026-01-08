# Production Run Report - Brizuela AMP Design Package

**Doc-Type:** Technical Report | Version 1.0 | 2026-01-05 | AI Whisperers

---

## Run Configuration

| Parameter | Value |
|-----------|-------|
| Date | 2026-01-05 |
| Generations | 30 |
| Population | 60 |
| Random Seed | 42 |
| Model | PeptideVAE (Spearman r=0.74) |

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total JSON files | 6 |
| Total candidates | 350 |
| Unique sequences | 213 |
| Pipelines run | 6 |

---

## B1: Pathogen-Specific Design

### A. baumannii (Critical Priority, Gram-negative)

| Metric | Value |
|--------|-------|
| WHO Priority | Critical |
| Resistance | Carbapenem-resistant |
| Gram Type | Negative |
| Candidates | 60 |
| MIC Range | 0.80 - 1.18 ug/mL |
| MIC Median | 0.87 ug/mL |
| Length Range | 11 - 28 AA |
| Net Charge Range | +0.5 to +4.0 |

**Top 5 Candidates:**

| Rank | Sequence | MIC (ug/mL) | Length | Charge |
|------|----------|-------------|--------|--------|
| 1 | GIGGHTSHTQCTS | 0.80 | 13 | +0.5 |
| 2 | GIGGHTQHTQTS | 0.80 | 12 | +0.5 |
| 3 | GIGGHTQHTQTS | 0.80 | 12 | +0.5 |
| 4 | AGGGHTQHTQTS | 0.80 | 12 | +0.5 |
| 5 | GGTGHTQHTQTS | 0.80 | 12 | +0.5 |

### S. aureus (High Priority, Gram-positive)

| Metric | Value |
|--------|-------|
| WHO Priority | High |
| Resistance | MRSA |
| Gram Type | Positive |
| Candidates | 60 |
| MIC Range | 0.81 - 3.61 ug/mL |
| MIC Median | 0.92 ug/mL |
| Length Range | 8 - 22 AA |
| Net Charge Range | 0.0 to +3.0 |

**Top 5 Candidates:**

| Rank | Sequence | MIC (ug/mL) | Length | Charge |
|------|----------|-------------|--------|--------|
| 1 | QCCANNCCNNAGS | 0.81 | 13 | 0 |
| 2 | QCCANNGGS | 0.81 | 9 | 0 |
| 3 | QCCANNGANAGS | 0.81 | 12 | 0 |
| 4 | QCCAMGCCNAGQ | 0.81 | 12 | 0 |
| 5 | QCCANNCCNGA | 0.81 | 11 | 0 |

### P. aeruginosa (Critical Priority, Gram-negative)

| Metric | Value |
|--------|-------|
| WHO Priority | Critical |
| Resistance | MDR |
| Gram Type | Negative |
| Candidates | 60 |
| MIC Range | 0.80 - 1.08 ug/mL |
| MIC Median | 0.86 ug/mL |
| Length Range | 10 - 27 AA |
| Net Charge Range | -1.0 to +5.0 |

**Top 5 Candidates:**

| Rank | Sequence | MIC (ug/mL) | Length | Charge |
|------|----------|-------------|--------|--------|
| 1 | DLTAQSTQQYCQQS | 0.80 | 14 | -0.5 |
| 2 | DLTSATSTQQYTQQS | 0.80 | 15 | -0.5 |
| 3 | CQCWSTQTYTQQS | 0.80 | 13 | 0 |
| 4 | DLTAQSTQQYCQFS | 0.80 | 14 | -0.5 |
| 5 | QSTQTYTDAQQFS | 0.80 | 13 | -0.5 |

---

## B8: Microbiome-Safe Design

### GUT Microbiome Context

| Metric | Value |
|--------|-------|
| Pathogens | C_difficile, E_coli_pathogenic, Salmonella |
| Commensals | Lactobacillus, Bifidobacterium, Bacteroides |
| Candidates | 60 |
| Selectivity Range | 0.00 - 1.40 |
| Selectivity Median | 0.43 |

**Top 5 Selective Candidates:**

| Rank | Sequence | SI | MIC (ug/mL) | Confidence |
|------|----------|-----|-------------|------------|
| 1 | CVTFYKTKTFFKLVTFFKFVRTFVR | 1.40 | 3.31 | Medium |
| 2 | RLKKTFFTVVTFTVFTKKTFFVKLFV | 0.93 | 3.36 | Medium |
| 3 | CVTNFPFFKFVRTVKKTF | 0.75 | 3.22 | Low |
| 4 | RLAKKTFFTVRKFFTV | 0.72 | 2.48 | Medium |
| 5 | TLKFVKKLFFTKFTFVKF | 0.62 | 1.89 | Medium |

**Interpretation:** Selectivity Index > 1.0 indicates peptide kills pathogens more effectively than it harms commensals. Top candidate (SI=1.40) shows good gut pathogen selectivity.

### SKIN Microbiome Context

| Metric | Value |
|--------|-------|
| Pathogens | S_aureus, MRSA, P_acnes_pathogenic |
| Commensals | S_epidermidis, C_acnes, Corynebacterium |
| Candidates | 60 |
| Selectivity Range | 0.00 - 0.77 |
| Selectivity Median | 0.00 |

**Top 5 Selective Candidates:**

| Rank | Sequence | SI | MIC (ug/mL) | Confidence |
|------|----------|-----|-------------|------------|
| 1 | WLALAAKKLAKLAKLTK | 0.77 | 3.03 | Medium |
| 2 | WVALAKLAAKKTLTK | 0.66 | 2.55 | Medium |
| 3 | WLALAKAKVAKKLVLGAKKL | 0.61 | 3.62 | Medium |
| 4 | WLALAKAKVAKKLVLGAKKL | 0.58 | 3.79 | Medium |
| 5 | VALAKLWAKKLALKLT | 0.54 | 2.43 | Medium |

**Interpretation:** Skin microbiome selectivity is more challenging (max SI=0.77 < 1.0). This reflects the biological reality that skin pathogens (S. aureus) and commensals (S. epidermidis) are closely related species.

---

## B10: Synthesis Optimization

| Metric | Value |
|--------|-------|
| Candidates | 50 |
| Objectives | MIC, Difficulty, Coupling, Cost |
| MIC Range | -0.1026 to -0.0802 (log10) |
| Cost Range | $10.00 - $34.00 |
| Coupling Efficiency | 51% - 87% |

**Grade Distribution:**

| Grade | Count | Percentage |
|-------|-------|------------|
| EXCELLENT | 0 | 0% |
| GOOD | 0 | 0% |
| MODERATE | 0 | 0% |
| CHALLENGING | 0 | 0% |
| DIFFICULT | 50 | 100% |

**Top 5 Synthesis-Optimized Candidates:**

| Rank | Sequence | MIC (log10) | Cost | Coupling |
|------|----------|-------------|------|----------|
| 1 | TGMSENQSHSHSMAQ | -0.1026 | $34.0 | 51% |
| 2 | TGMSENNQSHSHSAQ | -0.1025 | $32.5 | 53% |
| 3 | CPSSHTTSMAQ | -0.1018 | $25.5 | 57% |
| 4 | TAQYASHSMS | -0.1014 | $24.0 | 54% |
| 5 | TAQYASHSMCS | -0.1004 | $26.0 | 55% |

**Observation:** All candidates received "DIFFICULT" synthesis grade. This suggests the optimization is biased toward sequences that predict well for MIC but contain synthesis-challenging residues. Further tuning of objective weights may be needed.

---

## Cross-Pipeline Analysis

### MIC Comparison by Pipeline

| Pipeline | Best MIC (ug/mL) | Median MIC | Confidence |
|----------|------------------|------------|------------|
| B1: A_baumannii | 0.80 | 0.87 | Low |
| B1: S_aureus | 0.81 | 0.92 | Low |
| B1: P_aeruginosa | 0.80 | 0.86 | Low |
| B8: Gut | 1.63* | 2.75* | Medium |
| B8: Skin | 2.43* | 3.03* | Medium |
| B10: Synthesis | 0.79* | 0.82* | Unknown |

*MIC values for B8/B10 converted from log10 where applicable

### Key Observations

1. **B1 Pathogen-Specific:** All three pathogens show similar MIC predictions (~0.80 ug/mL). This suggests the PeptideVAE model may have limited discriminative power across pathogen types.

2. **B8 Microbiome-Safe:** Gut microbiome shows better selectivity potential (max SI=1.40) compared to skin (max SI=0.77). This is biologically plausible given the greater taxonomic diversity in gut microbiome.

3. **B10 Synthesis:** All candidates marked "DIFFICULT" indicates the synthesis difficulty heuristics may need recalibration. The NSGA-II appears to optimize MIC at the expense of synthesis feasibility.

4. **Confidence Levels:** B1 candidates predominantly "Low" confidence, B8 "Medium" confidence. This reflects the exploratory nature of sequence-space optimization.

---

## Output Files

| File | Format | Contents | Size |
|------|--------|----------|------|
| A_baumannii_results.json | JSON | Full results + candidates | 60 candidates |
| A_baumannii_candidates.csv | CSV | Tabular format | 60 rows |
| A_baumannii_peptides.fasta | FASTA | Sequences only | 60 sequences |
| S_aureus_results.json | JSON | Full results + candidates | 60 candidates |
| S_aureus_candidates.csv | CSV | Tabular format | 60 rows |
| S_aureus_peptides.fasta | FASTA | Sequences only | 60 sequences |
| P_aeruginosa_results.json | JSON | Full results + candidates | 60 candidates |
| P_aeruginosa_candidates.csv | CSV | Tabular format | 60 rows |
| P_aeruginosa_peptides.fasta | FASTA | Sequences only | 60 sequences |
| microbiome_safe_gut_results.json | JSON | Full results + candidates | 60 candidates |
| microbiome_safe_gut_candidates.csv | CSV | Tabular format | 60 rows |
| microbiome_safe_gut_peptides.fasta | FASTA | Sequences only | 60 sequences |
| microbiome_safe_skin_results.json | JSON | Full results + candidates | 60 candidates |
| microbiome_safe_skin_candidates.csv | CSV | Tabular format | 60 rows |
| microbiome_safe_skin_peptides.fasta | FASTA | Sequences only | 60 sequences |
| synthesis_optimized_results.json | JSON | Full results + candidates | 50 candidates |
| synthesis_optimized_candidates.csv | CSV | Tabular format | 50 rows |
| synthesis_optimized.fasta | FASTA | Sequences only | 50 sequences |

---

## Recommendations

### For Wet-Lab Validation

1. **Priority Candidates:** Focus on B1 top candidates for A. baumannii and P. aeruginosa (critical WHO priority)
2. **Selectivity Testing:** Validate B8 gut candidates (SI > 1.0) against real microbiome panels
3. **Synthesis:** Order shorter peptides first (8-12 AA) before longer candidates

### For Model Improvement

1. **Confidence Calibration:** Investigate why most predictions are "Low" confidence
2. **Synthesis Heuristics:** Recalibrate difficulty scoring - current model may be overly pessimistic
3. **Pathogen Specificity:** Consider pathogen-specific fine-tuning of PeptideVAE

### For Next Run

1. Increase generations to 50-100 for better convergence
2. Test additional pathogens (Enterobacteriaceae, H_pylori)
3. Run oral and urinary microbiome contexts
4. Experiment with different NSGA-II crossover/mutation rates

---

## Reproducibility

```bash
# Reproduce this run
cd deliverables/partners/carlos_brizuela/scripts

# B1 Pathogen-specific
python B1_pathogen_specific_design.py --pathogen A_baumannii --generations 30 --population 60 --seed 42
python B1_pathogen_specific_design.py --pathogen S_aureus --generations 30 --population 60 --seed 42
python B1_pathogen_specific_design.py --pathogen P_aeruginosa --generations 30 --population 60 --seed 42

# B8 Microbiome-safe
python B8_microbiome_safe_amps.py --context gut --generations 30 --population 60 --seed 42
python B8_microbiome_safe_amps.py --context skin --generations 30 --population 60 --seed 42

# B10 Synthesis-optimized
python B10_synthesis_optimization.py --generations 30 --population 60 --seed 42
```

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-05 | 1.0 | Initial production run report |
