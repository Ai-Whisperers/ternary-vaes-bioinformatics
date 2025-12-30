# Easy Research Ideas - Implementation Documentation

> **Complete implementation guide for the 8 "Easy" research ideas (Score 1-2)**

**Document Version:** 1.0
**Last Updated:** December 29, 2025
**Total Implementations:** 8 scripts across 4 partner projects

---

## Overview

This document provides comprehensive documentation for all 8 research ideas classified as "Easy" (difficulty score 1-2) in the Implementation Difficulty Analysis. These ideas leverage existing codebase components and require minimal new development.

---

## Quick Reference

| ID | Idea | Partner | Script | Lines |
|----|------|---------|--------|-------|
| A2 | Pan-Arbovirus Primer Library | Alejandra Rojas | `A2_pan_arbovirus_primers.py` | ~450 |
| B1 | Pathogen-Specific AMP Design | Carlos Brizuela | `B1_pathogen_specific_design.py` | ~480 |
| B8 | Microbiome-Safe AMPs | Carlos Brizuela | `B8_microbiome_safe_amps.py` | ~380 |
| B10 | Synthesis Optimization | Carlos Brizuela | `B10_synthesis_optimization.py` | ~350 |
| C1 | Rosetta-Blind Detection | José Colbes | `C1_rosetta_blind_detection.py` | ~400 |
| C4 | Mutation Effect Predictor | José Colbes | `C4_mutation_effect_predictor.py` | ~420 |
| H6 | TDR Screening Tool | HIV Package | `H6_tdr_screening.py` | ~380 |
| H7 | LA Injectable Selection | HIV Package | `H7_la_injectable_selection.py` | ~400 |

---

## A2: Pan-Arbovirus Primer Library

### Location
`deliverables/alejandra_rojas/scripts/A2_pan_arbovirus_primers.py`

### Purpose
Design a comprehensive RT-PCR primer library covering all major arboviruses circulating in Paraguay: Dengue (all 4 serotypes), Zika, Chikungunya, and Mayaro virus.

### Key Features
- Multi-virus sequence processing
- Cross-reactivity checking (ensure specificity)
- Serotype-specific primers for Dengue
- Primer pair design with Tm matching

### Usage
```bash
cd deliverables/alejandra_rojas
python scripts/A2_pan_arbovirus_primers.py --output_dir results/pan_arbovirus_primers/
```

### Output Structure
```
results/pan_arbovirus_primers/
├── DENV-1_primers.csv        # Ranked primer candidates
├── DENV-1_pairs.csv          # Primer pair combinations
├── DENV-1_primers.fasta      # Specific primers in FASTA
├── DENV-2_primers.csv
├── ...
├── ZIKV_primers.csv
├── CHIKV_primers.csv
├── MAYV_primers.csv
└── library_summary.json      # Complete metadata
```

### Technical Details
- **Window size:** 20 nt
- **GC range:** 40-60%
- **Tm range:** 55-65°C
- **Cross-reactivity threshold:** <70% homology
- **Amplicon size:** 100-300 bp

---

## B1: Pathogen-Specific AMP Design

### Location
`deliverables/carlos_brizuela/scripts/B1_pathogen_specific_design.py`

### Purpose
Design antimicrobial peptides targeting WHO priority pathogens using NSGA-II multi-objective optimization.

### Target Pathogens
| Priority | Pathogen | Resistance |
|----------|----------|------------|
| Critical | *A. baumannii* | Carbapenem-resistant |
| Critical | *P. aeruginosa* | MDR |
| Critical | Enterobacteriaceae | Carbapenem-resistant |
| High | *S. aureus* (MRSA) | Methicillin-resistant |
| High | *H. pylori* | Clarithromycin-resistant |

### Usage
```bash
cd deliverables/carlos_brizuela

# Single pathogen
python scripts/B1_pathogen_specific_design.py --pathogen A_baumannii

# All pathogens
python scripts/B1_pathogen_specific_design.py --all_pathogens
```

### Objectives Optimized
1. **Activity:** Minimize MIC against target pathogen
2. **Toxicity:** Minimize host cell toxicity
3. **Stability:** Maximize sequence validity in latent space

### Output
- `{pathogen}_results.json` - Complete optimization results
- `{pathogen}_candidates.csv` - Top peptide candidates
- `{pathogen}_peptides.fasta` - Sequences in FASTA format

---

## B8: Microbiome-Safe AMPs

### Location
`deliverables/carlos_brizuela/scripts/B8_microbiome_safe_amps.py`

### Purpose
Design AMPs that selectively kill pathogens while sparing beneficial skin/gut microbiome members.

### Selectivity Targets
| Target | Role | Desired MIC |
|--------|------|-------------|
| *S. aureus* | Pathogen | < 4 μg/mL |
| MRSA | Pathogen | < 8 μg/mL |
| *S. epidermidis* | Commensal | > 64 μg/mL |
| *C. acnes* | Commensal | > 64 μg/mL |
| *Malassezia* | Commensal | > 64 μg/mL |

### Key Metric: Selectivity Index
```
SI = geometric_mean(commensal MICs) / geometric_mean(pathogen MICs)
```
Higher SI = more selective (kills pathogens, spares commensals)

### Usage
```bash
python scripts/B8_microbiome_safe_amps.py --population 200 --generations 50
```

### Output
- `microbiome_safe_results.json` - Full results with MIC predictions
- `microbiome_safe_candidates.csv` - Top selective candidates
- `microbiome_safe_peptides.fasta` - Sequences

---

## B10: Synthesis Optimization

### Location
`deliverables/carlos_brizuela/scripts/B10_synthesis_optimization.py`

### Purpose
Optimize AMPs for ease of solid-phase peptide synthesis (SPPS) by predicting and minimizing synthesis difficulty.

### Synthesis Challenges Addressed
| Challenge | Cause | Prediction Target |
|-----------|-------|-------------------|
| Aggregation | Hydrophobic stretches | Aggregation propensity |
| Deletion peptides | Steric hindrance | Coupling efficiency |
| Racemization | Base-sensitive residues | Epimerization risk |
| Aspartimide | Asp-Xxx motifs | Side reaction probability |

### Metrics
- **Aggregation propensity:** Based on hydrophobic runs
- **Coupling efficiency:** Product of per-residue coupling
- **Cost estimation:** Based on amino acid reagent costs
- **Difficulty score:** Composite metric

### Usage
```bash
python scripts/B10_synthesis_optimization.py --population 200 --generations 50
```

### Output
- `synthesis_optimized_results.json` - Full synthesis metrics
- `synthesis_optimized_candidates.csv` - Easy-to-synthesize candidates

---

## C1: Rosetta-Blind Detection

### Location
`deliverables/jose_colbes/scripts/C1_rosetta_blind_detection.py`

### Purpose
Identify protein conformations that Rosetta scores as stable but geometric scoring flags as unstable ("Rosetta-blind spots").

### Classification Categories
| Category | Rosetta | Geometry | Interpretation |
|----------|---------|----------|----------------|
| Concordant stable | Stable | Stable | Agreement |
| Concordant unstable | Unstable | Unstable | Agreement |
| **Rosetta-blind** | Stable | Unstable | **KEY FINDING** |
| Geometry-blind | Unstable | Stable | Geometry misses |

### Discordance Score
```python
discordance = geom_norm × rosetta_norm
# High when: geometry says unstable AND Rosetta says stable
```

### Usage
```bash
python scripts/C1_rosetta_blind_detection.py --n_demo 500 --output results/rosetta_blind/
```

### Output
- `rosetta_blind_report.json` - Complete analysis
- Top Rosetta-blind residues ranked by discordance

---

## C4: Mutation Effect Predictor (ΔΔG)

### Location
`deliverables/jose_colbes/scripts/C4_mutation_effect_predictor.py`

### Purpose
Predict the effect of point mutations on protein stability using geometric features.

### Features Used
| Feature | Description | Weight |
|---------|-------------|--------|
| Δ Volume | Amino acid size change | 0.015 (core) |
| Δ Hydrophobicity | Burial preference change | 0.5 (core) |
| Δ Charge | Electrostatic change | 1.5 (core) |
| Δ Geometric | Rotamer landscape change | 1.2 |
| Δ Entropy | Rotamer entropy change | 1.0 |

### Classification
| ΔΔG Range | Classification |
|-----------|---------------|
| > 1.0 kcal/mol | Destabilizing |
| -0.5 to 1.0 | Neutral |
| < -0.5 kcal/mol | Stabilizing |

### Usage
```bash
# Demo mutations
python scripts/C4_mutation_effect_predictor.py

# Specific mutations
python scripts/C4_mutation_effect_predictor.py --mutations "G45A,M184V,K103N"

# From file
python scripts/C4_mutation_effect_predictor.py --mutations mutations.txt
```

### Output
- `mutation_effects.json` - Predicted ΔΔG values
- Classification and confidence for each mutation

---

## H6: TDR Screening Tool

### Location
`deliverables/hiv_research_package/scripts/H6_tdr_screening.py`

### Purpose
Screen treatment-naive HIV patients for transmitted drug resistance (TDR) to guide first-line regimen selection.

### TDR Mutations Database
| Class | Key Mutations | Prevalence |
|-------|---------------|------------|
| NRTI | M184V, K65R, T215Y | 5-10% |
| NNRTI | K103N, Y181C, G190A | 3-8% |
| INSTI | N155H, Q148H (rare) | <1% |

### First-Line Regimens Evaluated
1. TDF/3TC/DTG (WHO preferred)
2. TDF/FTC/DTG
3. TAF/FTC/DTG
4. TDF/3TC/EFV
5. ABC/3TC/DTG

### Usage
```bash
# Demo mode
python scripts/H6_tdr_screening.py --demo

# With sequence
python scripts/H6_tdr_screening.py --sequence patient.fasta --patient_id P001
```

### Output
- TDR status (positive/negative)
- Detected mutations
- Drug susceptibility profile
- Recommended regimen
- Alternative regimens

---

## H7: LA Injectable Selection

### Location
`deliverables/hiv_research_package/scripts/H7_la_injectable_selection.py`

### Purpose
Predict which patients will maintain viral suppression on long-acting injectables (CAB-LA/RPV-LA).

### LA Drugs
| Drug | Class | Half-life | Administration |
|------|-------|-----------|----------------|
| Cabotegravir (CAB) | INSTI | 25 days | IM gluteal |
| Rilpivirine (RPV) | NNRTI | 45 days | IM gluteal |

### Risk Factors Evaluated
1. **Baseline resistance** (CAB/RPV mutations)
2. **BMI** (affects pharmacokinetics)
3. **Adherence history**
4. **Prior NNRTI exposure**
5. **Psychiatric history** (RPV caution)

### Success Probability Model
```
P(success) = 0.95 - CAB_risk×0.3 - RPV_risk×0.4 - PK_penalty - adherence_penalty
```

### Usage
```bash
python scripts/H7_la_injectable_selection.py --demo
```

### Output
- Eligibility status
- Success probability
- Risk factors identified
- Recommendation (eligible/caution/not recommended)
- Monitoring plan

---

## Running All Implementations

### Quick Test (Demo Mode)
```bash
# Navigate to project root
cd ternary-vaes-bioinformatics

# A2: Pan-Arbovirus Primers
python deliverables/alejandra_rojas/scripts/A2_pan_arbovirus_primers.py

# B1: Pathogen-Specific Design
python deliverables/carlos_brizuela/scripts/B1_pathogen_specific_design.py --pathogen A_baumannii

# B8: Microbiome-Safe AMPs
python deliverables/carlos_brizuela/scripts/B8_microbiome_safe_amps.py

# B10: Synthesis Optimization
python deliverables/carlos_brizuela/scripts/B10_synthesis_optimization.py

# C1: Rosetta-Blind Detection
python deliverables/jose_colbes/scripts/C1_rosetta_blind_detection.py

# C4: Mutation Effect Predictor
python deliverables/jose_colbes/scripts/C4_mutation_effect_predictor.py

# H6: TDR Screening
python deliverables/hiv_research_package/scripts/H6_tdr_screening.py --demo

# H7: LA Injectable Selection
python deliverables/hiv_research_package/scripts/H7_la_injectable_selection.py --demo
```

---

## Dependencies

All implementations require:
- Python 3.8+
- NumPy

Optional (for full functionality):
- pandas (CSV export)
- BioPython (FASTA parsing for A2)
- PyTorch (for loading VAE checkpoints)

---

## Integration with Core Framework

These implementations are designed to integrate with the Ternary VAE framework:

```python
# Example: Using VAE latent space for B1
from src.models import TernaryVAEV5_11_PartialFreeze

vae = TernaryVAEV5_11_PartialFreeze(latent_dim=16)
vae.load_state_dict(torch.load('checkpoints/best.pt'))

# Decode latent vectors from optimization
sequence = vae.decode(optimal_latent_vector)
```

---

## Next Steps

After validating these implementations:
1. Connect to real VAE checkpoints for sequence decoding
2. Train pathogen-specific activity predictors (B1)
3. Integrate with experimental validation pipelines
4. Deploy clinical tools (H6, H7) in pilot settings

---

*Documentation for the Ternary VAE Bioinformatics Partnership*
*Implementation of 8 "Easy" Research Ideas (Difficulty Score 1-2)*
