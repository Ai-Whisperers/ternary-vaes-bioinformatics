# Diseases Module

**Multi-disease drug resistance and escape prediction framework using p-adic hyperbolic geometry.**

## Overview

This module provides specialized analyzers for predicting drug resistance, antibody escape, and treatment outcomes across **11 disease domains** using the Ternary VAE framework's p-adic encoding.

## Disease Coverage

| Disease | Type | Analyzer | Drugs/Targets | Key Features |
|---------|------|----------|---------------|--------------|
| **HIV** | Viral | `hiv_analyzer.py` | 23 ARVs | Transfer learning, ESM-2, 0.89 Spearman |
| **SARS-CoV-2** | Viral | `sars_cov2_analyzer.py` | Paxlovid, mAbs | Mpro resistance, Spike escape |
| **Tuberculosis** | Bacterial | `tuberculosis_analyzer.py` | 13 drugs | MDR/XDR classification |
| **Influenza** | Viral | `influenza_analyzer.py` | NAIs, baloxavir | Vaccine strain selection |
| **HCV** | Viral | `hcv_analyzer.py` | DAAs | NS3/NS5A/NS5B RAS |
| **HBV** | Viral | `hbv_analyzer.py` | NAs | S-gene overlap analysis |
| **Malaria** | Parasitic | `malaria_analyzer.py` | ACTs | K13 artemisinin resistance |
| **MRSA** | Bacterial | `mrsa_analyzer.py` | Multiple | mecA/mecC, MDR profiling |
| **Candida auris** | Fungal | `candida_analyzer.py` | Echinocandins, azoles | Pan-resistance alerts |
| **RSV** | Viral | `rsv_analyzer.py` | Nirsevimab, palivizumab | mAb escape |
| **Cancer** | Oncology | `cancer_analyzer.py` | TKIs | EGFR/BRAF/KRAS/ALK |

## Quick Start

### HIV Drug Resistance (Original)

```python
from src.diseases import DiseaseRegistry, get_disease_config

# Get HIV configuration
hiv_config = get_disease_config("hiv")

# Run prediction (uses trained VAE)
from src.models import TernaryVAE
model = TernaryVAE.load("outputs/best_hiv_model.pt")
```

### SARS-CoV-2 Analysis

```python
from src.diseases import SARSCoV2Analyzer, SARSCoV2Gene

analyzer = SARSCoV2Analyzer()

# Analyze Mpro sequences for Paxlovid resistance
results = analyzer.analyze(
    sequences={SARSCoV2Gene.NSP5: ["SGFRKMAFPS..."]},
    gene=SARSCoV2Gene.NSP5
)

print(results["drug_resistance"]["nirmatrelvir"])
# {'scores': [0.85], 'classifications': ['resistant'], 'mutations': [...]}
```

### Tuberculosis MDR Detection

```python
from src.diseases import TuberculosisAnalyzer, TBGene, TBDrug

analyzer = TuberculosisAnalyzer()

# Analyze multiple genes
results = analyzer.analyze(
    sequences={
        TBGene.RPOB: ["...rpoB sequence..."],
        TBGene.KATG: ["...katG sequence..."],
        TBGene.GYRA: ["...gyrA sequence..."],
    }
)

# Get MDR/XDR classification
for classification in results["mdr_classification"]:
    print(f"Isolate {classification['isolate']}: {classification['classification']}")
    # Output: Isolate 0: MDR-TB (or DS-TB, pre-XDR-TB, XDR-TB)
```

### Influenza Vaccine Strain Selection

```python
from src.diseases import InfluenzaAnalyzer, InfluenzaSubtype, InfluenzaGene

analyzer = InfluenzaAnalyzer()

# Compare vaccine candidates to circulating strains
recommendation = analyzer.recommend_vaccine_strain(
    candidate_sequences=["...HA seq 1...", "...HA seq 2..."],
    circulating_sequences=["...circulating 1...", "...circulating 2..."],
    subtype=InfluenzaSubtype.H3N2
)

print(f"Recommended candidate: {recommendation['recommended_index']}")
print(f"Average distance: {recommendation['recommended_score']:.3f}")
```

### Malaria Artemisinin Resistance

```python
from src.diseases import MalariaAnalyzer, MalariaGene, PlasmodiumSpecies

analyzer = MalariaAnalyzer()

# Analyze K13 propeller domain
results = analyzer.analyze(
    sequences={MalariaGene.KELCH13: ["...K13 sequence..."]},
    species=PlasmodiumSpecies.P_FALCIPARUM
)

# Check WHO-validated resistance markers
print(results["artemisinin_status"])  # ['artemisinin_resistant']
print(results["drug_resistance"]["artemisinin"]["who_category"])  # ['WHO_validated']
```

### Cancer Targeted Therapy

```python
from src.diseases import CancerAnalyzer, CancerGene, CancerType

analyzer = CancerAnalyzer()

# Analyze EGFR for TKI resistance
results = analyzer.analyze(
    sequences={CancerGene.EGFR: ["...EGFR sequence..."]},
    cancer_type=CancerType.NSCLC
)

# Get treatment recommendations
for rec in results["treatment_recommendations"]:
    print(f"Recommended: {rec['recommended_therapies']}")
    print(f"Avoid: {rec['avoid_therapies']}")
    print(f"Rationale: {rec['rationale']}")
```

## Mutation Databases

Each analyzer includes curated mutation databases from authoritative sources:

| Disease | Source | Reference |
|---------|--------|-----------|
| TB | WHO Mutation Catalogue 2021/2023 | [WHO](https://www.who.int/publications/i/item/9789240028173) |
| SARS-CoV-2 | Stanford CoVDB | [CoVDB](https://covdb.stanford.edu/) |
| Influenza | WHO GISRS | [GISAID](https://gisaid.org/) |
| HCV | EASL/AASLD Guidelines | [EASL](https://easl.eu/) |
| HBV | HBVdb | [HBVdb](https://hbvdb.ibcp.fr/) |
| Malaria | WHO Artemisinin Markers | [MalariaGEN](https://www.malariagen.net/) |
| MRSA | CLSI/EUCAST | [CARD](https://card.mcmaster.ca/) |
| C. auris | CDC AR Lab Network | [CDC](https://www.cdc.gov/candida-auris/) |
| RSV | FDA/CDC Surveillance | [GISAID](https://gisaid.org/) |
| Cancer | OncoKB/COSMIC | [OncoKB](https://www.oncokb.org/) |

## Clinical Classifications

### Tuberculosis
- **DS-TB**: Drug-susceptible
- **RR-TB**: Rifampicin-resistant only
- **MDR-TB**: RIF + INH resistant
- **pre-XDR-TB**: MDR + fluoroquinolone resistant
- **XDR-TB**: pre-XDR + bedaquiline/linezolid resistant

### Candida auris Alert Levels
- **Low**: Susceptible to all classes
- **Moderate**: Single class resistance
- **High**: Two class resistance
- **Critical**: Pan-resistant (all three classes)

### Cancer Treatment Recommendations
- Sensitizing mutations → targeted therapy
- Resistance mutations → alternative agents or clinical trials
- Treatment recommendations based on OncoKB evidence levels

## Files Structure

```
src/diseases/
├── __init__.py                 # Public API exports
├── base.py                     # DiseaseAnalyzer base class
├── registry.py                 # Disease registry
├── losses.py                   # Multi-disease loss functions
├── variant_escape.py           # Variant escape prediction heads
│
├── # Core Analyzers (HIV ecosystem)
├── hiv_analyzer.py             # HIV drug resistance
│
├── # Viral Pathogens
├── sars_cov2_analyzer.py       # SARS-CoV-2 (Paxlovid, mAbs)
├── influenza_analyzer.py       # Influenza (NAIs, vaccines)
├── hcv_analyzer.py             # HCV (DAAs)
├── hbv_analyzer.py             # HBV (NAs)
├── rsv_analyzer.py             # RSV (mAbs)
│
├── # Bacterial Pathogens
├── tuberculosis_analyzer.py    # TB (13 drugs, MDR/XDR)
├── mrsa_analyzer.py            # MRSA (mecA, MDR)
│
├── # Other Pathogens
├── malaria_analyzer.py         # Malaria (K13, ACTs)
├── candida_analyzer.py         # Candida auris (antifungals)
│
├── # Oncology
├── cancer_analyzer.py          # Cancer (EGFR, BRAF, KRAS, ALK)
│
└── README.md                   # This file
```

## Creating Synthetic Datasets

Each analyzer includes a synthetic dataset generator for testing:

```python
from src.diseases import (
    create_sars_cov2_dataset,
    create_tb_synthetic_dataset,
    create_influenza_synthetic_dataset,
    create_malaria_synthetic_dataset,
    create_cancer_synthetic_dataset,
)

# Create TB dataset for rifampicin
X, y, ids = create_tb_synthetic_dataset(drug=TBDrug.RIFAMPICIN)
print(f"Samples: {len(y)}, Features: {X.shape[1]}")
```

## Integration with Experiment Framework

```python
from src.experiments import CrossDiseaseExperiment, ExperimentConfig

# Run experiments across multiple diseases
experiment = CrossDiseaseExperiment(
    diseases=["hiv", "sars_cov_2", "tuberculosis"],
    base_config=ExperimentConfig(n_folds=5, n_repeats=3)
)

results = experiment.run_all()
print(experiment.generate_comparison_report())
```

## Key Concepts

### P-adic Distance in Drug Resistance

P-adic distance captures how mutations affect protein structure:
- **High p-adic distance**: Major structural change → likely resistance
- **Low p-adic distance**: Subtle change → may preserve susceptibility

### Cross-Disease Transfer Learning

The framework supports transfer learning across diseases:
1. Pre-train on large dataset (e.g., HIV with 200K+ sequences)
2. Fine-tune on smaller dataset (e.g., HCV with limited data)
3. Shared mutation patterns improve predictions

### Thermodynamics vs Kinetics

P-adic encoding captures **thermodynamic** properties:
- Drug binding (equilibrium)
- Protein stability (ΔΔG)
- Evolutionary distance

But NOT:
- Folding rates (kinetics)
- Aggregation rates

## Clinical Disclaimer

These analyses are for **research purposes only** and should not be used for clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical decisions.

## References

1. WHO TB Mutation Catalogue (2021, 2023)
2. EASL HCV Treatment Guidelines
3. WHO Artemisinin Resistance Markers
4. CLSI/EUCAST Antimicrobial Breakpoints
5. OncoKB Precision Oncology Knowledge Base
6. Stanford HIVdb Drug Resistance Database

---

*Last updated: December 28, 2024*
*Multi-disease expansion: 11 disease domains, 100+ drug targets*
