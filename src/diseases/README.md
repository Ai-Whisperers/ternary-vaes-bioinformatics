# Diseases Module

Disease-specific analysis using the p-adic hyperbolic framework.

## Purpose

This module provides specialized analyzers for understanding diseases through the geometric lens of the Ternary VAE framework, connecting molecular-level codon/protein changes to disease phenotypes.

## Available Disease Analyzers

### Repeat Expansion Diseases

Analyze trinucleotide repeat expansion diseases (Huntington's, Spinocerebellar Ataxias, etc.):

```python
from src.diseases import RepeatExpansionAnalyzer, TrinucleotideRepeat

analyzer = RepeatExpansionAnalyzer()

# Analyze Huntington's disease risk by CAG repeat count
result = analyzer.analyze_repeat_padic_distance(
    disease="huntington",
    repeat_count=42
)

# Get threshold information
info = analyzer.get_disease_info("huntington")
print(info.normal_range)      # (6, 35)
print(info.pathogenic_range)  # (40, None)
```

### Long COVID Analysis

Analyze SARS-CoV-2 spike protein variants and Long COVID:

```python
from src.diseases import LongCOVIDAnalyzer, SpikeVariantComparator

# Analyze spike protein mutations
analyzer = LongCOVIDAnalyzer()
risk = analyzer.analyze_spike_mutations(variant="omicron_ba5")

# Compare variants in hyperbolic space
comparator = SpikeVariantComparator()
distance = comparator.compare("delta", "omicron")
```

### Multiple Sclerosis

Analyze molecular mimicry and demyelination in MS:

```python
from src.diseases import (
    MultipleSclerosisAnalyzer,
    MolecularMimicryDetector,
    HLABindingPredictor,
)

# Detect molecular mimicry between pathogens and myelin
detector = MolecularMimicryDetector()
mimics = detector.find_mimics(
    pathogen_sequence="VVYWMV...",
    myelin_protein="MBP"
)

# Analyze HLA-disease associations
predictor = HLABindingPredictor()
binding = predictor.predict(peptide="VVYWMV", hla="DRB1*15:01")

# Full MS risk analysis
analyzer = MultipleSclerosisAnalyzer()
profile = analyzer.analyze_risk(
    hla_alleles=["DRB1*15:01", "A*02:01"],
    vitamin_d_status="deficient"
)
```

### Rheumatoid Arthritis

Analyze citrullination and the Goldilocks Zone in RA:

```python
from src.diseases import (
    RheumatoidArthritisAnalyzer,
    CitrullinationPredictor,
    GoldilocksZoneDetector,
    PAdicCitrullinationShift,
)

# Predict citrullination sites
predictor = CitrullinationPredictor()
sites = predictor.predict_sites(protein_sequence="MVSKGEEDNM...")

# Detect Goldilocks Zone for autoimmunity
detector = GoldilocksZoneDetector()
zones = detector.find_zones(
    native_affinity=0.3,
    citrullinated_affinity=0.7,
    hla_allele="DRB1*04:01"
)

# Analyze p-adic shift from citrullination
shift = PAdicCitrullinationShift()
distance = shift.compute(
    original_codon="CGG",  # Arginine
    modified=True          # Citrullinated
)
```

## Key Concepts

### Goldilocks Zone

The "Goldilocks Zone" concept in autoimmune disease refers to the observation that some modified peptides (e.g., citrullinated) bind HLA molecules "just right" - strong enough to trigger T cells but weak enough to escape tolerance:

- **Too weak**: No immune response
- **Too strong**: Deleted during thymic selection
- **Just right**: Escapes tolerance, triggers autoimmunity

### P-adic Distance in Disease

P-adic distance captures how mutations propagate through the codon hierarchy:
- **High valuation mutations**: Subtle changes, often tolerated
- **Low valuation mutations**: Major structural disruptions

### Hyperbolic Embedding

Disease variants naturally cluster in hyperbolic space:
- **Central**: Wild-type / normal alleles
- **Peripheral**: Pathogenic variants
- **Distance from center**: Correlates with severity

## Files

| File | Description |
|------|-------------|
| `repeat_expansion.py` | Trinucleotide repeat diseases |
| `long_covid.py` | SARS-CoV-2 and Long COVID analysis |
| `multiple_sclerosis.py` | MS molecular mimicry analysis |
| `rheumatoid_arthritis.py` | RA citrullination analysis |

## Clinical Disclaimer

These analyses are for research purposes only and should not be used for clinical diagnosis. Always consult qualified healthcare professionals for medical decisions.
