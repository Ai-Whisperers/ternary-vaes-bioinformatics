# Biology Module

Centralized biological constants - the **Single Source of Truth** for all biology-related data.

## Purpose

This module contains ALL biology-related constants used throughout the codebase, eliminating duplicate definitions and ensuring consistency. Import from here instead of defining constants in individual modules.

## Contents

### Genetic Code

```python
from src.biology import GENETIC_CODE, codon_to_amino_acid

# Standard genetic code: codon -> amino acid
amino_acid = GENETIC_CODE["ATG"]  # Returns "M" (Methionine)

# Or use the function
aa = codon_to_amino_acid("ATG")  # Returns "M"
```

### Nucleotide Mappings

```python
from src.biology import BASE_TO_IDX, IDX_TO_BASE

# Convert nucleotides to indices
idx = BASE_TO_IDX["A"]  # Returns 0
base = IDX_TO_BASE[0]   # Returns "A"
```

### Codon Indexing

```python
from src.biology import CODON_TO_INDEX, INDEX_TO_CODON

# 64 codons indexed 0-63
idx = CODON_TO_INDEX["ATG"]  # Returns index for ATG
codon = INDEX_TO_CODON[idx]  # Returns "ATG"

# Convert between triplet indices and codon index
from src.biology import triplet_to_codon_index, codon_index_to_triplet

codon_idx = triplet_to_codon_index(0, 3, 2)  # A=0, T=3, G=2
triplet = codon_index_to_triplet(codon_idx)  # Returns (0, 3, 2)
```

### Amino Acid Properties

```python
from src.biology import AMINO_ACID_PROPERTIES, get_amino_acid_property

# Available properties: hydrophobicity, charge, volume, polarity
hydro = get_amino_acid_property("M", "hydrophobicity")

# Or access the full dictionary
props = AMINO_ACID_PROPERTIES["M"]
```

## Files

| File | Description |
|------|-------------|
| `codons.py` | Genetic code and codon/nucleotide mappings |
| `amino_acids.py` | Amino acid properties and classifications |

## Usage Guidelines

1. **Always import from `src.biology`**, not from individual submodules
2. **Never duplicate** these constants in other modules
3. **Add new constants** to the appropriate file in this module
4. **Update imports** in `__init__.py` when adding new exports
