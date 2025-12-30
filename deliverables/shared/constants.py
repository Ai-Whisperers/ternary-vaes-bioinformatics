# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Shared constants for bioinformatics calculations.

Contains amino acid properties, codon tables, and other reference data.
"""

from __future__ import annotations

# Standard amino acid single-letter codes
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Standard genetic code
CODON_TABLE = {
    "TTT": "F", "TTC": "F",
    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I",
    "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y",
    "TAA": "*", "TAG": "*", "TGA": "*",
    "CAT": "H", "CAC": "H",
    "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D",
    "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C",
    "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# Amino acid charges at pH 7.4
CHARGES = {
    "A": 0, "C": 0, "D": -1, "E": -1, "F": 0,
    "G": 0, "H": 0.5, "I": 0, "K": 1, "L": 0,
    "M": 0, "N": 0, "P": 0, "Q": 0, "R": 1,
    "S": 0, "T": 0, "V": 0, "W": 0, "Y": 0,
}

# Amino acid molecular volumes (Angstrom^3)
VOLUMES = {
    "A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9,
    "G": 60.1, "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7,
    "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4,
    "S": 89.0, "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6,
}

# Amino acid molecular weights (Daltons)
MOLECULAR_WEIGHTS = {
    "A": 89.1, "C": 121.2, "D": 133.1, "E": 147.1, "F": 165.2,
    "G": 75.1, "H": 155.2, "I": 131.2, "K": 146.2, "L": 131.2,
    "M": 149.2, "N": 132.1, "P": 115.1, "Q": 146.2, "R": 174.2,
    "S": 105.1, "T": 119.1, "V": 117.1, "W": 204.2, "Y": 181.2,
}

# Amino acid flexibility (B-factor proxy)
FLEXIBILITY = {
    "A": 0.36, "C": 0.35, "D": 0.51, "E": 0.50, "F": 0.31,
    "G": 0.54, "H": 0.32, "I": 0.46, "K": 0.47, "L": 0.40,
    "M": 0.30, "N": 0.46, "P": 0.51, "Q": 0.49, "R": 0.53,
    "S": 0.51, "T": 0.44, "V": 0.39, "W": 0.31, "Y": 0.42,
}

# Secondary structure propensities (Chou-Fasman)
HELIX_PROPENSITY = {
    "A": 1.42, "C": 0.70, "D": 1.01, "E": 1.51, "F": 1.13,
    "G": 0.57, "H": 1.00, "I": 1.08, "K": 1.16, "L": 1.21,
    "M": 1.45, "N": 0.67, "P": 0.57, "Q": 1.11, "R": 0.98,
    "S": 0.77, "T": 0.83, "V": 1.06, "W": 1.08, "Y": 0.69,
}

SHEET_PROPENSITY = {
    "A": 0.83, "C": 1.19, "D": 0.54, "E": 0.37, "F": 1.38,
    "G": 0.75, "H": 0.87, "I": 1.60, "K": 0.74, "L": 1.30,
    "M": 1.05, "N": 0.89, "P": 0.55, "Q": 1.10, "R": 0.93,
    "S": 0.75, "T": 1.19, "V": 1.70, "W": 1.37, "Y": 1.47,
}

# WHO Priority Pathogens (for AMP targeting)
WHO_CRITICAL_PATHOGENS = [
    "Acinetobacter baumannii",
    "Pseudomonas aeruginosa",
    "Enterobacteriaceae (CRE)",
]

WHO_HIGH_PATHOGENS = [
    "Enterococcus faecium",
    "Staphylococcus aureus (MRSA)",
    "Helicobacter pylori",
    "Campylobacter species",
    "Salmonella species",
    "Neisseria gonorrhoeae",
]

# Arbovirus taxonomy IDs (NCBI)
ARBOVIRUS_TAXIDS = {
    "DENV-1": 11053,
    "DENV-2": 11060,
    "DENV-3": 11069,
    "DENV-4": 11070,
    "ZIKV": 64320,
    "CHIKV": 37124,
    "MAYV": 59301,
}

# HIV Drug Classes
HIV_DRUG_CLASSES = {
    "NRTI": ["TDF", "TAF", "ABC", "3TC", "FTC", "AZT", "d4T", "ddI"],
    "NNRTI": ["EFV", "NVP", "RPV", "ETR", "DOR"],
    "PI": ["LPV", "DRV", "ATV", "SQV"],
    "INSTI": ["DTG", "RAL", "EVG", "BIC", "CAB"],
}

# WHO Surveillance Drug Resistance Mutations (SDRMs)
WHO_SDRM_NRTI = [
    "M41L", "K65R", "D67N", "D67G", "D67E", "T69ins", "K70R", "K70E",
    "L74V", "L74I", "Y115F", "Q151M", "M184V", "M184I", "L210W",
    "T215Y", "T215F", "K219Q", "K219E",
]

WHO_SDRM_NNRTI = [
    "L100I", "K101E", "K101P", "K103N", "K103S", "V106A", "V106M",
    "Y181C", "Y181I", "Y181V", "Y188L", "Y188C", "Y188H", "G190A",
    "G190S", "G190E", "M230L",
]

WHO_SDRM_INSTI = [
    "T66I", "T66A", "T66K", "E92Q", "G118R", "F121Y", "G140S", "G140A",
    "G140C", "Y143R", "Y143H", "Y143C", "S147G", "Q148H", "Q148K",
    "Q148R", "N155H",
]

# Nucleotide mappings for primers
NUCLEOTIDES = "ACGT"
IUPAC_AMBIGUOUS = {
    "R": "AG", "Y": "CT", "S": "GC", "W": "AT", "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT",
}

# Primer design parameters
PRIMER_DEFAULTS = {
    "min_length": 18,
    "max_length": 25,
    "min_gc": 40,
    "max_gc": 60,
    "min_tm": 55,
    "max_tm": 65,
    "max_homopolymer": 4,
    "max_self_complementarity": 4,
}
