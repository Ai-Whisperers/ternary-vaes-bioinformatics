# Different Patterns of Codon Usage and Amino Acid Composition Across Primate Lentiviruses

**ID:** MOL-007
**Year:** 2023
**Journal:** Viruses
**DOI:** [10.3390/v15071580](https://www.mdpi.com/1999-4915/15/7/1580)
**PubMed:** [37515266](https://pubmed.ncbi.nlm.nih.gov/37515266/)

---

## Abstract

This comprehensive study analyzes codon usage patterns and amino acid composition across primate lentiviruses, revealing a characteristic A-rich genome bias in HIV-1 and related viruses. Using principal component analysis on sequences from SIV, HIV-2, and HIV-1 groups, the research demonstrates evolutionary trends toward increased A/T-ending codons and cell-type specific codon usage correlations with host monocytes, providing insights into viral adaptation and host interactions.

---

## Key Concepts

- **A-Rich Genome**: HIV-1 genomes contain 31.7-38.2% adenine, with cytosine extremely low (13.9-21.2%)
- **Codon Bias**: Striking difference between synonymous codon frequency in lentiviruses vs hosts
- **Selection Pressure**: Primate lentiviruses more affected by selection than mutation
- **Cell-Type Adaptation**: Codon usage correlates with monocyte gene expression

---

## Nucleotide Composition Analysis

### A-Rich Genome Characteristics
| Nucleotide | Lentivirus Frequency | Host Frequency |
|:-----------|:---------------------|:---------------|
| Adenine | 31.7-38.2% (up to 40%) | ~25% |
| Cytosine | 13.9-21.2% | ~25% |
| Thymine | Variable | ~25% |
| Guanine | Variable | ~25% |

### Evolutionary Trends
| Lentivirus | A/T in 3rd Position | Trend |
|:-----------|:--------------------|:------|
| SIV (Old World) | Lower | Ancestral |
| HIV-2 | Intermediate | Transitional |
| HIV-1 groups | **Highest** | Most derived |

---

## Principal Component Analysis Findings

### Codon Position Effects
| Position | HIV-1 Enrichment | Function |
|:---------|:-----------------|:---------|
| 1st position | A and G | Nonsynonymous |
| 2nd position | A and G | Nonsynonymous |
| **3rd position** | **A/T bias** | Synonymous |

### Gene-Specific Patterns
| Gene | A/T Bias Level | Significance |
|:-----|:---------------|:-------------|
| gag | **Strong** | Structural constraints |
| pol | **Strong** | Enzyme function |
| env | **Strong** | Surface protein |
| Regulatory genes | Moderate | Variable |

---

## Host Cell Correlation

### Cell-Type Specific Codon Usage
| Cell Type | Correlation with HIV-1 | Relative Strength |
|:----------|:-----------------------|:------------------|
| **Monocytes** | Significant | **30x > B cells** |
| T lymphocytes | Moderate | 5x < monocytes |
| B lymphocytes | Weak | Baseline |

> "Using the overall pattern of codon usage from genes expressed in human monocytes, B and T lymphocytes, the use of synonyms in all primate Lentiviruses is significantly correlated with monocytes but not with lymphocytes."

### Pandemic Group M Specifics
| Comparison | Correlation Ratio |
|:-----------|:------------------|
| Monocytes vs B lymphocytes | 30:1 |
| Monocytes vs T lymphocytes | 5:1 |

---

## Biological Explanations for A-Rich Genomes

### Proposed Mechanisms
| Mechanism | Explanation |
|:----------|:------------|
| RNA secondary structure | Less structure improves translation |
| APOBEC3 evasion | High A content may limit cytosine editing |
| Translational efficiency | A-rich codons may enhance expression |
| Amino acid selection | Preferential use of A-rich encoded residues |

### APOBEC3 Coevolution Hypothesis
> "It is conceivable that primate lentiviruses may have learned to coexist with these cellular enzymes (APOBEC3 family) through a different strategy, namely by evolving a genome with high A content."

---

## Selection vs Mutation Analysis

### ENC-GC3s Analysis
| Finding | Interpretation |
|:--------|:--------------|
| ENC-GC3s correlation | Weak |
| Neutrality index | Low |
| Dominant force | **Selection pressure** |

> "The ENC-GC3s plot and neutral evolution analysis showed that all primate lentiviruses were more affected by selection pressure than by mutation caused by the GC composition of the gene."

---

## Implications for Viral Evolution

### Evolutionary Trajectory
```
SIV (ancestral codon usage) →
HIV-2 (intermediate A/T bias) →
HIV-1 groups (maximal A/T in 3rd position) →
Pandemic group M (strongest monocyte correlation)
```

### Adaptive Significance
| Adaptation | Consequence |
|:-----------|:------------|
| Monocyte optimization | Efficient replication in key reservoir |
| Reduced RNA structure | Enhanced genome packaging |
| APOBEC3 resistance | Immune evasion |
| Codon optimization | Host adaptation |

---

## Related 2024 Research

### Schlafen14 Study (2024)
| Finding | Implication |
|:--------|:------------|
| Schlafen14 impairs HIV-1 | Codon usage-dependent restriction |
| Host restriction factor | New antiviral mechanism |
| Codon-based vulnerability | Potential therapeutic target |

---

## Relevance to Project

This codon usage research directly informs the Ternary VAE project:
- **Codon encoding**: A-rich bias as evolutionary constraint
- **Fitness landscapes**: Selection pressure vs mutational drift
- **Sequence generation**: Biologically realistic codon patterns
- **Host adaptation**: Cell-type specific optimization patterns
- **Evolutionary modeling**: Trajectory from SIV to pandemic HIV-1

---

*Added: 2025-12-24*
