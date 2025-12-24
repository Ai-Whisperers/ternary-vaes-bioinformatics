# Actionable Data Checklist

## 1. Video-Specific Data Targets

Based on Anton Petrov videos search topics:

### Nobel Prize 2025 (Immune System)

- **Source**: Nobel Prize website + laureate papers
- **URL**: https://www.nobelprize.org/prizes/medicine/2025/
- **Data Needed**: Molecular distance thresholds

### Bennu Asteroid

- **Source**: NASA OSIRIS-REx
- **URL**: https://www.nasa.gov/osiris-rex
- **Data Needed**: Amino acid composition ratios

### Fire Amoeba

- **Search**: "extreme temperature amoeba 2025"
- **Data Needed**: Genome sequence, codon usage

### Long COVID Blood Structures

- **Source**: Pretorius et al. publications
- **Search**: "long COVID microclots fibrin"
- **Data Needed**: Protein composition, PTM data

## 2. Download Checklist

### Priority 1 (Immediate)

- [ ] Nobel Prize 2025 Medicine papers
- [ ] Bennu amino acid data
- [ ] Long COVID protein data
- [ ] Stanford HIVDB mutation data

### Priority 2 (Short-term)

- [ ] Thermophile genomes (5+ species)
- [ ] AlphaFold HIV glycan structures
- [ ] GISAID SARS-CoV-2 spike sequences
- [ ] Citrullinome dataset for RA

## 3. API Access Examples

```python
# NCBI E-utilities
def get_ncbi_sequence(accession):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    url = f"{base}efetch.fcgi?db=nucleotide&id={accession}&rettype=fasta"
    return requests.get(url).text
```

## 4. Data Storage Plan

Recommended directory structure:

```
data/
├── external/
│   ├── genomes/
│   │   ├── extremophiles/
│   │   ├── viruses/
│   ├── structures/
│   │   ├── alphafold/
│   ├── ptms/
└── processed/
    ├── codon_tables/
    ├── padic_embeddings/
```
