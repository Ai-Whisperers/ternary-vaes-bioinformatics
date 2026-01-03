# Centralized Datasets Index

**Doc-Type:** Reference Index · Version 1.0 · Updated 2026-01-03 · AI Whisperers

---

## Purpose

This document provides a **single source of truth** for all external datasets, databases, and resources used or planned for the Ternary VAE project. It distinguishes between:

- **VALIDATED** - Datasets we have used and validated
- **PLANNED** - Datasets identified for future validation
- **REFERENCE** - External resources for context/literature

---

## 1. Protein Stability (ΔΔG)

### VALIDATED

| Dataset | URL | Description | Usage |
|---------|-----|-------------|-------|
| **ProThermDB** | https://web.iitm.ac.in/bioinfo2/prothermdb/ | Thermodynamic database, 219 curated mutations | `jose_colbes/` DDG training |

### PLANNED

| Dataset | URL | Description | Priority |
|---------|-----|-------------|----------|
| **DDG EMB Datasets** | https://ddgemb.biocomp.unibo.it/datasets/ | Curated ΔΔG benchmarks (S669, S2648, etc.) | HIGH |
| **ThermoMutDB** | https://biosig.lab.uq.edu.au/thermomutdb/ | Curated mutation effects | MEDIUM |
| **FireProtDB** | https://loschmidt.chemi.muni.cz/fireprotdb/ | Protein stability engineering | LOW |

---

## 2. Contact Prediction & Structure

### PLANNED

| Dataset | URL | Description | Priority |
|---------|-----|-------------|----------|
| **PSICOV Benchmark** | http://bioinfadmin.cs.ucl.ac.uk/downloads/PSICOV/ | 150 proteins with contact maps (.aln, .contacts) | HIGH |
| **PconsC4** | https://github.com/ElofssonLab/PconsC4 | Contact prediction benchmark (CASP/CAMEO domains) | HIGH |
| **AlphaFold Database** | https://alphafold.ebi.ac.uk/download | Complete protein structures | MEDIUM |

---

## 3. Codon/CDS Databases

### PLANNED

| Dataset | URL | Description | Priority |
|---------|-----|-------------|----------|
| **CoDNaS** | http://ufq.unq.edu.ar/codnas/ | Curated CDS-structure pairs (actual codons) | HIGH |
| **UniProt CDS** | https://www.uniprot.org/downloads | Complete proteomes with EMBL/GenBank CDS | MEDIUM |
| **NCBI CDS** | https://www.ncbi.nlm.nih.gov/datasets/ | Coding sequences via datasets API | MEDIUM |

---

## 4. Antimicrobial Peptides (AMP)

### VALIDATED

| Dataset | URL | Description | Usage |
|---------|-----|-------------|-------|
| **APD3** | https://aps.unmc.edu/ | >3,000 validated AMPs | `carlos_brizuela/` training |
| **DRAMP** | http://dramp.cpu-bioinfor.org/ | Comprehensive activity data | `carlos_brizuela/` training |
| **DBAASP** | https://dbaasp.org/ | Structure-activity relationships | Hemolysis data |

### REFERENCE

| Dataset | URL | Description |
|---------|-----|-------------|
| **HemoPI** | https://webs.iiitd.edu.in/raghava/hemopi/ | Hemolytic peptide database |
| **CAMPR3** | http://www.camp.bicnirrh.res.in/ | Collection of AMPs |

---

## 5. HIV/Drug Resistance

### VALIDATED

| Dataset | URL | Description | Usage |
|---------|-----|-------------|-------|
| **Stanford HIVdb** | https://hivdb.stanford.edu/ | Drug resistance scores, mutation comments | `hiv_research_package/` |
| **Los Alamos HIV** | https://www.hiv.lanl.gov/ | Sequences, epitopes, alignments | Vaccine targets |

### VALIDATED (Downloaded)

| Source | Repository/URL | Local Path |
|--------|----------------|------------|
| **HuggingFace** | `damlab/human_hiv_ppi` | `data/external/huggingface/human_hiv_ppi/` |
| **HuggingFace** | `damlab/HIV_V3_coreceptor` | `data/external/huggingface/HIV_V3_coreceptor/` |
| **GitHub** | `lucblassel/HIV-DRM-machine-learning` | Drug resistance ML data |

### PLANNED

| Dataset | URL | Description | Priority |
|---------|-----|-------------|----------|
| **Kaggle HIV-1/2** | https://www.kaggle.com/datasets/protobioengineering/hiv-1-and-hiv-2-rna-sequences | RNA sequences | LOW |
| **Zenodo Tropism** | https://zenodo.org/record/6475667 | CCR5/CXCR4 sequences | MEDIUM |

See [`data/external/HIV_DATASETS_DOWNLOAD_GUIDE.md`](../data/external/HIV_DATASETS_DOWNLOAD_GUIDE.md) for complete HIV dataset guide.

---

## 6. Arbovirus (DENV, ZIKV, CHIKV)

### REFERENCE

| Dataset | URL | Description | Usage |
|---------|-----|-------------|-------|
| **NCBI Virus** | https://www.ncbi.nlm.nih.gov/labs/virus/ | Viral sequences by taxon | `alejandra_rojas/` primers |
| **ViPR** | https://www.viprbrc.org/ | Virus Pathogen Resource | Sequence retrieval |

---

## 7. Protein Language Models

### REFERENCE

| Model | URL | Description |
|-------|-----|-------------|
| **ESM-2** | https://github.com/facebookresearch/esm | Meta AI protein embeddings |
| **ProtTrans** | https://github.com/agemagician/ProtTrans | Protein transformers |

---

## 8. General Bioinformatics

### REFERENCE

| Database | URL | Description |
|----------|-----|-------------|
| **UniProt** | https://www.uniprot.org/ | Protein sequences and annotations |
| **PDB** | https://www.rcsb.org/ | Protein structures |
| **NCBI** | https://www.ncbi.nlm.nih.gov/ | Sequences, literature, taxonomy |
| **Primer3** | https://primer3.org/ | Primer design tool |

---

## Download Locations

All external data should be downloaded to standardized paths:

```
data/
├── external/
│   ├── github/           # Git repositories
│   ├── kaggle/           # Kaggle datasets
│   ├── huggingface/      # HuggingFace datasets
│   ├── zenodo/           # Zenodo archives
│   └── manual/           # Manual downloads
│
research/
└── contact-prediction/
    └── data/
        └── validation/   # PSICOV, PconsC benchmarks
```

---

## Validation Status Legend

| Status | Meaning |
|--------|---------|
| **VALIDATED** | Downloaded, tested, integrated into pipeline |
| **PLANNED** | Identified for future use, not yet downloaded |
| **REFERENCE** | External resource for context, not direct input |

---

## Adding New Datasets

When adding a new dataset:

1. **Categorize** - Determine which section it belongs to
2. **Document** - Add URL, description, and intended usage
3. **Download** - Place in appropriate `data/external/` subdirectory
4. **Validate** - Test loading and format compatibility
5. **Update Status** - Move from PLANNED to VALIDATED

---

## Related Documentation

- [`data/external/HIV_DATASETS_DOWNLOAD_GUIDE.md`](../data/external/HIV_DATASETS_DOWNLOAD_GUIDE.md) - HIV dataset details
- [`deliverables/docs/CURATED_DATABASES.md`](../deliverables/docs/CURATED_DATABASES.md) - Curated training data
- [`deliverables/docs/INDEX.md`](../deliverables/docs/INDEX.md) - Deliverables documentation index

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-03 | 1.0 | Initial centralized index with 8 categories |
