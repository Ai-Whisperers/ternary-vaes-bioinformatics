# Data Acquisition Guide: ALS, Parkinson's, Dengue Research

**Doc-Type:** Data Acquisition Reference · Version 1.0 · Created 2026-01-03

---

## Quick Reference: All Data Sources

| Dataset | URL | Access | Size | Priority | Week |
|---------|-----|--------|------|:--------:|:----:|
| GTEx Motor Cortex RNA-seq | https://gtexportal.org/home/datasets | Free (dbGaP) | ~20 GB | **HIGH** | 1 |
| ALS Patient Transcriptomics | GEO GSE124439, GSE67196 | Free | ~10 GB | **HIGH** | 1 |
| PhosphoSitePlus PTMs | https://www.phosphosite.org/ | Free (account) | ~500 MB | **HIGH** | 1 |
| Dengue NS1 Sequences | NCBI Virus | Free | ~1 GB | **HIGH** | 1 |
| IEDB Dengue Epitopes | https://www.iedb.org/ | Free | ~50 MB | MEDIUM | 2 |
| PDB Structures | https://www.rcsb.org/ | Free | ~500 MB | MEDIUM | 3 |
| UniProt Annotations | https://www.uniprot.org/ | Free | ~100 MB | LOW | 1 |
| AlphaFold Structures | https://alphafold.ebi.ac.uk/ | Free | ~1 GB | LOW | 17 |

**Total Storage Needed:** ~35 GB (recommend 50 GB with processing space)

---

## 1. GTEx (Genotype-Tissue Expression Project)

### Overview
- **Purpose:** Tissue-specific gene expression and codon usage analysis
- **Data Type:** RNA-seq from 54 human tissues (17,382 samples)
- **Key Tissues:** Brain - Frontal Cortex (BA9), Brain - Substantia nigra, Brain - Cerebellum

### Access Instructions

**Step 1: Register for dbGaP Access**
```bash
# URL: https://dbgap.ncbi.nlm.nih.gov/aa/wga.cgi?page=login
# Study: phs000424.v8.p2 (GTEx)
# Application: Educational/Research use
# Approval Time: 1-2 weeks
```

**Step 2: Download RNA-seq Data**
```bash
# Option A: Gene-level counts (RECOMMENDED for codon bias)
# Download from GTEx Portal: https://gtexportal.org/home/datasets
# File: GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct.gz
# Size: ~500 MB compressed

# Option B: BAM files (for codon-level analysis)
# Download via SRA Toolkit:
module load sratoolkit
prefetch SRR1234567  # Example accession
sam-dump SRR1234567 > motor_cortex_sample.sam
```

**Step 3: Extract Tissue-Specific Samples**
```python
import pandas as pd

# Load GTEx sample annotations
gtex_annotations = pd.read_csv('GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt', sep='\t')

# Filter for motor cortex (Brain - Frontal Cortex BA9)
motor_cortex_samples = gtex_annotations[gtex_annotations['SMTSD'] == 'Brain - Frontal Cortex (BA9)']

# Filter for substantia nigra (Parkinson's)
substantia_nigra_samples = gtex_annotations[gtex_annotations['SMTSD'] == 'Brain - Substantia nigra']

# Filter for cerebellum (control - less affected in ALS/PD)
cerebellum_samples = gtex_annotations[gtex_annotations['SMTSD'] == 'Brain - Cerebellum']

print(f"Motor cortex: {len(motor_cortex_samples)} samples")
print(f"Substantia nigra: {len(substantia_nigra_samples)} samples")
print(f"Cerebellum: {len(cerebellum_samples)} samples")
```

**Expected Samples:**
- Motor Cortex (BA9): ~200 samples
- Substantia Nigra: ~100 samples
- Cerebellum: ~200 samples

### Processing Pipeline

**Codon Usage Calculation from Gene Counts:**
```python
# scripts/utils/gtex_codon_usage.py

from Bio import SeqIO
from collections import defaultdict
import pandas as pd

def load_gene_counts(gtex_file, tissue_samples):
    """Load GTEx gene counts for specific tissue samples."""
    counts = pd.read_csv(gtex_file, sep='\t', skiprows=2, index_col=0)
    tissue_counts = counts[tissue_samples]
    return tissue_counts

def calculate_codon_usage(gene_id, transcript_fasta, expression_level):
    """
    Calculate codon usage for a gene weighted by expression.

    Args:
        gene_id: Ensembl gene ID (e.g., ENSG00000120948 for TARDBP)
        transcript_fasta: Path to Ensembl transcript sequences
        expression_level: TPM or read count from GTEx

    Returns:
        dict: {codon: weighted_count}
    """
    codon_counts = defaultdict(float)

    # Load transcript sequences for gene
    for record in SeqIO.parse(transcript_fasta, 'fasta'):
        if gene_id in record.id:
            seq = str(record.seq)

            # Extract codons
            for i in range(0, len(seq) - 2, 3):
                codon = seq[i:i+3]
                if len(codon) == 3:  # Valid codon
                    codon_counts[codon] += expression_level

    return dict(codon_counts)

# Example usage
tardbp_counts = calculate_codon_usage(
    gene_id='ENSG00000120948',  # TDP-43 gene
    transcript_fasta='Homo_sapiens.GRCh38.cdna.all.fa',
    expression_level=motor_cortex_tpm['TARDBP']
)
```

### Data Files to Download

1. **Gene-level expression matrix:**
   - File: `GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct.gz`
   - Size: ~500 MB
   - Format: GCT (gene × sample matrix)

2. **Sample annotations:**
   - File: `GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt`
   - Size: ~5 MB
   - Contains: Tissue type, demographics, RIN scores

3. **Transcript sequences (for codon extraction):**
   - Source: Ensembl (https://www.ensembl.org/)
   - File: `Homo_sapiens.GRCh38.cdna.all.fa.gz`
   - Size: ~50 MB
   - Download: `wget ftp://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz`

---

## 2. GEO (Gene Expression Omnibus) - ALS Patient Data

### Key ALS Studies

| GEO ID | Title | Samples | Tissue | Year |
|--------|-------|---------|--------|------|
| **GSE124439** | ALS spinal cord RNA-seq | 148 (ALS) + 34 (control) | Spinal cord | 2019 |
| **GSE67196** | ALS motor cortex RNA-seq | 37 (ALS) + 10 (control) | Motor cortex | 2015 |
| **GSE153960** | iPSC motor neurons (C9ORF72) | 24 samples | iPSC-derived | 2020 |
| **GSE137810** | TDP-43 aggregation model | 18 samples | Cell line | 2019 |

### Download Instructions

**Using GEOquery (R):**
```r
# Install if needed
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("GEOquery")

library(GEOquery)

# Download GSE124439 (ALS spinal cord)
gse <- getGEO("GSE124439", GSEMatrix = TRUE)
als_data <- exprs(gse[[1]])  # Expression matrix

# Save for Python processing
write.csv(als_data, "als_spinal_cord_expression.csv")
```

**Using SRA Toolkit (command line):**
```bash
# Install SRA Toolkit
# Ubuntu/Debian:
sudo apt-get install sra-toolkit

# macOS:
brew install sra-tools

# Download GSE124439 metadata
esearch -db gds -query "GSE124439" | efetch -format docsum > GSE124439_metadata.xml

# Get SRA run IDs
esearch -db sra -query "GSE124439" | efetch -format runinfo > GSE124439_runinfo.csv

# Download first sample (example)
prefetch SRR8571930
fastq-dump --split-files SRR8571930
```

### Processing Pipeline

**Extract ALS-specific gene expression:**
```python
# scripts/utils/geo_als_processor.py

import pandas as pd
import numpy as np

def load_geo_expression(geo_file):
    """Load GEO expression matrix."""
    expr = pd.read_csv(geo_file, index_col=0)
    return expr

def filter_als_genes(expr_matrix):
    """Filter for ALS-associated genes."""
    als_genes = {
        'TARDBP': 'ENSG00000120948',  # TDP-43
        'SOD1': 'ENSG00000142168',     # Superoxide dismutase
        'FUS': 'ENSG00000089280',      # Fused in sarcoma
        'C9ORF72': 'ENSG00000147894',  # C9orf72
        'OPTN': 'ENSG00000123240',     # Optineurin
        'VCP': 'ENSG00000165280',      # Valosin-containing protein
        'SQSTM1': 'ENSG00000161011',   # p62/sequestosome-1
        'UBQLN2': 'ENSG00000111275',   # Ubiquilin-2
    }

    # Filter expression matrix
    als_expr = expr_matrix.loc[list(als_genes.values())]
    als_expr.index = [k for k, v in als_genes.items() if v in expr_matrix.index]

    return als_expr

def compare_als_vs_control(expr_matrix, sample_metadata):
    """Compare ALS patients vs healthy controls."""
    als_samples = sample_metadata[sample_metadata['disease'] == 'ALS']['sample_id']
    control_samples = sample_metadata[sample_metadata['disease'] == 'Control']['sample_id']

    als_expr = expr_matrix[als_samples]
    control_expr = expr_matrix[control_samples]

    # Differential expression
    from scipy.stats import mannwhitneyu

    results = []
    for gene in expr_matrix.index:
        stat, pval = mannwhitneyu(als_expr.loc[gene], control_expr.loc[gene])
        fc = als_expr.loc[gene].mean() / control_expr.loc[gene].mean()
        results.append({'gene': gene, 'fold_change': fc, 'p_value': pval})

    return pd.DataFrame(results)
```

---

## 3. PhosphoSitePlus - PTM Database

### Overview
- **Purpose:** Curated PTM sites for TDP-43, SOD1, Alpha-synuclein
- **Data Type:** Phosphorylation, ubiquitination, acetylation sites
- **Coverage:** >330,000 PTM sites across 50,000+ proteins

### Access Instructions

**Step 1: Register (Free)**
- URL: https://www.phosphosite.org/
- Create account (academic email recommended)
- Approval: Instant

**Step 2: Download PTM Datasets**
```bash
# Navigate to: Downloads → Datasets
# Download these files:

# 1. Phosphorylation sites
wget --user=your_email@university.edu --password=your_password \
  https://www.phosphosite.org/downloads/Phosphorylation_site_dataset.gz

# 2. Ubiquitination sites
wget --user=your_email --password=your_password \
  https://www.phosphosite.org/downloads/Ubiquitination_site_dataset.gz

# 3. Acetylation sites (if relevant)
wget --user=your_email --password=your_password \
  https://www.phosphosite.org/downloads/Acetylation_site_dataset.gz
```

### Protein-Specific PTM Extraction

**TDP-43 (Q13148) PTM Sites:**
```python
# scripts/utils/phosphosite_extractor.py

import pandas as pd

def extract_protein_ptms(phosphosite_file, uniprot_id):
    """
    Extract PTM sites for a specific protein.

    Args:
        phosphosite_file: Path to PhosphoSitePlus dataset
        uniprot_id: UniProt accession (e.g., 'Q13148' for TDP-43)

    Returns:
        DataFrame with PTM positions and annotations
    """
    ptms = pd.read_csv(phosphosite_file, sep='\t', skiprows=4)

    # Filter for protein
    protein_ptms = ptms[ptms['ACC_ID'] == uniprot_id]

    # Parse modification position
    protein_ptms['position'] = protein_ptms['MOD_RSD'].str.extract(r'(\d+)').astype(int)
    protein_ptms['residue'] = protein_ptms['MOD_RSD'].str.extract(r'([A-Z])')[0]

    return protein_ptms[['position', 'residue', 'MOD_RSD', 'ORGANISM', 'LT_LIT']]

# Example: TDP-43 phosphorylation sites
tdp43_phospho = extract_protein_ptms('Phosphorylation_site_dataset', 'Q13148')
print(f"TDP-43 has {len(tdp43_phospho)} phosphorylation sites")
print(tdp43_phospho.head())
```

**Expected Output for TDP-43:**
```
   position residue     MOD_RSD  LT_LIT
0       403       S  S403-p      15
1       404       S  S404-p       8
2       409       S  S409-p      25
3       410       S  S410-p      32
4       379       S  S379-p       5
...
```

### Creating Disease-Specific PTM Databases

**ALS PTM Database Template:**
```python
# src/research/bioinformatics/codon_encoder_research/als/data/tdp43_ptm_database.py

TDP43_PTM_DATABASE = {
    # Aggregation-associated phosphorylation sites
    403: {
        'residue': 'S',
        'type': 'phosphorylation',
        'pathology': 'aggregation',
        'frequency_in_inclusions': 0.85,
        'citations': 15,
        'kinase': 'Casein kinase 1',
        'notes': 'C-terminal phosphorylation cluster'
    },
    404: {
        'residue': 'S',
        'type': 'phosphorylation',
        'pathology': 'aggregation',
        'frequency_in_inclusions': 0.82,
        'citations': 8,
        'kinase': 'Casein kinase 1',
    },
    409: {
        'residue': 'S',
        'type': 'phosphorylation',
        'pathology': 'aggregation',
        'frequency_in_inclusions': 0.90,
        'citations': 25,
        'kinase': 'Casein kinase 1 / Casein kinase 2',
        'notes': 'Most abundant in ALS patient samples'
    },
    410: {
        'residue': 'S',
        'type': 'phosphorylation',
        'pathology': 'aggregation',
        'frequency_in_inclusions': 0.88,
        'citations': 32,
        'kinase': 'Casein kinase 1',
        'notes': 'Biomarker for ALS pathology'
    },

    # Nuclear export-related sites
    48: {
        'residue': 'S',
        'type': 'phosphorylation',
        'pathology': 'nuclear_export',
        'citations': 6,
        'kinase': 'Unknown',
        'notes': 'N-terminal, affects nuclear import'
    },

    # Ubiquitination sites
    95: {
        'residue': 'K',
        'type': 'ubiquitination',
        'pathology': 'degradation',
        'citations': 4,
        'notes': 'May promote proteasomal degradation'
    },
    181: {
        'residue': 'K',
        'type': 'ubiquitination',
        'pathology': 'aggregation',
        'citations': 7,
        'notes': 'Found in ALS inclusions, may impair clearance'
    },
}

# Metadata
TDP43_UNIPROT_ID = 'Q13148'
TDP43_GENE_NAME = 'TARDBP'
TDP43_PROTEIN_LENGTH = 414
TDP43_DOMAINS = {
    'NTD': (1, 102),          # N-terminal domain
    'RRM1': (103, 175),       # RNA recognition motif 1
    'RRM2': (191, 259),       # RNA recognition motif 2
    'CTD': (274, 414),        # C-terminal domain (low complexity)
}
```

---

## 4. NCBI Virus - Dengue Sequences

### Overview
- **Purpose:** Dengue virus genome sequences for all 4 serotypes
- **Data Type:** Complete genomes, NS1 gene sequences
- **Geographic Focus:** Paraguay 2011-2024 (Alejandra Rojas focus)

### Access Instructions

**Web Interface:**
```
1. Navigate to: https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/
2. Filter by:
   - Virus: Dengue virus
   - Host: Homo sapiens
   - Country: Paraguay
   - Collection Date: 2011-01-01 to 2024-12-31
   - Nucleotide Completeness: Complete
3. Select Columns: Accession, Length, Serotype, Collection Date, Isolate
4. Download: FASTA (nucleotide) + Metadata (CSV)
```

**Command Line (Entrez Direct):**
```bash
# Install Entrez Direct
sh -c "$(curl -fsSL ftp://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)"

# Search for Dengue genomes from Paraguay
esearch -db nucleotide \
  -query "Dengue virus[Organism] AND Paraguay[Country] AND 2011:2024[PDAT] AND complete genome" \
  | efetch -format fasta > dengue_paraguay_genomes.fasta

# Get metadata
esearch -db nucleotide \
  -query "Dengue virus[Organism] AND Paraguay[Country] AND 2011:2024[PDAT]" \
  | efetch -format docsum > dengue_metadata.xml
```

### Serotype-Specific Download

**Download Each Serotype Separately:**
```python
# scripts/utils/ncbi_dengue_downloader.py

from Bio import Entrez
import pandas as pd

Entrez.email = "your_email@university.edu"

def download_dengue_serotype(serotype, country='Paraguay', start_year=2011, end_year=2024):
    """
    Download Dengue sequences for a specific serotype.

    Args:
        serotype: 1, 2, 3, or 4
        country: Geographic filter
        start_year: Collection start year
        end_year: Collection end year

    Returns:
        List of SeqRecord objects
    """
    # Construct search query
    query = f"Dengue virus {serotype}[Organism] AND {country}[Country] AND {start_year}:{end_year}[PDAT] AND complete genome"

    # Search
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=1000)
    record = Entrez.read(handle)
    handle.close()

    id_list = record["IdList"]
    print(f"Found {len(id_list)} sequences for DENV-{serotype}")

    # Fetch sequences
    if len(id_list) > 0:
        handle = Entrez.efetch(db="nucleotide", id=id_list, rettype="fasta", retmode="text")
        sequences = handle.read()
        handle.close()

        # Save to file
        with open(f"dengue_{serotype}_{country}.fasta", 'w') as f:
            f.write(sequences)

        return id_list
    else:
        return []

# Download all 4 serotypes
for serotype in [1, 2, 3, 4]:
    ids = download_dengue_serotype(serotype)
```

### Extract NS1 Gene Sequences

**NS1 Gene Coordinates (approximate, verify per serotype):**
- DENV-1: ~2500-3500 nt
- DENV-2: ~2500-3500 nt
- DENV-3: ~2500-3500 nt
- DENV-4: ~2500-3500 nt

```python
from Bio import SeqIO

def extract_ns1_gene(genome_fasta, output_fasta):
    """
    Extract NS1 gene from full Dengue genomes.

    NS1 coordinates (CDS):
        Start: ~2475 (after capsid, prM, E)
        End: ~3525 (before NS2A)
        Length: ~1050 nt (350 aa)
    """
    ns1_sequences = []

    for record in SeqIO.parse(genome_fasta, 'fasta'):
        # Extract NS1 region (adjust coordinates if needed)
        ns1_seq = record.seq[2474:3525]  # 0-indexed

        # Create new record
        ns1_record = record[:]
        ns1_record.seq = ns1_seq
        ns1_record.id = record.id + "_NS1"
        ns1_record.description = "NS1 gene"

        ns1_sequences.append(ns1_record)

    # Save NS1 sequences
    SeqIO.write(ns1_sequences, output_fasta, 'fasta')
    print(f"Extracted {len(ns1_sequences)} NS1 sequences")

# Example
extract_ns1_gene('dengue_1_Paraguay.fasta', 'dengue_1_NS1_Paraguay.fasta')
```

### Expected Data Volume

| Serotype | Expected Sequences (Paraguay 2011-2024) | Storage |
|----------|----------------------------------------|---------|
| DENV-1 | ~50-100 | ~500 KB |
| DENV-2 | ~100-200 | ~1 MB |
| DENV-3 | ~20-50 | ~200 KB |
| DENV-4 | ~10-30 | ~100 KB |
| **Total** | ~180-380 | ~2 MB |

---

## 5. IEDB (Immune Epitope Database) - T-cell Epitopes

### Overview
- **Purpose:** Validated T-cell epitopes for Dengue, ALS, Parkinson's
- **Data Type:** Epitope sequences, HLA restrictions, assay results
- **Coverage:** >1 million epitopes

### Access Instructions

**Web Interface:**
```
1. URL: https://www.iedb.org/
2. Advanced Search → T Cell Assays
3. Filters:
   - Organism: Dengue virus (or Homo sapiens for self-antigens)
   - MHC Restriction: HLA-DR (Class II)
   - Assay: Positive only
4. Download: CSV format
```

**API Access (Programmatic):**
```python
# scripts/utils/iedb_downloader.py

import requests
import pandas as pd

def download_iedb_epitopes(organism, mhc_class='II', allele_filter=None):
    """
    Download epitopes from IEDB API.

    Args:
        organism: 'Dengue virus', 'Homo sapiens', etc.
        mhc_class: 'I' or 'II'
        allele_filter: List of HLA alleles (e.g., ['DRB1*01:01'])

    Returns:
        DataFrame with epitope data
    """
    # IEDB API endpoint
    base_url = "https://www.iedb.org/downloader.php"

    params = {
        'file_type': 'csv',
        'receptor_type': 'tcr',
        'organism': organism,
        'mhc_class': mhc_class,
    }

    # Download
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        # Parse CSV
        from io import StringIO
        epitopes = pd.read_csv(StringIO(response.text))

        # Filter by allele if specified
        if allele_filter:
            epitopes = epitopes[epitopes['Allele Name'].isin(allele_filter)]

        return epitopes
    else:
        print(f"Error downloading: {response.status_code}")
        return None

# Example: Dengue epitopes restricted by HLA-DRB1*01:01
dengue_epitopes = download_iedb_epitopes(
    organism='Dengue virus',
    mhc_class='II',
    allele_filter=['HLA-DRB1*01:01', 'HLA-DRB1*04:01']
)

print(f"Downloaded {len(dengue_epitopes)} Dengue epitopes")
```

### Dengue-Specific Epitope Extraction

**NS1 T-cell Epitopes:**
```python
def extract_ns1_epitopes(iedb_dataframe):
    """Extract NS1-specific epitopes from IEDB data."""
    ns1_epitopes = iedb_dataframe[
        iedb_dataframe['Antigen Name'].str.contains('NS1', case=False, na=False)
    ]

    # Group by peptide sequence
    unique_epitopes = ns1_epitopes.groupby('Description').agg({
        'Allele Name': lambda x: list(set(x)),
        'Assay Result': 'mean',
        'PubMed ID': lambda x: list(set(x))
    }).reset_index()

    unique_epitopes.rename(columns={'Description': 'epitope_sequence'}, inplace=True)

    return unique_epitopes

ns1_tcell = extract_ns1_epitopes(dengue_epitopes)
print(f"Found {len(ns1_tcell)} unique NS1 epitopes")
```

### Self-Antigen Epitopes (ALS, Parkinson's)

**TDP-43 and Alpha-Synuclein Epitopes:**
```python
# For neuroinflammation research
# NOTE: These are predicted, not experimentally validated

# Search for published autoimmune epitopes
self_epitopes = download_iedb_epitopes(
    organism='Homo sapiens',
    mhc_class='II'
)

# Filter for neurological proteins
neuro_proteins = ['TDP-43', 'TARDBP', 'alpha-synuclein', 'SNCA', 'SOD1']
neuro_epitopes = self_epitopes[
    self_epitopes['Antigen Name'].str.contains('|'.join(neuro_proteins), case=False, na=False)
]

print(f"Found {len(neuro_epitopes)} neuro-related self-epitopes")
# NOTE: This will likely be sparse; most will need prediction
```

---

## 6. PDB (Protein Data Bank) - 3D Structures

### Key Structures for Validation

| Protein | PDB ID | Description | Resolution | Use Case |
|---------|--------|-------------|------------|----------|
| TDP-43 RRM1 | 4BS2 | RNA recognition motif | 1.90 Å | ALS structural validation |
| SOD1 | 1SPD | WT homodimer | 2.00 Å | ALS aggregation reference |
| Alpha-synuclein fibril | 6A6B | Lewy body structure | 3.4 Å cryo-EM | Parkinson's aggregation |
| NS1 hexamer | 4O6B | Dengue NS1 | 2.40 Å | Dengue antibody binding |
| Dengue E protein | 1OKE | Envelope protein | 2.50 Å | Dengue ADE modeling |

### Download Instructions

**Using PDB API:**
```python
# scripts/utils/pdb_downloader.py

import requests
from Bio.PDB import PDBParser, PDBIO

def download_pdb_structure(pdb_id, output_dir='./structures'):
    """
    Download PDB structure file.

    Args:
        pdb_id: 4-character PDB ID (e.g., '4BS2')
        output_dir: Directory to save files

    Returns:
        Path to downloaded file
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)

    if response.status_code == 200:
        output_path = f"{output_dir}/{pdb_id}.pdb"
        with open(output_path, 'w') as f:
            f.write(response.text)
        print(f"Downloaded {pdb_id} to {output_path}")
        return output_path
    else:
        print(f"Error downloading {pdb_id}: {response.status_code}")
        return None

# Download key structures
key_structures = ['4BS2', '1SPD', '6A6B', '4O6B', '1OKE']
for pdb_id in key_structures:
    download_pdb_structure(pdb_id)
```

**Batch Download:**
```bash
# Command-line batch download
mkdir -p structures
cd structures

# TDP-43
wget https://files.rcsb.org/download/4BS2.pdb

# SOD1
wget https://files.rcsb.org/download/1SPD.pdb

# Alpha-synuclein fibril
wget https://files.rcsb.org/download/6A6B.pdb

# Dengue NS1
wget https://files.rcsb.org/download/4O6B.pdb

# Dengue E protein
wget https://files.rcsb.org/download/1OKE.pdb
```

---

## 7. AlphaFold Predicted Structures

### AlphaFold Database

**Access:**
- URL: https://alphafold.ebi.ac.uk/
- Coverage: 200+ million structures

**Key Proteins:**
```bash
# Download AlphaFold predictions
wget https://alphafold.ebi.ac.uk/files/AF-Q13148-F1-model_v4.pdb  # TDP-43
wget https://alphafold.ebi.ac.uk/files/AF-P00441-F1-model_v4.pdb  # SOD1
wget https://alphafold.ebi.ac.uk/files/AF-P37840-F1-model_v4.pdb  # Alpha-synuclein
```

### AlphaFold3 (For PTM Predictions)

**Access via AlphaFold Server:**
- URL: https://alphafoldserver.com/
- Requires Google account
- Free tier: 10 predictions/day

**For PTM modeling, use JSON input:**
```json
{
  "name": "TDP-43_S409_phosphorylated",
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "MSEYIRVTED...409 residues...",
        "modifications": [
          {
            "ptmType": "phosphorylation",
            "ptmPosition": 409
          }
        ]
      }
    }
  ]
}
```

---

## 8. Data Processing Checklist

### Week 1 Deliverables

- [ ] **GTEx**: Download gene counts + sample annotations (~500 MB)
- [ ] **GTEx**: Extract motor cortex, substantia nigra, cerebellum samples
- [ ] **Ensembl**: Download human transcript sequences (~50 MB)
- [ ] **GEO**: Download GSE124439 (ALS spinal cord, ~10 GB)
- [ ] **PhosphoSitePlus**: Register account, download phosphorylation dataset
- [ ] **PhosphoSitePlus**: Extract TDP-43, SOD1, alpha-synuclein PTMs
- [ ] **NCBI Virus**: Download Dengue genomes for all 4 serotypes (Paraguay)
- [ ] **NCBI Virus**: Extract NS1 gene sequences
- [ ] **IEDB**: Download Dengue T-cell epitopes (HLA-DR restricted)
- [ ] **UniProt**: Download full sequences for TDP-43, SOD1, SNCA, NS1
- [ ] **PDB**: Download key structures (4BS2, 1SPD, 6A6B, 4O6B)

### Data Organization

```
research/cross-disease-validation/data/
├── gtex/
│   ├── GTEx_gene_reads.gct.gz
│   ├── GTEx_sample_annotations.txt
│   ├── motor_cortex_samples.csv
│   ├── substantia_nigra_samples.csv
│   └── cerebellum_samples.csv
├── geo/
│   ├── GSE124439_als_spinal_cord.csv
│   ├── GSE67196_als_motor_cortex.csv
│   └── metadata/
├── phosphosite/
│   ├── Phosphorylation_site_dataset
│   ├── Ubiquitination_site_dataset
│   ├── tdp43_ptms.csv
│   ├── sod1_ptms.csv
│   └── alpha_synuclein_ptms.csv
├── dengue/
│   ├── serotypes/
│   │   ├── dengue_1_Paraguay.fasta
│   │   ├── dengue_2_Paraguay.fasta
│   │   ├── dengue_3_Paraguay.fasta
│   │   └── dengue_4_Paraguay.fasta
│   ├── ns1/
│   │   ├── dengue_1_NS1.fasta
│   │   └── ... (NS1 for all serotypes)
│   └── metadata/
│       └── dengue_paraguay_metadata.csv
├── iedb/
│   ├── dengue_tcell_epitopes.csv
│   └── self_antigen_epitopes.csv
├── pdb/
│   ├── 4BS2.pdb  # TDP-43
│   ├── 1SPD.pdb  # SOD1
│   ├── 6A6B.pdb  # Alpha-synuclein
│   ├── 4O6B.pdb  # NS1
│   └── 1OKE.pdb  # Dengue E
└── alphafold/
    ├── AF-Q13148-F1.pdb  # TDP-43 prediction
    ├── AF-P00441-F1.pdb  # SOD1 prediction
    └── AF-P37840-F1.pdb  # Alpha-synuclein prediction
```

---

## 9. Troubleshooting

### Common Issues

**Issue: dbGaP access denied**
- Solution: Ensure institutional affiliation is listed, reapply with PI approval

**Issue: SRA download too slow**
- Solution: Use Aspera Connect for faster downloads:
  ```bash
  ascp -i ~/.aspera/connect/etc/asperaweb_id_dsa.openssh -k1 -Tr -l100m \
    anonftp@ftp.ncbi.nlm.nih.gov:/sra/sra-instant/reads/ByRun/sra/SRR/SRR857/SRR8571930 .
  ```

**Issue: IEDB API returns empty results**
- Solution: Use web interface download, API may have stricter filters

**Issue: AlphaFold Server queue too long**
- Solution: Submit jobs overnight, or use local AlphaFold installation

---

## 10. Data Citation Requirements

When using these datasets, cite:

**GTEx:**
> GTEx Consortium. "The GTEx Consortium atlas of genetic regulatory effects across human tissues." Science 369.6509 (2020): 1318-1330.

**GEO:**
> Barrett, Tanya, et al. "NCBI GEO: archive for functional genomics data sets—update." Nucleic acids research 41.D1 (2012): D991-D995.

**PhosphoSitePlus:**
> Hornbeck, Peter V., et al. "PhosphoSitePlus, 2014: mutations, PTMs and recalibrations." Nucleic acids research 43.D1 (2015): D512-D520.

**IEDB:**
> Vita, Randi, et al. "The immune epitope database (IEDB): 2018 update." Nucleic acids research 47.D1 (2019): D339-D343.

**PDB:**
> Berman, Helen M., et al. "The protein data bank." Nucleic acids research 28.1 (2000): 235-242.

**AlphaFold:**
> Jumper, John, et al. "Highly accurate protein structure prediction with AlphaFold." Nature 596.7873 (2021): 583-589.

---

**Next Steps:**
1. Begin Week 1 data acquisition following checklist above
2. Set up data/ directory structure
3. Run initial processing scripts to validate data quality
4. Proceed to Phase 1 analysis (codon bias)

**Questions or Issues:** Contact AI Whisperers research team
