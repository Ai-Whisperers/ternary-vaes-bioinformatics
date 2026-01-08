"""
Download ALS gene sequences from NCBI RefSeq

This script downloads coding sequences for ALS-associated genes
for use in codon bias analysis.
"""

from Bio import Entrez, SeqIO
from pathlib import Path
import time

# IMPORTANT: Set your email for NCBI
Entrez.email = "noreply@example.com"  # Replace with actual email

# Output directory
OUTPUT_DIR = Path('research/cross-disease-validation/data/gene_sequences')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ALS genes to download
ALS_GENES = {
    'TARDBP': 'NM_007375.4',  # TDP-43
    'SOD1': 'NM_000454.5',    # Superoxide dismutase 1
    'FUS': 'NM_004960.4'      # Fused in sarcoma
}

def download_sequence(gene_name, accession):
    """
    Download coding sequence from NCBI RefSeq.

    Args:
        gene_name: Gene symbol (e.g., 'TARDBP')
        accession: RefSeq accession (e.g., 'NM_007375.4')

    Returns:
        Sequence string or None if failed
    """
    print(f"Downloading {gene_name} ({accession})...")

    try:
        # Fetch the sequence
        handle = Entrez.efetch(
            db="nucleotide",
            id=accession,
            rettype="fasta_cds_na",  # Get coding sequence (CDS) nucleotide
            retmode="text"
        )

        # Read the sequence
        record = SeqIO.read(handle, "fasta")
        handle.close()

        sequence = str(record.seq)
        print(f"  Downloaded: {len(sequence)} bp")

        return sequence

    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def main():
    print("="*80)
    print("Downloading ALS Gene Sequences from NCBI RefSeq")
    print("="*80)
    print()

    for gene_name, accession in ALS_GENES.items():
        # Download sequence
        sequence = download_sequence(gene_name, accession)

        if sequence:
            # Save to FASTA file
            output_file = OUTPUT_DIR / f"{gene_name}.fasta"

            with open(output_file, 'w') as f:
                f.write(f">{gene_name}|{accession}\n")
                # Write sequence in 60 bp lines (standard FASTA format)
                for i in range(0, len(sequence), 60):
                    f.write(sequence[i:i+60] + '\n')

            print(f"  Saved to: {output_file}")
        else:
            print(f"  FAILED to download {gene_name}")

        # Be nice to NCBI servers
        time.sleep(0.5)

    print()
    print("="*80)
    print("Download Complete")
    print("="*80)
    print(f"Sequences saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
