● Bioinformatics Scripts - Ingestion & Models Summary

  **Last Updated**: 2025-12-26

  Directory Structure

  bioinformatics/
  ├── rheumatoid_arthritis/   # 29 scripts - HLA alleles, citrullination
  ├── neurodegeneration/alzheimers/  # 6 scripts - tau phosphorylation
  ├── hiv/                    # HIV analysis (200K+ sequences)
  │   ├── scripts/            # Analysis scripts including validate_datasets.py
  │   └── results/            # Comprehensive analysis results
  └── sars_cov_2/             # 2 scripts - spike glycan analysis

  Top-Level Analysis Scripts (NEW - 2025-12-26)

  scripts/
  ├── analyze_all_datasets.py   # Comprehensive multi-dataset analysis
  ├── clinical_applications.py  # Clinical decision support generation
  ├── research_discoveries.py   # Research findings pipeline
  ├── train_codon_vae_hiv.py    # HIV-specific codon VAE training
  └── train/train.py            # Main training entry point

  Primary Model: 3-Adic Codon Encoder (V5.11.3)

  Path: ../genetic_code/data/codon_encoder_3adic.pt

  Architecture:
  - Input: 12-dim one-hot codon encoding (3 nucleotides × 4 bases)
  - Output: 16-dim hyperbolic embeddings (Poincaré ball)
  - Network: nn.Sequential(Linear → ReLU → ReLU → Linear) + clustering head

  Loading pattern in hyperbolic_utils.py:
  encoder, mapping, native_hyperbolic = load_codon_encoder(
      device='cpu',
      version='3adic'
  )

  Data Ingestion Patterns

  | Source Type         | Examples                                          |
  |---------------------|---------------------------------------------------|
  | Hardcoded sequences | HLA-DRB1 alleles, HIV epitopes, tau sites         |
  | Downloaded          | Human proteome (UniProt) → results/proteome_wide/ |
  | JSON configs        | AlphaFold3 batch inputs/outputs                   |
  | Model checkpoints   | torch.load() for codon encoder                    |

  AlphaFold3 Integration

  All modules generate AF3 inputs and consume .cif structure predictions:
  - HIV: BG505 gp120 glycan variants
  - SARS-CoV-2: RBD-ACE2 complexes
  - RA: HLA-peptide-TCR complexes

  Key Dependencies

  torch, numpy, scipy, sklearn, pandas, matplotlib, seaborn

  Shared Infrastructure

  rheumatoid_arthritis/scripts/hyperbolic_utils.py - used by all disease modules for:
  - codon_to_onehot(), poincare_distance(), encode_sequence_hyperbolic()
  - 21 clusters (20 amino acids + stop codon)
  - Hyperbolic curvature c=1.0

  Centralized Modules (NEW - 2025-12-26)

  src/biology/                    # Single Source of Truth for biology constants
  ├── amino_acids.py              # AMINO_ACID_PROPERTIES, hydrophobicity, charge
  └── codons.py                   # GENETIC_CODE, CODON_TO_INDEX, conversions

  src/analysis/immunology/        # Shared immunology utilities
  ├── epitope_encoding.py         # encode_amino_acid_sequence()
  ├── genetic_risk.py             # compute_hla_genetic_risk(), HLARiskProfile
  ├── padic_utils.py              # compute_padic_valuation(), compute_goldilocks_score()
  └── types.py                    # EpitopeAnalysisResult, HLAAlleleRisk

  Results Directories (NEW - 2025-12-26)

  results/
  ├── clinical_applications/      # Clinical decision support reports
  │   ├── CLINICAL_REPORT.md      # Human-readable clinical summary
  │   └── clinical_decision_support.json  # Machine-readable for integration
  └── research_discoveries/       # Research findings
      ├── RESEARCH_FINDINGS.md    # Key discoveries summary
      └── research_discoveries_report.json  # Full analysis results