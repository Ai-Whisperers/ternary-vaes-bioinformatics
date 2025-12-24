# Task: Rheumatoid Arthritis & Citrullination Analysis

**Objective**: Validate the "Goldilocks Zone" hypothesis in Rheumatoid Arthritis (RA) using the Codon Encoder.
**Source**: `01_PROJECT_KNOWLEDGE_BASE/03_EXPERIMENTS_AND_LABS/experiments_and_labs/bioinformatics/codon_encoder_research/rheumatoid_arthritis`

## High-Level Goals

- [ ] **Visualize HLA Risk**: Generate risk/safety charts for shared epitope alleles (HLA-DRB1).
- [ ] **Citrullination Shift**: Quantify the geometric shift caused by Arginine -> Citrulline conversion in p-adic space.
- [ ] **AlphaFold Validation**: Correlate geometric shifts with structural instability predictions.

## Detailed Tasks (Implementation)

- [ ] **Fix Syntax Error**: Repair `scripts/19_alphafold_structure_mapping.py` (See `P1_SECURITY_FIXES`).
- [ ] **Run Pipeline**: Execute `visualizations/generate_all.py` to produce the full chart suite.
- [ ] **Optimize Codons**: Run `scripts/04_codon_optimizer.py` to identify "safe" synonymous mutations for therapeutic peptides.
- [ ] **Data Loader**: Refactor `visualizations/utils/data_loader.py` to be robust against missing JSON files.

## Deliverables

- [ ] HLA Risk Heatmaps (Pitch Deck Asset).
- [ ] Structural deviation report (RMSD vs Hyperbolic Distance).
