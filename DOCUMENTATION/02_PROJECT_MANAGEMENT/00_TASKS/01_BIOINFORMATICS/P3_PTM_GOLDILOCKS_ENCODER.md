# Task: PTM-Goldilocks Encoder Implementation

**Objective**: Train a unified PTM-Goldilocks encoder to classify therapeutic potential using V5.11.3 embeddings.
**Source**: `01_PROJECT_KNOWLEDGE_BASE/03_EXPERIMENTS_AND_LABS/bioinformatics/genetic_code/PTM_GOLDILOCKS_ENCODER_ROADMAP.md`

## High-Level Goals

- [ ] **Unified Encoder**: A single model predicting therapeutic potential across HIV, SARS-CoV-2, and RA.
- [ ] **Goldilocks Zone**: Predict 15-30% geometric shift for optimal immunogenicity.

## Detailed Tasks (Implementation)

### 1. Data Construction (Phase 1)

- [ ] **Consolidate Ground Truth**: Create `research/genetic_code/scripts/10_consolidate_ptm_ground_truth.py`.
  - Extract 24 HIV sites (N->Q).
  - Extract 40+ SARS-CoV-2 targets (S->D, T->D, etc.).
  - Extract 20-50 RA sites (R->Q).
- [ ] **Augment Dataset**: Generate ~5000 samples from the V5.11.3 lattice (19,683 points).

### 2. Model Architecture (Phase 2)

- [ ] **Implement Encoder**: Create `PTMGoldilocksEncoder` class in `src/models/ptm_encoder.py`.
  - Input: 20D (12D codon + 8D PTM type).
  - Hidden: 32 -> 32 -> 16 (Poincare).
  - Heads: Cluster (21-class), Goldilocks (3-class), Asymmetry (Binary), Shift (Reg).
- [ ] **Loss Function**: Implement the multi-objective loss (Cluster + Goldilocks + Asymmetry + Shift + Contrastive).

### 3. Training & Validation (Phase 3)

- [ ] **Pre-training Script**: `research/genetic_code/scripts/11_train_ptm_goldilocks_encoder.py` (Augmented data).
- [ ] **Fine-tuning**: Fine-tune on the consolidated ground truth.
- [ ] **Evaluation**: `research/genetic_code/scripts/12_evaluate_ptm_encoder.py`.
  - Target: >85% Goldilocks accuracy.

## Deliverables

- [ ] `ptm_goldilocks_encoder.pt` (Trained Weights).
- [ ] `ptm_training_dataset.json` (The consolidated dataset).
