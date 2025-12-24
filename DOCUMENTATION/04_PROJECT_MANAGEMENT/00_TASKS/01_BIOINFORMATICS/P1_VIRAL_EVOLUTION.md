# Task: Viral Evolution & Glycan Shielding

**Objective**: Apply Ternary VAE to predict viral escape mutations and analyze glycan shielding mechanisms.
**Source**: `01_PROJECT_KNOWLEDGE_BASE/03_EXPERIMENTS_AND_LABS/experiments_and_labs/bioinformatics/codon_encoder_research/sars_cov_2`

## High-Level Goals

- [ ] **Glycan "Handshake"**: Prove that glycosylation sites form specific geometric barriers in the embedding space.
- [ ] **EVE Comparison**: Benchmark Ternary VAE fitness predictions against EVE (Evolutionary Variance Explanation).
- [ ] **Sentinel Analysis**: Identify "Sentinel" residues that predict major variants of concern.

## Detailed Tasks (Implementation)

- [ ] **HIV Sentinel**: Run `hiv/glycan_shield/01_glycan_sentinel_analysis.py`.
- [ ] **SARS-CoV-2 Handshake**: Run `sars_cov_2/glycan_shield/02_handshake_interface_analysis.py`.
- [ ] **PyR0 Integration**: Research feasibility of integrating PyR0 (Obermeyer 2022) likelihoods as a calibration target.
- [ ] **AlphaFold Input Gen**: Finalize `02_alphafold3_input_generator.py` for batch structural validation.

## Deliverables

- [ ] Glycan density heatmaps.
- [ ] Escape mutation prediction list for HIV Env.
