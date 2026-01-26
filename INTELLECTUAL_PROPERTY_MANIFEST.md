# Intellectual Property Manifest - Ternary VAE Project

**Doc-Type:** IP Protection · Version 1.0 · Created 2026-01-26 · AI Whisperers

---

## Purpose

This document establishes **cryptographic proof of prior art** for the research findings, algorithms, and implementations in this repository. All timestamps are verifiable through:
1. Git commit history (immutable once pushed)
2. SHA-256 hashes of key files
3. Optional: OpenTimestamps blockchain anchoring

**IMPORTANT:** This manifest should be updated and committed BEFORE making the repository public.

---

## Core Scientific Claims - Priority Established

### Claim 1: P-adic Hyperbolic VAE for Genetic Code

**First Published:** Git commit `41b2a14f` (2026-01-XX)
**Core Innovation:** Variational Autoencoder that learns 3-adic hierarchical structure over ternary operations, embedding 19,683 operations into hyperbolic Poincaré ball where radial position encodes 3-adic valuation.

**Key Files:**
| File | SHA-256 | Lines |
|------|---------|-------|
| `src/models/ternary_vae.py` | *computed at build* | ~1,500 |
| `src/geometry/poincare.py` | *computed at build* | 356 |
| `src/core/padic_math.py` | *computed at build* | 489 |

---

### Claim 2: TrainableCodonEncoder with Hyperbolic Embeddings

**First Published:** Git commit history shows development 2025-12 to 2026-01
**Core Innovation:** Neural encoder mapping 64 codons to 16-dimensional Poincaré ball, achieving LOO Spearman ρ=0.61 on DDG prediction (outperforming sequence-only baselines).

**Architecture (NOVEL):**
```
Input: 12-dim one-hot (4 bases × 3 positions)
→ MLP (12→64→64→16) with LayerNorm, SiLU
→ Exponential map to Poincaré ball
→ Output: 16-dim hyperbolic embeddings
```

**Key Files:**
| File | Purpose |
|------|---------|
| `src/encoders/trainable_codon_encoder.py` | Core implementation |
| `research/codon-encoder/training/train_codon_encoder.py` | Training script |
| `research/codon-encoder/training/results/trained_codon_encoder.pt` | Trained weights |

---

### Claim 3: 13-adic Viral Geometry Discovery (2026-01-26)

**First Published:** Git commit `63b52ebc` (2026-01-26)
**Core Discovery:** DENV-4 viral evolutionary space operates in **13-adic geometry** (R²=0.96), NOT 3-adic. This explains why 3-adic codon embeddings show ρ≈0 correlation with viral conservation.

**Statistical Evidence:**
- 13-adic single prime: R² = 0.9605
- Adelic (2+13): R² = 0.9888
- Dominant weights: 13-adic (0.095), 2-adic (0.077)
- F-statistic: 2568.57, p < 1e-300

**Key Files:**
| File | Purpose |
|------|---------|
| `deliverables/partners/arbovirus_surveillance/research/padic_structure_analysis/` | Analysis scripts |
| `deliverables/partners/arbovirus_surveillance/src/viral_projection_module.py` | Output module |

---

### Claim 4: Dual-Metric Framework (Shannon + Hyperbolic)

**Core Innovation:** Conservation metrics using BOTH Shannon entropy (nucleotide-level) and hyperbolic variance (codon-level) capture ORTHOGONAL information because they operate in different p-adic spaces.

**Application:** Primer design for highly variable pathogens (DENV-4 with 71.7% within-serotype identity).

---

### Claim 5: Contact Prediction from Codon Embeddings

**First Published:** 2026-01-03 (documented in CLAUDE.md)
**Core Discovery:** Pairwise hyperbolic distances between codon embeddings predict residue-residue 3D contacts (AUC-ROC = 0.67 on insulin B-chain).

---

## File Hash Registry

The following script generates SHA-256 hashes for all key files:

```bash
# Run: python scripts/generate_ip_hashes.py
```

### Critical Algorithm Files

```
src/models/ternary_vae.py
src/encoders/trainable_codon_encoder.py
src/encoders/peptide_encoder.py
src/geometry/poincare.py
src/core/padic_math.py
src/losses/peptide_losses.py
```

### Critical Research Files

```
deliverables/partners/arbovirus_surveillance/research/padic_structure_analysis/
deliverables/partners/protein_stability_ddg/src/validated_ddg_predictor.py
research/codon-encoder/training/train_codon_encoder.py
```

### Trained Model Checkpoints

```
checkpoints/v5_12_4/best_Q.pt
checkpoints/homeostatic_rich/best.pt
research/codon-encoder/training/results/trained_codon_encoder.pt
deliverables/partners/antimicrobial_peptides/checkpoints_definitive/best_production.pt
```

---

## Timestamp Verification Methods

### 1. Git Commit History (Primary)

Every commit is cryptographically signed with SHA-1 hash. The commit chain provides immutable timestamps.

```bash
# Verify any claim's timestamp
git log --follow --oneline -- <file_path>
git show <commit_hash>
```

### 2. GitHub Archive (Secondary)

GitHub maintains archives. Push commits frequently to establish public record.

### 3. OpenTimestamps (Recommended for Key Discoveries)

For major discoveries, create OpenTimestamps proof:

```bash
# Install: pip install opentimestamps-client
ots stamp INTELLECTUAL_PROPERTY_MANIFEST.md
# Creates .ots file with Bitcoin blockchain anchor
```

### 4. Archive.org Wayback Machine

Submit repository URL to Wayback Machine after making public.

---

## Citation Requirements

If you use any part of this work, you MUST cite:

```bibtex
@software{ternary_vae_2026,
  author = {AI Whisperers},
  title = {Ternary VAE: P-adic Hyperbolic Embeddings for Bioinformatics},
  year = {2026},
  url = {https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics},
  note = {13-adic viral geometry discovery, TrainableCodonEncoder}
}
```

---

## Legal Notice

This work is protected under the **PolyForm Noncommercial License 1.0.0**.

Key restrictions:
- **Commercial use prohibited** without explicit license
- **Attribution required** for any derivative work
- **Modification allowed** for noncommercial purposes only

The cryptographic timestamps in this document and git history establish **priority of invention** for all claims listed above.

---

## Update Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-26 | 1.0 | Initial manifest with 5 core claims |

---

*This document is part of the official IP protection strategy for the Ternary VAE project.*
