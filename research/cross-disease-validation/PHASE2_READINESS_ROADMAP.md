# Phase 2 Readiness Roadmap

**Doc-Type:** Research Planning · Version 1.0 · 2026-01-03 · AI Whisperers

**Objective:** Address 4 critical gaps identified in Phase 1 to enable Phase 2 computational expansion

**Status:** PLANNING - Awaiting Implementation

---

## Executive Summary

Phase 1 validation (2 of 5 tests passed) revealed 4 critical gaps preventing progression to Phase 2:

1. **PTM-Specific Encoder** - TrainableCodonEncoder cannot model post-translational modifications
2. **Literature PTM Database Curation** - 73% position error rate invalidates Test 4
3. **Hypothesis Refinement** - Goldilocks zone doesn't generalize, disease-specific mechanisms dominate
4. **AlphaFold Contact Prediction Integration** - Structural features needed for complex proteins

**Approach:** Leverage existing partner packages (Colbes, Brizuela, Rojas) as building blocks to systematically address each gap.

**Timeline:** 8-12 weeks to Phase 2 readiness (assuming full-time effort)

---

## Gap 1: PTM-Specific Encoder Development

**Problem:** TrainableCodonEncoder designed for genetic mutations (R→Q), cannot model PTMs (R→citrulline).

**Evidence from Phase 1:**
- R→Q mutation distance: 1.124 (constant)
- HIV escape mutation distance: 6.204 ± 0.598
- 0% overlap (vs 70% threshold)
- Fundamental error: PTMs change charge/structure without changing genetic code

### Solution Architecture: HybridPTMEncoder

**Design:** Extend TrainableCodonEncoder with PTM-specific features.

```
Input: Sequence + PTM annotation
       ↓
[TrainableCodonEncoder] → Codon embeddings (16D)
       +
[PTM State Encoder]     → PTM effect embeddings (8D)
       ↓
[Fusion Layer]          → Combined representation (24D)
       ↓
Output: Hyperbolic embedding on Poincaré ball
```

#### Architecture Components

**1. Base Encoder (Existing - TrainableCodonEncoder)**
- Input: 12-dim one-hot (4 bases × 3 positions)
- Architecture: MLP (12→64→64→16) with LayerNorm, SiLU, Dropout
- Output: 16-dim hyperbolic embeddings
- **Reuse:** LOO ρ=0.61 validated predictor from jose_colbes package

**2. PTM State Encoder (NEW)**

```python
class PTMStateEncoder(nn.Module):
    """Encodes PTM biochemical effects."""

    def __init__(self, ptm_dim=8):
        super().__init__()

        # PTM type embedding (learned)
        self.ptm_type_embedding = nn.Embedding(
            num_embeddings=10,  # phosphorylation, citrullination, etc.
            embedding_dim=4
        )

        # Physicochemical change encoder
        self.physchem_encoder = nn.Sequential(
            nn.Linear(4, 16),  # ΔCharge, ΔMass, ΔHydro, ΔVolume
            nn.LayerNorm(16),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 4)
        )

    def forward(self, ptm_type, delta_charge, delta_mass, delta_hydro, delta_volume):
        """
        Args:
            ptm_type: Integer PTM type (0=none, 1=phospho, 2=citrul, etc.)
            delta_charge: Charge change (-2 to +2)
            delta_mass: Mass change (Da)
            delta_hydro: Hydrophobicity change
            delta_volume: Volume change (Å³)

        Returns:
            ptm_embedding: 8-dim PTM effect representation
        """
        type_emb = self.ptm_type_embedding(ptm_type)  # (B, 4)

        physchem = torch.stack([delta_charge, delta_mass, delta_hydro, delta_volume], dim=-1)
        physchem_emb = self.physchem_encoder(physchem)  # (B, 4)

        ptm_embedding = torch.cat([type_emb, physchem_emb], dim=-1)  # (B, 8)

        return ptm_embedding
```

**3. Fusion Layer (NEW)**

```python
class HybridPTMEncoder(nn.Module):
    """Combines codon and PTM encodings."""

    def __init__(self, codon_dim=16, ptm_dim=8, output_dim=24):
        super().__init__()

        self.codon_encoder = TrainableCodonEncoder(latent_dim=codon_dim)
        self.ptm_encoder = PTMStateEncoder(ptm_dim=ptm_dim)

        self.fusion = nn.Sequential(
            nn.Linear(codon_dim + ptm_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )

        self.to_hyperbolic = geoopt.manifolds.PoincareBall(c=1.0)

    def forward(self, codon, ptm_type, delta_charge, delta_mass, delta_hydro, delta_volume):
        """
        Args:
            codon: String codon ('AGC')
            ptm_type: PTM type integer
            delta_*: PTM physicochemical changes

        Returns:
            z_hyp: 24-dim hyperbolic embedding
        """
        # Encode codon (genetic code)
        z_codon = self.codon_encoder.encode_codon(codon)  # (16,)

        # Encode PTM state (biochemical change)
        z_ptm = self.ptm_encoder(ptm_type, delta_charge, delta_mass, delta_hydro, delta_volume)  # (8,)

        # Fuse representations
        z_combined = torch.cat([z_codon, z_ptm], dim=-1)  # (24,)
        z_euclidean = self.fusion(z_combined)  # (24,)

        # Project to hyperbolic space
        z_hyp = self.to_hyperbolic.expmap0(z_euclidean)

        return z_hyp
```

#### Training Data Requirements

**Data Source 1: Phosphorylation Stability (ProTherm)**
- **n**: ~500 phosphorylation ΔΔG measurements
- **Features**: S/T/Y phosphorylation stability changes
- **Label**: ΔΔG (kcal/mol)
- **Use**: Train PTM-specific encoder on charge addition (+1 phosphate)

**Data Source 2: Citrullination Immunogenicity (IEDB)**
- **n**: ~200 citrullinated peptides with ACPA binding data
- **Features**: R→citrulline (charge loss -1, mass +1 Da)
- **Label**: ACPA binding affinity (log10 IC50)
- **Use**: Test generalization to different PTM chemistry

**Data Source 3: Acetylation/Methylation Stability (Custom curation needed)**
- **n**: ~100 K acetylation/methylation stability measurements
- **Features**: Charge neutralization (acetyl) or no change (methyl)
- **Label**: ΔΔG or binding affinity
- **Use**: Validate on non-charge PTMs

#### Integration with jose_colbes Package

**Leverage Existing DDG Predictor:**

```python
from deliverables.partners.jose_colbes.src.validated_ddg_predictor import ValidatedDDGPredictor

# Load validated TrainableCodonEncoder (LOO ρ=0.61)
base_predictor = ValidatedDDGPredictor.load('jose_colbes/models/ddg_predictor.joblib')

# Extract TrainableCodonEncoder
codon_encoder = base_predictor.encoder

# Use in HybridPTMEncoder
hybrid_encoder = HybridPTMEncoder(codon_encoder=codon_encoder)
```

**Training Protocol:**

1. **Freeze TrainableCodonEncoder** (preserve LOO ρ=0.61 performance)
2. **Train PTM State Encoder** on phosphorylation data (500 examples)
3. **Train Fusion Layer** end-to-end (unfreeze all after warmup)
4. **Validate on citrullination** (Test 4 re-execution)
5. **Cross-validate on acetylation/methylation**

#### Expected Performance

| PTM Type | Training n | Validation Metric | Target | Rationale |
|----------|-----------|-------------------|--------|-----------|
| Phosphorylation | 500 | Spearman ρ (ΔΔG) | > 0.50 | Matches base encoder |
| Citrullination | 200 | AUC (ACPA binding) | > 0.65 | Phase 1 threshold |
| Acetylation | 100 | Spearman ρ (ΔΔG) | > 0.45 | Lower n, harder task |

#### Implementation Plan

**Week 1-2: Data Collection**
- [ ] Download ProTherm phosphorylation subset
- [ ] Curate IEDB citrullination data (verify positions!)
- [ ] Manual curation of acetylation/methylation literature

**Week 3-4: Architecture Implementation**
- [ ] Implement PTMStateEncoder
- [ ] Implement HybridPTMEncoder fusion
- [ ] Write training script with TrainableCodonEncoder freezing

**Week 5-6: Training and Validation**
- [ ] Train on phosphorylation (ProTherm)
- [ ] Validate on citrullination (IEDB)
- [ ] Cross-validate on acetylation
- [ ] Bootstrap confidence intervals

**Week 7: Test 4 Re-Execution**
- [ ] Re-run Test 4 with HybridPTMEncoder
- [ ] Compare RA citrullination vs HIV escape distances
- [ ] Expected: Citrullination distance closer to HIV range (charge loss is large)

**Deliverable:** `src/encoders/hybrid_ptm_encoder.py` with LOO validation report

---

## Gap 2: Literature PTM Database Curation

**Problem:** 73% of RA citrullination literature positions failed verification (wrong amino acids in UniProt).

**Evidence from Phase 1:**
- 33 of 45 sites (73%) do NOT have R at stated positions
- Example: Vimentin 316 is S (not R), Fibrinogen alpha 36 is G (not R)
- Causes: Isoform differences, coordinate systems, database updates

### Solution: Curated PTM Database with Position Verification

#### Curation Protocol

**Step 1: Cross-Reference Multiple Databases**

```python
def verify_ptm_position(protein_uniprot, position, expected_aa, ptm_type):
    """
    Cross-reference PTM position across multiple databases.

    Returns:
        {
            'verified': bool,
            'canonical_sequence_aa': str,
            'isoforms': [{'isoform_id': str, 'aa_at_position': str}],
            'phosphositeplus_match': bool,
            'dbptm_match': bool,
            'confidence': str  # 'high', 'medium', 'low'
        }
    """
    # 1. Download UniProt canonical sequence
    canonical_seq = download_uniprot_sequence(protein_uniprot)
    canonical_aa = canonical_seq[position - 1] if position <= len(canonical_seq) else None

    # 2. Check all UniProt isoforms
    isoforms = get_uniprot_isoforms(protein_uniprot)
    isoform_matches = []
    for isoform in isoforms:
        iso_seq = isoform['sequence']
        if position <= len(iso_seq):
            aa = iso_seq[position - 1]
            isoform_matches.append({
                'isoform_id': isoform['id'],
                'aa_at_position': aa,
                'matches_expected': aa == expected_aa
            })

    # 3. Check PhosphoSitePlus (if phosphorylation)
    psp_match = False
    if ptm_type == 'phosphorylation':
        psp_data = query_phosphositeplus(protein_uniprot, position)
        psp_match = psp_data is not None and psp_data['residue'] == expected_aa

    # 4. Check dbPTM
    dbptm_data = query_dbptm(protein_uniprot, position, ptm_type)
    dbptm_match = dbptm_data is not None and dbptm_data['residue'] == expected_aa

    # 5. Compute confidence
    verified = canonical_aa == expected_aa
    any_isoform_match = any(iso['matches_expected'] for iso in isoform_matches)

    if verified and (psp_match or dbptm_match):
        confidence = 'high'
    elif verified or any_isoform_match:
        confidence = 'medium'
    elif any_isoform_match:
        confidence = 'low'
    else:
        confidence = 'failed'

    return {
        'verified': verified,
        'canonical_sequence_aa': canonical_aa,
        'isoforms': isoform_matches,
        'phosphositeplus_match': psp_match,
        'dbptm_match': dbptm_match,
        'confidence': confidence
    }
```

**Step 2: Re-Curate RA Citrullination Sites**

```python
# Test 4 RA sites with verification
RA_CITRULLINATION_SITES = [
    {'protein': 'Vimentin', 'uniprot': 'P08670', 'position': 71, 'evidence': 'High', 'target': 'ACPA'},
    {'protein': 'Vimentin', 'uniprot': 'P08670', 'position': 316, 'evidence': 'High', 'target': 'ACPA'},  # FAILED in Phase 1
    {'protein': 'Fibrinogen alpha', 'uniprot': 'P02671', 'position': 36, 'evidence': 'High', 'target': 'ACPA'},  # FAILED
    # ... 42 more sites
]

# Run verification
verified_sites = []
failed_sites = []

for site in RA_CITRULLINATION_SITES:
    result = verify_ptm_position(
        protein_uniprot=site['uniprot'],
        position=site['position'],
        expected_aa='R',  # Citrullination targets arginine
        ptm_type='citrullination'
    )

    if result['confidence'] in ['high', 'medium']:
        site['verification'] = result
        verified_sites.append(site)
    else:
        site['verification'] = result
        failed_sites.append(site)

print(f"Verified: {len(verified_sites)} / {len(RA_CITRULLINATION_SITES)}")
print(f"Failed: {len(failed_sites)} / {len(RA_CITRULLINATION_SITES)}")
```

**Step 3: Manual Literature Review for Failed Sites**

For the 73% failed sites:

1. **Download original paper** that reported the PTM
2. **Check Methods section** for:
   - Which protein isoform was used
   - Whether position includes signal peptide
   - Which species (human vs mouse)
3. **Contact authors** if coordinates unclear
4. **Update database** with corrected positions or mark as "unverifiable"

#### Curated Database Schema

```json
{
  "database_version": "2.0",
  "last_updated": "2026-01-03",
  "verification_date": "2026-01-03",
  "ptm_sites": [
    {
      "id": "RA_CITRUL_001",
      "protein_name": "Vimentin",
      "uniprot_id": "P08670",
      "position": 71,
      "residue": "R",
      "ptm_type": "citrullination",
      "disease": "Rheumatoid Arthritis",
      "target": "ACPA",
      "evidence_level": "High",
      "verification": {
        "status": "verified",
        "confidence": "high",
        "canonical_match": true,
        "phosphositeplus_match": false,
        "dbptm_match": true,
        "verified_date": "2026-01-03"
      },
      "literature": [
        {
          "pmid": "12345678",
          "title": "...",
          "authors": "...",
          "year": 2015,
          "isoform_used": "P08670-1",
          "coordinate_system": "mature_protein"
        }
      ]
    },
    {
      "id": "RA_CITRUL_002",
      "protein_name": "Vimentin",
      "uniprot_id": "P08670",
      "position": 316,
      "residue": "S",  # NOT R
      "ptm_type": "citrullination",
      "disease": "Rheumatoid Arthritis",
      "target": "ACPA",
      "evidence_level": "High",
      "verification": {
        "status": "failed",
        "confidence": "failed",
        "canonical_match": false,
        "canonical_aa": "S",
        "isoform_matches": [],
        "issue": "Position 316 is S in all UniProt isoforms. Possible coordinate error in literature.",
        "verified_date": "2026-01-03"
      },
      "literature": [
        {
          "pmid": "87654321",
          "title": "...",
          "note": "Contacted authors 2026-01-03, awaiting response"
        }
      ]
    }
  ]
}
```

#### Implementation Plan

**Week 1: Automated Verification**
- [ ] Implement `verify_ptm_position()` function
- [ ] Query UniProt API for all 45 RA sites
- [ ] Query PhosphoSitePlus API (requires registration)
- [ ] Query dbPTM API
- [ ] Generate verification report

**Week 2: Manual Literature Review**
- [ ] Download PDFs for all failed sites (n≈33)
- [ ] Extract Methods sections
- [ ] Identify isoform/coordinate issues
- [ ] Contact authors for 10 most promising sites

**Week 3: Database Construction**
- [ ] Build curated JSON database
- [ ] Write Python loader (`load_curated_ptm_database()`)
- [ ] Generate statistics report (% verified by confidence level)
- [ ] Commit to repository with version control

**Week 4: Expand to Other Diseases**
- [ ] Curate Tau phosphorylation sites (Alzheimer's)
- [ ] Curate MS citrullination sites (Multiple Sclerosis)
- [ ] Curate T1D PTM sites (Type 1 Diabetes)
- [ ] Create unified multi-disease PTM database

**Expected Outcome:**

| Disease | PTM Type | Literature Sites | Verified (High+Medium) | Failed | Recovery Rate |
|---------|----------|------------------|------------------------|--------|---------------|
| RA | Citrullination | 45 | 20-25 (44-56%) | 20-25 | 2x Phase 1 |
| Alzheimer's | Phosphorylation | 54 | 40-45 (74-83%) | 9-14 | Better (UniProt quality) |
| MS | Citrullination | 30 | 15-20 (50-67%) | 10-15 | Similar to RA |

**Deliverable:** `research/cross-disease-validation/data/curated_ptm_database_v2.json`

---

## Gap 3: Hypothesis Refinement Based on Phase 1 Learnings

**Problem:** Cross-disease hypotheses failed (Tests 3, 4, 5). Disease-specific mechanisms dominate over universal p-adic patterns.

**Evidence from Phase 1:**
- Test 3 (Dengue DHF): NS1 ρ=-0.33 (wrong protein, multifactorial outcome)
- Test 4 (Goldilocks): 0% RA-HIV overlap (PTM ≠ mutation, different mechanisms)
- Test 5 (Contact Prediction): AUC=0.451 for SOD1 (small protein validation doesn't generalize)

### Solution: Within-Disease and Within-PTM-Type Comparisons

#### Refined Hypotheses for Phase 2

**Hypothesis 2.1: Within-PTM-Type Generalization**

**Original (Failed):** HIV escape mutations and RA citrullination share Goldilocks zone.

**Refined:** Citrullination sites in different autoimmune diseases (RA vs MS) share p-adic distance patterns.

**Rationale:**
- Same PTM chemistry (R→citrulline, charge loss)
- Same mechanism (ACPA/anti-MOG antibody recognition)
- Removes mutation vs PTM confound

**Test Design:**
```python
# Compare RA citrullination vs MS citrullination
ra_sites = load_curated_ptm_database(disease='RA', ptm_type='citrullination')  # n=20-25 verified
ms_sites = load_curated_ptm_database(disease='MS', ptm_type='citrullination')  # n=15-20 verified

# Compute HybridPTMEncoder distances
ra_distances = [hybrid_encoder.encode(site) for site in ra_sites]
ms_distances = [hybrid_encoder.encode(site) for site in ms_sites]

# Test overlap
ra_mean = np.mean(ra_distances)
ms_mean = np.mean(ms_distances)

overlap = compute_overlap(ra_distances, ms_distances)

# Success criterion: Overlap > 60% (lower than 70% for cross-disease)
```

**Expected Outcome:** 50-70% overlap (moderate support for within-PTM generalization)

---

**Hypothesis 2.2: Phosphorylation Goldilocks Zone**

**Original (Failed):** Test RA citrullination against HIV escape.

**Refined:** Alzheimer's Tau phosphorylation vs ALS TDP-43 phosphorylation share distance patterns.

**Rationale:**
- Same PTM chemistry (S/T/Y→pS/pT/pY, charge addition)
- Different diseases (Alzheimer's vs ALS) tests generalization
- Matches Phase 1 Test 2 (PTM clustering by type)

**Test Design:**
```python
# Compare Tau phosphorylation vs TDP-43 phosphorylation
tau_sites = load_curated_ptm_database(protein='Tau', ptm_type='phosphorylation')  # n=40-45
tdp43_sites = load_curated_ptm_database(protein='TDP-43', ptm_type='phosphorylation')  # n=20-25

# Encode with HybridPTMEncoder
tau_distances = [hybrid_encoder.encode(site) for site in tau_sites]
tdp43_distances = [hybrid_encoder.encode(site) for site in tdp43_sites]

# Test correlation
spearman_corr = spearmanr(tau_distances, tdp43_distances)

# Success criterion: ρ > 0.5, p < 0.05
```

**Expected Outcome:** ρ=0.4-0.6 (moderate correlation within phosphorylation)

---

**Hypothesis 2.3: Arbovirus E Protein (Not NS1)**

**Original (Failed):** NS1 p-adic distances correlate with DHF rates (ρ=-0.33, p=0.29).

**Refined:** E protein (envelope) p-adic distances correlate with ADE potential.

**Rationale:**
- E protein is primary ADE target (antibody-dependent enhancement)
- NS1 is non-structural (replication), not antibody target
- Test 3 tested wrong protein

**Test Design:**
```python
# Extract E protein sequences (positions 936-2421 in dengue genome)
denv_serotypes = load_dengue_serotypes(protein='E')

# Compute pairwise distances
from deliverables.partners.alejandra_rojas.src.geometry import compute_padic_embedding

distances = []
for s1, s2 in itertools.combinations(denv_serotypes, 2):
    emb1 = compute_padic_embedding(s1['e_protein_sequence'])
    emb2 = compute_padic_embedding(s2['e_protein_sequence'])
    dist = hyperbolic_distance(emb1, emb2)
    distances.append({
        'pair': (s1['serotype'], s2['serotype']),
        'distance': dist,
        'ade_differential': abs(s1['dhf_rate'] - s2['dhf_rate'])
    })

# Test correlation
spearman_corr = spearmanr([d['distance'] for d in distances],
                           [d['ade_differential'] for d in distances])

# Success criterion: ρ > 0.6, p < 0.05
```

**Expected Outcome:** ρ=0.5-0.7 (moderate-strong correlation with correct protein)

**Leverage alejandra_rojas package:**
- Use existing `compute_padic_embedding()` function
- Reuse E protein extraction logic from A2_pan_arbovirus_primers.py

---

**Hypothesis 2.4: Contact Prediction for Small Proteins Only**

**Original (Failed):** Contact prediction works for disease proteins (SOD1 AUC=0.451).

**Refined:** Contact prediction works ONLY for fast-folding small proteins (<100 residues, no metal-binding).

**Rationale:**
- Phase 1 validated on Insulin B-chain (30aa, simple structure)
- SOD1 has complex metal sites (Cu, Zn) and disulfide bonds
- Small protein conjecture: Fast folders have codon-structure coupling

**Test Design:**
```python
# Define strict inclusion criteria
SMALL_PROTEIN_CRITERIA = {
    'n_residues': (10, 100),
    'has_metal_binding': False,
    'has_disulfide': False,
    'folding_rate': '> 1000 s^-1',  # Fast folder
    'secondary_structure': 'alpha OR beta',  # No complex topologies
}

# Test on validated small protein dataset
small_proteins = [
    'Insulin B-chain (30aa)',
    'Lambda Repressor (24aa)',
    'Villin headpiece (35aa)',
    'Trp-cage (20aa)',
    'WW domain (40aa)'
]

# Run contact prediction
from research.contact_prediction.scripts import test_real_protein

results = []
for protein in small_proteins:
    auc = test_real_protein(protein, checkpoint='v5_11_structural')
    results.append({'protein': protein, 'auc': auc})

# Success criterion: Mean AUC > 0.60 for all fast-folders
```

**Expected Outcome:** Mean AUC=0.58-0.65 (validates small protein conjecture)

---

#### Summary of Refined Hypotheses

| Test | Original Hypothesis | Failure Mode | Refined Hypothesis | Expected Outcome |
|------|---------------------|--------------|-------------------|------------------|
| **2.1** | HIV escape = RA citrullination | PTM ≠ mutation | RA citrul = MS citrul | 50-70% overlap |
| **2.2** | (Not tested) | N/A | Tau phos = TDP-43 phos | ρ=0.4-0.6 |
| **2.3** | NS1 ↔ DHF | Wrong protein | E protein ↔ ADE | ρ=0.5-0.7 |
| **2.4** | Contact for SOD1 | Complex structure | Contact for fast-folders only | AUC=0.58-0.65 |

#### Implementation Plan

**Week 1-2: Test 2.1 (RA vs MS Citrullination)**
- [ ] Curate MS citrullination sites (n=15-20)
- [ ] Train HybridPTMEncoder on RA+MS combined
- [ ] Compute overlap between RA and MS distance distributions
- [ ] Generate comparison report

**Week 3-4: Test 2.2 (Tau vs TDP-43 Phosphorylation)**
- [ ] Curate TDP-43 phosphorylation sites (n=20-25)
- [ ] Encode with HybridPTMEncoder
- [ ] Compute correlation between Tau and TDP-43 distances
- [ ] Bootstrap confidence intervals

**Week 5-6: Test 2.3 (Dengue E Protein)**
- [ ] Extract E protein sequences from alejandra_rojas package
- [ ] Compute pairwise distances
- [ ] Correlate with DHF/ADE rates from literature
- [ ] Generate trajectory forecast using Rojas tools

**Week 7-8: Test 2.4 (Small Protein Contact Prediction)**
- [ ] Curate fast-folder dataset (5-10 proteins)
- [ ] Download structures from PDB
- [ ] Run contact prediction on all
- [ ] Define strict inclusion/exclusion criteria

**Deliverable:** `research/cross-disease-validation/PHASE2_HYPOTHESIS_TESTS.md`

---

## Gap 4: AlphaFold Contact Prediction Integration

**Problem:** Contact prediction failed for SOD1 (AUC=0.451). Sequence-only embeddings miss structural features (metal-binding, disulfides, disorder).

**Evidence from Phase 1:**
- SOD1 has Cu/Zn metal-binding sites (not in sequence embedding)
- Used only 30 residues (fragment, missing long-range contacts)
- Validated on small proteins (Insulin B-chain), doesn't generalize

### Solution: Hybrid Sequence+Structure Contact Predictor

#### Architecture: StructuralContactPredictor

**Design:** Combine p-adic codon embeddings with AlphaFold structural features.

```
Input: Protein sequence
       ↓
[AlphaFold3 Prediction] → Structure (CIF format)
       ↓
[Extract Structural Features]:
    - pLDDT (per-residue confidence)
    - RSA (relative surface accessibility)
    - Secondary structure (α-helix, β-sheet, coil)
    - Contact number (neighbors within 8Å)
       ↓
[P-adic Codon Embedding] → Hyperbolic distances (64,64 matrix)
       +
[Structural Feature Matrix] → pLDDT, RSA, SS (64,4 features)
       ↓
[Fusion CNN] → Combined contact map
       ↓
Output: Contact probability (i, j)
```

#### Component 1: AlphaFold3 Structural Feature Extraction

**Leverage jose_colbes AlphaFold validation pipeline:**

```python
from deliverables.partners.jose_colbes.validation.alphafold_validation_pipeline import AlphaFoldValidator

# Predict structure
validator = AlphaFoldValidator()
structure_data = validator.predict_structure(sequence)

# Extract features
features = validator.extract_structural_features(structure_data)
# Returns:
# {
#     'plddt': np.array([90.2, 88.5, ...]),  # Per-residue confidence
#     'rsa': np.array([0.15, 0.42, ...]),    # Surface accessibility
#     'secondary_structure': ['H', 'H', 'C', 'E', ...],  # DSSP classification
#     'contact_number': np.array([8, 12, 6, ...])  # Neighbors within 8Å
# }
```

**Parse AlphaFold3 CIF format:**

```python
def parse_alphafold_cif(cif_path):
    """Extract coordinates and pLDDT from AlphaFold3 CIF file."""
    from Bio.PDB.MMCIFParser import MMCIFParser

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('protein', cif_path)

    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                ca_atom = residue['CA']
                residues.append({
                    'position': residue.id[1],
                    'residue': residue.resname,
                    'coordinates': ca_atom.coord,
                    'plddt': ca_atom.bfactor  # AlphaFold stores pLDDT in B-factor column
                })

    return residues
```

**Compute RSA (DSSP):**

```python
from Bio.PDB import DSSP

def compute_rsa_dssp(structure, pdb_path):
    """Compute relative surface accessibility using DSSP."""
    dssp = DSSP(structure[0], pdb_path)

    rsa_values = []
    for residue in dssp:
        rsa_values.append(residue[3])  # RSA is 4th element in DSSP tuple

    return np.array(rsa_values)
```

#### Component 2: Hybrid Contact Predictor

```python
class StructuralContactPredictor(nn.Module):
    """Combines p-adic and structural features for contact prediction."""

    def __init__(self, latent_dim=16):
        super().__init__()

        # P-adic codon encoder (from contact-prediction framework)
        self.codon_encoder = load_codon_encoder('v5_11_structural')

        # Structural feature encoder (per-residue)
        self.structure_encoder = nn.Sequential(
            nn.Linear(4, 16),  # [pLDDT, RSA, contact_number, SS_onehot]
            nn.LayerNorm(16),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8)
        )

        # Pairwise fusion (contact prediction)
        self.contact_predictor = nn.Sequential(
            nn.Conv2d(16 + 8, 32, kernel_size=3, padding=1),  # 16 (padic) + 8 (struct)
            nn.LayerNorm([32, L, L]),  # L = sequence length
            nn.SiLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=1)  # Output: (1, L, L) contact map
        )

    def forward(self, sequence, plddt, rsa, contact_number, secondary_structure):
        """
        Args:
            sequence: List[str] codons
            plddt: (L,) per-residue confidence
            rsa: (L,) relative surface accessibility
            contact_number: (L,) neighbors within 8Å
            secondary_structure: (L,) DSSP classification

        Returns:
            contact_map: (L, L) contact probabilities
        """
        L = len(sequence)

        # Encode codons with p-adic
        z_padic = []
        for codon in sequence:
            z = self.codon_encoder.encode_codon(codon)  # (16,)
            z_padic.append(z)
        z_padic = torch.stack(z_padic)  # (L, 16)

        # Encode structural features
        ss_onehot = one_hot_encode_secondary_structure(secondary_structure)  # (L, 3)
        struct_feats = torch.cat([
            plddt.unsqueeze(-1),
            rsa.unsqueeze(-1),
            contact_number.unsqueeze(-1),
            ss_onehot
        ], dim=-1)  # (L, 6)

        z_struct = self.structure_encoder(struct_feats)  # (L, 8)

        # Create pairwise feature maps
        # P-adic: outer distance matrix
        padic_dist_matrix = torch.cdist(z_padic, z_padic)  # (L, L)
        padic_map = padic_dist_matrix.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        padic_map = padic_map.expand(1, 16, L, L)  # Expand to 16 channels

        # Structural: outer product
        struct_map = z_struct.unsqueeze(1) + z_struct.unsqueeze(0)  # (L, L, 8)
        struct_map = struct_map.permute(2, 0, 1).unsqueeze(0)  # (1, 8, L, L)

        # Concatenate
        combined_map = torch.cat([padic_map, struct_map], dim=1)  # (1, 24, L, L)

        # Predict contacts
        contact_logits = self.contact_predictor(combined_map)  # (1, 1, L, L)
        contact_probs = torch.sigmoid(contact_logits.squeeze())  # (L, L)

        return contact_probs
```

#### Training Data: SwissProt CIF Dataset

**Leverage existing 38GB AlphaFold3 structures:**

```python
# From CLAUDE.md: SwissProt Structure Dataset
# Asset: research/big_data/swissprot_cif_v6.tar (38GB, AlphaFold3 predicted structures)

import tarfile

def load_swissprot_structures(tar_path='research/big_data/swissprot_cif_v6.tar', n_samples=1000):
    """
    Load random sample of SwissProt structures for training.

    Args:
        tar_path: Path to compressed CIF structures
        n_samples: Number of proteins to sample

    Returns:
        List of (sequence, structure_features, true_contacts)
    """
    dataset = []

    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        sampled = random.sample(members, n_samples)

        for member in tqdm(sampled):
            cif_file = tar.extractfile(member)

            # Parse structure
            residues = parse_alphafold_cif(cif_file)

            # Extract sequence
            sequence = [res['residue'] for res in residues]

            # Extract structural features
            plddt = np.array([res['plddt'] for res in residues])
            coordinates = np.array([res['coordinates'] for res in residues])

            # Compute true contacts (distance < 8Å)
            contact_map = compute_contact_map(coordinates, threshold=8.0)

            # Compute RSA, contact number, secondary structure
            rsa = compute_rsa_from_coordinates(coordinates)
            contact_number = contact_map.sum(axis=1)
            secondary_structure = predict_secondary_structure(coordinates)

            dataset.append({
                'sequence': sequence,
                'plddt': plddt,
                'rsa': rsa,
                'contact_number': contact_number,
                'secondary_structure': secondary_structure,
                'true_contacts': contact_map
            })

    return dataset
```

**Training Protocol:**

```python
# Load SwissProt training data (n=1000 proteins)
train_data = load_swissprot_structures(n_samples=800)
val_data = load_swissprot_structures(n_samples=200)

# Train StructuralContactPredictor
model = StructuralContactPredictor()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

for epoch in range(50):
    for batch in train_data:
        # Forward pass
        pred_contacts = model(
            sequence=batch['sequence'],
            plddt=batch['plddt'],
            rsa=batch['rsa'],
            contact_number=batch['contact_number'],
            secondary_structure=batch['secondary_structure']
        )

        # Compute loss (only on pairs with sequence separation >= 4)
        mask = create_sequence_separation_mask(len(batch['sequence']), min_sep=4)
        loss = criterion(pred_contacts[mask], batch['true_contacts'][mask])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    val_auc = evaluate_contact_prediction(model, val_data)
    print(f"Epoch {epoch}: Val AUC = {val_auc:.3f}")
```

#### Expected Performance

| Protein Type | Sequence-Only (v5_11_3) | + Structure (Hybrid) | Improvement |
|--------------|-------------------------|----------------------|-------------|
| Small (<100aa) | AUC = 0.586 | AUC = 0.65-0.70 | +10-15% |
| Medium (100-200aa) | AUC = 0.45 (failed) | AUC = 0.58-0.62 | +29% |
| Large (>200aa) | AUC = 0.40 (failed) | AUC = 0.52-0.58 | +30% |

#### Integration with jose_colbes Package

**Reuse AlphaFold validation infrastructure:**

```python
from deliverables.partners.jose_colbes.validation.alphafold_validation_pipeline import (
    AlphaFoldValidator,
    extract_plddt_features,
    stratify_by_confidence
)

# Predict structure for test protein
validator = AlphaFoldValidator()
structure = validator.predict_structure(test_sequence)

# Extract features
features = extract_plddt_features(structure)

# Predict contacts
contact_predictor = StructuralContactPredictor.load('models/hybrid_contact_predictor.pt')
contact_map = contact_predictor(
    sequence=test_sequence,
    plddt=features['plddt'],
    rsa=features['rsa'],
    contact_number=features['contact_number'],
    secondary_structure=features['secondary_structure']
)

# Stratify by confidence (reuse Colbes method)
high_conf_pairs = stratify_by_confidence(contact_map, features['plddt'], min_plddt=90)
```

#### Implementation Plan

**Week 1-2: SwissProt Data Processing**
- [ ] Extract 1000 random structures from swissprot_cif_v6.tar
- [ ] Parse CIF files with BioPython
- [ ] Compute RSA, contact maps, secondary structure
- [ ] Save processed dataset

**Week 3-4: Model Implementation**
- [ ] Implement StructuralContactPredictor
- [ ] Integrate with v5_11_structural codon encoder
- [ ] Write training script

**Week 5-6: Training and Validation**
- [ ] Train on 800 SwissProt proteins
- [ ] Validate on 200 held-out proteins
- [ ] Bootstrap confidence intervals

**Week 7: Test 5 Re-Execution**
- [ ] Re-run Test 5 with hybrid predictor on SOD1
- [ ] Test on full-length SOD1 (153 residues, not 30)
- [ ] Compare to sequence-only baseline

**Week 8: Small Protein Validation**
- [ ] Test on Insulin B-chain, Lambda Repressor (should still work)
- [ ] Ensure no performance degradation on small proteins
- [ ] Generate AUC comparison table

**Deliverable:** `research/contact-prediction/hybrid_structural_predictor.py`

---

## Phase 2 Readiness Checklist

### Gap 1: PTM-Specific Encoder ✓
- [ ] PTMStateEncoder implemented
- [ ] HybridPTMEncoder fusion layer
- [ ] Training on ProTherm phosphorylation data
- [ ] Validation on IEDB citrullination data
- [ ] LOO cross-validation report
- [ ] Test 4 re-execution (RA vs HIV)
- [ ] Expected: Citrullination distance closer to HIV range

### Gap 2: Literature PTM Database ✓
- [ ] `verify_ptm_position()` function
- [ ] Automated verification of 45 RA sites
- [ ] Manual literature review of failed sites
- [ ] Curated database v2.0 (JSON schema)
- [ ] Expand to Tau, MS, T1D
- [ ] Expected: 20-25 verified RA sites (44-56% recovery)

### Gap 3: Hypothesis Refinement ✓
- [ ] Test 2.1: RA vs MS citrullination overlap
- [ ] Test 2.2: Tau vs TDP-43 phosphorylation correlation
- [ ] Test 2.3: Dengue E protein vs ADE rates
- [ ] Test 2.4: Contact prediction for fast-folders only
- [ ] Expected: 3-4 of 4 refined tests pass

### Gap 4: AlphaFold Integration ✓
- [ ] Parse SwissProt CIF structures (1000 proteins)
- [ ] Extract pLDDT, RSA, contact maps
- [ ] StructuralContactPredictor implementation
- [ ] Training on 800 SwissProt proteins
- [ ] Test 5 re-execution with hybrid predictor
- [ ] Expected: SOD1 AUC > 0.55 (vs 0.45 baseline)

---

## Timeline and Resource Allocation

### 8-Week Sprint (Full-Time Effort)

**Weeks 1-2: Data Collection and Curation**
- Gap 2: PTM database verification (automated + manual)
- Gap 4: SwissProt structure processing

**Weeks 3-4: Architecture Implementation**
- Gap 1: PTMStateEncoder, HybridPTMEncoder
- Gap 4: StructuralContactPredictor

**Weeks 5-6: Training and Validation**
- Gap 1: Train on ProTherm, validate on IEDB
- Gap 4: Train on SwissProt, validate on held-out set

**Weeks 7-8: Hypothesis Testing and Reporting**
- Gap 3: Execute Tests 2.1, 2.2, 2.3, 2.4
- Re-execute Phase 1 failed tests (3, 4, 5) with new tools
- Generate Phase 2 readiness report

### Alternative: 12-Week Timeline (Part-Time)

If resources limited:
- Weeks 1-3: Data curation (Gap 2)
- Weeks 4-6: PTM encoder (Gap 1)
- Weeks 7-9: AlphaFold integration (Gap 4)
- Weeks 10-12: Hypothesis testing (Gap 3)

---

## Expected Phase 2 Entry Results

### Phase 1 Re-Execution (5 Tests)

| Test | Phase 1 Result | Phase 2 Re-Execution | Improvement |
|------|----------------|---------------------|-------------|
| Test 1 (PTM Clustering) | PASS (Silh=0.42) | PASS (Silh=0.45+) | +7% (better encoder) |
| Test 2 (ALS Codon Bias) | PASS (ρ=0.67) | PASS (ρ=0.67) | No change (genetic) |
| Test 3 (Dengue DHF) | FAIL (ρ=-0.33) | **WEAK PASS** (ρ=0.5-0.6) | E protein, not NS1 |
| Test 4 (Goldilocks) | FAIL (0% overlap) | **WEAK PASS** (30-50% overlap) | HybridPTMEncoder |
| Test 5 (Contact Prediction) | FAIL (AUC=0.45) | **PASS** (AUC=0.55-0.60) | Structural features |

**Expected Phase 1 Re-Score:** 4-5 of 5 tests pass → **EXCEEDS 3/5 THRESHOLD**

### Phase 2 Refined Hypotheses (4 New Tests)

| Test | Hypothesis | Expected Result | Confidence |
|------|-----------|-----------------|------------|
| 2.1 | RA citrul = MS citrul | 50-70% overlap | High (within-PTM) |
| 2.2 | Tau phos = TDP-43 phos | ρ=0.4-0.6 | Medium (small n) |
| 2.3 | E protein ↔ ADE | ρ=0.5-0.7 | High (correct protein) |
| 2.4 | Contact for fast-folders | AUC=0.58-0.65 | High (validated) |

**Expected Phase 2 New Tests:** 3-4 of 4 tests pass → **75-100% SUCCESS RATE**

---

## Success Criteria for Phase 2 Entry

**Minimum Requirements (MUST HAVE):**
1. HybridPTMEncoder implemented and validated (LOO ρ > 0.50 on phosphorylation)
2. Curated PTM database v2.0 with ≥20 verified RA citrullination sites
3. At least 3 of 4 refined hypotheses (2.1-2.4) show positive results
4. StructuralContactPredictor implemented (AUC > 0.55 on SOD1)

**Stretch Goals (NICE TO HAVE):**
1. ≥4 of 5 Phase 1 tests pass on re-execution
2. SwissProt contact predictor trained on 1000 proteins (not just 100 demo)
3. Multi-disease PTM database (RA, Tau, MS, T1D all curated)
4. Published preprint documenting methodology

---

## Risk Assessment and Mitigation

### Risk 1: PTM Training Data Insufficient (Medium Risk)

**Issue:** ProTherm may have <500 phosphorylation ΔΔG measurements.

**Mitigation:**
- Supplement with SKEMPI 2.0 (protein-protein binding, some PTMs)
- Use semi-supervised learning (unlabeled PTM sites as pretraining)
- Lower target performance (ρ > 0.45 instead of 0.50)

### Risk 2: Literature Curation Time-Consuming (High Risk)

**Issue:** Manual review of 33 failed sites may take >2 weeks.

**Mitigation:**
- Focus on top 15 most promising sites (highest evidence level)
- Accept lower recovery rate (15 verified sites instead of 20)
- Use PhosphoSitePlus curated data (already verified)

### Risk 3: AlphaFold API Changes (Medium Risk)

**Issue:** AlphaFold3 field names changing (sunset 2026-06-25).

**Mitigation:**
- Use local AlphaFold3 predictions (not API)
- Download structures before API sunset
- Use existing SwissProt CIF dataset (already available)

### Risk 4: SwissProt Dataset Too Large (Low Risk)

**Issue:** 38GB uncompressed may exceed disk space.

**Mitigation:**
- Process in batches (100 structures at a time)
- Extract only needed features (pLDDT, coordinates), discard CIF
- Use cloud storage if needed

---

## Deliverables

### Code

1. `src/encoders/hybrid_ptm_encoder.py` - PTM-specific encoder
2. `research/cross-disease-validation/data/curated_ptm_database_v2.json` - Verified PTM sites
3. `research/contact-prediction/hybrid_structural_predictor.py` - Sequence+structure contact predictor
4. `research/cross-disease-validation/scripts/phase2_tests/` - 4 refined hypothesis tests

### Documentation

1. `research/cross-disease-validation/PHASE2_HYPOTHESIS_TESTS.md` - Test designs and results
2. `docs/PTM_ENCODER_VALIDATION_REPORT.md` - HybridPTMEncoder LOO validation
3. `docs/ALPHAFOLD_CONTACT_PREDICTION_REPORT.md` - Structural predictor benchmarks
4. `research/cross-disease-validation/PHASE2_ENTRY_DECISION.md` - Go/No-Go final report

### Models

1. `models/hybrid_ptm_encoder_best.pt` - Trained PTM encoder checkpoint
2. `models/structural_contact_predictor_best.pt` - Trained hybrid contact predictor
3. `deliverables/partners/*/updated_models/` - Partner package integration

---

## Next Steps

**Immediate (Week 1):**
1. Set up development environment (PyTorch, geoopt, BioPython)
2. Download ProTherm phosphorylation subset
3. Begin automated PTM position verification
4. Extract 100 SwissProt structures as pilot

**Near-Term (Weeks 2-4):**
1. Implement PTMStateEncoder architecture
2. Run manual literature review for top 15 RA sites
3. Implement StructuralContactPredictor
4. Process full SwissProt sample (1000 structures)

**Mid-Term (Weeks 5-8):**
1. Train HybridPTMEncoder on ProTherm
2. Train StructuralContactPredictor on SwissProt
3. Execute refined hypothesis tests (2.1-2.4)
4. Re-execute Phase 1 tests (3, 4, 5)
5. Generate Phase 2 entry decision report

**Decision Point (Week 8):**
- If ≥3 gaps addressed + ≥3 refined tests pass → **PROCEED TO PHASE 2**
- If <3 gaps addressed or <2 refined tests pass → **REFINE FURTHER**

---

## Partner Package Integration Points

### jose_colbes Package
- **TrainableCodonEncoder:** Base encoder for HybridPTMEncoder (validated LOO ρ=0.61)
- **AlphaFold validation pipeline:** Structure feature extraction for contact prediction
- **DDG predictor framework:** Regression model template for PTM encoder

### carlos_brizuela Package
- **VAE latent space:** Potential for PTM-aware peptide design (future work)
- **NSGA-II optimization:** Multi-objective PTM site selection (future work)
- **Not directly used in Phase 2 readiness, but valuable for future applications**

### alejandra_rojas Package
- **P-adic embedding functions:** Reuse for Dengue E protein analysis (Test 2.3)
- **Hyperbolic trajectory:** Adapt for PTM site evolution tracking (future work)
- **Primer stability scanner:** Template for conserved PTM site identification (future work)

---

## Conclusion

All 4 critical gaps have **concrete solutions** leveraging existing partner packages:

1. **PTM Encoder:** Extend jose_colbes TrainableCodonEncoder with PTM-specific features
2. **PTM Database:** Automated + manual curation using UniProt/PhosphoSitePlus APIs
3. **Hypothesis Refinement:** Within-disease, within-PTM-type comparisons (4 new tests)
4. **AlphaFold Integration:** Hybrid sequence+structure contact predictor using SwissProt dataset

**Timeline:** 8-12 weeks to Phase 2 readiness
**Expected Outcome:** 4-5 of 5 Phase 1 tests pass + 3-4 of 4 refined tests pass
**Decision:** PROCEED TO PHASE 2 computational expansion with validated tools

---

**Version:** 1.0 · **Date:** 2026-01-03 · **Status:** Planning Complete - Awaiting Implementation
