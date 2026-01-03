#!/usr/bin/env python3
"""Amino Acid Functional Profiles for GO Validation.

This module builds functional profiles for each amino acid based on:
1. Physicochemical properties (from biochemistry literature)
2. Active site propensities (from enzyme databases)
3. Enzyme class enrichments (from EC classification)
4. Structural propensities (from PDB statistics)

These profiles are used to compute functional similarity between amino acids,
which is then compared to the groupoid morphism structure.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
from pathlib import Path


# =============================================================================
# Amino Acid Properties (from established biochemistry literature)
# =============================================================================

@dataclass
class AminoAcidFunctionalProfile:
    """Comprehensive functional profile for an amino acid."""
    code: str
    name: str

    # Physicochemical (Kyte-Doolittle, pKa, MW, etc.)
    hydrophobicity: float      # Kyte-Doolittle scale (-4.5 to 4.5)
    charge_ph7: float          # Net charge at pH 7 (-1, 0, +1)
    molecular_weight: float    # Daltons
    volume: float              # Å³ (van der Waals)
    polarity: float            # 0-1 scale
    aromaticity: float         # 0 or 1

    # Active site propensities (from CSA - Catalytic Site Atlas)
    catalytic_propensity: float    # Frequency in catalytic sites
    nucleophile_propensity: float  # Nucleophilic attack capability
    acid_base_propensity: float    # Acid-base catalysis
    metal_binding: float           # Metal coordination

    # Enzyme class enrichments (from UniProt EC statistics)
    ec1_oxidoreductase: float
    ec2_transferase: float
    ec3_hydrolase: float
    ec4_lyase: float
    ec5_isomerase: float
    ec6_ligase: float

    # Structural propensities (from PDB statistics)
    helix_propensity: float
    sheet_propensity: float
    turn_propensity: float
    disorder_propensity: float

    # Binding propensities
    dna_binding: float
    rna_binding: float
    atp_binding: float
    cofactor_binding: float

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for similarity computation."""
        return np.array([
            self.hydrophobicity / 4.5,  # Normalize to ~[-1, 1]
            self.charge_ph7,
            self.molecular_weight / 200,  # Normalize
            self.volume / 230,
            self.polarity,
            self.aromaticity,
            self.catalytic_propensity,
            self.nucleophile_propensity,
            self.acid_base_propensity,
            self.metal_binding,
            self.ec1_oxidoreductase,
            self.ec2_transferase,
            self.ec3_hydrolase,
            self.ec4_lyase,
            self.ec5_isomerase,
            self.ec6_ligase,
            self.helix_propensity,
            self.sheet_propensity,
            self.turn_propensity,
            self.disorder_propensity,
            self.dna_binding,
            self.rna_binding,
            self.atp_binding,
            self.cofactor_binding,
        ])


# =============================================================================
# Amino Acid Functional Data
# Data sources:
# - Physicochemical: Kyte & Doolittle 1982, Zamyatnin 1972
# - Catalytic: Bartlett et al. 2002 (Catalytic Site Atlas)
# - EC enrichment: UniProt enzyme statistics
# - Structural: Chou & Fasman 1978, Dunbrack 2002
# - Binding: Luscombe et al. 2001, Jones et al. 2003
# =============================================================================

AMINO_ACID_PROFILES: Dict[str, AminoAcidFunctionalProfile] = {
    'A': AminoAcidFunctionalProfile(
        code='A', name='Alanine',
        hydrophobicity=1.8, charge_ph7=0, molecular_weight=89.1, volume=88.6,
        polarity=0.0, aromaticity=0.0,
        catalytic_propensity=0.05, nucleophile_propensity=0.0, acid_base_propensity=0.02, metal_binding=0.01,
        ec1_oxidoreductase=0.08, ec2_transferase=0.07, ec3_hydrolase=0.06, ec4_lyase=0.07, ec5_isomerase=0.08, ec6_ligase=0.07,
        helix_propensity=1.42, sheet_propensity=0.83, turn_propensity=0.66, disorder_propensity=0.06,
        dna_binding=0.04, rna_binding=0.05, atp_binding=0.06, cofactor_binding=0.05,
    ),
    'R': AminoAcidFunctionalProfile(
        code='R', name='Arginine',
        hydrophobicity=-4.5, charge_ph7=1, molecular_weight=174.2, volume=173.4,
        polarity=1.0, aromaticity=0.0,
        catalytic_propensity=0.12, nucleophile_propensity=0.02, acid_base_propensity=0.15, metal_binding=0.03,
        ec1_oxidoreductase=0.05, ec2_transferase=0.08, ec3_hydrolase=0.07, ec4_lyase=0.06, ec5_isomerase=0.05, ec6_ligase=0.09,
        helix_propensity=0.98, sheet_propensity=0.93, turn_propensity=0.95, disorder_propensity=0.18,
        dna_binding=0.22, rna_binding=0.18, atp_binding=0.12, cofactor_binding=0.08,
    ),
    'N': AminoAcidFunctionalProfile(
        code='N', name='Asparagine',
        hydrophobicity=-3.5, charge_ph7=0, molecular_weight=132.1, volume=114.1,
        polarity=0.85, aromaticity=0.0,
        catalytic_propensity=0.08, nucleophile_propensity=0.05, acid_base_propensity=0.06, metal_binding=0.08,
        ec1_oxidoreductase=0.06, ec2_transferase=0.09, ec3_hydrolase=0.08, ec4_lyase=0.05, ec5_isomerase=0.06, ec6_ligase=0.08,
        helix_propensity=0.67, sheet_propensity=0.89, turn_propensity=1.56, disorder_propensity=0.12,
        dna_binding=0.08, rna_binding=0.10, atp_binding=0.07, cofactor_binding=0.12,
    ),
    'D': AminoAcidFunctionalProfile(
        code='D', name='Aspartate',
        hydrophobicity=-3.5, charge_ph7=-1, molecular_weight=133.1, volume=111.1,
        polarity=1.0, aromaticity=0.0,
        catalytic_propensity=0.18, nucleophile_propensity=0.12, acid_base_propensity=0.22, metal_binding=0.25,
        ec1_oxidoreductase=0.09, ec2_transferase=0.11, ec3_hydrolase=0.15, ec4_lyase=0.12, ec5_isomerase=0.08, ec6_ligase=0.10,
        helix_propensity=1.01, sheet_propensity=0.54, turn_propensity=1.46, disorder_propensity=0.15,
        dna_binding=0.06, rna_binding=0.08, atp_binding=0.09, cofactor_binding=0.18,
    ),
    'C': AminoAcidFunctionalProfile(
        code='C', name='Cysteine',
        hydrophobicity=2.5, charge_ph7=0, molecular_weight=121.2, volume=108.5,
        polarity=0.3, aromaticity=0.0,
        catalytic_propensity=0.15, nucleophile_propensity=0.25, acid_base_propensity=0.08, metal_binding=0.35,
        ec1_oxidoreductase=0.12, ec2_transferase=0.08, ec3_hydrolase=0.10, ec4_lyase=0.09, ec5_isomerase=0.11, ec6_ligase=0.07,
        helix_propensity=0.70, sheet_propensity=1.19, turn_propensity=1.19, disorder_propensity=0.04,
        dna_binding=0.08, rna_binding=0.06, atp_binding=0.05, cofactor_binding=0.22,
    ),
    'Q': AminoAcidFunctionalProfile(
        code='Q', name='Glutamine',
        hydrophobicity=-3.5, charge_ph7=0, molecular_weight=146.2, volume=143.8,
        polarity=0.85, aromaticity=0.0,
        catalytic_propensity=0.06, nucleophile_propensity=0.04, acid_base_propensity=0.05, metal_binding=0.06,
        ec1_oxidoreductase=0.05, ec2_transferase=0.08, ec3_hydrolase=0.07, ec4_lyase=0.06, ec5_isomerase=0.07, ec6_ligase=0.09,
        helix_propensity=1.11, sheet_propensity=1.10, turn_propensity=0.98, disorder_propensity=0.14,
        dna_binding=0.07, rna_binding=0.09, atp_binding=0.08, cofactor_binding=0.10,
    ),
    'E': AminoAcidFunctionalProfile(
        code='E', name='Glutamate',
        hydrophobicity=-3.5, charge_ph7=-1, molecular_weight=147.1, volume=138.4,
        polarity=1.0, aromaticity=0.0,
        catalytic_propensity=0.20, nucleophile_propensity=0.15, acid_base_propensity=0.25, metal_binding=0.22,
        ec1_oxidoreductase=0.10, ec2_transferase=0.12, ec3_hydrolase=0.14, ec4_lyase=0.15, ec5_isomerase=0.09, ec6_ligase=0.11,
        helix_propensity=1.51, sheet_propensity=0.37, turn_propensity=0.74, disorder_propensity=0.16,
        dna_binding=0.05, rna_binding=0.07, atp_binding=0.10, cofactor_binding=0.15,
    ),
    'G': AminoAcidFunctionalProfile(
        code='G', name='Glycine',
        hydrophobicity=-0.4, charge_ph7=0, molecular_weight=75.1, volume=60.1,
        polarity=0.0, aromaticity=0.0,
        catalytic_propensity=0.08, nucleophile_propensity=0.02, acid_base_propensity=0.03, metal_binding=0.05,
        ec1_oxidoreductase=0.09, ec2_transferase=0.10, ec3_hydrolase=0.08, ec4_lyase=0.11, ec5_isomerase=0.12, ec6_ligase=0.08,
        helix_propensity=0.57, sheet_propensity=0.75, turn_propensity=1.56, disorder_propensity=0.22,
        dna_binding=0.06, rna_binding=0.08, atp_binding=0.15, cofactor_binding=0.12,
    ),
    'H': AminoAcidFunctionalProfile(
        code='H', name='Histidine',
        hydrophobicity=-3.2, charge_ph7=0.1, molecular_weight=155.2, volume=153.2,
        polarity=0.7, aromaticity=1.0,
        catalytic_propensity=0.22, nucleophile_propensity=0.18, acid_base_propensity=0.28, metal_binding=0.35,
        ec1_oxidoreductase=0.11, ec2_transferase=0.10, ec3_hydrolase=0.18, ec4_lyase=0.08, ec5_isomerase=0.09, ec6_ligase=0.08,
        helix_propensity=1.00, sheet_propensity=0.87, turn_propensity=0.95, disorder_propensity=0.10,
        dna_binding=0.10, rna_binding=0.08, atp_binding=0.12, cofactor_binding=0.25,
    ),
    'I': AminoAcidFunctionalProfile(
        code='I', name='Isoleucine',
        hydrophobicity=4.5, charge_ph7=0, molecular_weight=131.2, volume=166.7,
        polarity=0.0, aromaticity=0.0,
        catalytic_propensity=0.02, nucleophile_propensity=0.0, acid_base_propensity=0.01, metal_binding=0.01,
        ec1_oxidoreductase=0.08, ec2_transferase=0.06, ec3_hydrolase=0.05, ec4_lyase=0.06, ec5_isomerase=0.07, ec6_ligase=0.05,
        helix_propensity=1.08, sheet_propensity=1.60, turn_propensity=0.47, disorder_propensity=0.03,
        dna_binding=0.02, rna_binding=0.03, atp_binding=0.04, cofactor_binding=0.03,
    ),
    'L': AminoAcidFunctionalProfile(
        code='L', name='Leucine',
        hydrophobicity=3.8, charge_ph7=0, molecular_weight=131.2, volume=166.7,
        polarity=0.0, aromaticity=0.0,
        catalytic_propensity=0.02, nucleophile_propensity=0.0, acid_base_propensity=0.01, metal_binding=0.01,
        ec1_oxidoreductase=0.07, ec2_transferase=0.06, ec3_hydrolase=0.05, ec4_lyase=0.05, ec5_isomerase=0.06, ec6_ligase=0.06,
        helix_propensity=1.21, sheet_propensity=1.30, turn_propensity=0.59, disorder_propensity=0.04,
        dna_binding=0.03, rna_binding=0.04, atp_binding=0.05, cofactor_binding=0.04,
    ),
    'K': AminoAcidFunctionalProfile(
        code='K', name='Lysine',
        hydrophobicity=-3.9, charge_ph7=1, molecular_weight=146.2, volume=168.6,
        polarity=1.0, aromaticity=0.0,
        catalytic_propensity=0.14, nucleophile_propensity=0.08, acid_base_propensity=0.12, metal_binding=0.05,
        ec1_oxidoreductase=0.06, ec2_transferase=0.09, ec3_hydrolase=0.08, ec4_lyase=0.10, ec5_isomerase=0.07, ec6_ligase=0.12,
        helix_propensity=1.16, sheet_propensity=0.74, turn_propensity=1.01, disorder_propensity=0.20,
        dna_binding=0.18, rna_binding=0.15, atp_binding=0.14, cofactor_binding=0.10,
    ),
    'M': AminoAcidFunctionalProfile(
        code='M', name='Methionine',
        hydrophobicity=1.9, charge_ph7=0, molecular_weight=149.2, volume=162.9,
        polarity=0.0, aromaticity=0.0,
        catalytic_propensity=0.03, nucleophile_propensity=0.02, acid_base_propensity=0.02, metal_binding=0.08,
        ec1_oxidoreductase=0.09, ec2_transferase=0.07, ec3_hydrolase=0.06, ec4_lyase=0.05, ec5_isomerase=0.08, ec6_ligase=0.06,
        helix_propensity=1.45, sheet_propensity=1.05, turn_propensity=0.60, disorder_propensity=0.05,
        dna_binding=0.04, rna_binding=0.05, atp_binding=0.06, cofactor_binding=0.07,
    ),
    'F': AminoAcidFunctionalProfile(
        code='F', name='Phenylalanine',
        hydrophobicity=2.8, charge_ph7=0, molecular_weight=165.2, volume=189.9,
        polarity=0.0, aromaticity=1.0,
        catalytic_propensity=0.04, nucleophile_propensity=0.01, acid_base_propensity=0.02, metal_binding=0.02,
        ec1_oxidoreductase=0.08, ec2_transferase=0.07, ec3_hydrolase=0.06, ec4_lyase=0.06, ec5_isomerase=0.07, ec6_ligase=0.05,
        helix_propensity=1.13, sheet_propensity=1.38, turn_propensity=0.60, disorder_propensity=0.04,
        dna_binding=0.06, rna_binding=0.08, atp_binding=0.07, cofactor_binding=0.06,
    ),
    'P': AminoAcidFunctionalProfile(
        code='P', name='Proline',
        hydrophobicity=-1.6, charge_ph7=0, molecular_weight=115.1, volume=112.7,
        polarity=0.0, aromaticity=0.0,
        catalytic_propensity=0.03, nucleophile_propensity=0.0, acid_base_propensity=0.02, metal_binding=0.02,
        ec1_oxidoreductase=0.05, ec2_transferase=0.06, ec3_hydrolase=0.05, ec4_lyase=0.07, ec5_isomerase=0.08, ec6_ligase=0.05,
        helix_propensity=0.57, sheet_propensity=0.55, turn_propensity=1.52, disorder_propensity=0.25,
        dna_binding=0.05, rna_binding=0.06, atp_binding=0.04, cofactor_binding=0.05,
    ),
    'S': AminoAcidFunctionalProfile(
        code='S', name='Serine',
        hydrophobicity=-0.8, charge_ph7=0, molecular_weight=105.1, volume=89.0,
        polarity=0.7, aromaticity=0.0,
        catalytic_propensity=0.16, nucleophile_propensity=0.20, acid_base_propensity=0.10, metal_binding=0.06,
        ec1_oxidoreductase=0.07, ec2_transferase=0.12, ec3_hydrolase=0.14, ec4_lyase=0.06, ec5_isomerase=0.08, ec6_ligase=0.09,
        helix_propensity=0.77, sheet_propensity=0.75, turn_propensity=1.43, disorder_propensity=0.15,
        dna_binding=0.09, rna_binding=0.11, atp_binding=0.10, cofactor_binding=0.12,
    ),
    'T': AminoAcidFunctionalProfile(
        code='T', name='Threonine',
        hydrophobicity=-0.7, charge_ph7=0, molecular_weight=119.1, volume=116.1,
        polarity=0.6, aromaticity=0.0,
        catalytic_propensity=0.10, nucleophile_propensity=0.12, acid_base_propensity=0.08, metal_binding=0.08,
        ec1_oxidoreductase=0.08, ec2_transferase=0.11, ec3_hydrolase=0.10, ec4_lyase=0.07, ec5_isomerase=0.09, ec6_ligase=0.08,
        helix_propensity=0.83, sheet_propensity=1.19, turn_propensity=0.96, disorder_propensity=0.12,
        dna_binding=0.07, rna_binding=0.09, atp_binding=0.11, cofactor_binding=0.14,
    ),
    'W': AminoAcidFunctionalProfile(
        code='W', name='Tryptophan',
        hydrophobicity=-0.9, charge_ph7=0, molecular_weight=204.2, volume=227.8,
        polarity=0.3, aromaticity=1.0,
        catalytic_propensity=0.06, nucleophile_propensity=0.02, acid_base_propensity=0.04, metal_binding=0.03,
        ec1_oxidoreductase=0.10, ec2_transferase=0.06, ec3_hydrolase=0.07, ec4_lyase=0.05, ec5_isomerase=0.06, ec6_ligase=0.04,
        helix_propensity=1.08, sheet_propensity=1.37, turn_propensity=0.96, disorder_propensity=0.03,
        dna_binding=0.08, rna_binding=0.10, atp_binding=0.06, cofactor_binding=0.08,
    ),
    'Y': AminoAcidFunctionalProfile(
        code='Y', name='Tyrosine',
        hydrophobicity=-1.3, charge_ph7=0, molecular_weight=181.2, volume=193.6,
        polarity=0.5, aromaticity=1.0,
        catalytic_propensity=0.10, nucleophile_propensity=0.08, acid_base_propensity=0.12, metal_binding=0.05,
        ec1_oxidoreductase=0.11, ec2_transferase=0.08, ec3_hydrolase=0.09, ec4_lyase=0.06, ec5_isomerase=0.07, ec6_ligase=0.06,
        helix_propensity=0.69, sheet_propensity=1.47, turn_propensity=1.14, disorder_propensity=0.05,
        dna_binding=0.09, rna_binding=0.11, atp_binding=0.08, cofactor_binding=0.10,
    ),
    'V': AminoAcidFunctionalProfile(
        code='V', name='Valine',
        hydrophobicity=4.2, charge_ph7=0, molecular_weight=117.1, volume=140.0,
        polarity=0.0, aromaticity=0.0,
        catalytic_propensity=0.02, nucleophile_propensity=0.0, acid_base_propensity=0.01, metal_binding=0.01,
        ec1_oxidoreductase=0.08, ec2_transferase=0.06, ec3_hydrolase=0.05, ec4_lyase=0.06, ec5_isomerase=0.07, ec6_ligase=0.05,
        helix_propensity=1.06, sheet_propensity=1.70, turn_propensity=0.50, disorder_propensity=0.03,
        dna_binding=0.02, rna_binding=0.03, atp_binding=0.04, cofactor_binding=0.03,
    ),
}


# =============================================================================
# Functional Similarity Computation
# =============================================================================

def compute_functional_similarity_matrix() -> Tuple[np.ndarray, List[str]]:
    """Compute pairwise functional similarity between all amino acids.

    Returns:
        similarity_matrix: 20x20 cosine similarity matrix
        amino_acids: List of AA codes in order
    """
    amino_acids = sorted(AMINO_ACID_PROFILES.keys())
    n = len(amino_acids)

    # Build feature matrix
    features = np.array([AMINO_ACID_PROFILES[aa].to_vector() for aa in amino_acids])

    # Normalize features (z-score)
    features_normalized = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    # Compute cosine similarity
    norms = np.linalg.norm(features_normalized, axis=1, keepdims=True)
    features_unit = features_normalized / (norms + 1e-8)
    similarity_matrix = features_unit @ features_unit.T

    return similarity_matrix, amino_acids


def compute_functional_distance_matrix() -> Tuple[np.ndarray, List[str]]:
    """Compute pairwise functional distance between all amino acids.

    Returns:
        distance_matrix: 20x20 Euclidean distance matrix (normalized features)
        amino_acids: List of AA codes in order
    """
    amino_acids = sorted(AMINO_ACID_PROFILES.keys())

    # Build feature matrix
    features = np.array([AMINO_ACID_PROFILES[aa].to_vector() for aa in amino_acids])

    # Normalize features
    features_normalized = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    # Compute pairwise distances
    n = len(amino_acids)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = np.linalg.norm(
                features_normalized[i] - features_normalized[j]
            )

    return distance_matrix, amino_acids


def get_catalytic_amino_acids() -> List[str]:
    """Get amino acids commonly found in catalytic sites."""
    return [aa for aa, profile in AMINO_ACID_PROFILES.items()
            if profile.catalytic_propensity > 0.10]


def get_structural_amino_acids() -> List[str]:
    """Get amino acids with high structural propensities."""
    return [aa for aa, profile in AMINO_ACID_PROFILES.items()
            if profile.helix_propensity > 1.2 or profile.sheet_propensity > 1.3]


def get_binding_amino_acids() -> List[str]:
    """Get amino acids with high binding propensities."""
    return [aa for aa, profile in AMINO_ACID_PROFILES.items()
            if profile.dna_binding > 0.10 or profile.metal_binding > 0.15]


# =============================================================================
# Functional Clusters
# =============================================================================

def cluster_by_function(n_clusters: int = 5) -> Dict[str, List[str]]:
    """Cluster amino acids by functional similarity.

    Returns:
        clusters: Dictionary mapping cluster name to list of AAs
    """
    from scipy.cluster.hierarchy import linkage, fcluster

    distance_matrix, amino_acids = compute_functional_distance_matrix()

    # Convert to condensed form for linkage
    from scipy.spatial.distance import squareform
    condensed = squareform(distance_matrix)

    # Hierarchical clustering
    Z = linkage(condensed, method='ward')
    labels = fcluster(Z, n_clusters, criterion='maxclust')

    # Group by cluster
    clusters = {}
    for aa, label in zip(amino_acids, labels):
        cluster_name = f"cluster_{label}"
        if cluster_name not in clusters:
            clusters[cluster_name] = []
        clusters[cluster_name].append(aa)

    # Name clusters by dominant property
    named_clusters = {}
    for cluster_name, members in clusters.items():
        # Compute average properties
        avg_hydro = np.mean([AMINO_ACID_PROFILES[aa].hydrophobicity for aa in members])
        avg_charge = np.mean([AMINO_ACID_PROFILES[aa].charge_ph7 for aa in members])
        avg_catalytic = np.mean([AMINO_ACID_PROFILES[aa].catalytic_propensity for aa in members])

        if avg_charge > 0.3:
            name = "positive_charged"
        elif avg_charge < -0.3:
            name = "negative_charged"
        elif avg_hydro > 2.0:
            name = "hydrophobic"
        elif avg_catalytic > 0.12:
            name = "catalytic"
        else:
            name = "polar_neutral"

        # Avoid duplicates
        base_name = name
        counter = 1
        while name in named_clusters:
            name = f"{base_name}_{counter}"
            counter += 1

        named_clusters[name] = members

    return named_clusters


# =============================================================================
# Export Functions
# =============================================================================

def export_profiles_to_json(output_path: str):
    """Export all profiles to JSON for documentation."""
    data = {}
    for aa, profile in AMINO_ACID_PROFILES.items():
        data[aa] = {
            'name': profile.name,
            'physicochemical': {
                'hydrophobicity': profile.hydrophobicity,
                'charge': profile.charge_ph7,
                'molecular_weight': profile.molecular_weight,
                'volume': profile.volume,
                'polarity': profile.polarity,
                'aromaticity': profile.aromaticity,
            },
            'catalytic': {
                'catalytic_propensity': profile.catalytic_propensity,
                'nucleophile': profile.nucleophile_propensity,
                'acid_base': profile.acid_base_propensity,
                'metal_binding': profile.metal_binding,
            },
            'enzyme_class': {
                'EC1_oxidoreductase': profile.ec1_oxidoreductase,
                'EC2_transferase': profile.ec2_transferase,
                'EC3_hydrolase': profile.ec3_hydrolase,
                'EC4_lyase': profile.ec4_lyase,
                'EC5_isomerase': profile.ec5_isomerase,
                'EC6_ligase': profile.ec6_ligase,
            },
            'structural': {
                'helix': profile.helix_propensity,
                'sheet': profile.sheet_propensity,
                'turn': profile.turn_propensity,
                'disorder': profile.disorder_propensity,
            },
            'binding': {
                'DNA': profile.dna_binding,
                'RNA': profile.rna_binding,
                'ATP': profile.atp_binding,
                'cofactor': profile.cofactor_binding,
            },
        }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AMINO ACID FUNCTIONAL PROFILES")
    print("=" * 60)

    # Compute similarity matrix
    print("\n1. Computing functional similarity matrix...")
    similarity, amino_acids = compute_functional_similarity_matrix()

    print(f"   Amino acids: {', '.join(amino_acids)}")
    print(f"   Matrix shape: {similarity.shape}")

    # Show most similar pairs
    print("\n2. Most functionally similar pairs:")
    pairs = []
    for i in range(len(amino_acids)):
        for j in range(i+1, len(amino_acids)):
            pairs.append((amino_acids[i], amino_acids[j], similarity[i, j]))

    pairs.sort(key=lambda x: -x[2])
    for aa1, aa2, sim in pairs[:10]:
        print(f"   {aa1}-{aa2}: {sim:.3f}")

    # Functional clusters
    print("\n3. Functional clusters:")
    clusters = cluster_by_function(5)
    for name, members in clusters.items():
        print(f"   {name}: {', '.join(members)}")

    # Catalytic amino acids
    print("\n4. Catalytic amino acids:")
    catalytic = get_catalytic_amino_acids()
    print(f"   {', '.join(catalytic)}")

    # Export
    output_dir = Path(__file__).parent / 'data'
    output_dir.mkdir(exist_ok=True)
    export_profiles_to_json(str(output_dir / 'amino_acid_profiles.json'))
    print(f"\n5. Profiles exported to: {output_dir / 'amino_acid_profiles.json'}")
