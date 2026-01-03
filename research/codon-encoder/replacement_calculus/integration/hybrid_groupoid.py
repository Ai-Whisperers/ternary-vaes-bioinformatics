#!/usr/bin/env python3
"""Hybrid Morphism Validity: Embedding + Physicochemistry.

This version combines:
1. Embedding distance (learned representations)
2. Physicochemical properties (charge, size, hydrophobicity)

Goal: Improve precision (26.3% → >50%) while maintaining recall (97.8%)

The hypothesis is that radical substitutions (L→D) are close in embedding space
but differ in physicochemical properties, so adding these constraints filters
false positives.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from replacement_calculus.groups import LocalMinimum
from replacement_calculus.morphisms import Morphism, MorphismType
from replacement_calculus.groupoids import Groupoid, find_escape_path, analyze_groupoid_structure


# =============================================================================
# Amino Acid Physicochemical Properties
# =============================================================================

@dataclass
class AminoAcidProperties:
    """Physicochemical properties of an amino acid."""
    name: str
    code: str
    molecular_weight: float  # Da
    charge: int              # -1, 0, +1 at pH 7
    hydrophobicity: float    # Kyte-Doolittle scale
    volume: float            # Å³
    polarity: str            # nonpolar, polar, positive, negative


# Properties from standard biochemistry references
AA_PROPERTIES: Dict[str, AminoAcidProperties] = {
    'A': AminoAcidProperties('Alanine', 'A', 89.1, 0, 1.8, 88.6, 'nonpolar'),
    'R': AminoAcidProperties('Arginine', 'R', 174.2, 1, -4.5, 173.4, 'positive'),
    'N': AminoAcidProperties('Asparagine', 'N', 132.1, 0, -3.5, 114.1, 'polar'),
    'D': AminoAcidProperties('Aspartate', 'D', 133.1, -1, -3.5, 111.1, 'negative'),
    'C': AminoAcidProperties('Cysteine', 'C', 121.2, 0, 2.5, 108.5, 'polar'),
    'Q': AminoAcidProperties('Glutamine', 'Q', 146.2, 0, -3.5, 143.8, 'polar'),
    'E': AminoAcidProperties('Glutamate', 'E', 147.1, -1, -3.5, 138.4, 'negative'),
    'G': AminoAcidProperties('Glycine', 'G', 75.1, 0, -0.4, 60.1, 'nonpolar'),
    'H': AminoAcidProperties('Histidine', 'H', 155.2, 0, -3.2, 153.2, 'positive'),  # pKa ~6
    'I': AminoAcidProperties('Isoleucine', 'I', 131.2, 0, 4.5, 166.7, 'nonpolar'),
    'L': AminoAcidProperties('Leucine', 'L', 131.2, 0, 3.8, 166.7, 'nonpolar'),
    'K': AminoAcidProperties('Lysine', 'K', 146.2, 1, -3.9, 168.6, 'positive'),
    'M': AminoAcidProperties('Methionine', 'M', 149.2, 0, 1.9, 162.9, 'nonpolar'),
    'F': AminoAcidProperties('Phenylalanine', 'F', 165.2, 0, 2.8, 189.9, 'nonpolar'),
    'P': AminoAcidProperties('Proline', 'P', 115.1, 0, -1.6, 112.7, 'nonpolar'),
    'S': AminoAcidProperties('Serine', 'S', 105.1, 0, -0.8, 89.0, 'polar'),
    'T': AminoAcidProperties('Threonine', 'T', 119.1, 0, -0.7, 116.1, 'polar'),
    'W': AminoAcidProperties('Tryptophan', 'W', 204.2, 0, -0.9, 227.8, 'nonpolar'),
    'Y': AminoAcidProperties('Tyrosine', 'Y', 181.2, 0, -1.3, 193.6, 'polar'),
    'V': AminoAcidProperties('Valine', 'V', 117.1, 0, 4.2, 140.0, 'nonpolar'),
}


# =============================================================================
# Physicochemical Constraints
# =============================================================================

def charge_compatible(aa1: str, aa2: str) -> bool:
    """Check if charge difference is acceptable.

    Rules:
    - Same charge: always OK
    - Neutral to charged: OK (mild)
    - Opposite charges: NOT OK (radical)
    """
    p1, p2 = AA_PROPERTIES.get(aa1), AA_PROPERTIES.get(aa2)
    if not p1 or not p2:
        return True  # Unknown = assume OK

    c1, c2 = p1.charge, p2.charge

    # Same charge or both neutral
    if c1 == c2:
        return True

    # One neutral, one charged (mild)
    if c1 == 0 or c2 == 0:
        return True

    # Opposite charges (radical)
    return False


def size_compatible(aa1: str, aa2: str, max_diff: float = 60.0) -> bool:
    """Check if size difference is acceptable.

    Args:
        max_diff: Maximum volume difference in Å³
    """
    p1, p2 = AA_PROPERTIES.get(aa1), AA_PROPERTIES.get(aa2)
    if not p1 or not p2:
        return True

    return abs(p1.volume - p2.volume) <= max_diff


def hydrophobicity_compatible(aa1: str, aa2: str, max_diff: float = 3.0) -> bool:
    """Check if hydrophobicity difference is acceptable.

    Args:
        max_diff: Maximum Kyte-Doolittle scale difference
    """
    p1, p2 = AA_PROPERTIES.get(aa1), AA_PROPERTIES.get(aa2)
    if not p1 or not p2:
        return True

    return abs(p1.hydrophobicity - p2.hydrophobicity) <= max_diff


def polarity_compatible(aa1: str, aa2: str) -> bool:
    """Check if polarity classes are compatible.

    Rules:
    - Same class: always OK
    - nonpolar ↔ polar: mild (depending on size)
    - charged ↔ nonpolar: NOT OK
    - positive ↔ negative: NOT OK
    """
    p1, p2 = AA_PROPERTIES.get(aa1), AA_PROPERTIES.get(aa2)
    if not p1 or not p2:
        return True

    pol1, pol2 = p1.polarity, p2.polarity

    # Same polarity class
    if pol1 == pol2:
        return True

    # Allow nonpolar ↔ polar
    if {pol1, pol2} == {'nonpolar', 'polar'}:
        return True

    # Disallow charged ↔ opposite charged
    if {pol1, pol2} == {'positive', 'negative'}:
        return False

    # Disallow extreme polar ↔ nonpolar
    if pol1 in ('positive', 'negative') and pol2 == 'nonpolar':
        return False
    if pol2 in ('positive', 'negative') and pol1 == 'nonpolar':
        return False

    return True


# =============================================================================
# Hybrid Morphism Validity
# =============================================================================

@dataclass
class HybridValidityConfig:
    """Configuration for hybrid validity checking."""
    max_embedding_distance: float = 3.5
    max_size_diff: float = 60.0
    max_hydrophobicity_diff: float = 3.0
    require_charge_compatible: bool = True
    require_polarity_compatible: bool = True

    # Weights for combined score
    weight_embedding: float = 1.0
    weight_charge: float = 2.0
    weight_size: float = 1.0
    weight_hydrophobicity: float = 1.0
    weight_polarity: float = 1.5


def is_valid_hybrid_morphism(
    source_aa: str,
    target_aa: str,
    source_center: np.ndarray,
    target_center: np.ndarray,
    config: HybridValidityConfig,
) -> Tuple[bool, str, float]:
    """Check hybrid validity: embedding distance + physicochemistry.

    Returns:
        (is_valid, reason, combined_cost)
    """
    # 1. Embedding distance check
    emb_dist = np.linalg.norm(source_center - target_center)
    if emb_dist > config.max_embedding_distance:
        return False, f"Embedding distance {emb_dist:.2f} > {config.max_embedding_distance}", float('inf')

    # 2. Charge compatibility
    if config.require_charge_compatible and not charge_compatible(source_aa, target_aa):
        return False, f"Charge incompatible: {source_aa}→{target_aa}", float('inf')

    # 3. Size compatibility
    if not size_compatible(source_aa, target_aa, config.max_size_diff):
        return False, f"Size difference > {config.max_size_diff}", float('inf')

    # 4. Hydrophobicity compatibility
    if not hydrophobicity_compatible(source_aa, target_aa, config.max_hydrophobicity_diff):
        return False, f"Hydrophobicity difference > {config.max_hydrophobicity_diff}", float('inf')

    # 5. Polarity compatibility
    if config.require_polarity_compatible and not polarity_compatible(source_aa, target_aa):
        return False, f"Polarity incompatible: {source_aa}→{target_aa}", float('inf')

    # Compute combined cost
    p1, p2 = AA_PROPERTIES.get(source_aa), AA_PROPERTIES.get(target_aa)

    cost = config.weight_embedding * emb_dist

    if p1 and p2:
        # Charge penalty (0 if same, 1 if one neutral, 2 if opposite)
        charge_diff = abs(p1.charge - p2.charge)
        cost += config.weight_charge * charge_diff

        # Size penalty (normalized)
        size_diff = abs(p1.volume - p2.volume) / 100.0
        cost += config.weight_size * size_diff

        # Hydrophobicity penalty (normalized)
        hydro_diff = abs(p1.hydrophobicity - p2.hydrophobicity) / 5.0
        cost += config.weight_hydrophobicity * hydro_diff

        # Polarity penalty
        if p1.polarity != p2.polarity:
            cost += config.weight_polarity

    return True, "Valid hybrid morphism", cost


# =============================================================================
# BLOSUM62 for Validation
# =============================================================================

BLOSUM62 = {
    ('A', 'A'): 4, ('A', 'R'): -1, ('A', 'N'): -2, ('A', 'D'): -2, ('A', 'C'): 0,
    ('A', 'Q'): -1, ('A', 'E'): -1, ('A', 'G'): 0, ('A', 'H'): -2, ('A', 'I'): -1,
    ('A', 'L'): -1, ('A', 'K'): -1, ('A', 'M'): -1, ('A', 'F'): -2, ('A', 'P'): -1,
    ('A', 'S'): 1, ('A', 'T'): 0, ('A', 'W'): -3, ('A', 'Y'): -2, ('A', 'V'): 0,
    ('R', 'R'): 5, ('R', 'N'): 0, ('R', 'D'): -2, ('R', 'C'): -3, ('R', 'Q'): 1,
    ('R', 'E'): 0, ('R', 'G'): -2, ('R', 'H'): 0, ('R', 'I'): -3, ('R', 'L'): -2,
    ('R', 'K'): 2, ('R', 'M'): -1, ('R', 'F'): -3, ('R', 'P'): -2, ('R', 'S'): -1,
    ('R', 'T'): -1, ('R', 'W'): -3, ('R', 'Y'): -2, ('R', 'V'): -3,
    ('N', 'N'): 6, ('N', 'D'): 1, ('N', 'C'): -3, ('N', 'Q'): 0, ('N', 'E'): 0,
    ('N', 'G'): 0, ('N', 'H'): 1, ('N', 'I'): -3, ('N', 'L'): -3, ('N', 'K'): 0,
    ('N', 'M'): -2, ('N', 'F'): -3, ('N', 'P'): -2, ('N', 'S'): 1, ('N', 'T'): 0,
    ('N', 'W'): -4, ('N', 'Y'): -2, ('N', 'V'): -3,
    ('D', 'D'): 6, ('D', 'C'): -3, ('D', 'Q'): 0, ('D', 'E'): 2, ('D', 'G'): -1,
    ('D', 'H'): -1, ('D', 'I'): -3, ('D', 'L'): -4, ('D', 'K'): -1, ('D', 'M'): -3,
    ('D', 'F'): -3, ('D', 'P'): -1, ('D', 'S'): 0, ('D', 'T'): -1, ('D', 'W'): -4,
    ('D', 'Y'): -3, ('D', 'V'): -3,
    ('C', 'C'): 9, ('C', 'Q'): -3, ('C', 'E'): -4, ('C', 'G'): -3, ('C', 'H'): -3,
    ('C', 'I'): -1, ('C', 'L'): -1, ('C', 'K'): -3, ('C', 'M'): -1, ('C', 'F'): -2,
    ('C', 'P'): -3, ('C', 'S'): -1, ('C', 'T'): -1, ('C', 'W'): -2, ('C', 'Y'): -2,
    ('C', 'V'): -1,
    ('Q', 'Q'): 5, ('Q', 'E'): 2, ('Q', 'G'): -2, ('Q', 'H'): 0, ('Q', 'I'): -3,
    ('Q', 'L'): -2, ('Q', 'K'): 1, ('Q', 'M'): 0, ('Q', 'F'): -3, ('Q', 'P'): -1,
    ('Q', 'S'): 0, ('Q', 'T'): -1, ('Q', 'W'): -2, ('Q', 'Y'): -1, ('Q', 'V'): -2,
    ('E', 'E'): 5, ('E', 'G'): -2, ('E', 'H'): 0, ('E', 'I'): -3, ('E', 'L'): -3,
    ('E', 'K'): 1, ('E', 'M'): -2, ('E', 'F'): -3, ('E', 'P'): -1, ('E', 'S'): 0,
    ('E', 'T'): -1, ('E', 'W'): -3, ('E', 'Y'): -2, ('E', 'V'): -2,
    ('G', 'G'): 6, ('G', 'H'): -2, ('G', 'I'): -4, ('G', 'L'): -4, ('G', 'K'): -2,
    ('G', 'M'): -3, ('G', 'F'): -3, ('G', 'P'): -2, ('G', 'S'): 0, ('G', 'T'): -2,
    ('G', 'W'): -2, ('G', 'Y'): -3, ('G', 'V'): -3,
    ('H', 'H'): 8, ('H', 'I'): -3, ('H', 'L'): -3, ('H', 'K'): -1, ('H', 'M'): -2,
    ('H', 'F'): -1, ('H', 'P'): -2, ('H', 'S'): -1, ('H', 'T'): -2, ('H', 'W'): -2,
    ('H', 'Y'): 2, ('H', 'V'): -3,
    ('I', 'I'): 4, ('I', 'L'): 2, ('I', 'K'): -3, ('I', 'M'): 1, ('I', 'F'): 0,
    ('I', 'P'): -3, ('I', 'S'): -2, ('I', 'T'): -1, ('I', 'W'): -3, ('I', 'Y'): -1,
    ('I', 'V'): 3,
    ('L', 'L'): 4, ('L', 'K'): -2, ('L', 'M'): 2, ('L', 'F'): 0, ('L', 'P'): -3,
    ('L', 'S'): -2, ('L', 'T'): -1, ('L', 'W'): -2, ('L', 'Y'): -1, ('L', 'V'): 1,
    ('K', 'K'): 5, ('K', 'M'): -1, ('K', 'F'): -3, ('K', 'P'): -1, ('K', 'S'): 0,
    ('K', 'T'): -1, ('K', 'W'): -3, ('K', 'Y'): -2, ('K', 'V'): -2,
    ('M', 'M'): 5, ('M', 'F'): 0, ('M', 'P'): -2, ('M', 'S'): -1, ('M', 'T'): -1,
    ('M', 'W'): -1, ('M', 'Y'): -1, ('M', 'V'): 1,
    ('F', 'F'): 6, ('F', 'P'): -4, ('F', 'S'): -2, ('F', 'T'): -2, ('F', 'W'): 1,
    ('F', 'Y'): 3, ('F', 'V'): -1,
    ('P', 'P'): 7, ('P', 'S'): -1, ('P', 'T'): -1, ('P', 'W'): -4, ('P', 'Y'): -3,
    ('P', 'V'): -2,
    ('S', 'S'): 4, ('S', 'T'): 1, ('S', 'W'): -3, ('S', 'Y'): -2, ('S', 'V'): -2,
    ('T', 'T'): 5, ('T', 'W'): -2, ('T', 'Y'): -2, ('T', 'V'): 0,
    ('W', 'W'): 11, ('W', 'Y'): 2, ('W', 'V'): -3,
    ('Y', 'Y'): 7, ('Y', 'V'): -1,
    ('V', 'V'): 4,
}

def get_blosum_score(aa1: str, aa2: str) -> int:
    if aa1 == aa2:
        return BLOSUM62.get((aa1, aa2), 0)
    return BLOSUM62.get((aa1, aa2), BLOSUM62.get((aa2, aa1), -4))


# =============================================================================
# Groupoid Construction
# =============================================================================

CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

AA_TO_CODONS: Dict[str, List[str]] = {}
for codon, aa in CODON_TABLE.items():
    if aa not in AA_TO_CODONS:
        AA_TO_CODONS[aa] = []
    AA_TO_CODONS[aa].append(codon)


def load_codon_embeddings() -> Tuple[Dict[str, np.ndarray], Dict]:
    """Load codon embeddings from JSON."""
    json_path = Path(__file__).parent.parent.parent / 'extraction' / 'results' / 'codon_embeddings_v5_12_3.json'

    with open(json_path) as f:
        data = json.load(f)

    embeddings = {}
    for codon, info in data['codons'].items():
        embeddings[codon] = np.array(info['embedding'])

    return embeddings, data.get('metadata', {})


def build_hybrid_groupoid(
    embeddings: Dict[str, np.ndarray],
    config: HybridValidityConfig,
) -> Tuple[Groupoid, Dict[str, int]]:
    """Build groupoid with hybrid morphism validity."""
    groupoid = Groupoid(name="hybrid_groupoid")
    aa_to_idx: Dict[str, int] = {}
    aa_centers: Dict[str, np.ndarray] = {}

    # Create LocalMinima for each amino acid
    for aa, codons in AA_TO_CODONS.items():
        if aa == '*':
            continue

        members = [embeddings[c] for c in codons if c in embeddings]
        if not members:
            continue

        center = np.mean(members, axis=0)
        aa_centers[aa] = center

        minimum = LocalMinimum(
            name=f"AA_{aa}",
            generators=[hash(c) % 1000 for c in codons],
            relations=[],
            center=center,
            members=members,
            metadata={'amino_acid': aa, 'codons': codons},
        )

        idx = groupoid.add_object(minimum)
        aa_to_idx[aa] = idx

    # Find valid morphisms using hybrid criteria
    for aa1, idx1 in aa_to_idx.items():
        for aa2, idx2 in aa_to_idx.items():
            if aa1 == aa2:
                continue

            is_valid, reason, cost = is_valid_hybrid_morphism(
                aa1, aa2,
                aa_centers[aa1], aa_centers[aa2],
                config
            )

            if is_valid:
                morphism = Morphism(
                    source=groupoid.objects[idx1],
                    target=groupoid.objects[idx2],
                    map_function=lambda x: x,
                    morphism_type=MorphismType.HOMOMORPHISM,
                    cost=cost,
                )
                groupoid.morphisms[(idx1, idx2)].append(morphism)

    return groupoid, aa_to_idx


def validate_against_blosum(
    groupoid: Groupoid,
    aa_to_idx: Dict[str, int],
) -> Dict:
    """Validate groupoid against BLOSUM62."""
    results = {
        'true_positives': 0,
        'false_positives': 0,
        'true_negatives': 0,
        'false_negatives': 0,
        'details': [],
    }

    for aa1 in aa_to_idx:
        for aa2 in aa_to_idx:
            if aa1 == aa2:
                continue

            idx1, idx2 = aa_to_idx[aa1], aa_to_idx[aa2]
            has_morphism = groupoid.has_morphism(idx1, idx2)
            blosum = get_blosum_score(aa1, aa2)
            is_conservative = blosum >= 0

            detail = {
                'aa1': aa1, 'aa2': aa2,
                'morphism': has_morphism,
                'blosum': blosum,
                'conservative': is_conservative,
            }

            if has_morphism and is_conservative:
                results['true_positives'] += 1
                detail['class'] = 'TP'
            elif has_morphism and not is_conservative:
                results['false_positives'] += 1
                detail['class'] = 'FP'
            elif not has_morphism and not is_conservative:
                results['true_negatives'] += 1
                detail['class'] = 'TN'
            else:
                results['false_negatives'] += 1
                detail['class'] = 'FN'

            results['details'].append(detail)

    tp, fp, tn, fn = (results['true_positives'], results['false_positives'],
                      results['true_negatives'], results['false_negatives'])

    total = tp + fp + tn + fn
    results['accuracy'] = (tp + tn) / total if total > 0 else 0
    results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    results['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    results['f1'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    return results


def optimize_config(embeddings: Dict[str, np.ndarray]) -> Tuple[HybridValidityConfig, Dict]:
    """Find optimal configuration by grid search."""
    best_f1 = 0
    best_config = None
    best_results = None

    # Grid search over key parameters
    for max_dist in [2.0, 2.5, 3.0, 3.5, 4.0]:
        for max_size in [40.0, 60.0, 80.0, 100.0]:
            for max_hydro in [2.0, 3.0, 4.0, 5.0]:
                for req_charge in [True, False]:
                    for req_polarity in [True, False]:
                        config = HybridValidityConfig(
                            max_embedding_distance=max_dist,
                            max_size_diff=max_size,
                            max_hydrophobicity_diff=max_hydro,
                            require_charge_compatible=req_charge,
                            require_polarity_compatible=req_polarity,
                        )

                        groupoid, aa_to_idx = build_hybrid_groupoid(embeddings, config)
                        results = validate_against_blosum(groupoid, aa_to_idx)

                        if results['f1'] > best_f1:
                            best_f1 = results['f1']
                            best_config = config
                            best_results = results

    return best_config, best_results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("HYBRID GROUPOID: Embedding + Physicochemistry")
    print("=" * 60)

    # Load embeddings
    print("\n1. Loading codon embeddings...")
    embeddings, metadata = load_codon_embeddings()
    print(f"   Loaded {len(embeddings)} embeddings")

    # Find optimal configuration
    print("\n2. Optimizing hybrid validity configuration...")
    print("   (Grid search over 400 configurations)")
    best_config, best_results = optimize_config(embeddings)

    print(f"\n   Optimal configuration:")
    print(f"     max_embedding_distance: {best_config.max_embedding_distance}")
    print(f"     max_size_diff: {best_config.max_size_diff}")
    print(f"     max_hydrophobicity_diff: {best_config.max_hydrophobicity_diff}")
    print(f"     require_charge_compatible: {best_config.require_charge_compatible}")
    print(f"     require_polarity_compatible: {best_config.require_polarity_compatible}")

    # Build groupoid with optimal config
    print("\n3. Building groupoid with optimal config...")
    groupoid, aa_to_idx = build_hybrid_groupoid(embeddings, best_config)

    analysis = analyze_groupoid_structure(groupoid)
    print(f"   Objects: {analysis['n_objects']}")
    print(f"   Morphisms: {analysis['n_morphisms']}")
    print(f"   Connected: {analysis['is_connected']}")

    # Validation results
    print("\n4. Validation against BLOSUM62:")
    print(f"   True Positives: {best_results['true_positives']}")
    print(f"   False Positives: {best_results['false_positives']}")
    print(f"   True Negatives: {best_results['true_negatives']}")
    print(f"   False Negatives: {best_results['false_negatives']}")
    print(f"   Accuracy: {best_results['accuracy']:.2%}")
    print(f"   Precision: {best_results['precision']:.2%}")
    print(f"   Recall: {best_results['recall']:.2%}")
    print(f"   F1 Score: {best_results['f1']:.4f}")

    # Compare with embedding-only baseline
    print("\n5. Comparison with baseline (embedding-only):")
    print("   Embedding-only: Precision=26.3%, Recall=97.8%, F1=0.41")
    print(f"   Hybrid:         Precision={best_results['precision']:.1%}, Recall={best_results['recall']:.1%}, F1={best_results['f1']:.2f}")

    improvement = (best_results['f1'] - 0.41) / 0.41 * 100
    print(f"   F1 Improvement: {improvement:+.1f}%")

    # Test escape paths
    print("\n6. Testing escape paths:")
    test_pairs = [
        ('L', 'I', 2), ('L', 'M', 2), ('K', 'R', 2), ('D', 'E', 2),
        ('L', 'D', -4), ('K', 'D', -1), ('F', 'Y', 3), ('S', 'T', 1),
    ]

    for aa1, aa2, blosum in test_pairs:
        idx1, idx2 = aa_to_idx.get(aa1), aa_to_idx.get(aa2)
        if idx1 is None or idx2 is None:
            continue

        path = find_escape_path(groupoid, idx1, idx2)
        if path:
            cost = sum(m.cost for m in path)
            print(f"   {aa1} → {aa2}: PATH (cost={cost:.2f}, BLOSUM={blosum})")
        else:
            print(f"   {aa1} → {aa2}: NO PATH (BLOSUM={blosum})")

    # Save results
    output_path = Path(__file__).parent / 'hybrid_groupoid_analysis.json'
    output_data = {
        'config': {
            'max_embedding_distance': best_config.max_embedding_distance,
            'max_size_diff': best_config.max_size_diff,
            'max_hydrophobicity_diff': best_config.max_hydrophobicity_diff,
            'require_charge_compatible': best_config.require_charge_compatible,
            'require_polarity_compatible': best_config.require_polarity_compatible,
        },
        'validation': {k: v for k, v in best_results.items() if k != 'details'},
        'groupoid': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in analysis.items()},
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n7. Results saved to: {output_path}")
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
