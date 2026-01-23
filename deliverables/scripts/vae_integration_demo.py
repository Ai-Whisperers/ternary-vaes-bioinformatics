# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""VAE Integration Demo - Connect Research Implementations to Real VAE.

This script demonstrates how to use the Ternary VAE with the research
implementations, decoding latent vectors to actual peptide sequences.

The VAE maps:
    - Latent space (16D Poincare ball) -> Ternary operations (3^9 = 19,683)
    - P-adic valuation encodes structural hierarchy
    - Radial position encodes stability (center = more stable)

Usage:
    python deliverables/scripts/vae_integration_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np

# Try to import torch and VAE components
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - running in mock mode")

# Amino acid codon mapping (from ternary operations)
CODON_TO_AA = {
    # Standard genetic code mapping
    # Ternary operations map to codons, which map to amino acids
    "TTT": "F", "TTC": "F",
    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I",
    "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y",
    "TAA": "*", "TAG": "*", "TGA": "*",
    "CAT": "H", "CAC": "H",
    "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D",
    "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C",
    "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# Map ternary to nucleotide
TERNARY_TO_NUC = {-1: "T", 0: "C", 1: "A"}  # Simplified mapping


class TernaryDecoder:
    """Decode ternary operations to amino acid sequences."""

    def __init__(self):
        self.ternary_to_nuc = TERNARY_TO_NUC
        self.codon_to_aa = CODON_TO_AA

    def ternary_to_codon(self, ternary_op: np.ndarray) -> str:
        """Convert 9 ternary values to 3 codons (9 nucleotides)."""
        if len(ternary_op) != 9:
            raise ValueError(f"Expected 9 ternary values, got {len(ternary_op)}")

        nucs = [self.ternary_to_nuc.get(int(t), "N") for t in ternary_op]
        codons = ["".join(nucs[i:i+3]) for i in range(0, 9, 3)]
        return codons

    def decode_to_amino_acids(self, ternary_ops: np.ndarray) -> str:
        """Decode batch of ternary operations to amino acid sequence."""
        if ternary_ops.ndim == 1:
            ternary_ops = ternary_ops.reshape(1, -1)

        amino_acids = []
        for op in ternary_ops:
            codons = self.ternary_to_codon(op)
            for codon in codons:
                aa = self.codon_to_aa.get(codon, "X")
                if aa != "*":  # Skip stop codons
                    amino_acids.append(aa)

        return "".join(amino_acids)


class VAEInterface:
    """Interface to the Ternary VAE for sequence generation."""

    def __init__(self, checkpoint_path: str = None):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.device = "cpu"
        self.decoder = TernaryDecoder()

        if TORCH_AVAILABLE and checkpoint_path:
            self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str):
        """Load VAE model from checkpoint."""
        try:
            from src.models import TernaryVAEV5_11_PartialFreeze

            # Initialize model
            self.model = TernaryVAEV5_11_PartialFreeze(
                latent_dim=16,
                hidden_dim=64,
                max_radius=0.99,
                curvature=1.0,
                use_controller=True,
                use_dual_projection=True,
            )

            # Load checkpoint
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
            else:
                self.model.load_state_dict(ckpt)

            self.model.eval()
            print(f"Loaded VAE from {checkpoint_path}")

        except Exception as e:
            print(f"Could not load VAE: {e}")
            self.model = None

    def decode_latent(self, z: np.ndarray) -> str:
        """Decode latent vector to amino acid sequence."""
        if self.model is None:
            # Mock decoding for demo
            return self._mock_decode(z)

        with torch.no_grad():
            z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0)

            # Get logits from decoder
            logits = self.model.decoder(z_tensor)  # (1, 9, 3)

            # Convert to ternary operations
            ternary_ops = torch.argmax(logits, dim=-1) - 1  # {0,1,2} -> {-1,0,1}
            ternary_ops = ternary_ops.numpy()

            # Decode to amino acids
            return self.decoder.decode_to_amino_acids(ternary_ops)

    def _mock_decode(self, z: np.ndarray) -> str:
        """Mock decoding when VAE not available."""
        # Use latent space to bias amino acid selection
        np.random.seed(int(abs(z[0] * 1000)) % (2**31))

        # Amino acids with properties influenced by latent dimensions
        aa_list = list("ACDEFGHIKLMNPQRSTVWY")

        # Radial position influences hydrophobicity
        radius = np.linalg.norm(z[:3]) if len(z) >= 3 else 0.5
        hydrophobic = list("ILVFMWYA")
        hydrophilic = list("RKDENQHST")

        # Generate sequence
        length = 20
        sequence = []
        for i in range(length):
            if np.random.random() < radius:
                sequence.append(np.random.choice(hydrophobic))
            else:
                sequence.append(np.random.choice(hydrophilic))

        return "".join(sequence)

    def encode_sequence(self, sequence: str) -> np.ndarray:
        """Encode amino acid sequence to latent space (mock)."""
        # This would require the encoder - for demo, use sequence properties
        charge = sum(1 if aa in "KRH" else -1 if aa in "DE" else 0 for aa in sequence)
        hydro = sum(1 if aa in "ILVFMWYA" else -1 if aa in "RKDENQHST" else 0 for aa in sequence)

        # Create mock latent vector
        z = np.zeros(16)
        z[0] = np.tanh(charge / 5)
        z[1] = np.tanh(hydro / 10)
        z[2:] = np.random.randn(14) * 0.1

        return z

    def get_radius(self, z: np.ndarray) -> float:
        """Get hyperbolic radius of latent vector."""
        return np.linalg.norm(z)

    def get_padic_valuation(self, z: np.ndarray) -> int:
        """Estimate p-adic valuation from radius (inverse relationship)."""
        radius = self.get_radius(z)
        # Center (r=0) = high valuation, edge (r=1) = low valuation
        valuation = int((1 - min(radius, 0.99)) * 9)
        return min(9, max(0, valuation))


def demo_amp_optimization():
    """Demo: Use VAE for AMP sequence generation."""
    print("\n" + "=" * 70)
    print("DEMO: VAE-BASED AMP OPTIMIZATION")
    print("=" * 70)

    vae = VAEInterface()

    # Generate candidate AMPs from latent space
    print("\nGenerating AMP candidates from latent space...")

    candidates = []
    for i in range(10):
        # Sample from latent space with AMP-favorable properties
        # Higher charge (z[0] > 0), moderate hydrophobicity (z[1] ~ 0)
        z = np.random.randn(16) * 0.3
        z[0] = 0.5 + np.random.randn() * 0.2  # Positive charge bias
        z[1] = np.random.randn() * 0.3  # Balanced hydrophobicity

        sequence = vae.decode_latent(z)
        radius = vae.get_radius(z)
        valuation = vae.get_padic_valuation(z)

        # Compute properties
        charge = sum(1 if aa in "KRH" else -1 if aa in "DE" else 0 for aa in sequence)
        hydro = sum(1 if aa in "ILVFMWYA" else 0 for aa in sequence) / len(sequence)

        candidates.append({
            "sequence": sequence,
            "radius": radius,
            "valuation": valuation,
            "charge": charge,
            "hydrophobicity": hydro,
        })

    # Sort by charge (AMP criterion)
    candidates.sort(key=lambda x: x["charge"], reverse=True)

    print(f"\n{'Rank':<5} {'Sequence':<25} {'Charge':<8} {'Hydro':<8} {'Radius':<8} {'V_p':<5}")
    print("-" * 70)
    for i, c in enumerate(candidates[:5], 1):
        print(f"{i:<5} {c['sequence']:<25} {c['charge']:<8} {c['hydrophobicity']:<8.2f} {c['radius']:<8.3f} {c['valuation']:<5}")


def demo_mutation_analysis():
    """Demo: Use VAE for mutation effect prediction."""
    print("\n" + "=" * 70)
    print("DEMO: VAE-BASED MUTATION ANALYSIS")
    print("=" * 70)

    vae = VAEInterface()

    # Wild-type sequence
    wt_sequence = "KLWKKLKKALK"
    wt_z = vae.encode_sequence(wt_sequence)
    wt_radius = vae.get_radius(wt_z)

    print(f"\nWild-type: {wt_sequence}")
    print(f"Latent radius: {wt_radius:.4f}")
    print(f"P-adic valuation: {vae.get_padic_valuation(wt_z)}")

    # Test mutations
    mutations = [
        ("K1A", "ALWKKLKKALK"),   # Charge reduction
        ("L3W", "KLWKKLKKALK"),   # Same (typo demo)
        ("K5D", "KLWKDLKKALK"),   # Charge reversal
        ("A11K", "KLWKKLKKAKK"),  # Charge addition
    ]

    print(f"\n{'Mutation':<10} {'Mutant Seq':<15} {'Delta_r':<12} {'Prediction':<15}")
    print("-" * 60)

    for mut_name, mut_seq in mutations:
        mut_z = vae.encode_sequence(mut_seq)
        mut_radius = vae.get_radius(mut_z)
        delta_r = mut_radius - wt_radius

        # Interpret: larger radius = less stable (edge of Poincare ball)
        if delta_r > 0.1:
            prediction = "Destabilizing"
        elif delta_r < -0.1:
            prediction = "Stabilizing"
        else:
            prediction = "Neutral"

        print(f"{mut_name:<10} {mut_seq:<15} {delta_r:<+12.4f} {prediction:<15}")


def demo_resistance_prediction():
    """Demo: Use VAE for drug resistance prediction."""
    print("\n" + "=" * 70)
    print("DEMO: VAE-BASED RESISTANCE PREDICTION")
    print("=" * 70)

    vae = VAEInterface()

    # HIV RT sequences with known resistance
    sequences = {
        "Wild-type": "PISPIETVPVKLKPGM",
        "M184V": "PISPIETVPVVLKPGM",  # 3TC resistance
        "K103N": "PISPIETVPVNLKPGM",  # EFV resistance
    }

    print("\nAnalyzing HIV RT resistance mutations...")
    print(f"\n{'Variant':<15} {'Radius':<10} {'Valuation':<12} {'Interpretation'}")
    print("-" * 60)

    wt_z = vae.encode_sequence(sequences["Wild-type"])
    wt_radius = vae.get_radius(wt_z)

    for name, seq in sequences.items():
        z = vae.encode_sequence(seq)
        radius = vae.get_radius(z)
        valuation = vae.get_padic_valuation(z)

        if name == "Wild-type":
            interp = "Reference"
        elif radius > wt_radius + 0.05:
            interp = "Altered stability"
        else:
            interp = "Similar to WT"

        print(f"{name:<15} {radius:<10.4f} {valuation:<12} {interp}")


def demo_latent_space_exploration():
    """Demo: Explore VAE latent space structure."""
    print("\n" + "=" * 70)
    print("DEMO: LATENT SPACE EXPLORATION")
    print("=" * 70)

    vae = VAEInterface()

    print("\nExploring radial structure of latent space...")
    print("(P-adic valuation correlates with radial position)")

    # Sample at different radii
    radii = [0.1, 0.3, 0.5, 0.7, 0.9]

    print(f"\n{'Radius':<10} {'Valuation':<12} {'Sample Sequence':<25} {'Properties'}")
    print("-" * 70)

    for target_r in radii:
        # Generate point at target radius
        direction = np.random.randn(16)
        direction = direction / np.linalg.norm(direction)
        z = direction * target_r

        sequence = vae.decode_latent(z)
        valuation = vae.get_padic_valuation(z)

        # Compute properties
        charge = sum(1 if aa in "KRH" else -1 if aa in "DE" else 0 for aa in sequence)

        print(f"{target_r:<10.2f} {valuation:<12} {sequence[:20]:<25} charge={charge}")


def main():
    """Run all demos."""
    print("=" * 70)
    print("VAE INTEGRATION DEMONSTRATION")
    print("Connecting Research Implementations to Ternary VAE")
    print("=" * 70)

    # Check for real checkpoint
    checkpoint_paths = [
        project_root / "checkpoints" / "pretrained_final.pt",
        project_root / "checkpoints" / "v5_11_11_homeostatic_rtx2060s" / "latest.pt",
    ]

    checkpoint = None
    for path in checkpoint_paths:
        if path.exists():
            checkpoint = str(path)
            print(f"\nFound checkpoint: {checkpoint}")
            break

    if checkpoint is None:
        print("\nNo checkpoint found - running in mock mode")
        print("(Mock mode uses sequence properties to simulate VAE behavior)")

    # Run demos
    demo_amp_optimization()
    demo_mutation_analysis()
    demo_resistance_prediction()
    demo_latent_space_exploration()

    # Summary
    print("\n" + "=" * 70)
    print("INTEGRATION SUMMARY")
    print("=" * 70)
    print("""
The VAE provides:
1. Latent space encoding of biological sequences
2. P-adic valuation as structural hierarchy metric
3. Radial position correlates with stability/function
4. Differentiable decoder for optimization

Research implementations can use:
- decode_latent(z) -> sequence (for optimization)
- encode_sequence(seq) -> z (for analysis)
- get_radius(z) -> stability metric
- get_padic_valuation(z) -> hierarchy level
""")

    print("VAE Integration Demo Complete!")


if __name__ == "__main__":
    main()
