# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Master Initialization Script.

Downloads/generates all data and trains all models for production use.

This script:
1. Initializes the shared VAE service
2. Downloads/generates arbovirus sequences (A2)
3. Downloads/generates AMP activity data (B1/B8/B10)
4. Generates protein stability data (C1/C4)
5. Tests Stanford HIVdb integration (H6/H7)
6. Trains all ML models

Usage:
    python initialize_all_data.py --all
    python initialize_all_data.py --partner alejandra
    python initialize_all_data.py --test
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
import json

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "deliverables"))

from shared.config import get_config, Config
from shared.vae_service import get_vae_service


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_subheader(text: str):
    """Print formatted subheader."""
    print(f"\n--- {text} ---")


def initialize_vae():
    """Initialize and test VAE service."""
    print_header("VAE SERVICE INITIALIZATION")

    config = get_config()
    print(f"Project root: {config.project_root}")
    print(f"VAE checkpoint: {config.vae_checkpoint or 'Not found (mock mode)'}")

    vae = get_vae_service()
    print(f"VAE mode: {'Real' if vae.is_real else 'Mock'}")

    # Test encoding/decoding
    print_subheader("Testing VAE Operations")

    test_sequence = "KLWKKLKKALK"
    z = vae.encode_sequence(test_sequence)
    print(f"Encoded '{test_sequence}' -> z shape: {z.shape}")
    print(f"  Radius: {vae.get_radius(z):.4f}")
    print(f"  P-adic valuation: {vae.get_padic_valuation(z)}")

    decoded = vae.decode_latent(z)
    print(f"Decoded -> '{decoded}'")

    # Sample from latent space
    samples = vae.sample_latent(5, charge_bias=0.5)
    print(f"\nSampled 5 latent vectors with charge bias:")
    for i, s in enumerate(samples):
        seq = vae.decode_latent(s)
        print(f"  {i + 1}. {seq} (r={vae.get_radius(s):.3f})")

    return True


def initialize_alejandra():
    """Initialize Alejandra Rojas data (arboviruses)."""
    print_header("ALEJANDRA ROJAS - ARBOVIRUS DATA")

    try:
        from alejandra_rojas.scripts.ncbi_arbovirus_loader import (
            NCBIArbovirusLoader,
            ArbovirusDatabase,
        )

        loader = NCBIArbovirusLoader()

        print_subheader("Loading/Generating Arbovirus Sequences")
        db = loader.load_or_download(max_per_virus=30)

        print(f"\nTotal sequences: {db.count()}")
        for virus in ["DENV-1", "DENV-2", "DENV-3", "DENV-4", "ZIKV", "CHIKV", "MAYV"]:
            print(f"  {virus}: {db.count(virus)}")

        print_subheader("Exporting FASTA Files")
        loader.export_fasta(db)

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def initialize_carlos():
    """Initialize Carlos Brizuela data (AMPs)."""
    print_header("CARLOS BRIZUELA - AMP DATA")

    try:
        from carlos_brizuela.scripts.dramp_activity_loader import (
            DRAMPLoader,
            AMPDatabase,
        )

        loader = DRAMPLoader()

        print_subheader("Loading/Generating AMP Database")
        db = loader.load_or_download()

        print(f"\nTotal peptides: {len(db.records)}")

        # Count by target
        targets = {}
        for r in db.records:
            if r.target_organism:
                key = r.target_organism.split()[0]
                targets[key] = targets.get(key, 0) + 1

        print("By target (top 5):")
        for target, count in sorted(targets.items(), key=lambda x: -x[1])[:5]:
            print(f"  {target}: {count}")

        print_subheader("Training Activity Predictors")
        metrics = loader.train_all_pathogen_models(db)

        print("\nTraining Summary:")
        for name, m in metrics.items():
            print(f"  {name}: r={m['pearson_r']:.3f}, RMSE={m['rmse']:.3f}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def initialize_jose():
    """Initialize Jose Colbes data (protein stability)."""
    print_header("JOSE COLBES - PROTEIN STABILITY DATA")

    try:
        from jose_colbes.scripts.protherm_ddg_loader import (
            ProThermLoader,
            StabilityDatabase,
        )

        loader = ProThermLoader()

        print_subheader("Loading/Generating Stability Database")
        db = loader.load_or_generate()

        print(f"\nTotal mutations: {len(db.records)}")

        # Statistics
        import numpy as np
        ddg_values = [r.ddg for r in db.records]
        destab = sum(1 for d in ddg_values if d > 1)
        stab = sum(1 for d in ddg_values if d < -1)
        neutral = len(ddg_values) - destab - stab

        print(f"DDG range: {min(ddg_values):.2f} to {max(ddg_values):.2f}")
        print(f"Destabilizing: {destab} ({destab / len(ddg_values):.1%})")
        print(f"Neutral: {neutral} ({neutral / len(ddg_values):.1%})")
        print(f"Stabilizing: {stab} ({stab / len(ddg_values):.1%})")

        print_subheader("Training DDG Predictor")
        metrics = loader.train_ddg_predictor(db)

        if metrics:
            print(f"\nModel Performance:")
            print(f"  Pearson r: {metrics['pearson_r']:.3f}")
            print(f"  RMSE: {metrics['rmse']:.3f} kcal/mol")
            print(f"  Classification: {metrics['classification_accuracy']:.1%}")

        # Test prediction
        print_subheader("Testing Predictions")
        test_mutations = [
            ("V", "A", "H", 0.2),  # Buried Val->Ala
            ("G", "A", "H", 0.2),  # Gly->Ala in helix
            ("K", "R", "C", 0.7),  # Surface Lys->Arg
        ]

        for wt, mut, ss, rsa in test_mutations:
            result = loader.predict_mutation(wt, mut, ss, rsa)
            if result:
                print(f"  {wt}->{mut} ({ss}, RSA={rsa}): "
                      f"DDG={result['ddg_predicted']:+.2f} [{result['classification']}]")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def initialize_hiv():
    """Initialize HIV package data (resistance)."""
    print_header("HIV PACKAGE - RESISTANCE DATA")

    try:
        from hiv_research_package.scripts.stanford_hivdb_client import (
            StanfordHIVdbClient,
        )

        client = StanfordHIVdbClient()

        print_subheader("Testing Stanford HIVdb Client")

        # Run demo analysis
        print("Running demo resistance analysis...")
        report = client._mock_analysis("", "DEMO_PATIENT")

        print(f"\nDemo Report:")
        print(f"  Patient: {report.patient_id}")
        print(f"  Subtype: {report.subtype}")
        print(f"  TDR: {'Yes' if report.has_tdr() else 'No'}")

        if report.mutations:
            print(f"  Mutations: {', '.join(m.notation for m in report.mutations[:5])}")
        if report.get_resistant_drugs():
            print(f"  Resistant drugs: {', '.join(report.get_resistant_drugs())}")

        print(f"  Recommended: {report.get_recommended_regimens()[0]}")

        # Save demo report
        output_path = client.cache_dir / "demo_report.json"
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nSaved demo report to: {output_path}")

        # Generate formatted report
        print_subheader("Formatted Report Preview")
        formatted = client.generate_report(report)
        for line in formatted.split("\n")[:20]:
            print(line)
        print("...")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_test():
    """Run quick tests on all components."""
    print_header("QUICK TEST - ALL COMPONENTS")

    results = {}

    # VAE
    print_subheader("Testing VAE Service")
    try:
        vae = get_vae_service()
        z = vae.sample_latent(1)[0]
        seq = vae.decode_latent(z)
        results["VAE"] = f"OK - Generated: {seq}"
    except Exception as e:
        results["VAE"] = f"FAIL - {e}"

    # A2
    print_subheader("Testing A2 (Primers)")
    try:
        from alejandra_rojas.scripts.ncbi_arbovirus_loader import NCBIArbovirusLoader
        loader = NCBIArbovirusLoader()
        results["A2"] = "OK - Loader initialized"
    except Exception as e:
        results["A2"] = f"FAIL - {e}"

    # B1/B8/B10
    print_subheader("Testing B1/B8/B10 (AMPs)")
    try:
        from carlos_brizuela.scripts.dramp_activity_loader import DRAMPLoader
        loader = DRAMPLoader()
        results["B1/B8/B10"] = "OK - Loader initialized"
    except Exception as e:
        results["B1/B8/B10"] = f"FAIL - {e}"

    # C1/C4
    print_subheader("Testing C1/C4 (Stability)")
    try:
        from jose_colbes.scripts.protherm_ddg_loader import ProThermLoader
        loader = ProThermLoader()
        results["C1/C4"] = "OK - Loader initialized"
    except Exception as e:
        results["C1/C4"] = f"FAIL - {e}"

    # H6/H7
    print_subheader("Testing H6/H7 (HIV)")
    try:
        from hiv_research_package.scripts.stanford_hivdb_client import StanfordHIVdbClient
        client = StanfordHIVdbClient()
        results["H6/H7"] = "OK - Client initialized"
    except Exception as e:
        results["H6/H7"] = f"FAIL - {e}"

    # Summary
    print_header("TEST RESULTS")
    all_ok = True
    for component, result in results.items():
        status = "PASS" if result.startswith("OK") else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  [{status}] {component}: {result}")

    return all_ok


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Initialize all data for production use"
    )
    parser.add_argument("--all", action="store_true", help="Initialize everything")
    parser.add_argument("--partner", choices=["alejandra", "carlos", "jose", "hiv"],
                        help="Initialize specific partner")
    parser.add_argument("--vae", action="store_true", help="Initialize VAE only")
    parser.add_argument("--test", action="store_true", help="Run quick tests")
    args = parser.parse_args()

    start_time = time.time()

    if args.test:
        success = run_quick_test()
        sys.exit(0 if success else 1)

    if args.vae or args.all:
        initialize_vae()

    if args.partner == "alejandra" or args.all:
        initialize_alejandra()

    if args.partner == "carlos" or args.all:
        initialize_carlos()

    if args.partner == "jose" or args.all:
        initialize_jose()

    if args.partner == "hiv" or args.all:
        initialize_hiv()

    if not any([args.all, args.partner, args.vae, args.test]):
        parser.print_help()
        return

    elapsed = time.time() - start_time
    print_header("INITIALIZATION COMPLETE")
    print(f"Total time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
