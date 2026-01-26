#!/usr/bin/env python3
"""Generate SHA-256 hashes for IP protection manifest.

This script creates cryptographic proof of prior art by:
1. Computing SHA-256 hashes of all critical files
2. Recording file sizes and line counts
3. Capturing git commit history
4. Generating a timestamped manifest

The output can be:
- Committed to git (provides timestamp via commit)
- Submitted to OpenTimestamps for blockchain anchoring
- Archived on archive.org

Usage:
    python scripts/generate_ip_hashes.py
    python scripts/generate_ip_hashes.py --output hashes.json
    python scripts/generate_ip_hashes.py --ots  # Create OpenTimestamps proof
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


# Critical files for IP protection
CRITICAL_FILES = {
    "core_algorithms": [
        "src/models/ternary_vae.py",
        "src/models/ternary_vae_optionc.py",
        "src/models/improved_components.py",
        "src/geometry/poincare.py",
        "src/geometry/__init__.py",
        "src/core/padic_math.py",
        "src/core/__init__.py",
    ],
    "encoders": [
        "src/encoders/trainable_codon_encoder.py",
        "src/encoders/codon_encoder.py",
        "src/encoders/peptide_encoder.py",
        "src/encoders/padic_amino_acid_encoder.py",
        "src/encoders/holographic_encoder.py",
    ],
    "losses": [
        "src/losses/peptide_losses.py",
        "src/losses/base.py",
        "src/losses/consequence_predictor.py",
    ],
    "biology": [
        "src/biology/codons.py",
    ],
    "research_discoveries": [
        "deliverables/partners/arbovirus_surveillance/research/padic_structure_analysis/01_multi_prime_ultrametric_test.py",
        "deliverables/partners/arbovirus_surveillance/research/padic_structure_analysis/02_projection_deformation_analysis.py",
        "deliverables/partners/arbovirus_surveillance/research/padic_structure_analysis/03_adelic_decomposition_test.py",
        "deliverables/partners/arbovirus_surveillance/src/viral_projection_module.py",
        "deliverables/partners/protein_stability_ddg/src/validated_ddg_predictor.py",
        "research/codon-encoder/training/train_codon_encoder.py",
    ],
    "trained_models": [
        "checkpoints/v5_12_4/best_Q.pt",
        "checkpoints/homeostatic_rich/best.pt",
        "checkpoints/v5_5/best.pt",
        "research/codon-encoder/training/results/trained_codon_encoder.pt",
        "deliverables/partners/antimicrobial_peptides/checkpoints_definitive/best_production.pt",
    ],
    "documentation": [
        "CLAUDE.md",
        "INTELLECTUAL_PROPERTY_MANIFEST.md",
        "README.md",
    ],
    "key_results": [
        "deliverables/partners/arbovirus_surveillance/research/padic_structure_analysis/results/adelic_decomposition_results.json",
        "deliverables/partners/arbovirus_surveillance/research/padic_structure_analysis/results/FINAL_VERDICT.json",
        "deliverables/partners/protein_stability_ddg/validation/results/scientific_metrics.json",
    ],
}


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def count_lines(filepath: Path) -> int:
    """Count lines in a text file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return -1


def get_git_info(filepath: Path) -> dict:
    """Get git information for a file."""
    try:
        # First commit
        result = subprocess.run(
            ["git", "log", "--follow", "--format=%H|%ai|%s", "--diff-filter=A", "--", str(filepath)],
            capture_output=True, text=True, cwd=filepath.parent if filepath.parent.exists() else "."
        )
        first_commit = result.stdout.strip().split("\n")[-1] if result.stdout.strip() else ""

        # Last commit
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H|%ai|%s", "--", str(filepath)],
            capture_output=True, text=True, cwd=filepath.parent if filepath.parent.exists() else "."
        )
        last_commit = result.stdout.strip()

        return {
            "first_commit": first_commit.split("|")[0] if first_commit and "|" in first_commit else None,
            "first_commit_date": first_commit.split("|")[1] if first_commit and "|" in first_commit else None,
            "last_commit": last_commit.split("|")[0] if last_commit and "|" in last_commit else None,
            "last_commit_date": last_commit.split("|")[1] if last_commit and "|" in last_commit else None,
        }
    except Exception as e:
        return {"error": str(e)}


def get_current_commit() -> str:
    """Get current HEAD commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def generate_manifest(project_root: Path) -> dict:
    """Generate complete IP manifest."""
    manifest = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "generate_ip_hashes.py v1.0",
            "project": "Ternary VAE - P-adic Hyperbolic Bioinformatics",
            "organization": "AI Whisperers",
            "current_commit": get_current_commit(),
        },
        "files": {},
        "summary": {
            "total_files": 0,
            "total_lines": 0,
            "total_bytes": 0,
            "categories": {}
        }
    }

    for category, files in CRITICAL_FILES.items():
        manifest["files"][category] = []
        category_lines = 0
        category_bytes = 0
        category_count = 0

        for rel_path in files:
            filepath = project_root / rel_path

            if filepath.exists():
                file_size = filepath.stat().st_size
                line_count = count_lines(filepath)
                sha256 = compute_sha256(filepath)
                git_info = get_git_info(filepath)

                file_entry = {
                    "path": rel_path,
                    "sha256": sha256,
                    "size_bytes": file_size,
                    "lines": line_count,
                    "git": git_info,
                    "exists": True
                }

                category_lines += line_count if line_count > 0 else 0
                category_bytes += file_size
                category_count += 1
            else:
                file_entry = {
                    "path": rel_path,
                    "exists": False
                }

            manifest["files"][category].append(file_entry)

        manifest["summary"]["categories"][category] = {
            "file_count": category_count,
            "total_lines": category_lines,
            "total_bytes": category_bytes
        }
        manifest["summary"]["total_files"] += category_count
        manifest["summary"]["total_lines"] += category_lines
        manifest["summary"]["total_bytes"] += category_bytes

    return manifest


def generate_hash_summary(manifest: dict) -> str:
    """Generate human-readable hash summary."""
    lines = [
        "=" * 70,
        "INTELLECTUAL PROPERTY HASH MANIFEST",
        f"Generated: {manifest['metadata']['generated_at']}",
        f"Commit: {manifest['metadata']['current_commit'][:12]}",
        "=" * 70,
        ""
    ]

    for category, files in manifest["files"].items():
        lines.append(f"\n## {category.upper().replace('_', ' ')}")
        lines.append("-" * 50)

        for f in files:
            if f.get("exists"):
                lines.append(f"  {f['path']}")
                lines.append(f"    SHA-256: {f['sha256']}")
                lines.append(f"    Size: {f['size_bytes']:,} bytes, {f['lines']} lines")
                if f.get("git", {}).get("first_commit_date"):
                    lines.append(f"    First: {f['git']['first_commit_date']}")
            else:
                lines.append(f"  {f['path']} [NOT FOUND]")

    lines.extend([
        "",
        "=" * 70,
        "SUMMARY",
        "=" * 70,
        f"Total files: {manifest['summary']['total_files']}",
        f"Total lines: {manifest['summary']['total_lines']:,}",
        f"Total bytes: {manifest['summary']['total_bytes']:,}",
        "",
        "This manifest provides cryptographic proof of prior art.",
        "Verify any file: sha256sum <filepath>",
        "=" * 70,
    ])

    return "\n".join(lines)


def create_ots_proof(manifest_path: Path):
    """Create OpenTimestamps proof (requires opentimestamps-client)."""
    try:
        result = subprocess.run(
            ["ots", "stamp", str(manifest_path)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"OpenTimestamps proof created: {manifest_path}.ots")
            return True
        else:
            print(f"OTS error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("OpenTimestamps client not installed. Install with: pip install opentimestamps-client")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate IP protection hashes")
    parser.add_argument("--output", "-o", type=str, default="IP_HASH_MANIFEST.json",
                        help="Output JSON file")
    parser.add_argument("--summary", "-s", type=str, default="IP_HASH_MANIFEST.txt",
                        help="Output summary text file")
    parser.add_argument("--ots", action="store_true",
                        help="Create OpenTimestamps proof")
    parser.add_argument("--project-root", type=str, default=".",
                        help="Project root directory")

    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()

    print("Generating IP protection manifest...")
    print(f"Project root: {project_root}")

    manifest = generate_manifest(project_root)

    # Save JSON manifest
    json_path = project_root / args.output
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"JSON manifest: {json_path}")

    # Save text summary
    summary = generate_hash_summary(manifest)
    txt_path = project_root / args.summary
    with open(txt_path, "w") as f:
        f.write(summary)
    print(f"Text summary: {txt_path}")

    # Print summary
    print("\n" + summary)

    # Create OpenTimestamps proof if requested
    if args.ots:
        print("\nCreating OpenTimestamps proof...")
        create_ots_proof(json_path)

    # Compute hash of the manifest itself
    manifest_hash = compute_sha256(json_path)
    print(f"\nManifest SHA-256: {manifest_hash}")
    print("This hash can be used to verify the manifest integrity.")

    return manifest


if __name__ == "__main__":
    main()
