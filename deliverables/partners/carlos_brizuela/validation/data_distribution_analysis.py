#!/usr/bin/env python3
"""Data Distribution Analysis: What's missing and what we need."""

import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_package_dir = _script_dir.parent
_repo_root = _package_dir.parent.parent.parent
sys.path.insert(0, str(_package_dir))
sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd
from training.dataset import create_full_dataset


def main():
    dataset = create_full_dataset()

    # Collect data
    data = []
    for sample in dataset:
        seq = sample['sequence']
        props = sample['properties']
        # props[0]=length, props[1]=charge, props[2]=hydrophobicity
        data.append({
            'sequence': seq,
            'length': len(seq),
            'hydrophobicity': props[2].item(),  # Index 2 is hydrophobicity
            'pathogen': sample['pathogen_label'],
        })

    df = pd.DataFrame(data)

    # Define bins
    df['length_bin'] = pd.cut(
        df['length'],
        bins=[0, 15, 25, 100],
        labels=['Short (≤15)', 'Medium (16-25)', 'Long (>25)']
    )
    df['hydro_bin'] = pd.cut(
        df['hydrophobicity'],
        bins=[-np.inf, 0.2, 0.5, np.inf],
        labels=['Hydrophilic', 'Balanced', 'Hydrophobic']
    )

    print("=" * 70)
    print("DATA DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # 1. Length x Hydrophobicity matrix
    print("\n1. SAMPLE COUNT: Length × Hydrophobicity")
    print("-" * 70)
    pivot = pd.crosstab(df['length_bin'], df['hydro_bin'])
    print(pivot)
    print(f"\nTotal: {len(df)}")

    # 2. Percentage matrix
    print("\n2. PERCENTAGE: Length × Hydrophobicity")
    print("-" * 70)
    pivot_pct = pivot / len(df) * 100
    print(pivot_pct.round(1))

    # 3. Identify gaps
    print("\n3. DATA GAPS (< 15 samples)")
    print("-" * 70)
    for length_bin in pivot.index:
        for hydro_bin in pivot.columns:
            n = pivot.loc[length_bin, hydro_bin]
            if n < 15:
                print(f"  {length_bin} + {hydro_bin}: N={n}")

    # 4. Pathogen distribution within each cell
    print("\n4. PATHOGEN DISTRIBUTION BY CELL")
    print("-" * 70)
    pathogen_names = ['E. coli', 'P. aeruginosa', 'S. aureus', 'A. baumannii']

    for length_bin in ['Short (≤15)', 'Medium (16-25)', 'Long (>25)']:
        for hydro_bin in ['Hydrophilic', 'Balanced', 'Hydrophobic']:
            subset = df[(df['length_bin'] == length_bin) & (df['hydro_bin'] == hydro_bin)]
            if len(subset) > 0:
                # Count pathogens manually to avoid pandas issues
                path_counts = [0, 0, 0, 0]
                for p in subset['pathogen']:
                    if 0 <= p < 4:
                        path_counts[p] += 1
                path_str = ", ".join([f"{pathogen_names[i]}:{path_counts[i]}"
                                       for i in range(4)])
                print(f"{length_bin} + {hydro_bin} (N={len(subset)}): {path_str}")

    # 5. What we need for balanced design
    print("\n5. RECOMMENDATIONS FOR BALANCED DATASET")
    print("-" * 70)

    # Target: minimum 30 samples per cell for meaningful statistics
    target_n = 30
    total_needed = 0

    print(f"Target: {target_n} samples per cell")
    print()

    for length_bin in pivot.index:
        for hydro_bin in pivot.columns:
            n = pivot.loc[length_bin, hydro_bin]
            if n < target_n:
                needed = target_n - n
                total_needed += needed
                print(f"  {length_bin} + {hydro_bin}: Have {n}, need +{needed}")

    print(f"\nTotal additional samples needed: {total_needed}")

    # 6. Priority recommendations
    print("\n6. PRIORITY DATA COLLECTION")
    print("-" * 70)
    print("""
HIGHEST PRIORITY (isolate length effect):
- Short (≤15) + Hydrophobic: Currently 0 samples
  → Find short hydrophobic AMPs with validated MIC
  → This tests if length is independent of hydrophobicity

HIGH PRIORITY (isolate hydrophobicity effect):
- Long (>25) + Hydrophilic: Currently 35 samples
  → Add ~15 more to reach N=50 for robust statistics
  → This tests if hydrophobicity is independent of length

MEDIUM PRIORITY (balanced coverage):
- Short (≤15) + Balanced: Currently 10 samples
- Long (>25) + Balanced: Currently 15 samples
  → Expand both for intermediate behavior analysis

RATIONALE:
The current dataset has a LENGTH × HYDROPHOBICITY correlation:
- Short peptides tend to be more hydrophilic
- Long peptides tend to be more hydrophobic

This confounding makes it impossible to determine which factor
is actually causing the prediction failure. Filling the gaps
will allow:
1. Training separate length-aware models
2. Adding length/hydrophobicity interaction features
3. Validating if 3D structure features (helix propensity,
   amphipathic moment) improve long/hydrophobic predictions
""")


if __name__ == '__main__':
    main()
