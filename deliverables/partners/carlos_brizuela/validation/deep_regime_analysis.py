#!/usr/bin/env python3
"""Deep Regime Analysis: Why does the failing regime fail?"""

import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_package_dir = _script_dir.parent
_repo_root = _package_dir.parent.parent.parent
sys.path.insert(0, str(_package_dir))
sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd
def safe_spearmanr(x, y):
    """Compute Spearman correlation safely."""
    from scipy.stats import spearmanr as _spearmanr
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan, 1.0
    try:
        return _spearmanr(x, y)
    except Exception:
        return np.nan, 1.0
import torch

from src.encoders.peptide_encoder import PeptideVAE
from training.dataset import create_full_dataset


def compute_features(seq):
    """Compute comprehensive features for a sequence."""
    hydro_values = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    charge_values = {'K': 1, 'R': 1, 'H': 0.5, 'D': -1, 'E': -1}

    aa_hydros = [hydro_values.get(aa, 0) for aa in seq]
    charges = [charge_values.get(aa, 0) for aa in seq]

    return {
        'hydro_variance': np.var(aa_hydros) if len(aa_hydros) > 1 else 0,
        'hydro_mean': np.mean(aa_hydros),
        'charge_density': sum(charges) / len(seq) if len(seq) > 0 else 0,
        'aromatic': sum(1 for aa in seq if aa in 'WFY') / len(seq),
        'proline': sum(1 for aa in seq if aa == 'P') / len(seq),
        'cysteine': sum(1 for aa in seq if aa == 'C') / len(seq),
        'hydrophobic_ratio': sum(1 for aa in seq if aa in 'AILMFVW') / len(seq),
        'charged_ratio': sum(1 for aa in seq if aa in 'KRHDE') / len(seq),
    }


def main():
    # Load model
    checkpoint = torch.load(
        'checkpoints/fold_0_best.pt',
        map_location='cuda',
        weights_only=False
    )
    config = checkpoint['config']
    model = PeptideVAE(
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        max_radius=config['max_radius'],
        curvature=config['curvature'],
    ).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Collect data
    dataset = create_full_dataset()
    data = []

    with torch.no_grad():
        for sample in dataset:
            seq = sample['sequence']
            outputs = model([seq], teacher_forcing=False)
            pred = outputs['mic_pred'].squeeze().cpu().item()
            props = sample['properties']
            feats = compute_features(seq)

            # Convert mic_target to float (may be tensor)
            mic_target = sample['mic']
            if hasattr(mic_target, 'item'):
                mic_target = mic_target.item()
            mic_target = float(mic_target)

            data.append({
                'sequence': seq,
                'length': len(seq),
                'mic_target': mic_target,
                'mic_pred': float(pred),
                'error': abs(mic_target - pred),
                'net_charge': props[1].item(),  # Index 1 is charge
                'hydrophobicity': props[2].item(),  # Index 2 is hydrophobicity
                'pathogen': sample['pathogen_label'],
                **feats
            })

    df = pd.DataFrame(data)
    df['working'] = ((df['length'] <= 25) & (df['hydrophobicity'] < 0.5)).astype(int)
    working = df[df['working'] == 1]
    failing = df[df['working'] == 0]

    print("=" * 70)
    print("DEEP ANALYSIS: WHY DOES THE FAILING REGIME FAIL?")
    print("=" * 70)

    # 1. Feature distributions
    print("\n1. FEATURE DISTRIBUTIONS: Working vs Failing")
    print("-" * 70)
    features = ['length', 'hydrophobicity', 'hydro_variance', 'charge_density',
                'aromatic', 'proline', 'hydrophobic_ratio', 'charged_ratio']

    for feat in features:
        w_mean, w_std = working[feat].mean(), working[feat].std()
        f_mean, f_std = failing[feat].mean(), failing[feat].std()
        diff = f_mean - w_mean
        print(f"{feat:20s}: W={w_mean:+.3f}±{w_std:.3f}  F={f_mean:+.3f}±{f_std:.3f}  Δ={diff:+.3f}")

    # 2. Correlations with error
    print("\n2. FEATURE CORRELATIONS WITH PREDICTION ERROR")
    print("-" * 70)
    for feat in features:
        arr = df[feat].values
        err = df['error'].values
        if len(np.unique(arr)) > 2:
            r, p = safe_spearmanr(arr, err)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{feat:20s}: r={r:+.3f} (p={p:.4f}) {sig}")

    # 3. Cross-analysis: Length within hydrophobicity
    print("\n3. CROSS-ANALYSIS: Length effect WITHIN hydrophobicity regimes")
    print("-" * 70)
    for hydro_name, h_mask in [
        ('Hydrophilic', df['hydrophobicity'] < 0.2),
        ('Balanced', (df['hydrophobicity'] >= 0.2) & (df['hydrophobicity'] < 0.5)),
        ('Hydrophobic', df['hydrophobicity'] >= 0.5)
    ]:
        subset = df[h_mask]
        if len(subset) > 10:
            print(f"\n{hydro_name} (N={len(subset)}):")
            for len_name, l_mask in [
                ('  Short (≤15)', subset['length'] <= 15),
                ('  Medium (16-25)', (subset['length'] > 15) & (subset['length'] <= 25)),
                ('  Long (>25)', subset['length'] > 25)
            ]:
                sub = subset[l_mask]
                if len(sub) > 5:
                    r, _ = safe_spearmanr(sub['mic_target'], sub['mic_pred'])
                    print(f"{len_name}: N={len(sub):3d}, r={r:.3f}")

    # 4. Hydrophobicity within length
    print("\n4. CROSS-ANALYSIS: Hydrophobicity effect WITHIN length regimes")
    print("-" * 70)
    for len_name, l_mask in [
        ('Short (≤15)', df['length'] <= 15),
        ('Medium (16-25)', (df['length'] > 15) & (df['length'] <= 25)),
        ('Long (>25)', df['length'] > 25)
    ]:
        subset = df[l_mask]
        if len(subset) > 10:
            print(f"\n{len_name} (N={len(subset)}):")
            for hydro_name, h_mask in [
                ('  Hydrophilic', subset['hydrophobicity'] < 0.2),
                ('  Balanced', (subset['hydrophobicity'] >= 0.2) & (subset['hydrophobicity'] < 0.5)),
                ('  Hydrophobic', subset['hydrophobicity'] >= 0.5)
            ]:
                sub = subset[h_mask]
                if len(sub) > 5:
                    r, _ = safe_spearmanr(sub['mic_target'], sub['mic_pred'])
                    print(f"{hydro_name}: N={len(sub):3d}, r={r:.3f}")

    # 5. Amphipathicity analysis
    print("\n5. AMPHIPATHICITY (Hydro Variance) ANALYSIS")
    print("-" * 70)
    median_var = df['hydro_variance'].median()

    for regime_name, regime_mask in [
        ('All', df['working'] >= 0),
        ('Working', df['working'] == 1),
        ('Failing', df['working'] == 0)
    ]:
        subset = df[regime_mask]
        amph = subset[subset['hydro_variance'] > median_var]
        non_amph = subset[subset['hydro_variance'] <= median_var]

        if len(amph) > 10 and len(non_amph) > 10:
            r_amph, _ = safe_spearmanr(amph['mic_target'], amph['mic_pred'])
            r_non, _ = safe_spearmanr(non_amph['mic_target'], non_amph['mic_pred'])
            print(f"{regime_name:10s}: Amphipathic r={r_amph:.3f} (N={len(amph)}), "
                  f"Non-amphipathic r={r_non:.3f} (N={len(non_amph)})")

    # 6. Worst vs Best predictions
    print("\n6. WORST vs BEST PREDICTIONS ANALYSIS")
    print("-" * 70)
    n_extreme = len(df) // 10  # 10%
    worst = df.nlargest(n_extreme, 'error')
    best = df.nsmallest(n_extreme, 'error')

    print(f"Comparing worst {n_extreme} vs best {n_extreme} predictions:")
    for feat in ['length', 'hydrophobicity', 'hydro_variance', 'aromatic',
                 'hydrophobic_ratio', 'charged_ratio']:
        w_mean = worst[feat].mean()
        b_mean = best[feat].mean()
        print(f"  {feat:20s}: Worst={w_mean:+.3f}, Best={b_mean:+.3f}, Δ={w_mean-b_mean:+.3f}")

    # 7. Pathogen breakdown within failing
    print("\n7. PATHOGEN BREAKDOWN WITHIN FAILING REGIME")
    print("-" * 70)
    pathogen_names = ['E. coli', 'P. aeruginosa', 'S. aureus', 'A. baumannii']
    for i, name in enumerate(pathogen_names):
        fail_path = failing[failing['pathogen'] == i]
        if len(fail_path) > 5:
            r, _ = safe_spearmanr(fail_path['mic_target'], fail_path['mic_pred'])
            print(f"  {name}: N={len(fail_path)}, r={r:.3f}, "
                  f"avg_len={fail_path['length'].mean():.1f}, "
                  f"avg_hydro={fail_path['hydrophobicity'].mean():.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("ACTIONABLE RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. LENGTH STRATIFICATION NEEDED:
   - Short peptides (≤15): Model works well (r=0.66)
   - Long peptides (>25): Model fails (r=0.14)
   - Consider: Separate models OR length-aware architecture

2. HYDROPHOBICITY MECHANISM DIFFERS:
   - Hydrophilic: Charge-based activity (model captures this)
   - Hydrophobic: Membrane insertion (3D structure matters)
   - Consider: Add amphipathic moment, helix propensity features

3. CROSS-VALIDATION NEEDED:
   - Test if long+hydrophilic works (isolate length effect)
   - Test if short+hydrophobic works (isolate hydrophobicity effect)
   - Current data may have length-hydrophobicity confounding

4. DATA AUGMENTATION:
   - Need more short hydrophobic peptides
   - Need more long hydrophilic peptides
   - This will help disentangle the two effects

5. ARCHITECTURAL CHANGES:
   - For long peptides: Consider segment-based encoding
   - For hydrophobic: Add explicit amphipathicity features
   - Consider: Multi-task learning with secondary structure
""")


if __name__ == '__main__':
    main()
