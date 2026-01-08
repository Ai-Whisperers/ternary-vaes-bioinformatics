#!/usr/bin/env python3
"""Regime Analysis for PeptideVAE Predictions.

This module validates the hypothesis that PeptideVAE performance varies
by sequence characteristics (length, hydrophobicity, Gram type), similar
to the regime-dependent behavior found in the Colbes DDG predictor.

Key Hypotheses:
    1. Short peptides (≤15 AA) have higher prediction accuracy than long (>25 AA)
    2. Hydrophilic peptides have higher accuracy than hydrophobic
    3. Gram-negative pathogens are predicted differently than Gram-positive

Statistical Methods:
    - Bootstrap confidence intervals for correlations
    - Permutation tests for regime differences
    - Effect size (Cohen's d) for regime comparisons

Usage:
    python validation/regime_analysis.py --checkpoint checkpoints/fold_0_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add paths
_script_dir = Path(__file__).resolve().parent
_package_dir = _script_dir.parent
_deliverables_dir = _package_dir.parent.parent
_repo_root = _deliverables_dir.parent
sys.path.insert(0, str(_package_dir))
sys.path.insert(0, str(_deliverables_dir))
sys.path.insert(0, str(_repo_root))

from scipy.stats import spearmanr, pearsonr, mannwhitneyu, bootstrap
from scipy.stats import permutation_test as scipy_permutation_test

from src.encoders.peptide_encoder import PeptideVAE
from training.dataset import create_full_dataset, PATHOGEN_TO_LABEL


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RegimeStats:
    """Statistics for a single regime."""
    name: str
    n_samples: int
    spearman_r: float
    spearman_ci_low: float
    spearman_ci_high: float
    pearson_r: float
    mae: float
    target_mean: float
    target_std: float
    pred_mean: float
    pred_std: float


@dataclass
class RegimeComparison:
    """Statistical comparison between two regimes."""
    regime_a: str
    regime_b: str
    r_diff: float  # r_a - r_b
    r_diff_ci_low: float
    r_diff_ci_high: float
    p_value: float
    effect_size: float  # Cohen's d
    significant: bool


@dataclass
class RegimeAnalysisReport:
    """Complete regime analysis report."""
    model_name: str
    timestamp: str
    n_total: int
    overall_spearman: float
    overall_pearson: float

    # Regime statistics
    length_regimes: List[RegimeStats] = field(default_factory=list)
    hydrophobicity_regimes: List[RegimeStats] = field(default_factory=list)
    gram_regimes: List[RegimeStats] = field(default_factory=list)
    pathogen_regimes: List[RegimeStats] = field(default_factory=list)

    # Comparisons
    length_comparisons: List[RegimeComparison] = field(default_factory=list)
    hydrophobicity_comparisons: List[RegimeComparison] = field(default_factory=list)
    gram_comparisons: List[RegimeComparison] = field(default_factory=list)

    # Summary
    key_findings: List[str] = field(default_factory=list)
    regime_hypothesis_confirmed: bool = False


# =============================================================================
# Statistical Functions
# =============================================================================


def bootstrap_correlation(
    targets: np.ndarray,
    preds: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute Spearman correlation with bootstrap confidence interval.

    Returns:
        Tuple of (correlation, ci_low, ci_high)
    """
    n = len(targets)
    correlations = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        r, _ = spearmanr(targets[indices], preds[indices])
        if not np.isnan(r):
            correlations.append(r)

    if len(correlations) < 100:
        return spearmanr(targets, preds)[0], np.nan, np.nan

    correlations = np.array(correlations)
    alpha = (1 - confidence) / 2
    ci_low = np.percentile(correlations, alpha * 100)
    ci_high = np.percentile(correlations, (1 - alpha) * 100)

    return np.median(correlations), ci_low, ci_high


def permutation_test_correlation_diff(
    targets_a: np.ndarray,
    preds_a: np.ndarray,
    targets_b: np.ndarray,
    preds_b: np.ndarray,
    n_permutations: int = 1000,
) -> Tuple[float, float]:
    """Test if correlation difference between regimes is significant.

    Returns:
        Tuple of (observed_diff, p_value)
    """
    r_a, _ = spearmanr(targets_a, preds_a)
    r_b, _ = spearmanr(targets_b, preds_b)
    observed_diff = r_a - r_b

    # Combine data
    all_targets = np.concatenate([targets_a, targets_b])
    all_preds = np.concatenate([preds_a, preds_b])
    n_a = len(targets_a)
    n_total = len(all_targets)

    # Permutation test
    count_extreme = 0
    for _ in range(n_permutations):
        perm = np.random.permutation(n_total)
        perm_targets_a = all_targets[perm[:n_a]]
        perm_preds_a = all_preds[perm[:n_a]]
        perm_targets_b = all_targets[perm[n_a:]]
        perm_preds_b = all_preds[perm[n_a:]]

        r_perm_a, _ = spearmanr(perm_targets_a, perm_preds_a)
        r_perm_b, _ = spearmanr(perm_targets_b, perm_preds_b)
        perm_diff = r_perm_a - r_perm_b

        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)
    return observed_diff, p_value


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n_a, n_b = len(group_a), len(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group_a) - np.mean(group_b)) / pooled_std


# =============================================================================
# Regime Analysis
# =============================================================================


def compute_regime_stats(
    name: str,
    targets: np.ndarray,
    preds: np.ndarray,
    n_bootstrap: int = 1000,
) -> RegimeStats:
    """Compute statistics for a single regime."""
    r_spearman, ci_low, ci_high = bootstrap_correlation(targets, preds, n_bootstrap)
    r_pearson, _ = pearsonr(targets, preds)
    mae = np.abs(targets - preds).mean()

    return RegimeStats(
        name=name,
        n_samples=len(targets),
        spearman_r=r_spearman,
        spearman_ci_low=ci_low,
        spearman_ci_high=ci_high,
        pearson_r=r_pearson,
        mae=mae,
        target_mean=targets.mean(),
        target_std=targets.std(),
        pred_mean=preds.mean(),
        pred_std=preds.std(),
    )


def compare_regimes(
    name_a: str,
    targets_a: np.ndarray,
    preds_a: np.ndarray,
    name_b: str,
    targets_b: np.ndarray,
    preds_b: np.ndarray,
    n_permutations: int = 1000,
) -> RegimeComparison:
    """Compare two regimes statistically."""
    r_a, _ = spearmanr(targets_a, preds_a)
    r_b, _ = spearmanr(targets_b, preds_b)

    # Permutation test for correlation difference
    r_diff, p_value = permutation_test_correlation_diff(
        targets_a, preds_a, targets_b, preds_b, n_permutations
    )

    # Bootstrap CI for difference
    n_bootstrap = 500
    diffs = []
    for _ in range(n_bootstrap):
        idx_a = np.random.choice(len(targets_a), size=len(targets_a), replace=True)
        idx_b = np.random.choice(len(targets_b), size=len(targets_b), replace=True)
        r_boot_a, _ = spearmanr(targets_a[idx_a], preds_a[idx_a])
        r_boot_b, _ = spearmanr(targets_b[idx_b], preds_b[idx_b])
        if not np.isnan(r_boot_a) and not np.isnan(r_boot_b):
            diffs.append(r_boot_a - r_boot_b)

    if len(diffs) > 50:
        ci_low = np.percentile(diffs, 2.5)
        ci_high = np.percentile(diffs, 97.5)
    else:
        ci_low, ci_high = np.nan, np.nan

    # Effect size on prediction errors
    errors_a = np.abs(targets_a - preds_a)
    errors_b = np.abs(targets_b - preds_b)
    effect = cohens_d(errors_a, errors_b)

    return RegimeComparison(
        regime_a=name_a,
        regime_b=name_b,
        r_diff=r_diff,
        r_diff_ci_low=ci_low,
        r_diff_ci_high=ci_high,
        p_value=p_value,
        effect_size=effect,
        significant=p_value < 0.05,
    )


def run_regime_analysis(
    checkpoint_path: Path,
    device: torch.device,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
) -> RegimeAnalysisReport:
    """Run complete regime analysis on trained model."""

    from datetime import datetime

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = PeptideVAE(
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        max_radius=config['max_radius'],
        curvature=config['curvature'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load data and get predictions
    dataset = create_full_dataset()

    all_preds = []
    all_targets = []
    all_lengths = []
    all_hydros = []
    all_pathogens = []
    all_charges = []

    print("Collecting predictions...")
    with torch.no_grad():
        for sample in dataset:
            seq = sample['sequence']
            outputs = model([seq], teacher_forcing=False)
            pred = outputs['mic_pred'].squeeze().cpu().item()

            all_preds.append(pred)
            all_targets.append(sample['mic'])
            all_lengths.append(len(seq))
            all_hydros.append(sample['properties'][1].item())
            all_pathogens.append(sample['pathogen_label'])
            all_charges.append(sample['properties'][0].item())

    preds = np.array(all_preds)
    targets = np.array(all_targets)
    lengths = np.array(all_lengths)
    hydros = np.array(all_hydros)
    pathogens = np.array(all_pathogens)
    charges = np.array(all_charges)

    # Overall metrics
    r_overall, _ = spearmanr(targets, preds)
    r_pearson_overall, _ = pearsonr(targets, preds)

    print(f"Overall Spearman r: {r_overall:.3f}")
    print()

    # Initialize report
    report = RegimeAnalysisReport(
        model_name=checkpoint_path.stem,
        timestamp=datetime.now().isoformat(),
        n_total=len(targets),
        overall_spearman=r_overall,
        overall_pearson=r_pearson_overall,
    )

    # ==========================================================================
    # Length Regimes
    # ==========================================================================
    print("Analyzing LENGTH regimes...")
    length_masks = {
        'Short (≤15)': lengths <= 15,
        'Medium (16-25)': (lengths > 15) & (lengths <= 25),
        'Long (>25)': lengths > 25,
    }

    for name, mask in length_masks.items():
        if mask.sum() >= 10:
            stats = compute_regime_stats(
                name, targets[mask], preds[mask], n_bootstrap
            )
            report.length_regimes.append(stats)
            print(f"  {name}: N={stats.n_samples}, r={stats.spearman_r:.3f} "
                  f"[{stats.spearman_ci_low:.3f}, {stats.spearman_ci_high:.3f}]")

    # Compare short vs long
    short_mask = lengths <= 15
    long_mask = lengths > 25
    if short_mask.sum() >= 10 and long_mask.sum() >= 10:
        comp = compare_regimes(
            'Short (≤15)', targets[short_mask], preds[short_mask],
            'Long (>25)', targets[long_mask], preds[long_mask],
            n_permutations
        )
        report.length_comparisons.append(comp)
        print(f"  Short vs Long: Δr={comp.r_diff:.3f}, p={comp.p_value:.4f}, "
              f"effect={comp.effect_size:.2f}")

    print()

    # ==========================================================================
    # Hydrophobicity Regimes
    # ==========================================================================
    print("Analyzing HYDROPHOBICITY regimes...")
    hydro_masks = {
        'Hydrophilic (<0.2)': hydros < 0.2,
        'Balanced (0.2-0.5)': (hydros >= 0.2) & (hydros < 0.5),
        'Hydrophobic (>0.5)': hydros >= 0.5,
    }

    for name, mask in hydro_masks.items():
        if mask.sum() >= 10:
            stats = compute_regime_stats(
                name, targets[mask], preds[mask], n_bootstrap
            )
            report.hydrophobicity_regimes.append(stats)
            print(f"  {name}: N={stats.n_samples}, r={stats.spearman_r:.3f} "
                  f"[{stats.spearman_ci_low:.3f}, {stats.spearman_ci_high:.3f}]")

    # Compare hydrophilic vs hydrophobic
    philic_mask = hydros < 0.2
    phobic_mask = hydros >= 0.5
    if philic_mask.sum() >= 10 and phobic_mask.sum() >= 10:
        comp = compare_regimes(
            'Hydrophilic', targets[philic_mask], preds[philic_mask],
            'Hydrophobic', targets[phobic_mask], preds[phobic_mask],
            n_permutations
        )
        report.hydrophobicity_comparisons.append(comp)
        print(f"  Hydrophilic vs Hydrophobic: Δr={comp.r_diff:.3f}, p={comp.p_value:.4f}")

    print()

    # ==========================================================================
    # Gram Type Regimes
    # ==========================================================================
    print("Analyzing GRAM TYPE regimes...")
    gram_neg_mask = (pathogens == 0) | (pathogens == 1) | (pathogens == 3)
    gram_pos_mask = pathogens == 2

    for name, mask in [('Gram-negative', gram_neg_mask), ('Gram-positive', gram_pos_mask)]:
        if mask.sum() >= 10:
            stats = compute_regime_stats(
                name, targets[mask], preds[mask], n_bootstrap
            )
            report.gram_regimes.append(stats)
            print(f"  {name}: N={stats.n_samples}, r={stats.spearman_r:.3f} "
                  f"[{stats.spearman_ci_low:.3f}, {stats.spearman_ci_high:.3f}]")

    if gram_neg_mask.sum() >= 10 and gram_pos_mask.sum() >= 10:
        comp = compare_regimes(
            'Gram-negative', targets[gram_neg_mask], preds[gram_neg_mask],
            'Gram-positive', targets[gram_pos_mask], preds[gram_pos_mask],
            n_permutations
        )
        report.gram_comparisons.append(comp)
        print(f"  Gram- vs Gram+: Δr={comp.r_diff:.3f}, p={comp.p_value:.4f}")

    print()

    # ==========================================================================
    # Pathogen-Specific Regimes
    # ==========================================================================
    print("Analyzing PATHOGEN regimes...")
    pathogen_names = ['E. coli', 'P. aeruginosa', 'S. aureus', 'A. baumannii']

    for i, name in enumerate(pathogen_names):
        mask = pathogens == i
        if mask.sum() >= 10:
            stats = compute_regime_stats(
                name, targets[mask], preds[mask], n_bootstrap
            )
            report.pathogen_regimes.append(stats)
            print(f"  {name}: N={stats.n_samples}, r={stats.spearman_r:.3f} "
                  f"[{stats.spearman_ci_low:.3f}, {stats.spearman_ci_high:.3f}]")

    print()

    # ==========================================================================
    # Key Findings
    # ==========================================================================
    print("=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Check length hypothesis
    if len(report.length_comparisons) > 0:
        comp = report.length_comparisons[0]
        if comp.significant and comp.r_diff > 0.2:
            finding = (f"LENGTH REGIME CONFIRMED: Short peptides outperform long by "
                      f"Δr={comp.r_diff:.2f} (p={comp.p_value:.4f})")
            report.key_findings.append(finding)
            print(f"✓ {finding}")
            report.regime_hypothesis_confirmed = True

    # Check hydrophobicity hypothesis
    if len(report.hydrophobicity_comparisons) > 0:
        comp = report.hydrophobicity_comparisons[0]
        if comp.significant and comp.r_diff > 0.2:
            finding = (f"HYDROPHOBICITY REGIME CONFIRMED: Hydrophilic outperform hydrophobic by "
                      f"Δr={comp.r_diff:.2f} (p={comp.p_value:.4f})")
            report.key_findings.append(finding)
            print(f"✓ {finding}")
            report.regime_hypothesis_confirmed = True

    # Best and worst regimes
    all_regimes = (report.length_regimes + report.hydrophobicity_regimes +
                   report.pathogen_regimes)
    if all_regimes:
        best = max(all_regimes, key=lambda x: x.spearman_r)
        worst = min(all_regimes, key=lambda x: x.spearman_r)

        finding = f"BEST REGIME: {best.name} (r={best.spearman_r:.3f}, N={best.n_samples})"
        report.key_findings.append(finding)
        print(f"✓ {finding}")

        finding = f"WORST REGIME: {worst.name} (r={worst.spearman_r:.3f}, N={worst.n_samples})"
        report.key_findings.append(finding)
        print(f"✗ {finding}")

        if best.spearman_r - worst.spearman_r > 0.3:
            finding = f"REGIME GAP: {best.spearman_r - worst.spearman_r:.2f} correlation difference"
            report.key_findings.append(finding)
            print(f"! {finding}")

    return report


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Regime Analysis for PeptideVAE")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Number of bootstrap samples')
    parser.add_argument('--n-permutations', type=int, default=1000,
                        help='Number of permutation test iterations')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print()

    report = run_regime_analysis(
        checkpoint_path=checkpoint_path,
        device=device,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
    )

    # Save report
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = checkpoint_path.parent / f"{checkpoint_path.stem}_regime_analysis.json"

    # Convert to JSON-serializable format
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return convert_to_native(asdict(obj))
        return obj

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(convert_to_native(asdict(report)), f, indent=2)

    print()
    print(f"Report saved to: {output_path}")

    return 0 if report.regime_hypothesis_confirmed else 1


if __name__ == '__main__':
    sys.exit(main())
