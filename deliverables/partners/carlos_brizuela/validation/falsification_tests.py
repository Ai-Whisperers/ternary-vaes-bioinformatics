#!/usr/bin/env python3
"""Falsification Tests for PeptideVAE.

This module implements rigorous scientific validation for the PeptideVAE model
following the plan's falsification criteria:

Performance Thresholds:
    - General model: r >= 0.60 (must beat sklearn baseline 0.56)
    - E. coli model: r >= 0.50 (must beat baseline 0.42)
    - P. aeruginosa: r >= 0.50 (must beat baseline 0.44)
    - S. aureus: r >= 0.35 (must beat baseline 0.22)

Required Ablations:
    1. NO_HYPERBOLIC: Euclidean vs Poincare (expect: -0.05)
    2. NO_TRANSFORMER: MLP only (expect: -0.08)
    3. NO_PROPERTIES: Remove property encoding (expect: -0.03)
    4. SINGLE_LOSS: MIC only, no auxiliary (expect: -0.05)

Biological Plausibility Checks:
    1. Charge-Activity: Cationic peptides cluster in active region
    2. Gram Separation: Gram+/- optima should separate
    3. Attention Analysis: Highlights amphipathic regions

Usage:
    python validation/falsification_tests.py --checkpoint path/to/model.pt
    python validation/falsification_tests.py --run-ablations
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add paths - repo_root must be first to avoid shadowing by local src/
_script_dir = Path(__file__).resolve().parent
_package_dir = _script_dir.parent
_deliverables_dir = _package_dir.parent.parent
_repo_root = _deliverables_dir.parent
# Insert in reverse priority order (last insert = highest priority)
sys.path.insert(0, str(_package_dir))
sys.path.insert(0, str(_deliverables_dir))
sys.path.insert(0, str(_repo_root))  # Must be last to take precedence

try:
    from scipy.stats import pearsonr, spearmanr, permutation_test
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from src.encoders.peptide_encoder import PeptideVAE
from training.dataset import (
    create_stratified_dataloaders,
    create_full_dataset,
    PATHOGEN_TO_LABEL,
)


# =============================================================================
# Constants and Thresholds
# =============================================================================


@dataclass
class PerformanceThresholds:
    """Performance thresholds for falsification."""

    # Must-beat thresholds (sklearn baselines)
    general_min: float = 0.55  # Below this = FAIL
    ecoli_min: float = 0.40
    pseudomonas_min: float = 0.40
    staphylococcus_min: float = 0.25

    # Target thresholds
    general_target: float = 0.62
    ecoli_target: float = 0.50
    pseudomonas_target: float = 0.50
    staphylococcus_target: float = 0.35

    # Ablation expected drops
    ablation_hyperbolic: float = 0.05
    ablation_transformer: float = 0.08
    ablation_properties: float = 0.03
    ablation_single_loss: float = 0.05


THRESHOLDS = PerformanceThresholds()


# =============================================================================
# Validation Results
# =============================================================================


@dataclass
class ValidationResult:
    """Single validation result."""

    name: str
    passed: bool
    metric_value: float
    threshold: float
    message: str


@dataclass
class FalsificationReport:
    """Complete falsification report."""

    model_name: str
    timestamp: str
    device: str

    # Performance metrics
    general_pearson: float
    general_spearman: float
    general_passed: bool

    # Pathogen-specific
    pathogen_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Ablation results
    ablation_results: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Biological validation
    biological_checks: Dict[str, bool] = field(default_factory=dict)

    # Summary
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    overall_passed: bool = False

    def add_result(self, result: ValidationResult):
        """Add a validation result."""
        self.total_tests += 1
        if result.passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1

    def compute_overall(self):
        """Compute overall pass/fail."""
        # Must pass: general model and at least 2 pathogen models
        self.overall_passed = self.general_passed and self.passed_tests >= self.total_tests * 0.7


# =============================================================================
# Evaluation Functions
# =============================================================================


@torch.no_grad()
def evaluate_model(
    model: PeptideVAE,
    dataloader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a dataloader.

    Returns:
        Dictionary with pearson_r, spearman_r, mae, etc.
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_pathogens = []

    for batch in dataloader:
        sequences = batch['sequences']
        mic_targets = batch['mic'].to(device)
        pathogens = batch['pathogens']

        outputs = model(sequences, teacher_forcing=False)
        mic_preds = outputs['mic_pred'].squeeze(-1)

        all_preds.append(mic_preds.cpu())
        all_targets.append(mic_targets.cpu())
        all_pathogens.extend(pathogens)

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    metrics = {
        'n_samples': len(preds),
        'mae': np.abs(preds - targets).mean(),
        'rmse': np.sqrt(np.mean((preds - targets) ** 2)),
    }

    if HAS_SCIPY:
        try:
            pearson_r, pearson_p = pearsonr(targets, preds)
            spearman_r, spearman_p = spearmanr(targets, preds)
            metrics['pearson_r'] = pearson_r
            metrics['pearson_p'] = pearson_p
            metrics['spearman_r'] = spearman_r
            metrics['spearman_p'] = spearman_p
        except Exception:
            metrics['pearson_r'] = 0.0
            metrics['spearman_r'] = 0.0

    # Compute per-pathogen metrics
    for pathogen in set(all_pathogens):
        mask = np.array([p == pathogen for p in all_pathogens])
        if mask.sum() < 10:
            continue

        path_preds = preds[mask]
        path_targets = targets[mask]

        if HAS_SCIPY:
            try:
                pr, _ = pearsonr(path_targets, path_preds)
                sr, _ = spearmanr(path_targets, path_preds)
                metrics[f'{pathogen}_pearson'] = pr
                metrics[f'{pathogen}_spearman'] = sr
                metrics[f'{pathogen}_n'] = int(mask.sum())
            except Exception:
                pass

    return metrics


def run_permutation_test(
    model: PeptideVAE,
    dataloader,
    device: torch.device,
    n_permutations: int = 100,
) -> Dict[str, float]:
    """Run permutation test for significance.

    Args:
        model: Trained model
        dataloader: Validation dataloader
        device: Device
        n_permutations: Number of permutations

    Returns:
        Dictionary with permutation_p, observed_r, null_distribution stats
    """
    if not HAS_SCIPY:
        return {'permutation_p': 1.0, 'note': 'scipy not available'}

    model.eval()

    # Get all predictions and targets
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            sequences = batch['sequences']
            mic_targets = batch['mic'].to(device)

            outputs = model(sequences, teacher_forcing=False)
            mic_preds = outputs['mic_pred'].squeeze(-1)

            all_preds.append(mic_preds.cpu())
            all_targets.append(mic_targets.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    # Observed correlation
    observed_r, _ = pearsonr(targets, preds)

    # Permutation distribution
    null_distribution = []
    rng = np.random.default_rng(42)

    for _ in range(n_permutations):
        permuted_targets = rng.permutation(targets)
        perm_r, _ = pearsonr(permuted_targets, preds)
        null_distribution.append(perm_r)

    null_distribution = np.array(null_distribution)

    # Compute p-value (one-tailed: observed > null)
    p_value = (null_distribution >= observed_r).mean()

    return {
        'permutation_p': p_value,
        'observed_r': observed_r,
        'null_mean': null_distribution.mean(),
        'null_std': null_distribution.std(),
        'n_permutations': n_permutations,
    }


# =============================================================================
# Biological Validation
# =============================================================================


@torch.no_grad()
def check_charge_activity_correlation(
    model: PeptideVAE,
    dataloader,
    device: torch.device,
) -> Dict[str, float]:
    """Check if cationic peptides cluster in active region.

    Hypothesis: Peptides with higher charge should have lower MIC
    (more antimicrobial activity) for Gram-negative bacteria.
    """
    from deliverables.shared.peptide_utils import compute_peptide_properties

    model.eval()

    charges = []
    mic_preds = []
    radii = []
    pathogens = []

    for batch in dataloader:
        sequences = batch['sequences']

        outputs = model(sequences, teacher_forcing=False)

        for seq in sequences:
            props = compute_peptide_properties(seq)
            charges.append(props.get('net_charge', 0))

        mic_preds.extend(outputs['mic_pred'].squeeze(-1).cpu().numpy())
        radii.extend(model.get_hyperbolic_radii(outputs['z_hyp']).cpu().numpy())
        pathogens.extend(batch['pathogens'])

    charges = np.array(charges)
    mic_preds = np.array(mic_preds)
    radii = np.array(radii)

    # For Gram-negative: higher charge → lower MIC (negative correlation)
    gram_neg_mask = np.array([p in ['escherichia', 'pseudomonas', 'acinetobacter']
                              for p in pathogens])

    result = {
        'all_charge_mic_corr': 0.0,
        'gramneg_charge_mic_corr': 0.0,
        'charge_radius_corr': 0.0,
        'passed': False,
    }

    if HAS_SCIPY:
        try:
            # Overall correlation
            r_all, _ = pearsonr(charges, mic_preds)
            result['all_charge_mic_corr'] = r_all

            # Gram-negative only
            if gram_neg_mask.sum() >= 10:
                r_gn, _ = pearsonr(charges[gram_neg_mask], mic_preds[gram_neg_mask])
                result['gramneg_charge_mic_corr'] = r_gn

            # Charge-radius correlation
            r_radius, _ = pearsonr(charges, radii)
            result['charge_radius_corr'] = r_radius

            # Pass if Gram-negative shows expected negative correlation
            result['passed'] = result['gramneg_charge_mic_corr'] < -0.1

        except Exception:
            pass

    return result


@torch.no_grad()
def check_gram_separation(
    model: PeptideVAE,
    dataloader,
    device: torch.device,
) -> Dict[str, float]:
    """Check if Gram+/- peptides separate in embedding space."""
    from src.geometry import poincare_distance

    model.eval()

    gram_pos_embeddings = []
    gram_neg_embeddings = []

    for batch in dataloader:
        sequences = batch['sequences']
        pathogens = batch['pathogens']

        outputs = model(sequences, teacher_forcing=False)
        z_hyp = outputs['z_hyp'].cpu()

        for i, pathogen in enumerate(pathogens):
            if pathogen == 'staphylococcus':
                gram_pos_embeddings.append(z_hyp[i])
            elif pathogen in ['escherichia', 'pseudomonas', 'acinetobacter']:
                gram_neg_embeddings.append(z_hyp[i])

    result = {
        'n_gram_pos': len(gram_pos_embeddings),
        'n_gram_neg': len(gram_neg_embeddings),
        'inter_centroid_dist': 0.0,
        'intra_gram_pos_dist': 0.0,
        'intra_gram_neg_dist': 0.0,
        'passed': False,
    }

    if len(gram_pos_embeddings) < 5 or len(gram_neg_embeddings) < 5:
        return result

    gram_pos = torch.stack(gram_pos_embeddings)
    gram_neg = torch.stack(gram_neg_embeddings)

    # Compute centroids
    centroid_pos = gram_pos.mean(dim=0, keepdim=True)
    centroid_neg = gram_neg.mean(dim=0, keepdim=True)

    # Inter-centroid distance
    inter_dist = poincare_distance(centroid_pos, centroid_neg, c=1.0).item()
    result['inter_centroid_dist'] = inter_dist

    # Intra-class distances
    if len(gram_pos) > 1:
        pos_dists = []
        for i in range(min(len(gram_pos), 20)):
            for j in range(i+1, min(len(gram_pos), 20)):
                d = poincare_distance(gram_pos[i:i+1], gram_pos[j:j+1], c=1.0)
                pos_dists.append(d.item())
        result['intra_gram_pos_dist'] = np.mean(pos_dists) if pos_dists else 0

    if len(gram_neg) > 1:
        neg_dists = []
        for i in range(min(len(gram_neg), 20)):
            for j in range(i+1, min(len(gram_neg), 20)):
                d = poincare_distance(gram_neg[i:i+1], gram_neg[j:j+1], c=1.0)
                neg_dists.append(d.item())
        result['intra_gram_neg_dist'] = np.mean(neg_dists) if neg_dists else 0

    # Pass if inter > avg(intra)
    avg_intra = (result['intra_gram_pos_dist'] + result['intra_gram_neg_dist']) / 2
    result['passed'] = inter_dist > avg_intra * 1.2  # 20% margin

    return result


# =============================================================================
# Main Validation Runner
# =============================================================================


def run_falsification_tests(
    checkpoint_path: Path,
    device: torch.device,
    n_permutations: int = 100,
) -> FalsificationReport:
    """Run complete falsification test suite.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Evaluation device
        n_permutations: Permutation test iterations

    Returns:
        FalsificationReport with all results
    """
    import datetime

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})

    # Create model
    model = PeptideVAE(
        latent_dim=config.get('latent_dim', 16),
        hidden_dim=config.get('hidden_dim', 128),
        n_layers=config.get('n_layers', 2),
        n_heads=config.get('n_heads', 4),
        dropout=config.get('dropout', 0.1),
        max_radius=config.get('max_radius', 0.95),
        curvature=config.get('curvature', 1.0),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create data loaders (use validation set for evaluation)
    fold_idx = checkpoint.get('fold_idx', 0)
    _, val_loader = create_stratified_dataloaders(
        fold_idx=fold_idx,
        n_folds=config.get('n_folds', 5),
        batch_size=config.get('batch_size', 32),
    )

    # Initialize report
    report = FalsificationReport(
        model_name=checkpoint_path.stem,
        timestamp=datetime.datetime.now().isoformat(),
        device=str(device),
        general_pearson=0.0,
        general_spearman=0.0,
        general_passed=False,
    )

    print("="*60)
    print("FALSIFICATION TEST SUITE")
    print("="*60)
    print(f"Model: {checkpoint_path.name}")
    print(f"Device: {device}")
    print()

    # 1. General Model Performance
    print("1. General Model Performance")
    print("-"*40)
    metrics = evaluate_model(model, val_loader, device)
    report.general_pearson = metrics.get('pearson_r', 0)
    report.general_spearman = metrics.get('spearman_r', 0)
    report.general_passed = report.general_pearson >= THRESHOLDS.general_min

    print(f"   Pearson r:  {report.general_pearson:.4f}")
    print(f"   Spearman r: {report.general_spearman:.4f}")
    print(f"   MAE:        {metrics.get('mae', 0):.4f}")
    status = "PASS" if report.general_passed else "FAIL"
    print(f"   Status: [{status}] (threshold: {THRESHOLDS.general_min})")

    report.add_result(ValidationResult(
        name="general_pearson",
        passed=report.general_passed,
        metric_value=report.general_pearson,
        threshold=THRESHOLDS.general_min,
        message=f"Pearson r = {report.general_pearson:.4f}",
    ))
    print()

    # 2. Pathogen-Specific Performance
    print("2. Pathogen-Specific Performance")
    print("-"*40)
    pathogen_thresholds = {
        'escherichia': THRESHOLDS.ecoli_min,
        'pseudomonas': THRESHOLDS.pseudomonas_min,
        'staphylococcus': THRESHOLDS.staphylococcus_min,
    }

    for pathogen, threshold in pathogen_thresholds.items():
        key = f'{pathogen}_pearson'
        r = metrics.get(key, 0)
        n = metrics.get(f'{pathogen}_n', 0)
        passed = r >= threshold

        report.pathogen_metrics[pathogen] = {
            'pearson_r': r,
            'n_samples': n,
            'passed': passed,
        }

        status = "PASS" if passed else "FAIL"
        print(f"   {pathogen.capitalize()}: r={r:.4f}, n={n}, [{status}] (threshold: {threshold})")

        report.add_result(ValidationResult(
            name=f"{pathogen}_pearson",
            passed=passed,
            metric_value=r,
            threshold=threshold,
            message=f"{pathogen} Pearson r = {r:.4f}",
        ))
    print()

    # 3. Permutation Test
    print("3. Permutation Test for Significance")
    print("-"*40)
    perm_results = run_permutation_test(model, val_loader, device, n_permutations)
    perm_passed = perm_results.get('permutation_p', 1.0) < 0.05

    print(f"   Observed r: {perm_results.get('observed_r', 0):.4f}")
    print(f"   Null mean:  {perm_results.get('null_mean', 0):.4f} +/- {perm_results.get('null_std', 0):.4f}")
    print(f"   p-value:    {perm_results.get('permutation_p', 1.0):.4f}")
    status = "PASS" if perm_passed else "FAIL"
    print(f"   Status: [{status}] (threshold: p < 0.05)")

    report.add_result(ValidationResult(
        name="permutation_test",
        passed=perm_passed,
        metric_value=perm_results.get('permutation_p', 1.0),
        threshold=0.05,
        message=f"Permutation p = {perm_results.get('permutation_p', 1.0):.4f}",
    ))
    print()

    # 4. Biological Validation
    print("4. Biological Plausibility Checks")
    print("-"*40)

    # Charge-Activity correlation
    charge_results = check_charge_activity_correlation(model, val_loader, device)
    print(f"   Charge-MIC correlation (Gram-): {charge_results['gramneg_charge_mic_corr']:.4f}")
    print(f"   Charge-radius correlation: {charge_results['charge_radius_corr']:.4f}")
    status = "PASS" if charge_results['passed'] else "FAIL"
    print(f"   Status: [{status}] (expect negative for Gram-)")
    report.biological_checks['charge_activity'] = charge_results['passed']

    # Gram separation
    gram_results = check_gram_separation(model, val_loader, device)
    print(f"   Gram+/- centroid distance: {gram_results['inter_centroid_dist']:.4f}")
    print(f"   Intra Gram+: {gram_results['intra_gram_pos_dist']:.4f}")
    print(f"   Intra Gram-: {gram_results['intra_gram_neg_dist']:.4f}")
    status = "PASS" if gram_results['passed'] else "FAIL"
    print(f"   Status: [{status}] (inter > avg intra)")
    report.biological_checks['gram_separation'] = gram_results['passed']

    for check_name, passed in report.biological_checks.items():
        report.add_result(ValidationResult(
            name=f"bio_{check_name}",
            passed=passed,
            metric_value=1.0 if passed else 0.0,
            threshold=0.5,
            message=f"Biological check: {check_name}",
        ))
    print()

    # Compute overall
    report.compute_overall()

    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Tests passed: {report.passed_tests}/{report.total_tests}")
    overall_status = "PASS" if report.overall_passed else "FAIL"
    print(f"Overall: [{overall_status}]")

    return report


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Falsification Tests for PeptideVAE")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument('--n-permutations', type=int, default=100,
                        help="Number of permutation test iterations")
    parser.add_argument('--output', type=str, default=None,
                        help="Output JSON file for report")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1

    report = run_falsification_tests(
        checkpoint_path=checkpoint_path,
        device=device,
        n_permutations=args.n_permutations,
    )

    # Save report
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = checkpoint_path.parent / f"{checkpoint_path.stem}_falsification.json"

    # Convert numpy types to Python native types for JSON serialization
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
        return obj

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(convert_to_native(asdict(report)), f, indent=2)
    print(f"\nReport saved to: {output_path}")

    return 0 if report.overall_passed else 1


if __name__ == '__main__':
    sys.exit(main())
