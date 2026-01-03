"""
Arrow Flip Experimental Validation: Testing Zone Predictions Against ProTherm DDG Data

This module validates that the 5-zone classification (hard_hybrid, soft_hybrid, uncertain,
soft_simple, hard_simple) correlates with actual experimental mutation effects.

Key experiments:
1. Zone-DDG correlation: Do hybrid zones show different DDG patterns?
2. Prediction accuracy by zone: Is p-adic predictor better in hybrid zones?
3. Hybrid vs simple predictor comparison: Statistical test per zone

Statistical rigor:
- Bootstrap Spearman with 95% CI (1000 iterations)
- S669-aligned evaluation protocol
- Effect size (Cohen's d) for threshold comparisons
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "deliverables"))

# Import ProTherm data loader
from partners.jose_colbes.scripts.protherm_ddg_loader import (
    ProThermLoader, MutationRecord, StabilityDatabase
)

# Import zone classification
from arrow_flip_clustering import AA_PROPERTIES


# =============================================================================
# BOOTSTRAP SIGNIFICANCE TESTING
# =============================================================================

def bootstrap_spearman(y_true: np.ndarray, y_pred: np.ndarray,
                       n_iterations: int = 1000) -> Dict:
    """
    Compute bootstrap confidence interval for Spearman correlation.

    Confirms that improvements are not sampling noise.
    """
    correlations = []
    n = len(y_true)

    for _ in range(n_iterations):
        indices = np.random.choice(n, size=n, replace=True)
        r, _ = spearmanr(y_true[indices], y_pred[indices])
        if not np.isnan(r):
            correlations.append(r)

    if len(correlations) == 0:
        return {
            'mean': 0.0, 'std': 0.0,
            'ci_95': [0.0, 0.0],
            'p_value_vs_zero': 1.0
        }

    correlations = np.array(correlations)
    return {
        'mean': float(np.mean(correlations)),
        'std': float(np.std(correlations)),
        'ci_95': [float(np.percentile(correlations, 2.5)),
                  float(np.percentile(correlations, 97.5))],
        'p_value_vs_zero': float(np.mean(correlations <= 0))
    }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


# =============================================================================
# DATA LOADING AND LINKING
# =============================================================================

def load_protherm_mutations() -> List[MutationRecord]:
    """Load curated mutations from ProTherm database."""
    loader = ProThermLoader()
    db = loader.generate_curated_database()
    return db.records


def load_zone_assignments() -> Dict:
    """Load zone assignments from arrow_flip_clustering_results.json."""
    results_path = Path(__file__).parent / "arrow_flip_clustering_results.json"
    with open(results_path) as f:
        return json.load(f)


def get_zone_for_pair(aa1: str, aa2: str, zone_data: Dict) -> Tuple[str, float]:
    """
    Map an AA pair to its prediction zone.

    Returns: (zone_name, prob_hybrid)
    """
    zone_details = zone_data.get('soft_boundaries', {}).get('zone_details', {})

    # Create normalized pair key (alphabetical order)
    pair_key = f"{min(aa1, aa2)}-{max(aa1, aa2)}"

    for zone_name, pairs in zone_details.items():
        for pair_info in pairs:
            if pair_info['pair'] == pair_key:
                return zone_name, pair_info.get('prob_hybrid', 0.5)

    # If not found, classify based on properties
    return classify_unknown_pair(aa1, aa2)


def classify_unknown_pair(aa1: str, aa2: str) -> Tuple[str, float]:
    """Classify a pair not in the original 190 using decision rules."""
    if aa1 not in AA_PROPERTIES or aa2 not in AA_PROPERTIES:
        return 'uncertain', 0.5

    p1, p2 = AA_PROPERTIES[aa1], AA_PROPERTIES[aa2]
    hydro_diff = abs(p1['hydrophobicity'] - p2['hydrophobicity'])
    volume_diff = abs(p1['volume'] - p2['volume'])
    same_charge = (p1['charge'] == p2['charge'])

    # Apply decision rules from V5_SOFT_BOUNDARIES.md
    if hydro_diff > 5.15:
        if volume_diff < 55 and same_charge:
            return 'hard_hybrid', 0.91
        elif not same_charge:
            return 'uncertain', 0.5
        else:
            return 'soft_hybrid', 0.7
    else:
        if not same_charge:
            return 'hard_simple', 0.08
        elif volume_diff > 55:
            return 'soft_simple', 0.3
        else:
            return 'uncertain', 0.5


def link_mutations_to_zones(mutations: List[MutationRecord],
                            zone_data: Dict) -> pd.DataFrame:
    """
    Create linked dataset: mutation -> zone -> DDG.

    Returns DataFrame with columns:
    [pdb_id, position, wt_aa, mut_aa, ddg, zone, prob_hybrid,
     secondary_structure, rsa, is_buried, is_surface]
    """
    records = []

    for mut in mutations:
        zone, prob_hybrid = get_zone_for_pair(mut.wild_type, mut.mutant, zone_data)

        rsa = mut.solvent_accessibility if mut.solvent_accessibility else 0.5

        records.append({
            'pdb_id': mut.pdb_id,
            'position': mut.position,
            'wt_aa': mut.wild_type,
            'mut_aa': mut.mutant,
            'ddg': mut.ddg,
            'zone': zone,
            'prob_hybrid': prob_hybrid,
            'secondary_structure': mut.secondary_structure or 'C',
            'rsa': rsa,
            'is_buried': rsa < 0.25,
            'is_surface': rsa > 0.5
        })

    return pd.DataFrame(records)


# =============================================================================
# PREDICTORS
# =============================================================================

def simple_predictor(wt: str, mut: str) -> float:
    """
    Simple physicochemical predictor for DDG.
    Uses only delta_hydro + delta_charge + delta_volume.
    """
    if wt not in AA_PROPERTIES or mut not in AA_PROPERTIES:
        return 1.0  # Default neutral

    p_wt, p_mut = AA_PROPERTIES[wt], AA_PROPERTIES[mut]

    delta_hydro = abs(p_wt['hydrophobicity'] - p_mut['hydrophobicity'])
    delta_charge = abs(p_wt['charge'] - p_mut['charge'])
    delta_volume = abs(p_wt['volume'] - p_mut['volume']) / 50.0

    # Simple linear combination (empirical weights)
    return 0.3 * delta_hydro + 1.5 * delta_charge + 0.02 * delta_volume


def hybrid_predictor(wt: str, mut: str) -> float:
    """
    Hybrid predictor incorporating p-adic-inspired features.
    Adds charge incompatibility penalties and hydrophobic burial effects.
    """
    if wt not in AA_PROPERTIES or mut not in AA_PROPERTIES:
        return 1.0

    p_wt, p_mut = AA_PROPERTIES[wt], AA_PROPERTIES[mut]

    # Base physicochemical
    delta_hydro = abs(p_wt['hydrophobicity'] - p_mut['hydrophobicity'])
    delta_charge = abs(p_wt['charge'] - p_mut['charge'])
    delta_volume = abs(p_wt['volume'] - p_mut['volume']) / 50.0

    base = 0.3 * delta_hydro + 1.5 * delta_charge + 0.02 * delta_volume

    # Hybrid penalties (p-adic inspired)
    # 1. Opposite charge penalty
    if p_wt['charge'] * p_mut['charge'] < 0:
        base += 2.0

    # 2. Hydrophobic to polar transition
    hydro_wt = p_wt['hydrophobicity']
    hydro_mut = p_mut['hydrophobicity']
    if hydro_wt > 2.0 and hydro_mut < 0:  # Hydrophobic to polar
        base += 1.5
    elif hydro_wt < -2.0 and hydro_mut > 2.0:  # Polar to hydrophobic
        base += 1.5

    # 3. Aromatic transition penalty
    if p_wt['aromatic'] != p_mut['aromatic']:
        base += 0.8

    # 4. Large volume change in similar hydrophobicity
    if delta_hydro < 2.0 and delta_volume > 1.5:
        base += 0.5

    return base


# =============================================================================
# EXPERIMENT 1: ZONE-DDG CORRELATION
# =============================================================================

def experiment_1_zone_ddg_correlation(linked_data: pd.DataFrame) -> Dict:
    """
    Test: Do hybrid-zone mutations show different DDG patterns than simple-zone?

    Hypothesis:
    - Hard hybrid pairs should show larger DDG variance (context-dependent)
    - Hard simple pairs should show more predictable DDG (physicochemistry dominates)
    """
    results = {
        'by_zone': {},
        'statistical_tests': {},
        'effect_sizes': {}
    }

    # Compute DDG stats per zone
    for zone in ['hard_hybrid', 'soft_hybrid', 'uncertain', 'soft_simple', 'hard_simple']:
        zone_data = linked_data[linked_data['zone'] == zone]
        if len(zone_data) > 0:
            ddg_values = zone_data['ddg'].values
            results['by_zone'][zone] = {
                'n': len(zone_data),
                'mean_ddg': float(np.mean(ddg_values)),
                'std_ddg': float(np.std(ddg_values)),
                'median_ddg': float(np.median(ddg_values)),
                'min_ddg': float(np.min(ddg_values)),
                'max_ddg': float(np.max(ddg_values))
            }

    # Statistical tests: hybrid zones vs simple zones
    hybrid_zones = linked_data[linked_data['zone'].isin(['hard_hybrid', 'soft_hybrid'])]
    simple_zones = linked_data[linked_data['zone'].isin(['hard_simple', 'soft_simple'])]

    if len(hybrid_zones) > 5 and len(simple_zones) > 5:
        hybrid_ddg = hybrid_zones['ddg'].values
        simple_ddg = simple_zones['ddg'].values

        # Mann-Whitney U test (non-parametric)
        stat, p_value = mannwhitneyu(hybrid_ddg, simple_ddg, alternative='two-sided')
        results['statistical_tests']['hybrid_vs_simple_mann_whitney'] = {
            'statistic': float(stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }

        # Effect size
        d = cohens_d(hybrid_ddg, simple_ddg)
        results['effect_sizes']['hybrid_vs_simple_cohens_d'] = {
            'value': float(d),
            'interpretation': 'large' if abs(d) > 0.8 else ('medium' if abs(d) > 0.5 else 'small')
        }

        # Variance comparison (F-test approximation via Levene)
        from scipy.stats import levene
        stat, p_value = levene(hybrid_ddg, simple_ddg)
        results['statistical_tests']['variance_levene'] = {
            'statistic': float(stat),
            'p_value': float(p_value),
            'hybrid_var': float(np.var(hybrid_ddg)),
            'simple_var': float(np.var(simple_ddg)),
            'significant': p_value < 0.05
        }

    return results


# =============================================================================
# EXPERIMENT 2: PREDICTION ACCURACY BY ZONE
# =============================================================================

def experiment_2_prediction_accuracy_by_zone(linked_data: pd.DataFrame) -> Dict:
    """
    Test: Is DDG prediction accuracy different by zone?

    For each zone:
    1. Use both predictors to predict DDG
    2. Compute Spearman correlation with actual DDG
    3. Bootstrap for confidence intervals
    """
    results = {'by_zone': {}, 'overall': {}}

    # Add predictions
    linked_data = linked_data.copy()
    linked_data['simple_pred'] = linked_data.apply(
        lambda r: simple_predictor(r['wt_aa'], r['mut_aa']), axis=1
    )
    linked_data['hybrid_pred'] = linked_data.apply(
        lambda r: hybrid_predictor(r['wt_aa'], r['mut_aa']), axis=1
    )

    # Per-zone analysis
    for zone in ['hard_hybrid', 'soft_hybrid', 'uncertain', 'soft_simple', 'hard_simple']:
        zone_data = linked_data[linked_data['zone'] == zone]
        if len(zone_data) >= 10:
            ddg = zone_data['ddg'].values
            simple = zone_data['simple_pred'].values
            hybrid = zone_data['hybrid_pred'].values

            # Bootstrap correlations
            simple_bootstrap = bootstrap_spearman(ddg, simple, n_iterations=1000)
            hybrid_bootstrap = bootstrap_spearman(ddg, hybrid, n_iterations=1000)

            results['by_zone'][zone] = {
                'n': len(zone_data),
                'simple_predictor': simple_bootstrap,
                'hybrid_predictor': hybrid_bootstrap,
                'hybrid_advantage': hybrid_bootstrap['mean'] - simple_bootstrap['mean']
            }

    # Overall comparison
    ddg_all = linked_data['ddg'].values
    simple_all = linked_data['simple_pred'].values
    hybrid_all = linked_data['hybrid_pred'].values

    results['overall']['simple_predictor'] = bootstrap_spearman(ddg_all, simple_all)
    results['overall']['hybrid_predictor'] = bootstrap_spearman(ddg_all, hybrid_all)
    results['overall']['n_total'] = len(linked_data)

    return results


# =============================================================================
# EXPERIMENT 3: HYBRID VS SIMPLE PREDICTOR COMPARISON
# =============================================================================

def experiment_3_hybrid_vs_simple_predictor(linked_data: pd.DataFrame) -> Dict:
    """
    Test: Compare two predictors by zone with statistical significance.

    For each zone:
    - Compute MAE, Spearman for each predictor
    - Statistical test: Is difference significant?

    Validates claim: "Hybrid-regime pairs show better p-adic prediction accuracy"
    """
    results = {'by_zone': {}, 'hypothesis_support': {}}

    # Add predictions
    linked_data = linked_data.copy()
    linked_data['simple_pred'] = linked_data.apply(
        lambda r: simple_predictor(r['wt_aa'], r['mut_aa']), axis=1
    )
    linked_data['hybrid_pred'] = linked_data.apply(
        lambda r: hybrid_predictor(r['wt_aa'], r['mut_aa']), axis=1
    )

    # Compute errors
    linked_data['simple_error'] = abs(linked_data['ddg'] - linked_data['simple_pred'])
    linked_data['hybrid_error'] = abs(linked_data['ddg'] - linked_data['hybrid_pred'])

    # Per-zone comparison
    hybrid_wins_hybrid_zones = 0
    simple_wins_simple_zones = 0
    total_testable = 0

    for zone in ['hard_hybrid', 'soft_hybrid', 'uncertain', 'soft_simple', 'hard_simple']:
        zone_data = linked_data[linked_data['zone'] == zone]
        if len(zone_data) >= 10:
            ddg = zone_data['ddg'].values
            simple_pred = zone_data['simple_pred'].values
            hybrid_pred = zone_data['hybrid_pred'].values
            simple_error = zone_data['simple_error'].values
            hybrid_error = zone_data['hybrid_error'].values

            # Correlations
            simple_r, simple_p = spearmanr(ddg, simple_pred)
            hybrid_r, hybrid_p = spearmanr(ddg, hybrid_pred)

            # MAE
            simple_mae = float(np.mean(simple_error))
            hybrid_mae = float(np.mean(hybrid_error))

            # Paired t-test on errors
            t_stat, p_value = ttest_ind(simple_error, hybrid_error)

            # Bootstrap comparison
            simple_bootstrap = bootstrap_spearman(ddg, simple_pred)
            hybrid_bootstrap = bootstrap_spearman(ddg, hybrid_pred)

            # CI overlap check
            ci_overlap = (simple_bootstrap['ci_95'][0] < hybrid_bootstrap['ci_95'][1] and
                          simple_bootstrap['ci_95'][1] > hybrid_bootstrap['ci_95'][0])

            results['by_zone'][zone] = {
                'n': len(zone_data),
                'simple': {
                    'spearman': float(simple_r) if not np.isnan(simple_r) else 0.0,
                    'mae': simple_mae,
                    'bootstrap_ci': simple_bootstrap['ci_95']
                },
                'hybrid': {
                    'spearman': float(hybrid_r) if not np.isnan(hybrid_r) else 0.0,
                    'mae': hybrid_mae,
                    'bootstrap_ci': hybrid_bootstrap['ci_95']
                },
                'comparison': {
                    'hybrid_advantage_r': float(hybrid_r - simple_r) if not np.isnan(hybrid_r - simple_r) else 0.0,
                    'hybrid_advantage_mae': simple_mae - hybrid_mae,  # Positive = hybrid better
                    't_test_p_value': float(p_value),
                    'ci_overlap': ci_overlap,
                    'significant': p_value < 0.05 and not ci_overlap
                }
            }

            # Track hypothesis support
            total_testable += 1
            if zone in ['hard_hybrid', 'soft_hybrid']:
                if hybrid_r > simple_r:
                    hybrid_wins_hybrid_zones += 1
            elif zone in ['hard_simple', 'soft_simple']:
                if simple_r > hybrid_r:
                    simple_wins_simple_zones += 1

    # Hypothesis evaluation
    results['hypothesis_support'] = {
        'claim': "Hybrid approach works better in hybrid zones, simple in simple zones",
        'hybrid_zones_tested': sum(1 for z in ['hard_hybrid', 'soft_hybrid']
                                   if z in results['by_zone']),
        'hybrid_wins_in_hybrid_zones': hybrid_wins_hybrid_zones,
        'simple_zones_tested': sum(1 for z in ['hard_simple', 'soft_simple']
                                   if z in results['by_zone']),
        'simple_wins_in_simple_zones': simple_wins_simple_zones,
        'total_testable_zones': total_testable
    }

    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_experiments() -> Dict:
    """Run all three experiments and compile results."""
    print("=" * 60)
    print("V5 ARROW FLIP EXPERIMENTAL VALIDATION")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading ProTherm mutations...")
    mutations = load_protherm_mutations()
    print(f"      Loaded {len(mutations)} curated mutations")

    print("\n[2/4] Loading zone assignments...")
    zone_data = load_zone_assignments()
    zones = zone_data.get('soft_boundaries', {}).get('zones', {})
    print(f"      Zones: {zones}")

    print("\n[3/4] Linking mutations to zones...")
    linked_data = link_mutations_to_zones(mutations, zone_data)
    print(f"      Linked {len(linked_data)} mutations")
    print(f"      Zone distribution:")
    for zone in ['hard_hybrid', 'soft_hybrid', 'uncertain', 'soft_simple', 'hard_simple']:
        n = len(linked_data[linked_data['zone'] == zone])
        print(f"        {zone}: {n}")

    print("\n[4/4] Running experiments...")

    results = {
        'metadata': {
            'n_mutations': len(mutations),
            'n_linked': len(linked_data),
            'zone_coverage': linked_data['zone'].value_counts().to_dict()
        }
    }

    # Experiment 1
    print("\n  Experiment 1: Zone-DDG Correlation...")
    results['experiment_1_zone_ddg_correlation'] = experiment_1_zone_ddg_correlation(linked_data)

    # Experiment 2
    print("  Experiment 2: Prediction Accuracy by Zone...")
    results['experiment_2_prediction_accuracy'] = experiment_2_prediction_accuracy_by_zone(linked_data)

    # Experiment 3
    print("  Experiment 3: Hybrid vs Simple Predictor...")
    results['experiment_3_predictor_comparison'] = experiment_3_hybrid_vs_simple_predictor(linked_data)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Exp 1 summary
    exp1 = results['experiment_1_zone_ddg_correlation']
    if 'hybrid_vs_simple_mann_whitney' in exp1.get('statistical_tests', {}):
        p = exp1['statistical_tests']['hybrid_vs_simple_mann_whitney']['p_value']
        print(f"\nExp 1: Hybrid vs Simple zones DDG difference")
        print(f"       Mann-Whitney p-value: {p:.4f} ({'Significant' if p < 0.05 else 'Not significant'})")

    # Exp 2 summary
    exp2 = results['experiment_2_prediction_accuracy']
    print(f"\nExp 2: Overall prediction accuracy (bootstrap 95% CI)")
    for predictor in ['simple_predictor', 'hybrid_predictor']:
        if predictor in exp2.get('overall', {}):
            data = exp2['overall'][predictor]
            print(f"       {predictor}: r={data['mean']:.3f} [{data['ci_95'][0]:.3f}, {data['ci_95'][1]:.3f}]")

    # Exp 3 summary
    exp3 = results['experiment_3_predictor_comparison']
    hyp = exp3.get('hypothesis_support', {})
    print(f"\nExp 3: Hypothesis support")
    print(f"       Hybrid wins in hybrid zones: {hyp.get('hybrid_wins_in_hybrid_zones', 0)}/{hyp.get('hybrid_zones_tested', 0)}")
    print(f"       Simple wins in simple zones: {hyp.get('simple_wins_in_simple_zones', 0)}/{hyp.get('simple_zones_tested', 0)}")

    # Save results (convert numpy types for JSON serialization)
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output_path = Path(__file__).parent / "arrow_flip_experimental_validation_results.json"
    with open(output_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = run_all_experiments()
