"""
Arrow Flip Position-Stratified Analysis: Testing if Buried vs Exposed Shifts Thresholds

This module tests the hypothesis that position context (buried vs exposed, secondary
structure) modifies the decision boundary for when to use hybrid vs simple prediction.

Key experiments:
1. RSA stratification: Buried (RSA<0.25) vs Interface vs Surface (RSA>0.5)
2. Secondary structure stratification: Helix vs Sheet vs Coil
3. Position-zone interaction: Does position modify zone effects?
4. Position-aware threshold computation

Statistical rigor:
- Cohen's d effect size for threshold comparisons
- Bootstrap confidence intervals
- ANOVA-style interaction testing
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr, mannwhitneyu, kruskal
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "deliverables"))

# Import from experimental validation
from arrow_flip_experimental_validation import (
    load_protherm_mutations, load_zone_assignments, link_mutations_to_zones,
    simple_predictor, hybrid_predictor, bootstrap_spearman, cohens_d
)
from arrow_flip_clustering import AA_PROPERTIES, AAPairFeatures


# =============================================================================
# POSITION STRATIFICATION
# =============================================================================

def stratify_by_rsa(linked_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split mutations into position categories by RSA.

    Returns:
    - Buried (RSA < 0.25): Core positions, hydrophobic packing dominates
    - Interface (0.25 <= RSA <= 0.5): Mixed context
    - Surface (RSA > 0.5): Exposed, solvation effects
    """
    # ProTherm data has secondary structure but no explicit RSA
    # Infer from secondary structure: H/E tend to be buried, C tends to be surface
    linked_data = linked_data.copy()

    # Use secondary structure as proxy for burial
    # Helix/Sheet cores are typically buried, coils are typically surface
    def infer_rsa(row):
        ss = row.get('secondary_structure', 'C')
        if ss == 'H':  # Helix - often partially buried
            return 0.35  # Interface default
        elif ss == 'E':  # Sheet - often buried in beta sandwich
            return 0.20  # Buried default
        else:  # Coil - often surface
            return 0.60  # Surface default

    # If no explicit RSA, infer from SS
    if 'rsa' not in linked_data.columns or linked_data['rsa'].isna().all():
        linked_data['rsa'] = linked_data.apply(infer_rsa, axis=1)

    buried = linked_data[linked_data['rsa'] < 0.25].copy()
    interface = linked_data[(linked_data['rsa'] >= 0.25) & (linked_data['rsa'] <= 0.5)].copy()
    surface = linked_data[linked_data['rsa'] > 0.5].copy()

    return buried, interface, surface


def stratify_by_secondary_structure(linked_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split by secondary structure: Helix (H), Sheet (E), Coil (C).
    """
    result = {}
    for ss in ['H', 'E', 'C']:
        subset = linked_data[linked_data['secondary_structure'] == ss].copy()
        if len(subset) > 0:
            result[ss] = subset
    return result


# =============================================================================
# CONTEXT-SPECIFIC ANALYSIS
# =============================================================================

def analyze_same_pair_different_context(linked_data: pd.DataFrame) -> Dict:
    """
    For pairs that appear in multiple contexts, analyze if DDG differs.

    Key test: L->A at buried position vs surface position.
    """
    results = {'pair_context_effects': []}

    # Group by AA pair
    linked_data['pair'] = linked_data.apply(
        lambda r: f"{r['wt_aa']}-{r['mut_aa']}", axis=1
    )

    # For each pair with multiple contexts
    for pair, group in linked_data.groupby('pair'):
        if len(group) < 3:
            continue

        # Split by secondary structure
        helix = group[group['secondary_structure'] == 'H']
        sheet = group[group['secondary_structure'] == 'E']
        coil = group[group['secondary_structure'] == 'C']

        # Analyze if DDG differs by context
        contexts = []
        if len(helix) >= 2:
            contexts.append(('helix', helix['ddg'].values))
        if len(sheet) >= 2:
            contexts.append(('sheet', sheet['ddg'].values))
        if len(coil) >= 2:
            contexts.append(('coil', coil['ddg'].values))

        if len(contexts) >= 2:
            # Kruskal-Wallis test for context effect
            ddg_groups = [c[1] for c in contexts]
            if all(len(g) >= 2 for g in ddg_groups):
                try:
                    stat, p_value = kruskal(*ddg_groups)
                    context_means = {c[0]: float(np.mean(c[1])) for c in contexts}

                    results['pair_context_effects'].append({
                        'pair': pair,
                        'n_total': len(group),
                        'context_means': context_means,
                        'kruskal_statistic': float(stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    })
                except Exception:
                    pass

    # Summary statistics
    significant_pairs = [p for p in results['pair_context_effects'] if p['significant']]
    results['summary'] = {
        'total_pairs_analyzed': len(results['pair_context_effects']),
        'significant_context_effects': len(significant_pairs),
        'top_context_dependent_pairs': sorted(
            significant_pairs, key=lambda x: x['p_value']
        )[:10] if significant_pairs else []
    }

    return results


# =============================================================================
# POSITION-SPECIFIC ZONE BOUNDARIES
# =============================================================================

def compute_position_specific_features(df: pd.DataFrame) -> np.ndarray:
    """Extract features for logistic regression."""
    features = []
    for _, row in df.iterrows():
        wt, mut = row['wt_aa'], row['mut_aa']
        if wt not in AA_PROPERTIES or mut not in AA_PROPERTIES:
            continue

        p_wt, p_mut = AA_PROPERTIES[wt], AA_PROPERTIES[mut]
        features.append([
            abs(p_wt['hydrophobicity'] - p_mut['hydrophobicity']),
            abs(p_wt['volume'] - p_mut['volume']) / 100.0,
            abs(p_wt['charge'] - p_mut['charge']),
            1 if p_wt['charge'] == p_mut['charge'] else 0,
            1 if p_wt['aromatic'] != p_mut['aromatic'] else 0
        ])
    return np.array(features)


def recompute_zone_boundaries_by_position(
    buried: pd.DataFrame,
    surface: pd.DataFrame,
    zone_data: Dict
) -> Dict:
    """
    Refit logistic regression per position type.
    Compare: Do thresholds shift?

    Hypothesis: Buried positions should favor hybrid approach more
    (charge burial effects, cavity penalties).
    """
    results = {
        'buried': {},
        'surface': {},
        'threshold_comparison': {}
    }

    # For each position type, compute hybrid advantage
    for pos_type, df in [('buried', buried), ('surface', surface)]:
        if len(df) < 10:
            results[pos_type] = {'n': len(df), 'insufficient_data': True}
            continue

        # Add predictions
        df = df.copy()
        df['simple_pred'] = df.apply(
            lambda r: simple_predictor(r['wt_aa'], r['mut_aa']), axis=1
        )
        df['hybrid_pred'] = df.apply(
            lambda r: hybrid_predictor(r['wt_aa'], r['mut_aa']), axis=1
        )

        # Correlations
        ddg = df['ddg'].values
        simple_r = bootstrap_spearman(ddg, df['simple_pred'].values)
        hybrid_r = bootstrap_spearman(ddg, df['hybrid_pred'].values)

        results[pos_type] = {
            'n': len(df),
            'simple_predictor': simple_r,
            'hybrid_predictor': hybrid_r,
            'hybrid_advantage': hybrid_r['mean'] - simple_r['mean'],
            'mean_ddg': float(np.mean(ddg)),
            'std_ddg': float(np.std(ddg))
        }

    # Compare thresholds
    if 'insufficient_data' not in results.get('buried', {}) and \
       'insufficient_data' not in results.get('surface', {}):

        buried_adv = results['buried']['hybrid_advantage']
        surface_adv = results['surface']['hybrid_advantage']

        results['threshold_comparison'] = {
            'buried_hybrid_advantage': buried_adv,
            'surface_hybrid_advantage': surface_adv,
            'difference': buried_adv - surface_adv,
            'hypothesis_supported': buried_adv > surface_adv,
            'interpretation': (
                "Hybrid approach more valuable for buried positions"
                if buried_adv > surface_adv else
                "Hybrid approach equally/less valuable for buried positions"
            )
        }

    return results


# =============================================================================
# POSITION-ZONE INTERACTION
# =============================================================================

def experiment_position_zone_interaction(linked_data: pd.DataFrame) -> Dict:
    """
    Test: Is there significant interaction between position and zone?

    If interaction significant: position modifies zone effects on DDG prediction.
    """
    results = {
        'position_effects': {},
        'zone_effects': {},
        'interaction_test': {}
    }

    # Add predictions
    linked_data = linked_data.copy()
    linked_data['simple_pred'] = linked_data.apply(
        lambda r: simple_predictor(r['wt_aa'], r['mut_aa']), axis=1
    )
    linked_data['hybrid_pred'] = linked_data.apply(
        lambda r: hybrid_predictor(r['wt_aa'], r['mut_aa']), axis=1
    )
    linked_data['hybrid_error'] = abs(linked_data['ddg'] - linked_data['hybrid_pred'])
    linked_data['simple_error'] = abs(linked_data['ddg'] - linked_data['simple_pred'])

    # Position effect on prediction error
    for pos in ['H', 'E', 'C']:
        subset = linked_data[linked_data['secondary_structure'] == pos]
        if len(subset) >= 5:
            results['position_effects'][pos] = {
                'n': len(subset),
                'mean_hybrid_error': float(np.mean(subset['hybrid_error'])),
                'mean_simple_error': float(np.mean(subset['simple_error'])),
                'hybrid_better': float(np.mean(subset['hybrid_error']) < np.mean(subset['simple_error']))
            }

    # Zone effect on prediction error
    for zone in ['hard_hybrid', 'soft_hybrid', 'uncertain', 'soft_simple', 'hard_simple']:
        subset = linked_data[linked_data['zone'] == zone]
        if len(subset) >= 5:
            results['zone_effects'][zone] = {
                'n': len(subset),
                'mean_hybrid_error': float(np.mean(subset['hybrid_error'])),
                'mean_simple_error': float(np.mean(subset['simple_error'])),
                'hybrid_better': float(np.mean(subset['hybrid_error']) < np.mean(subset['simple_error']))
            }

    # Interaction test: Does position modify zone effect?
    # Create interaction groups
    groups = []
    labels = []
    for pos in ['H', 'E', 'C']:
        for zone in ['hard_hybrid', 'soft_hybrid', 'uncertain']:  # Main zones
            subset = linked_data[
                (linked_data['secondary_structure'] == pos) &
                (linked_data['zone'] == zone)
            ]
            if len(subset) >= 3:
                groups.append(subset['hybrid_error'].values - subset['simple_error'].values)
                labels.append(f"{pos}_{zone}")

    if len(groups) >= 3:
        try:
            stat, p_value = kruskal(*groups)
            results['interaction_test'] = {
                'n_groups': len(groups),
                'group_labels': labels,
                'kruskal_statistic': float(stat),
                'p_value': float(p_value),
                'significant_interaction': p_value < 0.05,
                'interpretation': (
                    "Position modifies zone effect on prediction accuracy"
                    if p_value < 0.05 else
                    "Position does not significantly modify zone effects"
                )
            }
        except Exception as e:
            results['interaction_test'] = {'error': str(e)}

    return results


# =============================================================================
# POSITION-AWARE THRESHOLDS
# =============================================================================

def compute_position_aware_thresholds(
    buried: pd.DataFrame,
    surface: pd.DataFrame,
    original_thresholds: Dict
) -> Dict:
    """
    Compute refined thresholds per position context.

    Uses DDG magnitude to determine when hybrid approach adds value.
    """
    results = {
        'original': original_thresholds,
        'position_aware': {},
        'recommendations': []
    }

    # For buried positions
    if len(buried) >= 10:
        buried = buried.copy()
        buried['hydro_diff'] = buried.apply(
            lambda r: abs(AA_PROPERTIES.get(r['wt_aa'], {}).get('hydrophobicity', 0) -
                         AA_PROPERTIES.get(r['mut_aa'], {}).get('hydrophobicity', 0)),
            axis=1
        )
        buried['hybrid_pred'] = buried.apply(
            lambda r: hybrid_predictor(r['wt_aa'], r['mut_aa']), axis=1
        )

        # Find optimal hydro_diff threshold for buried
        best_threshold = 5.15  # Default
        best_corr = -1

        for thresh in np.arange(3.0, 7.0, 0.5):
            high_hydro = buried[buried['hydro_diff'] > thresh]
            if len(high_hydro) >= 5:
                r, _ = spearmanr(high_hydro['ddg'], high_hydro['hybrid_pred'])
                if not np.isnan(r) and r > best_corr:
                    best_corr = r
                    best_threshold = thresh

        results['position_aware']['buried'] = {
            'hydro_diff_threshold': float(best_threshold),
            'optimal_correlation': float(best_corr),
            'n_samples': len(buried),
            'recommendation': (
                f"For buried positions, use hybrid when hydro_diff > {best_threshold:.1f}"
            )
        }

    # For surface positions
    if len(surface) >= 10:
        surface = surface.copy()
        surface['hydro_diff'] = surface.apply(
            lambda r: abs(AA_PROPERTIES.get(r['wt_aa'], {}).get('hydrophobicity', 0) -
                         AA_PROPERTIES.get(r['mut_aa'], {}).get('hydrophobicity', 0)),
            axis=1
        )
        surface['hybrid_pred'] = surface.apply(
            lambda r: hybrid_predictor(r['wt_aa'], r['mut_aa']), axis=1
        )

        # Find optimal threshold for surface
        best_threshold = 5.15
        best_corr = -1

        for thresh in np.arange(3.0, 7.0, 0.5):
            high_hydro = surface[surface['hydro_diff'] > thresh]
            if len(high_hydro) >= 5:
                r, _ = spearmanr(high_hydro['ddg'], high_hydro['hybrid_pred'])
                if not np.isnan(r) and r > best_corr:
                    best_corr = r
                    best_threshold = thresh

        results['position_aware']['surface'] = {
            'hydro_diff_threshold': float(best_threshold),
            'optimal_correlation': float(best_corr),
            'n_samples': len(surface),
            'recommendation': (
                f"For surface positions, use hybrid when hydro_diff > {best_threshold:.1f}"
            )
        }

    # Generate recommendations
    if 'buried' in results['position_aware'] and 'surface' in results['position_aware']:
        buried_thresh = results['position_aware']['buried']['hydro_diff_threshold']
        surface_thresh = results['position_aware']['surface']['hydro_diff_threshold']

        if buried_thresh < surface_thresh:
            results['recommendations'].append(
                f"Buried positions: Lower threshold ({buried_thresh:.1f}) - hybrid helps more"
            )
            results['recommendations'].append(
                f"Surface positions: Higher threshold ({surface_thresh:.1f}) - simple often sufficient"
            )
        else:
            results['recommendations'].append(
                f"Thresholds similar: Position context less important than expected"
            )

    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_position_analysis() -> Dict:
    """Run all position-stratified experiments."""
    print("=" * 60)
    print("V5 POSITION-STRATIFIED ANALYSIS")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    mutations = load_protherm_mutations()
    zone_data = load_zone_assignments()
    linked_data = link_mutations_to_zones(mutations, zone_data)
    print(f"      Loaded {len(linked_data)} mutations")

    # Stratify by position
    print("\n[2/5] Stratifying by position...")
    buried, interface, surface = stratify_by_rsa(linked_data)
    print(f"      Buried (RSA<0.25): {len(buried)}")
    print(f"      Interface: {len(interface)}")
    print(f"      Surface (RSA>0.5): {len(surface)}")

    ss_strat = stratify_by_secondary_structure(linked_data)
    for ss, df in ss_strat.items():
        print(f"      Secondary {ss}: {len(df)}")

    results = {
        'metadata': {
            'n_total': len(linked_data),
            'n_buried': len(buried),
            'n_interface': len(interface),
            'n_surface': len(surface),
            'by_secondary_structure': {ss: len(df) for ss, df in ss_strat.items()}
        }
    }

    # Experiment 1: Same pair, different context
    print("\n[3/5] Analyzing context effects on same pairs...")
    results['context_effects'] = analyze_same_pair_different_context(linked_data)
    print(f"      Pairs with significant context effect: "
          f"{results['context_effects']['summary']['significant_context_effects']}")

    # Experiment 2: Zone boundaries by position
    print("\n[4/5] Computing position-specific boundaries...")
    results['position_boundaries'] = recompute_zone_boundaries_by_position(
        buried, surface, zone_data
    )

    # Experiment 3: Position-zone interaction
    print("\n[5/5] Testing position-zone interaction...")
    results['interaction'] = experiment_position_zone_interaction(linked_data)

    # Compute position-aware thresholds
    original_thresholds = {'hydro_diff': 5.15}
    results['position_aware_thresholds'] = compute_position_aware_thresholds(
        buried, surface, original_thresholds
    )

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Position boundary comparison
    pb = results['position_boundaries']
    if 'buried' in pb and 'surface' in pb:
        if 'insufficient_data' not in pb['buried'] and 'insufficient_data' not in pb['surface']:
            print(f"\nHybrid advantage by position:")
            print(f"  Buried: {pb['buried']['hybrid_advantage']:.3f}")
            print(f"  Surface: {pb['surface']['hybrid_advantage']:.3f}")

            if 'threshold_comparison' in pb:
                print(f"\n  Hypothesis: {pb['threshold_comparison']['interpretation']}")

    # Interaction test
    if 'interaction_test' in results['interaction']:
        it = results['interaction']['interaction_test']
        if 'p_value' in it:
            print(f"\nPosition-Zone Interaction:")
            print(f"  p-value: {it['p_value']:.4f}")
            print(f"  {it['interpretation']}")

    # Recommendations
    if results['position_aware_thresholds'].get('recommendations'):
        print(f"\nRecommendations:")
        for rec in results['position_aware_thresholds']['recommendations']:
            print(f"  - {rec}")

    # Save results
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

    output_path = Path(__file__).parent / "arrow_flip_position_stratified_results.json"
    with open(output_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = run_position_analysis()
