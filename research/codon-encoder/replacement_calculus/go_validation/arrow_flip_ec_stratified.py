"""
Arrow Flip EC-Stratified Analysis: Investigating Enzyme Class-Specific Rules

This module investigates whether different enzyme classes (EC1-EC6) have
different decision boundaries for when to use hybrid vs simple prediction.

Key focus: EC1 (oxidoreductase) anomaly - why might redox enzymes behave differently?

Experiments:
1. EC class propensity analysis per AA pair
2. EC1-specific investigation (metal binding, redox-active residues)
3. Feature importance changes by EC context
4. EC-specific decision rules

Hypothesis: Metal-binding AAs (H, C, D, E, M, Y) may have lower hydro_diff
threshold because electronic effects dominate in oxidoreductases.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "deliverables"))

# Import functional profiles with EC data
from functional_profiles import AMINO_ACID_PROFILES

# Import from experimental validation
from arrow_flip_experimental_validation import (
    load_protherm_mutations, load_zone_assignments, link_mutations_to_zones,
    simple_predictor, hybrid_predictor, bootstrap_spearman, cohens_d
)
from arrow_flip_clustering import AA_PROPERTIES, build_pair_dataset


# =============================================================================
# EC CLASS DATA
# =============================================================================

# Amino acids relevant to EC1 (oxidoreductases) - metal binding and redox active
EC1_RELEVANT_AAS = ['H', 'C', 'D', 'E', 'M', 'Y']

# EC class names
EC_CLASSES = [
    ('ec1_oxidoreductase', 'EC1: Oxidoreductases'),
    ('ec2_transferase', 'EC2: Transferases'),
    ('ec3_hydrolase', 'EC3: Hydrolases'),
    ('ec4_lyase', 'EC4: Lyases'),
    ('ec5_isomerase', 'EC5: Isomerases'),
    ('ec6_ligase', 'EC6: Ligases')
]


def get_ec_class_weights(aa: str) -> Dict[str, float]:
    """
    Get EC class propensities from AMINO_ACID_PROFILES.

    Returns dictionary of EC class -> enrichment score.
    """
    if aa not in AMINO_ACID_PROFILES:
        return {ec[0]: 0.0 for ec in EC_CLASSES}

    profile = AMINO_ACID_PROFILES[aa]
    return {
        'ec1_oxidoreductase': profile.ec1_oxidoreductase,
        'ec2_transferase': profile.ec2_transferase,
        'ec3_hydrolase': profile.ec3_hydrolase,
        'ec4_lyase': profile.ec4_lyase,
        'ec5_isomerase': profile.ec5_isomerase,
        'ec6_ligase': profile.ec6_ligase,
    }


def get_dominant_ec_class(aa: str) -> Tuple[str, float]:
    """Get the EC class with highest propensity for an amino acid."""
    weights = get_ec_class_weights(aa)
    if not weights:
        return 'unknown', 0.0
    max_ec = max(weights.items(), key=lambda x: x[1])
    return max_ec[0], max_ec[1]


# =============================================================================
# EC CLASS PAIR FEATURES
# =============================================================================

def compute_ec_class_pair_features(aa1: str, aa2: str) -> Dict:
    """
    Compute EC-relevant features for an AA pair.

    Features:
    - ec_profile_distance: Euclidean distance in EC space
    - dominant_ec_change: Does dominant EC class change?
    - catalytic_propensity_change: From functional_profiles.py
    - ec1_involvement: Is either AA EC1-relevant (metal binding/redox)?
    """
    w1 = get_ec_class_weights(aa1)
    w2 = get_ec_class_weights(aa2)

    # EC profile distance
    ec_distance = np.sqrt(sum((w1[ec] - w2[ec])**2 for ec in w1.keys()))

    # Dominant EC change
    dom1, score1 = get_dominant_ec_class(aa1)
    dom2, score2 = get_dominant_ec_class(aa2)
    dominant_ec_change = dom1 != dom2

    # Catalytic propensity
    cat1 = AMINO_ACID_PROFILES[aa1].catalytic_propensity if aa1 in AMINO_ACID_PROFILES else 0
    cat2 = AMINO_ACID_PROFILES[aa2].catalytic_propensity if aa2 in AMINO_ACID_PROFILES else 0
    catalytic_change = abs(cat1 - cat2)

    # EC1 involvement (metal binding / redox active)
    ec1_involvement = aa1 in EC1_RELEVANT_AAS or aa2 in EC1_RELEVANT_AAS
    both_ec1_relevant = aa1 in EC1_RELEVANT_AAS and aa2 in EC1_RELEVANT_AAS

    # Metal binding propensity
    mb1 = AMINO_ACID_PROFILES[aa1].metal_binding if aa1 in AMINO_ACID_PROFILES else 0
    mb2 = AMINO_ACID_PROFILES[aa2].metal_binding if aa2 in AMINO_ACID_PROFILES else 0
    metal_binding_change = abs(mb1 - mb2)

    return {
        'ec_profile_distance': float(ec_distance),
        'dominant_ec_change': dominant_ec_change,
        'dominant_ec_1': dom1,
        'dominant_ec_2': dom2,
        'catalytic_propensity_change': float(catalytic_change),
        'ec1_involvement': ec1_involvement,
        'both_ec1_relevant': both_ec1_relevant,
        'metal_binding_change': float(metal_binding_change),
        'ec1_propensity_1': float(w1.get('ec1_oxidoreductase', 0)),
        'ec1_propensity_2': float(w2.get('ec1_oxidoreductase', 0))
    }


# =============================================================================
# EC1 ANOMALY INVESTIGATION
# =============================================================================

def investigate_ec1_anomaly(linked_data: pd.DataFrame) -> Dict:
    """
    Investigate why EC1 (oxidoreductases) might behave differently.

    Hypothesis: Metal-binding AAs have different substitution patterns
    because electronic effects (not hydrophobicity) dominate.

    Test:
    1. Filter pairs involving EC1-relevant AAs (H, C, D, E, M, Y)
    2. Compare zone distribution to overall
    3. Check if hydro_diff importance changes
    """
    results = {
        'ec1_relevant_aas': EC1_RELEVANT_AAS,
        'hypothesis': "Metal-binding AAs may have lower hydro_diff threshold",
        'analysis': {}
    }

    # Add EC1 involvement
    linked_data = linked_data.copy()
    linked_data['ec1_involved'] = linked_data.apply(
        lambda r: r['wt_aa'] in EC1_RELEVANT_AAS or r['mut_aa'] in EC1_RELEVANT_AAS,
        axis=1
    )
    linked_data['both_ec1'] = linked_data.apply(
        lambda r: r['wt_aa'] in EC1_RELEVANT_AAS and r['mut_aa'] in EC1_RELEVANT_AAS,
        axis=1
    )

    # Add predictions
    linked_data['simple_pred'] = linked_data.apply(
        lambda r: simple_predictor(r['wt_aa'], r['mut_aa']), axis=1
    )
    linked_data['hybrid_pred'] = linked_data.apply(
        lambda r: hybrid_predictor(r['wt_aa'], r['mut_aa']), axis=1
    )

    # Split by EC1 involvement
    ec1_mutations = linked_data[linked_data['ec1_involved']]
    non_ec1_mutations = linked_data[~linked_data['ec1_involved']]

    # Compare prediction accuracy
    if len(ec1_mutations) >= 10 and len(non_ec1_mutations) >= 10:
        # EC1-involved
        ddg_ec1 = ec1_mutations['ddg'].values
        ec1_simple_r = bootstrap_spearman(ddg_ec1, ec1_mutations['simple_pred'].values)
        ec1_hybrid_r = bootstrap_spearman(ddg_ec1, ec1_mutations['hybrid_pred'].values)

        # Non-EC1
        ddg_non = non_ec1_mutations['ddg'].values
        non_simple_r = bootstrap_spearman(ddg_non, non_ec1_mutations['simple_pred'].values)
        non_hybrid_r = bootstrap_spearman(ddg_non, non_ec1_mutations['hybrid_pred'].values)

        results['analysis']['ec1_involved'] = {
            'n': len(ec1_mutations),
            'simple_predictor': ec1_simple_r,
            'hybrid_predictor': ec1_hybrid_r,
            'hybrid_advantage': ec1_hybrid_r['mean'] - ec1_simple_r['mean']
        }

        results['analysis']['non_ec1'] = {
            'n': len(non_ec1_mutations),
            'simple_predictor': non_simple_r,
            'hybrid_predictor': non_hybrid_r,
            'hybrid_advantage': non_hybrid_r['mean'] - non_simple_r['mean']
        }

        # Compare
        ec1_adv = ec1_hybrid_r['mean'] - ec1_simple_r['mean']
        non_adv = non_hybrid_r['mean'] - non_simple_r['mean']

        results['analysis']['comparison'] = {
            'ec1_hybrid_advantage': float(ec1_adv),
            'non_ec1_hybrid_advantage': float(non_adv),
            'difference': float(ec1_adv - non_adv),
            'ec1_needs_hybrid_more': ec1_adv > non_adv,
            'interpretation': (
                "EC1-relevant mutations benefit MORE from hybrid approach"
                if ec1_adv > non_adv else
                "EC1-relevant mutations benefit LESS from hybrid approach"
            )
        }

    # Zone distribution comparison
    ec1_zones = ec1_mutations['zone'].value_counts().to_dict() if len(ec1_mutations) > 0 else {}
    non_ec1_zones = non_ec1_mutations['zone'].value_counts().to_dict() if len(non_ec1_mutations) > 0 else {}

    results['analysis']['zone_distribution'] = {
        'ec1_involved': ec1_zones,
        'non_ec1': non_ec1_zones
    }

    # Hydrophobicity analysis for EC1 pairs
    if len(ec1_mutations) >= 10:
        ec1_mutations = ec1_mutations.copy()
        ec1_mutations['hydro_diff'] = ec1_mutations.apply(
            lambda r: abs(AA_PROPERTIES.get(r['wt_aa'], {}).get('hydrophobicity', 0) -
                         AA_PROPERTIES.get(r['mut_aa'], {}).get('hydrophobicity', 0)),
            axis=1
        )

        # Correlation of hydro_diff with DDG for EC1 vs non-EC1
        ec1_hydro_corr, _ = spearmanr(ec1_mutations['hydro_diff'], ec1_mutations['ddg'])

        non_ec1_mutations = non_ec1_mutations.copy()
        non_ec1_mutations['hydro_diff'] = non_ec1_mutations.apply(
            lambda r: abs(AA_PROPERTIES.get(r['wt_aa'], {}).get('hydrophobicity', 0) -
                         AA_PROPERTIES.get(r['mut_aa'], {}).get('hydrophobicity', 0)),
            axis=1
        )
        non_ec1_hydro_corr, _ = spearmanr(non_ec1_mutations['hydro_diff'], non_ec1_mutations['ddg'])

        results['analysis']['hydrophobicity_importance'] = {
            'ec1_hydro_ddg_correlation': float(ec1_hydro_corr) if not np.isnan(ec1_hydro_corr) else 0.0,
            'non_ec1_hydro_ddg_correlation': float(non_ec1_hydro_corr) if not np.isnan(non_ec1_hydro_corr) else 0.0,
            'interpretation': (
                "Hydrophobicity LESS important for EC1-relevant pairs"
                if abs(ec1_hydro_corr) < abs(non_ec1_hydro_corr) else
                "Hydrophobicity equally/more important for EC1-relevant pairs"
            )
        }

    return results


# =============================================================================
# EC-SPECIFIC FEATURE IMPORTANCE
# =============================================================================

def compute_feature_importance_by_ec_context(linked_data: pd.DataFrame) -> Dict:
    """
    Compute feature importance for predicting DDG, stratified by EC context.

    Tests if different features matter for different enzyme classes.
    """
    results = {}

    # Add EC features
    linked_data = linked_data.copy()

    def add_ec_features(row):
        ec_feat = compute_ec_class_pair_features(row['wt_aa'], row['mut_aa'])
        return pd.Series(ec_feat)

    ec_features = linked_data.apply(add_ec_features, axis=1)
    linked_data = pd.concat([linked_data, ec_features], axis=1)

    # Add standard features
    linked_data['hydro_diff'] = linked_data.apply(
        lambda r: abs(AA_PROPERTIES.get(r['wt_aa'], {}).get('hydrophobicity', 0) -
                     AA_PROPERTIES.get(r['mut_aa'], {}).get('hydrophobicity', 0)),
        axis=1
    )
    linked_data['volume_diff'] = linked_data.apply(
        lambda r: abs(AA_PROPERTIES.get(r['wt_aa'], {}).get('volume', 100) -
                     AA_PROPERTIES.get(r['mut_aa'], {}).get('volume', 100)),
        axis=1
    )
    linked_data['charge_diff'] = linked_data.apply(
        lambda r: abs(AA_PROPERTIES.get(r['wt_aa'], {}).get('charge', 0) -
                     AA_PROPERTIES.get(r['mut_aa'], {}).get('charge', 0)),
        axis=1
    )

    # Feature columns
    feature_cols = ['hydro_diff', 'volume_diff', 'charge_diff',
                    'ec_profile_distance', 'catalytic_propensity_change',
                    'metal_binding_change']

    # Overall feature correlations with DDG
    overall_corrs = {}
    for col in feature_cols:
        if col in linked_data.columns:
            r, p = spearmanr(linked_data[col], linked_data['ddg'])
            if not np.isnan(r):
                overall_corrs[col] = {'correlation': float(r), 'p_value': float(p)}

    results['overall_feature_importance'] = overall_corrs

    # Stratified by EC1 involvement
    ec1_data = linked_data[linked_data['ec1_involvement'] == True]
    non_ec1_data = linked_data[linked_data['ec1_involvement'] == False]

    for subset_name, subset in [('ec1_involved', ec1_data), ('non_ec1', non_ec1_data)]:
        if len(subset) < 10:
            continue

        subset_corrs = {}
        for col in feature_cols:
            if col in subset.columns:
                r, p = spearmanr(subset[col], subset['ddg'])
                if not np.isnan(r):
                    subset_corrs[col] = {'correlation': float(r), 'p_value': float(p)}

        results[f'{subset_name}_feature_importance'] = subset_corrs

    return results


# =============================================================================
# EC-SPECIFIC DECISION RULES
# =============================================================================

def generate_ec_specific_rules(linked_data: pd.DataFrame) -> Dict:
    """
    Generate decision rules specific to EC context.

    Format: Same as arrow_flip_clustering decision_rules but with EC stratification.
    """
    results = {'rules': [], 'summary': {}}

    # Add features
    linked_data = linked_data.copy()
    linked_data['hydro_diff'] = linked_data.apply(
        lambda r: abs(AA_PROPERTIES.get(r['wt_aa'], {}).get('hydrophobicity', 0) -
                     AA_PROPERTIES.get(r['mut_aa'], {}).get('hydrophobicity', 0)),
        axis=1
    )
    linked_data['ec1_involved'] = linked_data.apply(
        lambda r: r['wt_aa'] in EC1_RELEVANT_AAS or r['mut_aa'] in EC1_RELEVANT_AAS,
        axis=1
    )

    # Binary target: hybrid better than simple?
    linked_data['simple_pred'] = linked_data.apply(
        lambda r: simple_predictor(r['wt_aa'], r['mut_aa']), axis=1
    )
    linked_data['hybrid_pred'] = linked_data.apply(
        lambda r: hybrid_predictor(r['wt_aa'], r['mut_aa']), axis=1
    )
    linked_data['simple_error'] = abs(linked_data['ddg'] - linked_data['simple_pred'])
    linked_data['hybrid_error'] = abs(linked_data['ddg'] - linked_data['hybrid_pred'])
    linked_data['hybrid_better'] = linked_data['hybrid_error'] < linked_data['simple_error']

    # Rule 1: EC1-involved with high hydro_diff
    ec1_high_hydro = linked_data[
        (linked_data['ec1_involved']) &
        (linked_data['hydro_diff'] > 3.0)
    ]
    if len(ec1_high_hydro) >= 5:
        hybrid_rate = ec1_high_hydro['hybrid_better'].mean()
        results['rules'].append({
            'name': 'EC1 + High Hydro',
            'conditions': ['ec1_involved = True', 'hydro_diff > 3.0'],
            'prediction': 'hybrid' if hybrid_rate > 0.5 else 'simple',
            'confidence': float(max(hybrid_rate, 1 - hybrid_rate)),
            'n_samples': len(ec1_high_hydro),
            'hybrid_win_rate': float(hybrid_rate)
        })

    # Rule 2: Non-EC1 with high hydro_diff
    non_ec1_high_hydro = linked_data[
        (~linked_data['ec1_involved']) &
        (linked_data['hydro_diff'] > 5.0)
    ]
    if len(non_ec1_high_hydro) >= 5:
        hybrid_rate = non_ec1_high_hydro['hybrid_better'].mean()
        results['rules'].append({
            'name': 'Non-EC1 + High Hydro',
            'conditions': ['ec1_involved = False', 'hydro_diff > 5.0'],
            'prediction': 'hybrid' if hybrid_rate > 0.5 else 'simple',
            'confidence': float(max(hybrid_rate, 1 - hybrid_rate)),
            'n_samples': len(non_ec1_high_hydro),
            'hybrid_win_rate': float(hybrid_rate)
        })

    # Rule 3: EC1 with low hydro_diff (where simple might win)
    ec1_low_hydro = linked_data[
        (linked_data['ec1_involved']) &
        (linked_data['hydro_diff'] <= 3.0)
    ]
    if len(ec1_low_hydro) >= 5:
        hybrid_rate = ec1_low_hydro['hybrid_better'].mean()
        results['rules'].append({
            'name': 'EC1 + Low Hydro',
            'conditions': ['ec1_involved = True', 'hydro_diff <= 3.0'],
            'prediction': 'hybrid' if hybrid_rate > 0.5 else 'simple',
            'confidence': float(max(hybrid_rate, 1 - hybrid_rate)),
            'n_samples': len(ec1_low_hydro),
            'hybrid_win_rate': float(hybrid_rate)
        })

    # Summary
    results['summary'] = {
        'total_rules': len(results['rules']),
        'ec1_threshold': 3.0,  # Lower threshold for EC1
        'non_ec1_threshold': 5.0,  # Standard threshold
        'recommendation': (
            "For EC1-relevant substitutions (involving H,C,D,E,M,Y), "
            "use hybrid approach when hydro_diff > 3.0. "
            "For non-EC1 substitutions, use standard threshold (5.0)."
        )
    }

    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_ec_analysis() -> Dict:
    """Run all EC-stratified experiments."""
    print("=" * 60)
    print("V5 EC-STRATIFIED ANALYSIS")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    mutations = load_protherm_mutations()
    zone_data = load_zone_assignments()
    linked_data = link_mutations_to_zones(mutations, zone_data)
    print(f"      Loaded {len(linked_data)} mutations")

    # Count EC1-relevant mutations
    ec1_count = sum(1 for _, r in linked_data.iterrows()
                    if r['wt_aa'] in EC1_RELEVANT_AAS or r['mut_aa'] in EC1_RELEVANT_AAS)
    print(f"      EC1-relevant mutations: {ec1_count}")

    results = {
        'metadata': {
            'n_total': len(linked_data),
            'n_ec1_relevant': ec1_count,
            'ec1_relevant_aas': EC1_RELEVANT_AAS
        }
    }

    # Experiment 1: EC1 anomaly investigation
    print("\n[2/5] Investigating EC1 anomaly...")
    results['ec1_anomaly'] = investigate_ec1_anomaly(linked_data)

    # Experiment 2: Feature importance by EC context
    print("\n[3/5] Computing feature importance by EC context...")
    results['feature_importance'] = compute_feature_importance_by_ec_context(linked_data)

    # Experiment 3: EC-specific decision rules
    print("\n[4/5] Generating EC-specific rules...")
    results['ec_rules'] = generate_ec_specific_rules(linked_data)

    # Experiment 4: EC class pair analysis
    print("\n[5/5] Analyzing EC class pair features...")
    ec_pair_analysis = []
    sample_pairs = [('H', 'A'), ('C', 'A'), ('D', 'A'), ('E', 'A'),
                    ('L', 'A'), ('V', 'A'), ('I', 'A'), ('F', 'A')]
    for aa1, aa2 in sample_pairs:
        feat = compute_ec_class_pair_features(aa1, aa2)
        feat['pair'] = f"{aa1}-{aa2}"
        ec_pair_analysis.append(feat)
    results['sample_ec_pair_features'] = ec_pair_analysis

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # EC1 anomaly
    ec1 = results['ec1_anomaly']
    if 'comparison' in ec1.get('analysis', {}):
        comp = ec1['analysis']['comparison']
        print(f"\nEC1 Anomaly Investigation:")
        print(f"  EC1-involved hybrid advantage: {comp['ec1_hybrid_advantage']:.3f}")
        print(f"  Non-EC1 hybrid advantage: {comp['non_ec1_hybrid_advantage']:.3f}")
        print(f"  {comp['interpretation']}")

    if 'hydrophobicity_importance' in ec1.get('analysis', {}):
        hydro = ec1['analysis']['hydrophobicity_importance']
        print(f"\nHydrophobicity importance:")
        print(f"  EC1: r={hydro['ec1_hydro_ddg_correlation']:.3f}")
        print(f"  Non-EC1: r={hydro['non_ec1_hydro_ddg_correlation']:.3f}")
        print(f"  {hydro['interpretation']}")

    # EC-specific rules
    if results['ec_rules'].get('rules'):
        print(f"\nEC-Specific Decision Rules:")
        for rule in results['ec_rules']['rules']:
            print(f"  {rule['name']}: {rule['prediction']} "
                  f"(conf={rule['confidence']:.2f}, n={rule['n_samples']})")

    print(f"\n{results['ec_rules']['summary']['recommendation']}")

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

    output_path = Path(__file__).parent / "arrow_flip_ec_stratified_results.json"
    with open(output_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = run_ec_analysis()
