# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Manifold organization evaluation framework for TernaryVAE.

This module provides type-aware evaluation of manifold organization,
supporting both valuation-optimal and frequency-optimal structures.
"""

from enum import Enum
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from scipy.stats import spearmanr

from src.core import TERNARY


class ManifoldType(Enum):
    """Types of manifold organization."""
    VALUATION_OPTIMAL = "valuation_optimal"
    FREQUENCY_OPTIMAL = "frequency_optimal"
    ADAPTIVE = "adaptive"


class ManifoldEvaluator:
    """Type-aware evaluator for manifold organization."""

    def __init__(self,
                 valuation_thresholds: Dict[str, float] = None,
                 frequency_thresholds: Dict[str, float] = None):
        """Initialize evaluator with performance thresholds.

        Args:
            valuation_thresholds: Quality thresholds for valuation-optimal manifolds
            frequency_thresholds: Quality thresholds for frequency-optimal manifolds
        """
        self.valuation_thresholds = valuation_thresholds or {
            'excellent': -0.80,
            'good': -0.60,
            'weak': -0.30,
            'ceiling': -0.8321  # Mathematical limit
        }

        self.frequency_thresholds = frequency_thresholds or {
            'excellent': 0.70,
            'good': 0.50,
            'weak': 0.20
        }

    def evaluate_hierarchy(self,
                          radii: Union[torch.Tensor, np.ndarray],
                          valuations: Union[torch.Tensor, np.ndarray],
                          intended_type: ManifoldType = ManifoldType.ADAPTIVE,
                          return_detailed: bool = False) -> Union[str, Dict]:
        """Evaluate manifold organization based on intended type.

        Args:
            radii: Radii of embeddings in hyperbolic space
            valuations: P-adic valuations of corresponding operations
            intended_type: Expected manifold organization type
            return_detailed: Whether to return detailed analysis

        Returns:
            Evaluation result (string summary or detailed dict)
        """
        # Convert to numpy for scipy
        if isinstance(radii, torch.Tensor):
            radii = radii.cpu().numpy()
        if isinstance(valuations, torch.Tensor):
            valuations = valuations.cpu().numpy()

        # Compute Spearman correlation
        corr, p_value = spearmanr(valuations, radii)

        # Basic analysis
        analysis = {
            'hierarchy_score': corr,
            'p_value': p_value,
            'manifold_type': self._classify_manifold_type(corr),
            'organization_quality': self._assess_quality(corr, intended_type),
            'data_distribution': self._analyze_distribution(valuations, radii)
        }

        if intended_type == ManifoldType.VALUATION_OPTIMAL:
            analysis['type_alignment'] = self._evaluate_valuation_optimal(corr)
        elif intended_type == ManifoldType.FREQUENCY_OPTIMAL:
            analysis['type_alignment'] = self._evaluate_frequency_optimal(corr)
        else:  # ADAPTIVE
            analysis['type_alignment'] = self._evaluate_adaptive(corr)

        # Additional metrics
        analysis['richness'] = self._compute_richness(radii, valuations)
        analysis['separation'] = self._compute_level_separation(radii, valuations)
        analysis['geometric_efficiency'] = self._compute_geometric_efficiency(radii, valuations)

        if return_detailed:
            return analysis
        else:
            return self._format_summary(analysis, intended_type)

    def _classify_manifold_type(self, corr: float) -> str:
        """Classify manifold type based on hierarchy score."""
        if corr <= -0.3:
            return "valuation_optimal"
        elif corr >= 0.3:
            return "frequency_optimal"
        else:
            return "unorganized"

    def _assess_quality(self, corr: float, intended_type: ManifoldType) -> str:
        """Assess organization quality relative to type."""
        abs_corr = abs(corr)

        if intended_type == ManifoldType.VALUATION_OPTIMAL:
            if corr >= self.valuation_thresholds['excellent']:
                return "excellent"
            elif corr >= self.valuation_thresholds['good']:
                return "good"
            elif corr >= self.valuation_thresholds['weak']:
                return "weak"
            else:
                return "poor"

        elif intended_type == ManifoldType.FREQUENCY_OPTIMAL:
            if corr >= self.frequency_thresholds['excellent']:
                return "excellent"
            elif corr >= self.frequency_thresholds['good']:
                return "good"
            elif corr >= self.frequency_thresholds['weak']:
                return "weak"
            else:
                return "poor"

        else:  # ADAPTIVE
            if abs_corr >= 0.75:
                return "excellent"
            elif abs_corr >= 0.55:
                return "good"
            elif abs_corr >= 0.35:
                return "weak"
            else:
                return "poor"

    def _evaluate_valuation_optimal(self, corr: float) -> Dict:
        """Detailed evaluation for valuation-optimal manifolds."""
        thresholds = self.valuation_thresholds

        if corr >= thresholds['excellent']:
            status = "✓ EXCELLENT"
            note = "Strong p-adic hierarchy preserved"
        elif corr >= thresholds['good']:
            status = "✓ GOOD"
            note = "Adequate p-adic structure"
        elif corr >= thresholds['weak']:
            status = "⚠ WEAK"
            note = "Some p-adic structure present"
        elif corr < 0:
            status = "⚠ POOR"
            note = "Very weak p-adic structure"
        else:
            status = "✗ FREQUENCY-OPTIMAL"
            note = f"Shows frequency organization (corr={corr:.3f}), not valuation"

        return {
            'status': status,
            'note': note,
            'distance_to_ceiling': abs(corr - thresholds['ceiling']),
            'achievable_improvement': max(0, thresholds['ceiling'] - corr) if corr < 0 else 0
        }

    def _evaluate_frequency_optimal(self, corr: float) -> Dict:
        """Detailed evaluation for frequency-optimal manifolds."""
        thresholds = self.frequency_thresholds

        if corr >= thresholds['excellent']:
            status = "✓ EXCELLENT"
            note = "Strong frequency-based organization"
        elif corr >= thresholds['good']:
            status = "✓ GOOD"
            note = "Good frequency hierarchy"
        elif corr >= thresholds['weak']:
            status = "⚠ WEAK"
            note = "Some frequency organization"
        elif corr > 0:
            status = "⚠ POOR"
            note = "Very weak frequency structure"
        else:
            status = "✗ VALUATION-OPTIMAL"
            note = f"Shows p-adic organization (corr={corr:.3f}), not frequency"

        return {
            'status': status,
            'note': note,
            'compression_efficiency': self._estimate_compression_efficiency(corr),
            'retrieval_optimization': self._estimate_retrieval_optimization(corr)
        }

    def _evaluate_adaptive(self, corr: float) -> Dict:
        """Evaluation for adaptive/unknown manifold types."""
        abs_corr = abs(corr)

        if corr < 0:
            org_type = "valuation-optimal"
            efficiency = "semantic reasoning"
        else:
            org_type = "frequency-optimal"
            efficiency = "compression/retrieval"

        if abs_corr >= 0.75:
            status = "✓ EXCELLENT"
        elif abs_corr >= 0.55:
            status = "✓ GOOD"
        elif abs_corr >= 0.35:
            status = "⚠ WEAK"
        else:
            status = "⚠ POOR"

        return {
            'status': status,
            'detected_type': org_type,
            'organization_strength': abs_corr,
            'optimal_for': efficiency,
            'note': f"Manifold shows {org_type} organization"
        }

    def _analyze_distribution(self, valuations: np.ndarray, radii: np.ndarray) -> Dict:
        """Analyze data distribution across valuation levels."""
        distribution = {}

        for v in range(10):  # v=0 to v=9
            mask = valuations == v
            if np.any(mask):
                v_radii = radii[mask]
                distribution[f'v{v}'] = {
                    'count': len(v_radii),
                    'mean_radius': float(np.mean(v_radii)),
                    'std_radius': float(np.std(v_radii)),
                    'frequency': len(v_radii) / len(valuations)
                }

        # Key statistics
        v0_freq = distribution.get('v0', {}).get('frequency', 0)
        high_v_freq = sum(distribution.get(f'v{v}', {}).get('frequency', 0) for v in range(8, 10))

        distribution['summary'] = {
            'v0_dominance': v0_freq,  # Should be ~0.667 (66.7%)
            'high_valuation_rarity': high_v_freq,  # Should be very small
            'mathematical_limit_factor': v0_freq  # Explains hierarchy ceiling
        }

        return distribution

    def _compute_richness(self, radii: np.ndarray, valuations: np.ndarray) -> float:
        """Compute richness (within-level variance)."""
        variances = []
        for v in range(10):
            mask = valuations == v
            if np.sum(mask) > 1:  # Need at least 2 points for variance
                variances.append(np.var(radii[mask]))

        return float(np.mean(variances)) if variances else 0.0

    def _compute_level_separation(self, radii: np.ndarray, valuations: np.ndarray) -> float:
        """Compute separation between valuation levels."""
        level_means = []
        for v in range(10):
            mask = valuations == v
            if np.any(mask):
                level_means.append(np.mean(radii[mask]))

        if len(level_means) < 2:
            return 0.0

        # Mean absolute difference between consecutive levels
        separations = [abs(level_means[i] - level_means[i+1])
                      for i in range(len(level_means)-1)]

        return float(np.mean(separations))

    def _compute_geometric_efficiency(self, radii: np.ndarray, valuations: np.ndarray) -> Dict:
        """Compute geometric efficiency metrics."""
        # Volume allocation efficiency
        level_frequencies = {}
        level_volumes = {}

        for v in range(10):
            mask = valuations == v
            if np.any(mask):
                freq = np.sum(mask) / len(valuations)
                mean_radius = np.mean(radii[mask])
                # Approximate hyperbolic volume (exponential growth)
                volume = np.exp(mean_radius)  # Simplified

                level_frequencies[v] = freq
                level_volumes[v] = volume

        # Efficiency metrics
        freq_array = np.array([level_frequencies.get(v, 0) for v in range(10)])
        vol_array = np.array([level_volumes.get(v, 1) for v in range(10)])

        # Correlation between frequency and allocated volume
        freq_vol_corr = np.corrcoef(freq_array, vol_array)[0, 1] if len(freq_array) > 1 else 0

        return {
            'frequency_volume_correlation': float(freq_vol_corr),
            'volume_efficiency': 'high' if freq_vol_corr > 0.5 else 'low',
            'interpretation': 'frequency_optimal' if freq_vol_corr > 0.3 else 'valuation_optimal'
        }

    def _estimate_compression_efficiency(self, corr: float) -> str:
        """Estimate compression efficiency based on hierarchy."""
        if corr >= 0.7:
            return "excellent"
        elif corr >= 0.5:
            return "good"
        elif corr >= 0.3:
            return "moderate"
        else:
            return "poor"

    def _estimate_retrieval_optimization(self, corr: float) -> str:
        """Estimate retrieval optimization based on hierarchy."""
        if corr >= 0.6:
            return "optimized_for_frequent_items"
        elif corr >= 0.3:
            return "moderately_optimized"
        else:
            return "not_optimized"

    def _format_summary(self, analysis: Dict, intended_type: ManifoldType) -> str:
        """Format analysis into human-readable summary."""
        corr = analysis['hierarchy_score']
        quality = analysis['organization_quality']
        manifold_type = analysis['manifold_type']
        alignment = analysis['type_alignment']

        summary = f"Hierarchy Score: {corr:.4f} ({quality} {manifold_type})\n"
        summary += f"Organization: {alignment['status']} - {alignment['note']}\n"

        # Additional insights
        if 'distance_to_ceiling' in alignment:
            ceiling_dist = alignment['distance_to_ceiling']
            summary += f"Distance to ceiling (-0.8321): {ceiling_dist:.4f}\n"

        richness = analysis['richness']
        separation = analysis['separation']
        summary += f"Richness: {richness:.6f}, Level Separation: {separation:.4f}\n"

        # Application recommendations
        if manifold_type == "valuation_optimal":
            summary += "✓ Optimal for: Semantic reasoning, compositional learning, p-adic applications\n"
            summary += "⚠ Suboptimal for: Compression, fast retrieval of frequent items\n"
        elif manifold_type == "frequency_optimal":
            summary += "✓ Optimal for: Compression, fast retrieval, statistical ML\n"
            summary += "⚠ Suboptimal for: Semantic reasoning, rare pattern detection\n"
        else:
            summary += "⚠ Organization unclear - consider explicit type selection\n"

        return summary


# Convenience functions for quick evaluation
def evaluate_valuation_optimal(radii: Union[torch.Tensor, np.ndarray],
                              valuations: Union[torch.Tensor, np.ndarray]) -> str:
    """Quick evaluation assuming valuation-optimal target."""
    evaluator = ManifoldEvaluator()
    return evaluator.evaluate_hierarchy(radii, valuations, ManifoldType.VALUATION_OPTIMAL)

def evaluate_frequency_optimal(radii: Union[torch.Tensor, np.ndarray],
                              valuations: Union[torch.Tensor, np.ndarray]) -> str:
    """Quick evaluation assuming frequency-optimal target."""
    evaluator = ManifoldEvaluator()
    return evaluator.evaluate_hierarchy(radii, valuations, ManifoldType.FREQUENCY_OPTIMAL)

def evaluate_adaptive(radii: Union[torch.Tensor, np.ndarray],
                     valuations: Union[torch.Tensor, np.ndarray]) -> str:
    """Type-agnostic evaluation."""
    evaluator = ManifoldEvaluator()
    return evaluator.evaluate_hierarchy(radii, valuations, ManifoldType.ADAPTIVE)

def detailed_manifold_analysis(radii: Union[torch.Tensor, np.ndarray],
                             valuations: Union[torch.Tensor, np.ndarray],
                             intended_type: ManifoldType = ManifoldType.ADAPTIVE) -> Dict:
    """Comprehensive manifold analysis."""
    evaluator = ManifoldEvaluator()
    return evaluator.evaluate_hierarchy(radii, valuations, intended_type, return_detailed=True)