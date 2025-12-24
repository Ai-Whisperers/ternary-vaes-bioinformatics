"""Shared utilities for benchmark scripts.

Extracted from measure_coupled_resolution.py and measure_manifold_resolution.py
to eliminate code duplication (D1.5 from DUPLICATION_REPORT).
"""

import numpy as np
from typing import Any


def convert_to_python_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object to convert (dict, list, numpy type, or other)

    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
