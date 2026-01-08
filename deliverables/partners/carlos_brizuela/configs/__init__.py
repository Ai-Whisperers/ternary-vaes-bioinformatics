"""Configuration loader for AMP Design Package.

Loads JSON configs for pathogens, microbiome contexts, and synthesis parameters.
Supports custom user configs via CLI args or environment variables.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

# Default config directory
CONFIG_DIR = Path(__file__).parent


def load_config(config_name: str, custom_path: Optional[str] = None) -> Dict:
    """Load a JSON configuration file.

    Args:
        config_name: Base name of config (e.g., 'pathogens', 'microbiome', 'synthesis')
        custom_path: Optional path to custom config file (overrides default)

    Returns:
        Dictionary with config data

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config is invalid JSON
    """
    if custom_path:
        config_path = Path(custom_path)
    else:
        config_path = CONFIG_DIR / f"{config_name}.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pathogens(custom_path: Optional[str] = None) -> Dict:
    """Load WHO priority pathogens configuration.

    Returns:
        Dictionary with 'pathogens' key containing pathogen definitions
    """
    config = load_config("pathogens", custom_path)
    return config.get("pathogens", config)


def load_microbiome(custom_path: Optional[str] = None) -> Dict:
    """Load microbiome contexts configuration.

    Returns:
        Dictionary with 'contexts' key containing skin/gut definitions
    """
    config = load_config("microbiome", custom_path)
    return config.get("contexts", config)


def load_synthesis(custom_path: Optional[str] = None) -> Dict:
    """Load synthesis parameters configuration.

    Returns:
        Dictionary with AA properties, dipeptide penalties, etc.
    """
    return load_config("synthesis", custom_path)


def get_pathogen_names() -> list:
    """Get list of available pathogen names."""
    config = load_pathogens()
    return list(config.keys())


def get_microbiome_contexts() -> list:
    """Get list of available microbiome contexts."""
    config = load_microbiome()
    return list(config.keys())
