# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Global visualization configuration and constants.

This module defines global settings for the visualization package including
figure sizes, DPI settings, and default parameters that ensure consistency
across all generated figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# =============================================================================
# Figure Size Constants (in inches)
# =============================================================================

# Journal column widths
COLUMN_WIDTH_SINGLE = 3.5  # Single column (Nature, Science)
COLUMN_WIDTH_DOUBLE = 7.0  # Double column / full page
COLUMN_WIDTH_IEEE = 3.5  # IEEE single column

# Presentation sizes
SLIDE_WIDTH = 13.33  # PowerPoint 16:9
SLIDE_HEIGHT = 7.5

# Aspect ratios
GOLDEN_RATIO = 1.618
ASPECT_SQUARE = 1.0
ASPECT_WIDE = 16 / 9
ASPECT_STANDARD = 4 / 3

# =============================================================================
# DPI Settings
# =============================================================================

DPI_SCREEN = 100  # For notebook/screen display
DPI_PRESENTATION = 150  # For presentations
DPI_PUBLICATION = 300  # For publication/print
DPI_POSTER = 150  # For posters (larger figures)

# =============================================================================
# Font Configuration
# =============================================================================

FONT_FAMILY_SANS = ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"]
FONT_FAMILY_SERIF = ["Times New Roman", "Times", "DejaVu Serif"]
FONT_FAMILY_MONO = ["Consolas", "Monaco", "DejaVu Sans Mono"]

# Font sizes by context
FONT_SIZES = {
    "paper": {"title": 10, "label": 9, "tick": 8, "legend": 8, "annotation": 7},
    "notebook": {"title": 12, "label": 11, "tick": 10, "legend": 10, "annotation": 9},
    "talk": {"title": 16, "label": 14, "tick": 12, "legend": 12, "annotation": 11},
    "poster": {"title": 24, "label": 20, "tick": 16, "legend": 16, "annotation": 14},
}

# =============================================================================
# Export Settings
# =============================================================================

DEFAULT_EXPORT_FORMATS = ("png", "svg")
PUBLICATION_EXPORT_FORMATS = ("png", "svg", "pdf")


class ExportFormat(str, Enum):
    """Supported export formats."""

    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    EPS = "eps"
    HTML = "html"  # For Plotly interactive


# =============================================================================
# Theme Enumeration
# =============================================================================


class Theme(str, Enum):
    """Available visualization themes."""

    SCIENTIFIC = "scientific"  # Publication-ready, minimal
    PITCH = "pitch"  # Presentation-ready, bold
    DARK = "dark"  # Dark background for demos
    NOTEBOOK = "notebook"  # Jupyter notebook optimized


class Context(str, Enum):
    """Visualization contexts affecting scale."""

    PAPER = "paper"  # Small figures for papers
    NOTEBOOK = "notebook"  # Medium for Jupyter
    TALK = "talk"  # Large for presentations
    POSTER = "poster"  # Extra large for posters


# =============================================================================
# Configuration Dataclass
# =============================================================================


@dataclass
class VisualizationConfig:
    """Global visualization configuration.

    Attributes:
        theme: Visual theme to use
        context: Size/scale context
        dpi: Resolution for raster output
        figure_width: Default figure width in inches
        aspect_ratio: Default aspect ratio (height = width / ratio)
        font_family: Font family preference
        export_formats: Default export formats
        output_dir: Default output directory
    """

    theme: Theme = Theme.SCIENTIFIC
    context: Context = Context.NOTEBOOK
    dpi: int = DPI_PRESENTATION
    figure_width: float = COLUMN_WIDTH_DOUBLE
    aspect_ratio: float = GOLDEN_RATIO
    font_family: list[str] = field(default_factory=lambda: FONT_FAMILY_SANS.copy())
    export_formats: tuple[str, ...] = DEFAULT_EXPORT_FORMATS
    output_dir: Path = field(default_factory=lambda: Path("outputs/figures"))

    @property
    def figure_height(self) -> float:
        """Calculate figure height from width and aspect ratio."""
        return self.figure_width / self.aspect_ratio

    @property
    def figsize(self) -> tuple[float, float]:
        """Return (width, height) tuple for matplotlib."""
        return (self.figure_width, self.figure_height)

    @property
    def font_sizes(self) -> dict[str, int]:
        """Get font sizes for current context."""
        return FONT_SIZES[self.context.value]


# =============================================================================
# Global Configuration Instance
# =============================================================================

# Default global configuration (can be modified at runtime)
_global_config = VisualizationConfig()


def get_config() -> VisualizationConfig:
    """Get the current global visualization configuration."""
    return _global_config


def set_config(config: VisualizationConfig) -> None:
    """Set the global visualization configuration."""
    global _global_config
    _global_config = config


def configure(
    theme: Theme | str | None = None,
    context: Context | str | None = None,
    dpi: int | None = None,
    figure_width: float | None = None,
    aspect_ratio: float | None = None,
    output_dir: Path | str | None = None,
) -> VisualizationConfig:
    """Configure global visualization settings.

    Args:
        theme: Visual theme (scientific, pitch, dark, notebook)
        context: Size context (paper, notebook, talk, poster)
        dpi: Resolution for raster output
        figure_width: Default figure width in inches
        aspect_ratio: Default aspect ratio
        output_dir: Default output directory

    Returns:
        Updated configuration object
    """
    global _global_config

    if theme is not None:
        _global_config.theme = Theme(theme) if isinstance(theme, str) else theme
    if context is not None:
        _global_config.context = Context(context) if isinstance(context, str) else context
    if dpi is not None:
        _global_config.dpi = dpi
    if figure_width is not None:
        _global_config.figure_width = figure_width
    if aspect_ratio is not None:
        _global_config.aspect_ratio = aspect_ratio
    if output_dir is not None:
        _global_config.output_dir = Path(output_dir)

    return _global_config
