# Visualization Module

Professional-quality visualization system for scientific publications and presentations.

## Purpose

This module provides a unified visualization system with:
- Scientific publication styles (Nature, Science, IEEE compatible)
- Presentation/pitch styles (bold, high-contrast)
- Colorblind-friendly palettes
- Multi-format export (PNG, SVG, PDF)

## Quick Start

```python
from src.visualization import (
    create_figure,
    use_scientific_style,
    SEMANTIC,
    save_figure,
)

# Apply scientific styling
use_scientific_style()

# Create figure with automatic styling
fig, ax = create_figure()
ax.plot([1, 2, 3], [1, 4, 9], color=SEMANTIC.primary)
ax.set_title("Example Plot")

# Save in publication-quality formats
save_figure(fig, "example", formats=["png", "svg", "pdf"])
```

## Themes

```python
from src.visualization import (
    use_scientific_style,  # For papers
    use_pitch_style,       # For presentations
    use_dark_style,        # For demos
    use_notebook_style,    # For Jupyter
)

# Or use context manager for temporary styling
from src.visualization import theme_context, Theme, Context

with theme_context(Theme.SCIENTIFIC, Context.PAPER):
    fig, ax = create_figure()
    # Scientific paper styling active
```

## Color Palettes

### Semantic Palette

```python
from src.visualization import SEMANTIC

SEMANTIC.primary    # Main color for emphasis
SEMANTIC.secondary  # Supporting color
SEMANTIC.success    # Positive outcomes (green)
SEMANTIC.warning    # Caution (yellow/orange)
SEMANTIC.danger     # Negative outcomes (red)
```

### Colorblind-Friendly Palettes

```python
from src.visualization import TOLMUTED, TOLVIBRANT, TABLEAU10

# Paul Tol's muted palette (8 colors)
colors = TOLMUTED

# Paul Tol's vibrant palette (7 colors)
colors = TOLVIBRANT

# Tableau 10 (colorblind-safe)
colors = TABLEAU10
```

### Colormaps

```python
from src.visualization import (
    get_risk_cmap,       # Red-yellow-green for risk
    get_safety_cmap,     # Safety/stability zones
    get_goldilocks_cmap, # Optimal range highlighting
    get_diverging_cmap,  # Centered diverging data
    get_sequential_cmap, # Single-hue progression
)

# Example: Risk colormap for mutation data
cmap = get_risk_cmap()
im = ax.imshow(risk_matrix, cmap=cmap)
```

## Figure Creation

```python
from src.visualization import (
    create_figure,           # Standard figure
    create_scientific_figure, # Publication-ready
    create_pitch_figure,     # Presentation-ready
    create_3d_figure,        # 3D visualization
    create_panel_figure,     # Multi-panel layout
)

# Multi-panel figure (e.g., 2x2 grid)
fig, axes = create_panel_figure(nrows=2, ncols=2)
```

## Export

```python
from src.visualization import (
    save_figure,
    save_publication_figure,  # 300 DPI, PDF/PNG
    save_presentation_figure, # Large, high-contrast
    save_web_figure,          # Optimized for web
)

# Publication export (300 DPI, multiple formats)
save_publication_figure(fig, "figure1")

# Batch export
save_figure_batch([fig1, fig2], ["panel_a", "panel_b"])
```

## Annotations

```python
from src.visualization import (
    add_significance_bracket,
    add_pvalue_annotation,
    add_goldilocks_zones,
    add_threshold_line,
    despine,
)

# Add statistical significance
add_significance_bracket(ax, x1=0, x2=1, y=10, pvalue=0.001)

# Add Goldilocks zone shading
add_goldilocks_zones(ax, optimal_range=(0.3, 0.7))

# Clean up spines
despine(ax)
```

## Files

| File | Description |
|------|-------------|
| `config.py` | Theme and export configuration |
| `styles.py` | Color palettes and theme functions |
| `core.py` | Figure creation and export utilities |
| `plots/` | Specialized plot types |
| `projections/` | Mathematical projections |

## Publication Standards

- **DPI**: 300 for print, 150 for screen
- **Font**: Sans-serif (Helvetica/Arial) for figures
- **Size**: Single column (3.5") or double column (7")
- **Format**: PDF for vector, PNG for raster
