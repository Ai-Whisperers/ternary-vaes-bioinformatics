# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Base figure and axes classes for visualization.

This module provides factory functions and helper classes for creating
figures with consistent styling and automatic theme application.
"""

from __future__ import annotations

from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from ..config import Context, Theme, get_config
from ..styles.themes import apply_theme

# =============================================================================
# Figure Factory Functions
# =============================================================================


def create_figure(
    figsize: tuple[float, float] | None = None,
    nrows: int = 1,
    ncols: int = 1,
    theme: Theme | str | None = None,
    context: Context | str | None = None,
    projection: Literal["2d", "3d"] | None = None,
    sharex: bool = False,
    sharey: bool = False,
    squeeze: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes | list[Axes]]:
    """Create a figure with automatic theme application.

    Args:
        figsize: Figure size as (width, height) in inches. If None, uses config default.
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        theme: Theme to apply (scientific, pitch, dark, notebook)
        context: Size context (paper, notebook, talk, poster)
        projection: '3d' for 3D axes, None or '2d' for 2D
        sharex: Share x-axis between subplots
        sharey: Share y-axis between subplots
        squeeze: If True, squeeze extra dimensions from axes array
        **kwargs: Additional arguments passed to plt.subplots()

    Returns:
        Tuple of (figure, axes) where axes may be single Axes or array
    """
    config = get_config()

    # Apply theme
    apply_theme(theme, context)

    # Determine figure size
    if figsize is None:
        figsize = config.figsize

    # Handle 3D projection
    subplot_kw = kwargs.pop("subplot_kw", {})
    if projection == "3d":
        subplot_kw["projection"] = "3d"

    # Create figure
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        subplot_kw=subplot_kw if subplot_kw else None,
        **kwargs,
    )

    return fig, axes


def create_scientific_figure(
    figsize: tuple[float, float] | None = None,
    nrows: int = 1,
    ncols: int = 1,
    **kwargs: Any,
) -> tuple[Figure, Axes | list[Axes]]:
    """Create a figure with scientific publication styling.

    Args:
        figsize: Figure size. Defaults to single column width.
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        **kwargs: Additional arguments passed to create_figure()

    Returns:
        Tuple of (figure, axes)
    """
    if figsize is None:
        from ..config import COLUMN_WIDTH_SINGLE, GOLDEN_RATIO

        figsize = (COLUMN_WIDTH_SINGLE, COLUMN_WIDTH_SINGLE / GOLDEN_RATIO)

    return create_figure(
        figsize=figsize,
        nrows=nrows,
        ncols=ncols,
        theme=Theme.SCIENTIFIC,
        context=Context.PAPER,
        **kwargs,
    )


def create_pitch_figure(
    figsize: tuple[float, float] = (12, 8),
    nrows: int = 1,
    ncols: int = 1,
    **kwargs: Any,
) -> tuple[Figure, Axes | list[Axes]]:
    """Create a figure with presentation styling.

    Args:
        figsize: Figure size. Defaults to presentation size.
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        **kwargs: Additional arguments passed to create_figure()

    Returns:
        Tuple of (figure, axes)
    """
    return create_figure(
        figsize=figsize,
        nrows=nrows,
        ncols=ncols,
        theme=Theme.PITCH,
        context=Context.TALK,
        **kwargs,
    )


def create_3d_figure(
    figsize: tuple[float, float] = (10, 8),
    nrows: int = 1,
    ncols: int = 1,
    theme: Theme | str | None = None,
    context: Context | str | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes3D | list[Axes3D]]:
    """Create a figure with 3D axes.

    Args:
        figsize: Figure size
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        theme: Theme to apply
        context: Size context
        **kwargs: Additional arguments passed to create_figure()

    Returns:
        Tuple of (figure, 3D axes)
    """
    return create_figure(
        figsize=figsize,
        nrows=nrows,
        ncols=ncols,
        theme=theme,
        context=context,
        projection="3d",
        **kwargs,
    )


def create_panel_figure(
    panels: list[dict[str, Any]],
    figsize: tuple[float, float] | None = None,
    theme: Theme | str | None = None,
    context: Context | str | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """Create a figure with named panels using gridspec.

    Args:
        panels: List of panel configurations, each with:
            - name: Panel identifier
            - row: Row position (0-indexed)
            - col: Column position (0-indexed)
            - rowspan: Number of rows to span (default 1)
            - colspan: Number of columns to span (default 1)
            - projection: '3d' for 3D axes (optional)
        figsize: Figure size
        theme: Theme to apply
        context: Size context

    Returns:
        Tuple of (figure, dict mapping panel names to axes)

    Example:
        panels = [
            {"name": "main", "row": 0, "col": 0, "rowspan": 2, "colspan": 2},
            {"name": "side", "row": 0, "col": 2},
            {"name": "bottom", "row": 2, "col": 0, "colspan": 3},
        ]
        fig, axes = create_panel_figure(panels)
        axes["main"].plot(...)
    """
    config = get_config()

    if figsize is None:
        figsize = config.figsize

    apply_theme(theme, context)

    # Determine grid dimensions
    max_row = max(p["row"] + p.get("rowspan", 1) for p in panels)
    max_col = max(p["col"] + p.get("colspan", 1) for p in panels)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(max_row, max_col)

    axes_dict = {}
    for panel in panels:
        name = panel["name"]
        row = panel["row"]
        col = panel["col"]
        rowspan = panel.get("rowspan", 1)
        colspan = panel.get("colspan", 1)
        projection = panel.get("projection")

        subplot_kw = {}
        if projection == "3d":
            subplot_kw["projection"] = "3d"

        ax = fig.add_subplot(
            gs[row : row + rowspan, col : col + colspan],
            **subplot_kw,
        )
        axes_dict[name] = ax

    return fig, axes_dict


# =============================================================================
# Axes Enhancement Functions
# =============================================================================


def despine(
    ax: Axes,
    top: bool = True,
    right: bool = True,
    left: bool = False,
    bottom: bool = False,
) -> None:
    """Remove spines from axes.

    Args:
        ax: Matplotlib axes
        top: Remove top spine
        right: Remove right spine
        left: Remove left spine
        bottom: Remove bottom spine
    """
    if top:
        ax.spines["top"].set_visible(False)
    if right:
        ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(False)


def set_axis_style(
    ax: Axes,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    grid: bool = False,
    despine_top: bool = True,
    despine_right: bool = True,
) -> None:
    """Apply common styling to an axes.

    Args:
        ax: Matplotlib axes
        title: Axes title
        xlabel: X-axis label
        ylabel: Y-axis label
        xlim: X-axis limits
        ylim: Y-axis limits
        grid: Show grid
        despine_top: Remove top spine
        despine_right: Remove right spine
    """
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if grid:
        ax.grid(True, alpha=0.3)

    despine(ax, top=despine_top, right=despine_right)


def add_panel_label(
    ax: Axes,
    label: str,
    loc: Literal["upper left", "upper right", "lower left", "lower right"] = "upper left",
    fontsize: int | None = None,
    fontweight: str = "bold",
    offset: tuple[float, float] = (0.02, 0.98),
) -> None:
    """Add panel label (e.g., 'A', 'B', 'C') to figure panel.

    Args:
        ax: Matplotlib axes
        label: Panel label text (e.g., 'A', 'B')
        loc: Label location
        fontsize: Font size (uses title size if None)
        fontweight: Font weight
        offset: (x, y) offset from corner in axes coordinates
    """
    if fontsize is None:
        fontsize = plt.rcParams["axes.titlesize"]

    # Determine position based on location
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"

    x = offset[0] if "left" in loc else 1 - offset[0]
    y = offset[1] if "upper" in loc else 1 - offset[1]

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        ha=ha,
        va=va,
    )


def add_inset_axes(
    ax: Axes,
    bounds: tuple[float, float, float, float],
    **kwargs: Any,
) -> Axes:
    """Add inset axes to a parent axes.

    Args:
        ax: Parent axes
        bounds: (x, y, width, height) in parent axes coordinates
        **kwargs: Additional arguments for inset_axes

    Returns:
        Inset axes
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    return inset_axes(
        ax,
        width="100%",
        height="100%",
        bbox_to_anchor=bounds,
        bbox_transform=ax.transAxes,
        **kwargs,
    )


# =============================================================================
# Color Bar Helpers
# =============================================================================


def add_colorbar(
    mappable: Any,
    ax: Axes,
    label: str | None = None,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    shrink: float = 1.0,
    aspect: int = 20,
    pad: float = 0.05,
    **kwargs: Any,
) -> Any:
    """Add colorbar to axes with consistent styling.

    Args:
        mappable: The mappable object (scatter, imshow, etc.)
        ax: Axes to attach colorbar to
        label: Colorbar label
        orientation: Colorbar orientation
        shrink: Shrink factor
        aspect: Aspect ratio
        pad: Padding between axes and colorbar
        **kwargs: Additional arguments for colorbar

    Returns:
        Colorbar object
    """
    fig = ax.get_figure()
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        orientation=orientation,
        shrink=shrink,
        aspect=aspect,
        pad=pad,
        **kwargs,
    )

    if label:
        cbar.set_label(label)

    return cbar


# =============================================================================
# Legend Helpers
# =============================================================================


def add_legend(
    ax: Axes,
    loc: str = "best",
    frameon: bool = True,
    framealpha: float = 0.9,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    """Add legend with consistent styling.

    Args:
        ax: Matplotlib axes
        loc: Legend location
        frameon: Show legend frame
        framealpha: Frame transparency
        title: Legend title
        **kwargs: Additional legend arguments

    Returns:
        Legend object
    """
    legend = ax.legend(
        loc=loc,
        frameon=frameon,
        framealpha=framealpha,
        title=title,
        **kwargs,
    )

    return legend
