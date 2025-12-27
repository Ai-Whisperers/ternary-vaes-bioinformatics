# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Ternary VAE Command Line Interface.

This module provides a modern CLI for the Ternary VAE package using Typer.

Usage:
    # After pip install -e .
    ternary-vae --help
    ternary-vae train --config configs/ternary.yaml
    ternary-vae analyze --dataset stanford-hivdb
    ternary-vae info

    # Short alias
    tvae train --epochs 100
"""

import typer
from rich.console import Console
from rich.table import Table

from . import train as train_module
from . import analyze as analyze_module
from . import data as data_module

__all__ = ["app", "main"]

# Create the main Typer app
app = typer.Typer(
    name="ternary-vae",
    help="Ternary VAE - Dual Neural VAEs for 3-adic algebraic structure learning",
    add_completion=False,
    rich_markup_mode="rich",
)

# Add sub-commands
app.add_typer(train_module.app, name="train", help="Train Ternary VAE models")
app.add_typer(analyze_module.app, name="analyze", help="Run analysis pipelines")
app.add_typer(data_module.app, name="data", help="Data management commands")

console = Console()


@app.command()
def info():
    """Display package information and system status."""
    from src import __version__, __author__, __license__
    import torch
    import sys
    from pathlib import Path

    table = Table(title="Ternary VAE Package Info")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Version", __version__)
    table.add_row("Author", __author__)
    table.add_row("License", __license__)
    table.add_row("Python", sys.version.split()[0])
    table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        table.add_row("CUDA Device", torch.cuda.get_device_name(0))

    # Check for data directories
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    research_datasets = project_root / "data" / "research" / "datasets"

    table.add_row("Data Directory", str(data_dir.exists()))
    table.add_row("Research Datasets", str(research_datasets.exists()))

    console.print(table)


@app.command()
def version():
    """Show version and exit."""
    from src import __version__
    console.print(f"ternary-vae version {__version__}")


@app.callback()
def callback():
    """
    Ternary VAE - Dual Neural VAEs for learning 3-adic algebraic structure.

    A bioinformatics toolkit for analyzing codon evolution, drug resistance,
    and immune escape using hyperbolic geometry.
    """
    pass


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
