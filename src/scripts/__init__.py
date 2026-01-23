# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""CLI entry points and executable scripts for the Ternary VAE project.

This package contains executable scripts organized by purpose.
Reusable library code lives in src/; these are CLI entry points only.

Subpackages:
    analysis/       Code quality audits, metrics, and validation scripts
    benchmark/      Performance benchmarking and profiling
    docs/           Documentation generation utilities
    epsilon_vae/    Epsilon-VAE research experiments
    eval/           Model evaluation and downstream validation
    hiv/            HIV-specific analysis entry points
    ingest/         Data ingestion and preprocessing
    legal/          Copyright and license management
    maintenance/    Codebase maintenance and migration
    optimization/   Latent space optimization scripts
    setup/          Environment setup and configuration
    training/       Training script variants
    visualization/  Plotting and visualization scripts

Main Entry Points:
    train.py        Main training script for Ternary VAE v5.11

Usage:
    python src/scripts/train.py --config src/configs/ternary.yaml
    python src/scripts/benchmark/run_benchmark.py
    python src/scripts/visualization/visualize_ternary_manifold.py
"""
