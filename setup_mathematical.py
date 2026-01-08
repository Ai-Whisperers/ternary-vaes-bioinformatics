"""
Setup script for Ternary VAE Mathematical Framework

This package provides the core mathematical substrate for training
Variational Autoencoders on 3-adic (p-adic) ternary structures.
"""

from setuptools import setup, find_packages
import os

# Read long description from README if it exists
long_description = "Mathematical framework for 3-adic hyperbolic VAEs"
readme_path = os.path.join(os.path.dirname(__file__), "README_mathematical.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="ternary-vae-framework",
    version="1.0.0",
    description="Mathematical framework for 3-adic hyperbolic VAEs",
    long_description=long_description,
    long_description_content_type="text/markdown",

    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},

    # Core dependencies
    install_requires=[
        "torch>=2.0.0",
        "geoopt>=0.5.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "pyyaml>=6.0",
        "tensorboard>=2.8.0",
    ],

    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },

    # Entry points
    entry_points={
        "console_scripts": [
            "ternary-train=scripts.train:main",
        ],
    },

    # Metadata
    author="AI Whisperers",
    author_email="support@aiwhisperers.com",
    url="https://github.com/Ai-Whisperers/ternary-vae-framework",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.10",

    # Include package data
    include_package_data=True,
    zip_safe=False,
)