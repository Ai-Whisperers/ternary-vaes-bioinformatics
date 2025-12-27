Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/>`_.


[5.11.0] - 2024
---------------

Added
~~~~~

- **Complete CLI interface** with ``ternary-vae`` / ``tvae`` commands

  - ``train run`` - Train Ternary VAE models
  - ``train hiv`` - HIV-specific training
  - ``analyze stanford`` - Stanford HIVDB analysis
  - ``analyze catnap`` - CATNAP neutralization analysis
  - ``data status`` - Dataset status checking

- **Unified HIV data loaders** in ``src/data/hiv/``

  - Stanford HIVDB loader (7,154 records across 4 drug classes)
  - LANL CTL epitope loader (2,116 epitopes)
  - CATNAP neutralization loader (189,879 records)
  - External dataset loaders (HuggingFace, Zenodo, Kaggle)
  - HXB2 position mapper

- **Analysis scripts** in ``scripts/hiv/analysis/``

  - Drug resistance analysis
  - CTL escape analysis
  - Antibody neutralization analysis
  - Tropism switching analysis
  - Cross-dataset integration
  - Vaccine target identification

- **Comprehensive documentation**

  - Sphinx-based documentation with autodoc
  - API reference for all modules
  - User guides and tutorials
  - Installation instructions

- **Package infrastructure**

  - Modern ``pyproject.toml`` with all dependencies
  - PEP 561 type checking support (``py.typed``)
  - Optional dependency groups (viz, dev, docs, bio, security)

Changed
~~~~~~~

- Reorganized project structure for better modularity
- Updated configuration management
- Improved error handling throughout

Fixed
~~~~~

- Broken links in README documentation
- Type annotations throughout codebase
- Linting issues and code quality improvements


[5.10.0] - 2024
---------------

Added
~~~~~

- Core Ternary VAE model architecture
- Dual VAE system (VAE-A explorer, VAE-B refiner)
- Hyperbolic geometry operations (Poincaré ball)
- P-adic valuation and distance functions
- Codon encoding and genetic code constants
- Basic visualization utilities

Changed
~~~~~~~

- Migrated to PyTorch 2.0+
- Updated minimum Python version to 3.10


[5.0.0] - 2024
--------------

Initial Release
~~~~~~~~~~~~~~~

- Foundational project structure
- Basic VAE implementation
- Codon encoding research
- Initial visualization tools


Previous Versions
-----------------

For changes prior to version 5.0.0, see the git commit history.
