Installation
============

Requirements
------------

- Python 3.10 or later
- PyTorch 2.0 or later
- CUDA (optional, for GPU acceleration)


From Source
-----------

Clone the repository and install in development mode:

.. code-block:: bash

    git clone https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics.git
    cd ternary-vaes-bioinformatics
    pip install -e .

With optional dependencies:

.. code-block:: bash

    # All optional dependencies
    pip install -e ".[all]"

    # Just visualization
    pip install -e ".[viz]"

    # Development tools
    pip install -e ".[dev]"

    # Documentation building
    pip install -e ".[docs]"

    # Bioinformatics tools
    pip install -e ".[bio]"


Optional Dependency Groups
--------------------------

- **viz**: matplotlib, seaborn, plotly, tensorboard
- **dev**: pytest, black, ruff, mypy
- **docs**: sphinx, sphinx-rtd-theme, myst-parser
- **bio**: biopython, pyarrow
- **security**: pip-audit, bandit


Verifying Installation
----------------------

.. code-block:: bash

    # Check CLI is working
    ternary-vae info

    # Run tests
    pytest tests/unit/


GPU Support
-----------

For GPU acceleration, ensure you have CUDA installed and PyTorch with CUDA support:

.. code-block:: bash

    # Check CUDA availability in Python
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"


Data Setup
----------

Some analysis features require external datasets:

.. code-block:: bash

    # Check data status
    ternary-vae data status

    # Download HuggingFace datasets
    ternary-vae data download huggingface
