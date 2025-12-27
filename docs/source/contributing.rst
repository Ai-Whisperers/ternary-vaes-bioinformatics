Contributing
============

We welcome contributions to Ternary VAE! This guide will help you get started.


Development Setup
-----------------

1. Fork and clone the repository:

.. code-block:: bash

    git clone https://github.com/YOUR-USERNAME/ternary-vaes-bioinformatics.git
    cd ternary-vaes-bioinformatics

2. Install development dependencies:

.. code-block:: bash

    pip install -e ".[dev]"

3. Set up pre-commit hooks (optional):

.. code-block:: bash

    pre-commit install


Code Style
----------

We use the following tools to maintain code quality:

- **Black** - Code formatting (line length 120)
- **isort** - Import sorting
- **Ruff** - Linting
- **mypy** - Type checking

Run all checks:

.. code-block:: bash

    black src/ tests/
    isort src/ tests/
    ruff check src/ tests/
    mypy src/


Testing
-------

We use pytest for testing:

.. code-block:: bash

    # Run all tests
    pytest

    # Run with coverage
    pytest --cov=src --cov-report=html

    # Run specific test file
    pytest tests/unit/test_models.py


Test Guidelines
~~~~~~~~~~~~~~~

1. Write tests for all new functionality
2. Maintain test coverage above 80%
3. Use meaningful test names
4. Include docstrings explaining what each test verifies


Pull Request Process
--------------------

1. Create a feature branch:

.. code-block:: bash

    git checkout -b feature/your-feature-name

2. Make your changes and commit:

.. code-block:: bash

    git add .
    git commit -m "feat: add your feature description"

3. Push and create a pull request:

.. code-block:: bash

    git push origin feature/your-feature-name


Commit Message Format
~~~~~~~~~~~~~~~~~~~~~

We use conventional commits:

- ``feat:`` - New features
- ``fix:`` - Bug fixes
- ``docs:`` - Documentation changes
- ``test:`` - Test additions/modifications
- ``refactor:`` - Code refactoring
- ``chore:`` - Maintenance tasks


Code Review
-----------

All submissions require code review:

1. At least one approval from a maintainer
2. All CI checks must pass
3. No merge conflicts
4. Documentation updated if needed


Documentation
-------------

Build and preview documentation locally:

.. code-block:: bash

    cd docs
    make html
    # Open _build/html/index.html in browser

Or use live reload:

.. code-block:: bash

    make livehtml


Adding Documentation
~~~~~~~~~~~~~~~~~~~~

- API docs are auto-generated from docstrings
- Use Google-style docstrings
- Add examples where helpful


Reporting Issues
----------------

When reporting bugs:

1. Check existing issues first
2. Include Python version and OS
3. Provide minimal reproduction steps
4. Include error messages and tracebacks


Feature Requests
----------------

We welcome feature requests! Please:

1. Describe the use case
2. Explain expected behavior
3. Consider implementation approach


Questions
---------

For questions, open a GitHub Discussion or reach out to maintainers.
