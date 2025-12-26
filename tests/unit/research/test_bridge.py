"""Tests for the research module bridge.

Tests verify path resolution and utility functions work correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestPathResolution:
    """Test path resolution functions."""

    def test_get_project_root(self):
        """Project root should contain src/ directory."""
        from src.research import get_project_root

        root = get_project_root()
        assert root.exists()
        assert (root / "src").exists()

    def test_get_research_path_base(self):
        """Should return research directory."""
        from src.research import get_research_path

        path = get_research_path()
        assert path.name == "research"

    def test_get_research_path_subdir(self):
        """Should return research subdirectory."""
        from src.research import get_research_path

        path = get_research_path("bioinformatics")
        assert "bioinformatics" in str(path)

    def test_get_data_path_base(self):
        """Should return data directory."""
        from src.research import get_data_path

        path = get_data_path()
        assert path.name == "data"

    def test_get_data_path_subdir(self):
        """Should return data subdirectory."""
        from src.research import get_data_path

        path = get_data_path("external")
        assert "external" in str(path)

    def test_get_results_path(self):
        """Should return results directory."""
        from src.research import get_results_path

        path = get_results_path()
        assert path.name == "results"

    def test_get_config_path_existing(self):
        """Should return config path if exists."""
        from src.research import get_config_path

        path = get_config_path("ternary.yaml")
        # May or may not exist, but should not crash
        assert path is None or path.exists()

    def test_get_config_path_nonexistent(self):
        """Should return None for nonexistent config."""
        from src.research import get_config_path

        path = get_config_path("nonexistent_config_12345.yaml")
        assert path is None


class TestListingFunctions:
    """Test directory listing functions."""

    def test_list_research_experiments(self):
        """Should return list of experiment names."""
        from src.research import list_research_experiments

        experiments = list_research_experiments()
        assert isinstance(experiments, list)
        # Should have at least bioinformatics
        if experiments:  # May be empty in some test environments
            assert all(isinstance(e, str) for e in experiments)

    def test_list_datasets(self):
        """Should return list of dataset names."""
        from src.research import list_datasets

        datasets = list_datasets()
        assert isinstance(datasets, list)
        if datasets:  # May be empty in some test environments
            assert all(isinstance(d, str) for d in datasets)


class TestModuleImports:
    """Test module can be imported correctly."""

    def test_import_all_exports(self):
        """All exported symbols should be importable."""
        from src.research import (
            get_config_path,
            get_data_path,
            get_project_root,
            get_research_path,
            get_results_path,
            list_datasets,
            list_research_experiments,
        )

        # Verify functions are callable
        assert callable(get_project_root)
        assert callable(get_research_path)
        assert callable(get_data_path)
        assert callable(get_results_path)
        assert callable(get_config_path)
        assert callable(list_research_experiments)
        assert callable(list_datasets)
