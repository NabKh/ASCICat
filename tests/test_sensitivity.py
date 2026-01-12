"""
Tests for ASCICat Sensitivity Analysis Module

Tests the SensitivityAnalyzer and SensitivityVisualizer classes for:
- Weight grid generation
- Sensitivity analysis calculations
- Rank statistics computation
- Sensitivity indices
- Statistical tests

Author: N. Khossossi
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil


class TestWeightGridGeneration:
    """Tests for weight grid generation methods."""

    def test_generate_weight_grid_basic(self):
        """Test basic weight grid generation."""
        from ascicat.sensitivity import SensitivityAnalyzer

        # Create mock calculator
        class MockCalculator:
            data = pd.DataFrame({'symbol': ['A', 'B', 'C']})

        analyzer = SensitivityAnalyzer(MockCalculator())
        weights = analyzer.generate_weight_grid(n_points=11)

        # Check that all weights are valid
        assert len(weights) > 0, "Should generate at least one weight combination"
        assert weights.shape[1] == 3, "Each weight should have 3 components"

        # Check weight constraints
        for w in weights:
            assert np.isclose(sum(w), 1.0, atol=1e-6), "Weights must sum to 1"
            assert all(w >= 0), "All weights must be non-negative"
            assert all(w <= 1), "All weights must be <= 1"

    def test_generate_weight_grid_bounds(self):
        """Test weight grid respects min/max bounds."""
        from ascicat.sensitivity import SensitivityAnalyzer

        class MockCalculator:
            data = pd.DataFrame({'symbol': ['A', 'B', 'C']})

        analyzer = SensitivityAnalyzer(MockCalculator())
        min_w, max_w = 0.2, 0.6
        weights = analyzer.generate_weight_grid(
            n_points=11, min_weight=min_w, max_weight=max_w
        )

        # Check bounds are respected
        for w in weights:
            assert all(w >= min_w - 1e-6), f"All weights should be >= {min_w}"
            assert all(w <= max_w + 1e-6), f"All weights should be <= {max_w}"

    def test_generate_full_simplex_grid(self):
        """Test full simplex grid generation."""
        from ascicat.sensitivity import SensitivityAnalyzer

        class MockCalculator:
            data = pd.DataFrame({'symbol': ['A', 'B', 'C']})

        analyzer = SensitivityAnalyzer(MockCalculator())
        n_points = 10
        weights = analyzer.generate_full_simplex_grid(n_points=n_points)

        # Check expected number of points (triangular number formula)
        expected_count = (n_points + 1) * (n_points + 2) // 2
        assert len(weights) == expected_count, f"Expected {expected_count} points"

        # Check all weights sum to 1
        for w in weights:
            assert np.isclose(sum(w), 1.0, atol=1e-4), "Weights must sum to 1"

    def test_generate_nature_weight_grid(self):
        """Test standard weight grid generation."""
        from ascicat.sensitivity import SensitivityAnalyzer

        class MockCalculator:
            data = pd.DataFrame({'symbol': ['A', 'B', 'C']})

        analyzer = SensitivityAnalyzer(MockCalculator())
        target = 100
        weights = analyzer.generate_nature_weight_grid(n_target=target)

        # Should be approximately the target count
        assert abs(len(weights) - target) < target * 0.5, \
            f"Should be approximately {target} points"


class TestRankStatistics:
    """Tests for rank statistics computation."""

    def test_compute_robustness_score(self):
        """Test robustness score computation."""
        from ascicat.sensitivity import SensitivityAnalyzer

        class MockCalculator:
            data = pd.DataFrame({'symbol': ['A', 'B', 'C']})

        analyzer = SensitivityAnalyzer(MockCalculator())

        # Perfect performer
        score = analyzer._compute_robustness_score(
            mean_rank=1.0, std_rank=0.0,
            top10_freq=1.0, top20_freq=1.0,
            n_catalysts=100
        )
        assert 0 <= score <= 1, "Score must be in [0, 1]"
        assert score > 0.9, "Perfect performer should have high robustness"

        # Poor performer
        score = analyzer._compute_robustness_score(
            mean_rank=90.0, std_rank=20.0,
            top10_freq=0.0, top20_freq=0.0,
            n_catalysts=100
        )
        assert 0 <= score <= 1, "Score must be in [0, 1]"
        assert score < 0.3, "Poor performer should have low robustness"


class TestKendallInterpretation:
    """Tests for Kendall's W interpretation."""

    def test_interpret_kendall_w(self):
        """Test Kendall's W interpretation strings."""
        from ascicat.sensitivity import SensitivityAnalyzer

        class MockCalculator:
            data = pd.DataFrame({'symbol': ['A']})

        analyzer = SensitivityAnalyzer(MockCalculator())

        # Test various ranges
        assert "Very low" in analyzer._interpret_kendall_w(0.05)
        assert "Low" in analyzer._interpret_kendall_w(0.2)
        assert "Moderate" in analyzer._interpret_kendall_w(0.4)
        assert "Good" in analyzer._interpret_kendall_w(0.6)
        assert "Strong" in analyzer._interpret_kendall_w(0.8)


class TestSensitivityVisualizer:
    """Tests for SensitivityVisualizer class."""

    def test_visualizer_initialization(self):
        """Test visualizer initializes correctly."""
        from ascicat.sensitivity import SensitivityVisualizer
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = SensitivityVisualizer(output_dir=tmpdir)

            assert viz.output_dir.exists(), "Output directory should be created"
            assert viz.dpi == 600, "Default DPI should be 600"
            assert 'activity' in viz.colors, "Should have activity color"
            assert 'stability' in viz.colors, "Should have stability color"
            assert 'cost' in viz.colors, "Should have cost color"

    def test_format_metric_name(self):
        """Test metric name formatting."""
        from ascicat.sensitivity import SensitivityVisualizer
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = SensitivityVisualizer(output_dir=tmpdir)

            assert viz._format_metric_name('best_asci') == 'Best ASCI Score'
            assert viz._format_metric_name('top10_mean_asci') == 'Mean ASCI (Top 10)'
            assert viz._format_metric_name('unknown_metric') == 'Unknown Metric'


class TestSensitivityIndices:
    """Tests for sensitivity index computation."""

    def test_sensitivity_indices_sum(self):
        """Test that sensitivity indices approximately sum to 1."""
        from ascicat.sensitivity import SensitivityAnalyzer

        class MockCalculator:
            data = pd.DataFrame({'symbol': ['A']})

        analyzer = SensitivityAnalyzer(MockCalculator())

        # Create mock sensitivity results
        n_catalysts = 10
        n_weights = 50
        mock_results = {
            'weights': np.random.dirichlet([1, 1, 1], n_weights),
            'rank_matrix': np.random.randint(1, n_catalysts + 1, (n_catalysts, n_weights)),
            'symbols': [f'Cat_{i}' for i in range(n_catalysts)],
        }

        indices = analyzer.compute_sensitivity_indices(mock_results)

        # Check structure
        assert 'S_activity' in indices
        assert 'S_stability' in indices
        assert 'S_cost' in indices

        # Check sum is approximately 1 (after normalization)
        total = sum(indices.values())
        assert np.isclose(total, 1.0, atol=0.01), \
            f"Indices should sum to ~1, got {total}"


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_run_enhanced_sensitivity_analysis_imports(self):
        """Test that convenience function can be imported."""
        from ascicat.sensitivity import run_enhanced_sensitivity_analysis
        from ascicat import run_enhanced_sensitivity_analysis as imported_func

        assert run_enhanced_sensitivity_analysis is not None
        assert imported_func is not None


class TestTernaryCoordinates:
    """Tests for ternary coordinate transformations."""

    def test_ternary_transformation(self):
        """Test ternary coordinate transformation is correct."""
        import numpy as np

        # Test vertices
        # w_a=1, w_s=0, w_c=0 should map to (1, 0)
        w = np.array([[1, 0, 0]])
        x = w[:, 0] + 0.5 * w[:, 1]
        y = (np.sqrt(3) / 2) * w[:, 1]
        assert np.isclose(x[0], 1.0), "Activity corner should be at x=1"
        assert np.isclose(y[0], 0.0), "Activity corner should be at y=0"

        # w_a=0, w_s=1, w_c=0 should map to (0.5, sqrt(3)/2)
        w = np.array([[0, 1, 0]])
        x = w[:, 0] + 0.5 * w[:, 1]
        y = (np.sqrt(3) / 2) * w[:, 1]
        assert np.isclose(x[0], 0.5), "Stability corner should be at x=0.5"
        assert np.isclose(y[0], np.sqrt(3)/2), "Stability corner should be at y=sqrt(3)/2"

        # w_a=0, w_s=0, w_c=1 should map to (0, 0)
        w = np.array([[0, 0, 1]])
        x = w[:, 0] + 0.5 * w[:, 1]
        y = (np.sqrt(3) / 2) * w[:, 1]
        assert np.isclose(x[0], 0.0), "Cost corner should be at x=0"
        assert np.isclose(y[0], 0.0), "Cost corner should be at y=0"

        # Center (1/3, 1/3, 1/3) should map to (0.5, sqrt(3)/6)
        w = np.array([[1/3, 1/3, 1/3]])
        x = w[:, 0] + 0.5 * w[:, 1]
        y = (np.sqrt(3) / 2) * w[:, 1]
        expected_x = 1/3 + 0.5 * 1/3
        expected_y = (np.sqrt(3) / 2) * 1/3
        assert np.isclose(x[0], expected_x), "Center x-coordinate"
        assert np.isclose(y[0], expected_y), "Center y-coordinate"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
