"""
Test Suite for Scoring Functions
=================================

Tests for mathematical scoring implementations.

Test Coverage:
- Activity scoring (linear and Gaussian)
- Stability scoring
- Cost scoring
- Combined ASCI calculation
- Edge cases and numerical stability

Author: Süleyman Er
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ascicat.scoring import (
    ScoringFunctions,
    score_activity,
    score_stability,
    score_cost,
    calculate_asci
)
from ascicat.config import ASCIConstants


class TestActivityScoring:
    """Test activity scoring functions"""
    
    def test_linear_activity_at_optimum(self):
        """Test linear scoring at optimal binding energy"""
        score = ScoringFunctions.activity_score(
            delta_E=-0.27,
            optimal_E=-0.27,
            sigma_a=0.15,
            method='linear'
        )
        assert np.isclose(score, 1.0)
    
    def test_gaussian_activity_at_optimum(self):
        """Test Gaussian scoring at optimal binding energy"""
        score = ScoringFunctions.activity_score(
            delta_E=-0.27,
            optimal_E=-0.27,
            sigma_a=0.15,
            method='gaussian'
        )
        assert np.isclose(score, 1.0)
    
    def test_linear_activity_decreases_with_deviation(self):
        """Test linear scoring decreases with deviation"""
        optimal = -0.27
        sigma = 0.15
        
        # Test points away from optimum
        deviations = [0.05, 0.10, 0.15, 0.20]
        scores = []
        
        for dev in deviations:
            score = ScoringFunctions.activity_score(
                delta_E=optimal + dev,
                optimal_E=optimal,
                sigma_a=sigma,
                method='linear'
            )
            scores.append(score)
        
        # Scores should decrease
        for i in range(len(scores)-1):
            assert scores[i] > scores[i+1]
    
    def test_gaussian_activity_exponential_decay(self):
        """Test Gaussian scoring exponential decay"""
        optimal = -0.27
        sigma = 0.15
        
        # At 1 sigma away
        score_1sigma = ScoringFunctions.activity_score(
            delta_E=optimal + sigma,
            optimal_E=optimal,
            sigma_a=sigma,
            method='gaussian'
        )
        
        # Should be exp(-0.5) ≈ 0.606
        assert np.isclose(score_1sigma, np.exp(-0.5), rtol=0.01)
    
    def test_activity_score_bounds(self):
        """Test activity scores are bounded [0, 1]"""
        for method in ['linear', 'gaussian']:
            for delta_E in np.linspace(-1.0, 0.0, 20):
                score = ScoringFunctions.activity_score(
                    delta_E=delta_E,
                    optimal_E=-0.27,
                    sigma_a=0.15,
                    method=method
                )
                assert 0.0 <= score <= 1.0
    
    def test_activity_vectorized(self):
        """Test vectorized activity scoring"""
        delta_E = np.array([-0.27, -0.30, -0.24, -0.35])
        
        scores = ScoringFunctions.activity_score(
            delta_E=delta_E,
            optimal_E=-0.27,
            sigma_a=0.15,
            method='linear'
        )
        
        assert len(scores) == len(delta_E)
        assert all(0.0 <= s <= 1.0 for s in scores)
    
    def test_invalid_method(self):
        """Test invalid scoring method"""
        with pytest.raises(ValueError):
            ScoringFunctions.activity_score(
                delta_E=-0.27,
                optimal_E=-0.27,
                sigma_a=0.15,
                method='invalid'
            )
    
    def test_invalid_sigma(self):
        """Test invalid sigma parameter"""
        with pytest.raises(ValueError):
            ScoringFunctions.activity_score(
                delta_E=-0.27,
                optimal_E=-0.27,
                sigma_a=-0.15,  # Negative sigma
                method='linear'
            )


class TestStabilityScoring:
    """Test stability scoring function"""
    
    def test_stability_at_minimum(self):
        """Test stability score at minimum surface energy"""
        score = ScoringFunctions.stability_score(
            surface_energy=0.1,
            gamma_min=0.1,
            gamma_max=5.0
        )
        assert np.isclose(score, 1.0)
    
    def test_stability_at_maximum(self):
        """Test stability score at maximum surface energy"""
        score = ScoringFunctions.stability_score(
            surface_energy=5.0,
            gamma_min=0.1,
            gamma_max=5.0
        )
        assert np.isclose(score, 0.0)
    
    def test_stability_linear_interpolation(self):
        """Test linear interpolation in stability scoring"""
        score_mid = ScoringFunctions.stability_score(
            surface_energy=2.55,  # Midpoint
            gamma_min=0.1,
            gamma_max=5.0
        )
        assert np.isclose(score_mid, 0.5, rtol=0.01)
    
    def test_stability_inverse_relationship(self):
        """Test inverse relationship: higher gamma → lower score"""
        gamma_min, gamma_max = 0.1, 5.0
        
        score_low = ScoringFunctions.stability_score(1.0, gamma_min, gamma_max)
        score_high = ScoringFunctions.stability_score(4.0, gamma_min, gamma_max)
        
        assert score_low > score_high
    
    def test_stability_score_bounds(self):
        """Test stability scores are bounded [0, 1]"""
        for gamma in np.linspace(0.1, 5.0, 50):
            score = ScoringFunctions.stability_score(
                surface_energy=gamma,
                gamma_min=0.1,
                gamma_max=5.0
            )
            assert 0.0 <= score <= 1.0
    
    def test_stability_vectorized(self):
        """Test vectorized stability scoring"""
        gamma = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        
        scores = ScoringFunctions.stability_score(
            surface_energy=gamma,
            gamma_min=0.1,
            gamma_max=5.0
        )
        
        assert len(scores) == len(gamma)
        assert all(0.0 <= s <= 1.0 for s in scores)


class TestCostScoring:
    """Test cost scoring function"""
    
    def test_cost_at_minimum(self):
        """Test cost score at minimum cost"""
        score = ScoringFunctions.cost_score(
            cost=1.0,
            cost_min=1.0,
            cost_max=200000.0
        )
        assert np.isclose(score, 1.0)
    
    def test_cost_at_maximum(self):
        """Test cost score at maximum cost"""
        score = ScoringFunctions.cost_score(
            cost=200000.0,
            cost_min=1.0,
            cost_max=200000.0
        )
        assert np.isclose(score, 0.0)
    
    def test_cost_logarithmic_scaling(self):
        """Test logarithmic scaling in cost scoring"""
        # Cost increase by order of magnitude should have consistent effect
        cost_min, cost_max = 1.0, 200000.0
        
        score_10 = ScoringFunctions.cost_score(10.0, cost_min, cost_max)
        score_100 = ScoringFunctions.cost_score(100.0, cost_min, cost_max)
        score_1000 = ScoringFunctions.cost_score(1000.0, cost_min, cost_max)
        
        # Check roughly equal spacing
        diff1 = score_10 - score_100
        diff2 = score_100 - score_1000
        
        assert np.isclose(diff1, diff2, rtol=0.1)
    
    def test_cost_score_bounds(self):
        """Test cost scores are bounded [0, 1]"""
        for cost in np.logspace(0, 5, 50):  # 1 to 100000
            score = ScoringFunctions.cost_score(
                cost=cost,
                cost_min=1.0,
                cost_max=200000.0
            )
            assert 0.0 <= score <= 1.0
    
    def test_cost_inverse_relationship(self):
        """Test inverse relationship: higher cost → lower score"""
        cost_min, cost_max = 1.0, 200000.0
        
        score_cheap = ScoringFunctions.cost_score(1000.0, cost_min, cost_max)
        score_expensive = ScoringFunctions.cost_score(100000.0, cost_min, cost_max)
        
        assert score_cheap > score_expensive
    
    def test_cost_vectorized(self):
        """Test vectorized cost scoring"""
        costs = np.array([10, 100, 1000, 10000, 100000])
        
        scores = ScoringFunctions.cost_score(
            cost=costs,
            cost_min=1.0,
            cost_max=200000.0
        )
        
        assert len(scores) == len(costs)
        assert all(0.0 <= s <= 1.0 for s in scores)


class TestCombinedASCI:
    """Test combined ASCI calculation"""
    
    def test_asci_equal_weights(self):
        """Test ASCI with equal weights"""
        asci = ScoringFunctions.combined_asci_score(
            activity_score=0.8,
            stability_score=0.6,
            cost_score=0.7,
            w_a=0.33,
            w_s=0.33,
            w_c=0.34
        )
        
        expected = 0.33 * 0.8 + 0.33 * 0.6 + 0.34 * 0.7
        assert np.isclose(asci, expected)
    
    def test_asci_activity_dominant(self):
        """Test ASCI with activity-dominant weights"""
        asci = ScoringFunctions.combined_asci_score(
            activity_score=0.9,
            stability_score=0.5,
            cost_score=0.5,
            w_a=0.6,
            w_s=0.2,
            w_c=0.2
        )
        
        # High activity should dominate
        expected = 0.6 * 0.9 + 0.2 * 0.5 + 0.2 * 0.5
        assert np.isclose(asci, expected)
    
    def test_asci_bounds(self):
        """Test ASCI is bounded [0, 1]"""
        # Test various combinations
        for _ in range(100):
            s_a = np.random.uniform(0, 1)
            s_s = np.random.uniform(0, 1)
            s_c = np.random.uniform(0, 1)
            
            asci = ScoringFunctions.combined_asci_score(
                s_a, s_s, s_c,
                w_a=0.33, w_s=0.33, w_c=0.34
            )
            
            assert 0.0 <= asci <= 1.0
    
    def test_asci_invalid_weight_sum(self):
        """Test ASCI with invalid weight sum"""
        with pytest.raises(ValueError):
            ScoringFunctions.combined_asci_score(
                0.8, 0.6, 0.7,
                w_a=0.5, w_s=0.3, w_c=0.3  # Sum = 1.1
            )
    
    def test_asci_negative_weights(self):
        """Test ASCI with negative weights"""
        with pytest.raises(ValueError):
            ScoringFunctions.combined_asci_score(
                0.8, 0.6, 0.7,
                w_a=-0.1, w_s=0.6, w_c=0.5
            )
    
    def test_asci_vectorized(self):
        """Test vectorized ASCI calculation"""
        s_a = np.array([0.8, 0.6, 0.9, 0.7])
        s_s = np.array([0.7, 0.8, 0.5, 0.6])
        s_c = np.array([0.6, 0.7, 0.8, 0.9])
        
        asci = ScoringFunctions.combined_asci_score(
            s_a, s_s, s_c,
            w_a=0.4, w_s=0.3, w_c=0.3
        )
        
        assert len(asci) == len(s_a)
        assert all(0.0 <= a <= 1.0 for a in asci)


class TestConvenienceFunctions:
    """Test convenience wrapper functions"""
    
    def test_score_activity_wrapper(self):
        """Test score_activity convenience function"""
        score = score_activity(
            delta_E=-0.27,
            optimal_E=-0.27,
            sigma_a=0.15,
            method='linear'
        )
        assert np.isclose(score, 1.0)
    
    def test_score_stability_wrapper(self):
        """Test score_stability convenience function"""
        score = score_stability(
            surface_energy=0.1,
            gamma_min=0.1,
            gamma_max=5.0
        )
        assert np.isclose(score, 1.0)
    
    def test_score_cost_wrapper(self):
        """Test score_cost convenience function"""
        score = score_cost(
            cost=1.0,
            cost_min=1.0,
            cost_max=200000.0
        )
        assert np.isclose(score, 1.0)
    
    def test_calculate_asci_wrapper(self):
        """Test calculate_asci convenience function"""
        asci = calculate_asci(
            activity_score=0.8,
            stability_score=0.6,
            cost_score=0.7,
            w_a=0.33,
            w_s=0.33,
            w_c=0.34
        )
        
        expected = 0.33 * 0.8 + 0.33 * 0.6 + 0.34 * 0.7
        assert np.isclose(asci, expected)


class TestNumericalStability:
    """Test numerical stability and edge cases"""
    
    def test_activity_very_far_from_optimum(self):
        """Test activity scoring far from optimum"""
        score = ScoringFunctions.activity_score(
            delta_E=-2.0,  # Very far from optimum
            optimal_E=-0.27,
            sigma_a=0.15,
            method='gaussian'
        )
        assert score >= 0.0  # Should not be negative
        assert score < 0.01  # Should be very small
    
    def test_stability_extrapolation(self):
        """Test stability scoring with out-of-range values"""
        # Below min
        score_low = ScoringFunctions.stability_score(
            surface_energy=0.0,  # Below min
            gamma_min=0.1,
            gamma_max=5.0
        )
        assert score_low <= 1.0
        
        # Above max
        score_high = ScoringFunctions.stability_score(
            surface_energy=6.0,  # Above max
            gamma_min=0.1,
            gamma_max=5.0
        )
        assert score_high >= 0.0
    
    def test_cost_extreme_values(self):
        """Test cost scoring with extreme values"""
        cost_min, cost_max = 1.0, 200000.0
        
        # Very cheap
        score_cheap = ScoringFunctions.cost_score(0.1, cost_min, cost_max)
        assert 0.9 <= score_cheap <= 1.0
        
        # Very expensive
        score_expensive = ScoringFunctions.cost_score(1e6, cost_min, cost_max)
        assert 0.0 <= score_expensive <= 0.1
    
    def test_nan_handling(self):
        """Test handling of NaN values"""
        # Activity scoring should handle NaN gracefully
        with pytest.warns(UserWarning):
            score = ScoringFunctions.activity_score(
                delta_E=np.nan,
                optimal_E=-0.27,
                sigma_a=0.15,
                method='linear'
            )
            assert np.isfinite(score)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
