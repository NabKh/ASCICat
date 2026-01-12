"""
Test Suite for ASCICalculator
==============================

Comprehensive tests for the main ASCI calculation engine.

Test Coverage:
- Initialization and configuration
- Data loading and validation
- ASCI score calculation
- Ranking and statistics
- Error handling

Author: Süleyman Er
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ascicat import ASCICalculator, ReactionConfig
from ascicat.config import ASCIConstants


class TestASCICalculatorInitialization:
    """Test calculator initialization"""
    
    def test_her_initialization(self):
        """Test HER calculator initialization"""
        calc = ASCICalculator(reaction='HER')
        assert calc.reaction == 'HER'
        assert calc.config.name == 'HER'
        assert calc.config.optimal_energy == -0.27
        assert calc.scoring_method == 'linear'
    
    def test_co2rr_initialization(self):
        """Test CO2RR calculator initialization"""
        calc = ASCICalculator(reaction='CO2RR', pathway='CO')
        assert calc.reaction == 'CO2RR'
        assert calc.pathway == 'CO'
        assert calc.config.optimal_energy == -0.67
    
    def test_invalid_reaction(self):
        """Test initialization with invalid reaction"""
        with pytest.raises(ValueError):
            ASCICalculator(reaction='INVALID')
    
    def test_invalid_scoring_method(self):
        """Test initialization with invalid scoring method"""
        with pytest.raises(ValueError):
            ASCICalculator(reaction='HER', scoring_method='invalid')
    
    def test_verbose_mode(self):
        """Test verbose mode initialization"""
        calc = ASCICalculator(reaction='HER', verbose=False)
        assert calc.verbose == False


class TestDataLoading:
    """Test data loading functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample catalyst data"""
        return pd.DataFrame({
            'DFT_ads_E': [-0.27, -0.30, -0.24, -0.35, -0.20],
            'surface_energy': [0.5, 1.0, 0.8, 1.5, 0.6],
            'Cost': [50000, 100000, 25000, 150000, 10000],
            'symbol': ['Pt', 'Ru', 'Ni', 'Ir', 'Cu'],
            'AandB': ['Pt_111', 'Ru_0001', 'Ni_111', 'Ir_111', 'Cu_111'],
            'reaction_type': ['HER'] * 5,
            'optimal_energy': [-0.27] * 5,
            'activity_width': [0.15] * 5
        })
    
    @pytest.fixture
    def temp_data_file(self, sample_data):
        """Create temporary data file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            return f.name
    
    def test_load_valid_data(self, temp_data_file):
        """Test loading valid data"""
        calc = ASCICalculator(reaction='HER', verbose=False)
        calc.load_data(temp_data_file)
        
        assert calc.data is not None
        assert len(calc.data) == 5
        assert 'DFT_ads_E' in calc.data.columns
        assert 'surface_energy' in calc.data.columns
        assert 'Cost' in calc.data.columns
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file"""
        calc = ASCICalculator(reaction='HER', verbose=False)
        
        with pytest.raises(FileNotFoundError):
            calc.load_data('nonexistent_file.csv')
    
    def test_missing_required_columns(self):
        """Test data with missing required columns"""
        calc = ASCICalculator(reaction='HER', verbose=False)
        
        # Create data with missing columns
        incomplete_data = pd.DataFrame({
            'DFT_ads_E': [-0.27, -0.30],
            'symbol': ['Pt', 'Ru']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            incomplete_data.to_csv(f.name, index=False)
            
            with pytest.raises(ValueError):
                calc.load_data(f.name, validate=True)


class TestASCICalculation:
    """Test ASCI score calculation"""
    
    @pytest.fixture
    def loaded_calculator(self, sample_data_file):
        """Create calculator with loaded data"""
        calc = ASCICalculator(reaction='HER', verbose=False)
        calc.load_data(sample_data_file)
        return calc
    
    @pytest.fixture
    def sample_data_file(self):
        """Create sample data file"""
        data = pd.DataFrame({
            'DFT_ads_E': [-0.27, -0.30, -0.24, -0.35, -0.20],
            'surface_energy': [0.5, 1.0, 0.8, 1.5, 0.6],
            'Cost': [50000, 100000, 25000, 150000, 10000],
            'symbol': ['Pt', 'Ru', 'Ni', 'Ir', 'Cu'],
            'AandB': ['Pt_111', 'Ru_0001', 'Ni_111', 'Ir_111', 'Cu_111'],
            'reaction_type': ['HER'] * 5,
            'optimal_energy': [-0.27] * 5,
            'activity_width': [0.15] * 5
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            return f.name
    
    def test_calculate_asci_default_weights(self, loaded_calculator):
        """Test ASCI calculation with default weights"""
        results = loaded_calculator.calculate_asci()
        
        assert 'ASCI' in results.columns
        assert 'activity_score' in results.columns
        assert 'stability_score' in results.columns
        assert 'cost_score' in results.columns
        assert 'rank' in results.columns
        
        # Check value ranges
        assert (results['ASCI'] >= 0).all() and (results['ASCI'] <= 1).all()
        assert (results['activity_score'] >= 0).all()
        assert (results['stability_score'] >= 0).all()
        assert (results['cost_score'] >= 0).all()
    
    def test_calculate_asci_custom_weights(self, loaded_calculator):
        """Test ASCI calculation with custom weights"""
        results = loaded_calculator.calculate_asci(w_a=0.5, w_s=0.3, w_c=0.2)
        
        assert loaded_calculator.current_weights == (0.5, 0.3, 0.2)
        assert 'ASCI' in results.columns
    
    def test_invalid_weights_sum(self, loaded_calculator):
        """Test calculation with invalid weight sum"""
        with pytest.raises(ValueError):
            loaded_calculator.calculate_asci(w_a=0.5, w_s=0.3, w_c=0.3)
    
    def test_negative_weights(self, loaded_calculator):
        """Test calculation with negative weights"""
        with pytest.raises(ValueError):
            loaded_calculator.calculate_asci(w_a=-0.1, w_s=0.6, w_c=0.5)
    
    def test_ranking_order(self, loaded_calculator):
        """Test that rankings are correctly ordered"""
        results = loaded_calculator.calculate_asci()
        
        # Rankings should be sequential
        ranks = sorted(results['rank'].values)
        expected_ranks = list(range(1, len(results) + 1))
        assert ranks == expected_ranks
        
        # ASCI should decrease with rank
        asci_values = results.sort_values('rank')['ASCI'].values
        assert all(asci_values[i] >= asci_values[i+1] for i in range(len(asci_values)-1))


class TestTopCatalysts:
    """Test top catalyst retrieval"""
    
    @pytest.fixture
    def calculated_results(self, sample_data_file):
        """Create calculator with calculated results"""
        calc = ASCICalculator(reaction='HER', verbose=False)
        calc.load_data(sample_data_file)
        calc.calculate_asci()
        return calc
    
    @pytest.fixture
    def sample_data_file(self):
        """Create sample data file"""
        data = pd.DataFrame({
            'DFT_ads_E': np.linspace(-0.35, -0.20, 20),
            'surface_energy': np.random.uniform(0.5, 2.0, 20),
            'Cost': np.random.uniform(10000, 150000, 20),
            'symbol': [f'Cat_{i}' for i in range(20)],
            'AandB': [f'Cat_{i}_111' for i in range(20)],
            'reaction_type': ['HER'] * 20,
            'optimal_energy': [-0.27] * 20,
            'activity_width': [0.15] * 20
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            return f.name
    
    def test_get_top_n_catalysts(self, calculated_results):
        """Test retrieving top N catalysts"""
        top5 = calculated_results.get_top_catalysts(n=5)
        
        assert len(top5) == 5
        assert top5['rank'].iloc[0] == 1
        assert top5['rank'].iloc[4] == 5
    
    def test_get_top_more_than_available(self, calculated_results):
        """Test requesting more catalysts than available"""
        all_catalysts = calculated_results.get_top_catalysts(n=100)
        
        assert len(all_catalysts) == len(calculated_results.results)


class TestStatistics:
    """Test statistical analysis"""
    
    @pytest.fixture
    def calculated_results(self, sample_data_file):
        """Create calculator with calculated results"""
        calc = ASCICalculator(reaction='HER', verbose=False)
        calc.load_data(sample_data_file)
        calc.calculate_asci()
        return calc
    
    @pytest.fixture
    def sample_data_file(self):
        """Create sample data file"""
        data = pd.DataFrame({
            'DFT_ads_E': np.linspace(-0.35, -0.20, 20),
            'surface_energy': np.random.uniform(0.5, 2.0, 20),
            'Cost': np.random.uniform(10000, 150000, 20),
            'symbol': [f'Cat_{i}' for i in range(20)],
            'AandB': [f'Cat_{i}_111' for i in range(20)],
            'reaction_type': ['HER'] * 20,
            'optimal_energy': [-0.27] * 20,
            'activity_width': [0.15] * 20
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            return f.name
    
    def test_get_statistics(self, calculated_results):
        """Test statistics generation"""
        stats = calculated_results.get_statistics()
        
        assert 'reaction' in stats
        assert 'total_catalysts' in stats
        assert 'asci' in stats
        assert 'activity_score' in stats
        assert 'stability_score' in stats
        assert 'cost_score' in stats
        
        # Check ASCI statistics
        assert 'mean' in stats['asci']
        assert 'std' in stats['asci']
        assert 'min' in stats['asci']
        assert 'max' in stats['asci']
    
    def test_statistics_without_calculation(self):
        """Test statistics before calculation"""
        calc = ASCICalculator(reaction='HER', verbose=False)
        
        with pytest.raises((ValueError, AttributeError)):
            calc.get_statistics()


class TestResultsSaving:
    """Test result saving functionality"""
    
    @pytest.fixture
    def calculated_results(self, sample_data_file):
        """Create calculator with calculated results"""
        calc = ASCICalculator(reaction='HER', verbose=False)
        calc.load_data(sample_data_file)
        calc.calculate_asci()
        return calc
    
    @pytest.fixture
    def sample_data_file(self):
        """Create sample data file"""
        data = pd.DataFrame({
            'DFT_ads_E': [-0.27, -0.30, -0.24],
            'surface_energy': [0.5, 1.0, 0.8],
            'Cost': [50000, 100000, 25000],
            'symbol': ['Pt', 'Ru', 'Ni'],
            'AandB': ['Pt_111', 'Ru_0001', 'Ni_111'],
            'reaction_type': ['HER'] * 3,
            'optimal_energy': [-0.27] * 3,
            'activity_width': [0.15] * 3
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            return f.name
    
    def test_save_results(self, calculated_results):
        """Test saving results to file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        calculated_results.save_results(output_path, include_metadata=True)
        
        # Verify file exists
        assert Path(output_path).exists()
        
        # Verify file can be read
        saved_data = pd.read_csv(output_path, comment='#')
        assert len(saved_data) == len(calculated_results.results)
    
    def test_save_without_calculation(self):
        """Test saving without prior calculation"""
        calc = ASCICalculator(reaction='HER', verbose=False)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            with pytest.raises(ValueError):
                calc.save_results(f.name)


class TestScoringMethods:
    """Test different scoring methods"""
    
    @pytest.fixture
    def sample_data_file(self):
        """Create sample data file"""
        data = pd.DataFrame({
            'DFT_ads_E': [-0.27, -0.30, -0.24, -0.35, -0.20],
            'surface_energy': [0.5, 1.0, 0.8, 1.5, 0.6],
            'Cost': [50000, 100000, 25000, 150000, 10000],
            'symbol': ['Pt', 'Ru', 'Ni', 'Ir', 'Cu'],
            'AandB': ['Pt_111', 'Ru_0001', 'Ni_111', 'Ir_111', 'Cu_111'],
            'reaction_type': ['HER'] * 5,
            'optimal_energy': [-0.27] * 5,
            'activity_width': [0.15] * 5
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            return f.name
    
    def test_linear_scoring(self, sample_data_file):
        """Test linear activity scoring"""
        calc = ASCICalculator(reaction='HER', scoring_method='linear', verbose=False)
        calc.load_data(sample_data_file)
        results = calc.calculate_asci()
        
        assert calc.scoring_method == 'linear'
        assert 'activity_score' in results.columns
    
    def test_gaussian_scoring(self, sample_data_file):
        """Test Gaussian activity scoring"""
        calc = ASCICalculator(reaction='HER', scoring_method='gaussian', verbose=False)
        calc.load_data(sample_data_file)
        results = calc.calculate_asci()
        
        assert calc.scoring_method == 'gaussian'
        assert 'activity_score' in results.columns
    
    def test_scoring_method_comparison(self, sample_data_file):
        """Test that linear and Gaussian give different results"""
        calc_linear = ASCICalculator(reaction='HER', scoring_method='linear', verbose=False)
        calc_linear.load_data(sample_data_file)
        results_linear = calc_linear.calculate_asci()
        
        calc_gaussian = ASCICalculator(reaction='HER', scoring_method='gaussian', verbose=False)
        calc_gaussian.load_data(sample_data_file)
        results_gaussian = calc_gaussian.calculate_asci()
        
        # Activity scores should differ
        assert not np.allclose(
            results_linear['activity_score'].values,
            results_gaussian['activity_score'].values
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
