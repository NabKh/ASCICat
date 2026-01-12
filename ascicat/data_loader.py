"""
ascicat/data_loader.py
Data loading and validation for ASCICat
Handles CSV files with robust error checking
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path
import warnings

from .config import ReactionConfig, ASCIConstants


class DataLoader:
    """Load and validate catalyst data for ASCI calculations"""
    
    def __init__(self, config: ReactionConfig, verbose: bool = True):
        """
        Initialize data loader
        
        Parameters
        ----------
        config : ReactionConfig
            Reaction configuration
        verbose : bool
            Print loading information
        """
        self.config = config
        self.verbose = verbose
        self.data = None
        self.original_size = 0
        self.filtered_size = 0
    
    def load(self, file_path: str, validate: bool = True) -> pd.DataFrame:
        """
        Load catalyst data from CSV file
        
        Parameters
        ----------
        file_path : str
            Path to CSV file
        validate : bool
            Perform validation checks
        
        Returns
        -------
        pd.DataFrame
            Loaded and validated data
        
        Raises
        ------
        FileNotFoundError
            If file doesn't exist
        ValueError
            If data validation fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Loading {self.config.name} Data")
            if self.config.pathway != 'default':
                print(f"Pathway: {self.config.pathway}")
            print('='*80)
            print(f"\nFile: {file_path}")
        
        # Load CSV
        try:
            data = pd.read_csv(file_path)
            self.original_size = len(data)
            if self.verbose:
                print(f"✓ Loaded {self.original_size:,} entries")
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {e}")
        
        # Validate if requested
        if validate:
            data = self._validate_data(data)
        
        self.data = data
        self.filtered_size = len(data)
        
        if self.verbose:
            retention = (self.filtered_size / self.original_size * 100) if self.original_size > 0 else 0
            print(f"\n✓ Final dataset: {self.filtered_size:,} entries ({retention:.2f}% retained)")
            print('='*80)
        
        return data
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data quality and structure
        
        Parameters
        ----------
        data : pd.DataFrame
            Raw data
        
        Returns
        -------
        pd.DataFrame
            Validated and cleaned data
        """
        if self.verbose:
            print(f"\n🔍 Validating data...")
        
        initial_size = len(data)
        
        # 1. Check required columns (only the essential ones)
        essential_cols = ['DFT_ads_E', 'surface_energy', 'Cost']
        missing_cols = [col for col in essential_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        # 2. Check data types
        self._check_data_types(data)
        
        # 3. Check for missing values in critical columns
        data = self._check_missing_values(data)
        
        # 4. Check value ranges
        data = self._check_value_ranges(data)
        
        # 5. Remove duplicates
        data = self._remove_duplicates(data)
        
        removed = initial_size - len(data)
        if removed > 0 and self.verbose:
            print(f"  ⚠️  Removed {removed:,} problematic entries during validation")
        
        return data
    
    def _check_data_types(self, data: pd.DataFrame) -> None:
        """Check data types are correct"""
        numeric_cols = ['DFT_ads_E', 'surface_energy', 'Cost']
        
        for col in numeric_cols:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    warnings.warn(f"Column '{col}' should be numeric, got {data[col].dtype}")
        
        if self.verbose:
            print(f"  ✓ Data types validated")
    
    def _check_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Check and handle missing values"""
        critical_cols = ['DFT_ads_E', 'surface_energy', 'Cost']
        
        missing_counts = {}
        for col in critical_cols:
            if col in data.columns:
                missing = data[col].isnull().sum()
                if missing > 0:
                    missing_counts[col] = missing
        
        if missing_counts:
            if self.verbose:
                print(f"  ⚠️  Missing values detected:")
                for col, count in missing_counts.items():
                    pct = (count / len(data)) * 100
                    print(f"      {col}: {count:,} ({pct:.2f}%)")
            
            # Remove rows with missing critical values
            data = data.dropna(subset=[col for col in critical_cols if col in data.columns])
            if self.verbose:
                print(f"  → Removed rows with missing values")
        else:
            if self.verbose:
                print(f"  ✓ No missing critical values")
        
        return data
    
    def _check_value_ranges(self, data: pd.DataFrame) -> pd.DataFrame:
        """Check if values are in physically reasonable ranges"""
        initial_size = len(data)
        issues = []
        
        # Check activity descriptor
        if 'DFT_ads_E' in data.columns:
            activity = data['DFT_ads_E']
            extreme_activity = ((activity < -5.0) | (activity > 5.0)).sum()
            if extreme_activity > 0:
                issues.append(f"activity outliers: {extreme_activity}")
                data = data[(activity >= -5.0) & (activity <= 5.0)]
        
        # Check surface energy (MUST be positive!)
        if 'surface_energy' in data.columns:
            gamma = data['surface_energy']
            negative_gamma = (gamma < 0).sum()
            if negative_gamma > 0:
                issues.append(f"negative surface energies: {negative_gamma}")
                data = data[gamma >= 0]
            
            very_high_gamma = (gamma > 15.0).sum()
            if very_high_gamma > 0:
                issues.append(f"very high surface energies: {very_high_gamma}")
                data = data[gamma <= 15.0]
        
        # Check cost (MUST be positive!)
        if 'Cost' in data.columns:
            cost = data['Cost']
            invalid_cost = (cost <= 0).sum()
            if invalid_cost > 0:
                issues.append(f"invalid costs: {invalid_cost}")
                data = data[cost > 0]
            
            extreme_cost = (cost > 1e6).sum()
            if extreme_cost > 0:
                issues.append(f"extreme costs: {extreme_cost}")
                data = data[cost <= 1e6]
        
        if issues and self.verbose:
            print(f"  ⚠️  Value range issues: {', '.join(issues)}")
        elif self.verbose:
            print(f"  ✓ All values in reasonable ranges")
        
        removed = initial_size - len(data)
        if removed > 0 and self.verbose:
            print(f"  → Removed {removed:,} entries with invalid values")
        
        return data
    
    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate entries"""
        critical_cols = ['DFT_ads_E', 'surface_energy', 'Cost']
        
        # Filter to existing columns
        critical_cols = [col for col in critical_cols if col in data.columns]
        
        if 'symbol' in data.columns:
            critical_cols.append('symbol')
        
        initial_size = len(data)
        data = data.drop_duplicates(subset=critical_cols, keep='first')
        removed = initial_size - len(data)
        
        if removed > 0 and self.verbose:
            print(f"  ⚠️  Removed {removed:,} duplicate entries")
        elif self.verbose:
            print(f"  ✓ No duplicates found")
        
        return data
    
    def get_statistics(self) -> dict:
        """
        Get data statistics
        
        Returns
        -------
        dict
            Statistics dictionary
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        stats = {
            'total_entries': len(self.data),
            'activity': {
                'min': float(self.data['DFT_ads_E'].min()),
                'max': float(self.data['DFT_ads_E'].max()),
                'mean': float(self.data['DFT_ads_E'].mean()),
                'std': float(self.data['DFT_ads_E'].std())
            },
            'stability': {
                'min': float(self.data['surface_energy'].min()),
                'max': float(self.data['surface_energy'].max()),
                'mean': float(self.data['surface_energy'].mean()),
                'std': float(self.data['surface_energy'].std())
            },
            'cost': {
                'min': float(self.data['Cost'].min()),
                'max': float(self.data['Cost'].max()),
                'mean': float(self.data['Cost'].mean()),
                'median': float(self.data['Cost'].median()),
                'std': float(self.data['Cost'].std())
            }
        }
        
        # Add composition info if available
        if 'Ametal' in self.data.columns:
            stats['unique_primary_elements'] = int(self.data['Ametal'].nunique())
        if 'Bmetal' in self.data.columns:
            stats['unique_secondary_elements'] = int(self.data['Bmetal'].nunique())
        
        return stats
    
    def print_summary(self) -> None:
        """Print data summary"""
        if self.data is None:
            print("No data loaded")
            return
        
        stats = self.get_statistics()
        
        print(f"\n{'='*80}")
        print(f"Data Summary - {self.config.name}")
        if self.config.pathway != 'default':
            print(f"Pathway: {self.config.pathway}")
        print('='*80)
        
        print(f"\nTotal entries: {stats['total_entries']:,}")
        
        print(f"\n📊 Descriptor Ranges:")
        print(f"  Activity (ΔE):")
        print(f"    Range: {stats['activity']['min']:.4f} to {stats['activity']['max']:.4f} eV")
        print(f"    Mean:  {stats['activity']['mean']:.4f} ± {stats['activity']['std']:.4f} eV")
        
        print(f"  Stability (γ):")
        print(f"    Range: {stats['stability']['min']:.4f} to {stats['stability']['max']:.4f} J/m²")
        print(f"    Mean:  {stats['stability']['mean']:.4f} ± {stats['stability']['std']:.4f} J/m²")
        
        print(f"  Cost:")
        print(f"    Range:  ${stats['cost']['min']:.2f} to ${stats['cost']['max']:.0f}/kg")
        print(f"    Mean:   ${stats['cost']['mean']:.2f} ± ${stats['cost']['std']:.2f}/kg")
        print(f"    Median: ${stats['cost']['median']:.2f}/kg")
        
        if 'unique_primary_elements' in stats:
            print(f"\n🧪 Composition:")
            print(f"  Unique primary elements: {stats['unique_primary_elements']}")
            if 'unique_secondary_elements' in stats:
                print(f"  Unique secondary elements: {stats['unique_secondary_elements']}")
        
        print('='*80)


def load_data(file_path: str, config: ReactionConfig, 
              validate: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    Convenience function to load data
    
    Parameters
    ----------
    file_path : str
        Path to CSV file
    config : ReactionConfig
        Reaction configuration
    validate : bool
        Perform validation
    verbose : bool
        Print information
    
    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    loader = DataLoader(config, verbose=verbose)
    data = loader.load(file_path, validate=validate)
    return data