"""
ascicat/analyzer.py
Analysis tools for ASCI results
Statistical analysis, filtering, and comparison
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json

from .config import ReactionConfig
from .utils import get_pareto_front, calculate_correlation_matrix


class Analyzer():
    """Analyzer for ASCI results"""
    
    def __init__(self, results_df: pd.DataFrame, config: ReactionConfig):
        """
        Initialize analyzer
        
        Parameters
        ----------
        results_df : pd.DataFrame
            ASCI calculation results
        config : ReactionConfig
            Reaction configuration
        """
        self.df = results_df.copy()
        self.config = config
    
    def get_top_catalysts(self, n: int = 10, metric: str = 'ASCI') -> pd.DataFrame:
        """
        Get top N catalysts by specified metric
        
        Parameters
        ----------
        n : int
            Number of catalysts
        metric : str
            Metric to rank by ('ASCI', 'activity_score', etc.)
        
        Returns
        -------
        pd.DataFrame
            Top N catalysts
        """
        return self.df.nlargest(n, metric)
    
    def get_pareto_optimal(self) -> pd.DataFrame:
        """
        Get Pareto optimal catalysts
        
        Returns
        -------
        pd.DataFrame
            Pareto optimal catalysts
        """
        objectives = ['activity_score', 'stability_score', 'cost_score']
        maximize = [True, True, True]
        
        pareto_df = get_pareto_front(self.df, objectives, maximize)
        
        return pareto_df
    
    def filter_by_threshold(self, metric: str, threshold: float, 
                          greater_than: bool = True) -> pd.DataFrame:
        """
        Filter catalysts by threshold
        
        Parameters
        ----------
        metric : str
            Metric to filter by
        threshold : float
            Threshold value
        greater_than : bool
            If True, keep values > threshold
        
        Returns
        -------
        pd.DataFrame
            Filtered catalysts
        """
        if greater_than:
            return self.df[self.df[metric] > threshold].copy()
        else:
            return self.df[self.df[metric] < threshold].copy()
    
    def get_statistics_by_element(self) -> Dict:
        """
        Get statistics grouped by element
        
        Returns
        -------
        dict
            Statistics by element
        """
        stats = {}
        
        if 'Ametal' in self.df.columns:
            stats['primary_element'] = self.df.groupby('Ametal').agg({
                'ASCI': ['mean', 'std', 'max', 'count'],
                'activity_score': 'mean',
                'stability_score': 'mean',
                'cost_score': 'mean'
            }).to_dict()
        
        if 'Bmetal' in self.df.columns:
            stats['secondary_element'] = self.df.groupby('Bmetal').agg({
                'ASCI': ['mean', 'std', 'max', 'count'],
                'activity_score': 'mean',
                'stability_score': 'mean',
                'cost_score': 'mean'
            }).to_dict()
        
        return stats
    
    def get_correlation_analysis(self) -> pd.DataFrame:
        """
        Get correlation matrix of key metrics
        
        Returns
        -------
        pd.DataFrame
            Correlation matrix
        """
        metrics = ['ASCI', 'activity_score', 'stability_score', 'cost_score',
                  'DFT_ads_E', 'surface_energy', 'Cost']
        
        # Filter to existing columns
        metrics = [m for m in metrics if m in self.df.columns]
        
        return calculate_correlation_matrix(self.df, metrics)
    
    def compare_weight_scenarios(self, weight_scenarios: List[Tuple[float, float, float]]) -> pd.DataFrame:
        """
        Compare different weight scenarios
        
        Parameters
        ----------
        weight_scenarios : list of tuples
            List of (w_a, w_s, w_c) weight combinations
        
        Returns
        -------
        pd.DataFrame
            Comparison results
        """
        from .scoring import ScoringFunctions
        
        results = []
        
        for w_a, w_s, w_c in weight_scenarios:
            # Calculate new ASCI scores
            asci = ScoringFunctions.combined_asci_score(
                self.df['activity_score'],
                self.df['stability_score'],
                self.df['cost_score'],
                w_a, w_s, w_c
            )
            
            # Get top catalyst
            top_idx = asci.argmax()
            top_catalyst = self.df.iloc[top_idx]
            
            results.append({
                'weights': f"({w_a:.2f}, {w_s:.2f}, {w_c:.2f})",
                'w_activity': w_a,
                'w_stability': w_s,
                'w_cost': w_c,
                'mean_asci': asci.mean(),
                'max_asci': asci.max(),
                'top_catalyst': top_catalyst.get('symbol', 'N/A'),
                'top_asci': asci.max()
            })
        
        return pd.DataFrame(results)
    
    def export_summary(self, output_path: str) -> None:
        """
        Export comprehensive summary
        
        Parameters
        ----------
        output_path : str
            Output file path
        """
        summary = {
            'reaction': self.config.name,
            'pathway': self.config.pathway,
            'total_catalysts': len(self.df),
            'statistics': {
                'asci': {
                    'mean': float(self.df['ASCI'].mean()),
                    'std': float(self.df['ASCI'].std()),
                    'min': float(self.df['ASCI'].min()),
                    'max': float(self.df['ASCI'].max())
                }
            },
            'top_10_catalysts': self.get_top_catalysts(10)[['symbol', 'ASCI']].to_dict('records'),
            'pareto_optimal_count': len(self.get_pareto_optimal())
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Summary exported to: {output_path}")


def analyze_results(results_df: pd.DataFrame, config: ReactionConfig) -> Analyzer:
    """
    Convenience function to create analyzer
    
    Parameters
    ----------
    results_df : pd.DataFrame
        ASCI results
    config : ReactionConfig
        Reaction configuration
    
    Returns
    -------
    Analyzer
        Initialized analyzer
    """
    return Analyzer(results_df, config)
