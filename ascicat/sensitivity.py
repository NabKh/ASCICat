"""
ASCICat Enhanced Sensitivity Analysis Module
=============================================

Sensitivity analysis for multi-objective catalyst screening.

This module provides rigorous statistical methods for analyzing:
1. Weight sensitivity using ternary diagrams and Sobol indices
2. Ranking robustness with bootstrap confidence intervals
3. Statistical significance of ranking differences

Key Methods:
- Ternary diagram visualization (proper 3-weight representation)
- Bootstrap confidence intervals for rankings
- Variance-based sensitivity indices
- Spearman rank correlation analysis
- Friedman test for ranking significance

Author: Nabil Khossossi
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import Counter
from scipy import stats
from pathlib import Path
import warnings


class SensitivityAnalyzer:
    """
    Enhanced sensitivity analysis for ASCI weight parameters.

    Provides comprehensive analysis including:
    - Ternary diagrams for weight space visualization
    - Bootstrap confidence intervals
    - Variance-based sensitivity indices
    - Rank correlation analysis
    """

    def __init__(self, calculator, n_bootstrap: int = 100, random_state: int = 42):
        """
        Initialize sensitivity analyzer.

        Parameters
        ----------
        calculator : ASCICalculator
            Initialized calculator with loaded data
        n_bootstrap : int
            Number of bootstrap iterations for confidence intervals
        random_state : int
            Random seed for reproducibility
        """
        self.calculator = calculator
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.results_cache = {}

        np.random.seed(random_state)

    def generate_weight_grid(self, n_points: int = 21,
                             min_weight: float = 0.1,
                             max_weight: float = 0.8) -> np.ndarray:
        """
        Generate a regular grid of weight combinations on the simplex.

        Uses barycentric coordinates to ensure uniform coverage of
        the valid weight space (w_a + w_s + w_c = 1).

        Parameters
        ----------
        n_points : int
            Number of points per edge of the simplex
        min_weight : float
            Minimum weight for any objective
        max_weight : float
            Maximum weight for any objective

        Returns
        -------
        np.ndarray
            Array of shape (n, 3) with weight combinations [w_a, w_s, w_c]
        """
        weights = []
        step = 1.0 / (n_points - 1)

        for i in range(n_points):
            for j in range(n_points - i):
                k = n_points - 1 - i - j
                w_a = i * step
                w_s = j * step
                w_c = k * step

                # Ensure within bounds
                if (min_weight <= w_a <= max_weight and
                    min_weight <= w_s <= max_weight and
                    min_weight <= w_c <= max_weight):
                    weights.append([w_a, w_s, w_c])

        return np.array(weights)

    def generate_nature_weight_grid(self, n_target: int = 200,
                                    min_weight: float = 0.1,
                                    max_weight: float = 0.6) -> np.ndarray:
        """
        Generate weight grid for Comprehensive sensitivity analysis.

        Systematically varies weights across w_a, w_s, w_c in [min_weight, max_weight]
        with constraint w_i = 1, producing approximately n_target combinations.

        Parameters
        ----------
        n_target : int
            Target number of weight combinations (default: 200)
        min_weight : float
            Minimum weight for any objective (default: 0.1)
        max_weight : float
            Maximum weight for any objective (default: 0.6)

        Returns
        -------
        np.ndarray
            Array of shape (n, 3) with weight combinations [w_a, w_s, w_c]
        """
        best_weights = None
        best_diff = float('inf')

        for n_steps in range(10, 100):
            step = (max_weight - min_weight) / n_steps
            weights = []

            w_a = min_weight
            while w_a <= max_weight + 1e-9:
                w_s = min_weight
                while w_s <= max_weight + 1e-9:
                    w_c = 1.0 - w_a - w_s
                    if min_weight <= w_c <= max_weight:
                        weights.append([round(w_a, 4), round(w_s, 4), round(w_c, 4)])
                    w_s += step
                w_a += step

            n_weights = len(weights)
            diff = abs(n_weights - n_target)

            if diff < best_diff:
                best_diff = diff
                best_weights = np.array(weights)

            if n_weights > n_target * 1.5:
                break

        print(f"Generated {len(best_weights)} weight combinations "
              f"(target: {n_target}, w in [{min_weight}, {max_weight}])")

        return best_weights

    def generate_full_simplex_grid(self, n_points: int = 25) -> np.ndarray:
        """
        Generate a FULL simplex grid covering the entire weight space.

        This creates a uniform triangular grid for complete ternary visualization.
        Each weight can range from 0 to 1, with constraint w_i = 1.

        Parameters
        ----------
        n_points : int
            Number of points along each edge (total ~n_points^2/2 combinations)

        Returns
        -------
        np.ndarray
            Array of shape (n, 3) with weight combinations [w_a, w_s, w_c]
        """
        weights = []

        for i in range(n_points + 1):
            for j in range(n_points + 1 - i):
                k = n_points - i - j
                w_a = i / n_points
                w_s = j / n_points
                w_c = k / n_points
                weights.append([round(w_a, 4), round(w_s, 4), round(w_c, 4)])

        weights = np.array(weights)
        print(f"Generated {len(weights)} weight combinations (full simplex, n={n_points})")

        return weights

    def run_full_sensitivity(self, weights: np.ndarray,
                            track_top_n: int = 50,
                            verbose: bool = True) -> Dict:
        """
        Run comprehensive sensitivity analysis across all weight combinations.

        Parameters
        ----------
        weights : np.ndarray
            Weight combinations to test
        track_top_n : int
            Number of top catalysts to track per combination
        verbose : bool
            Print progress

        Returns
        -------
        dict
            Comprehensive sensitivity results including:
            - 'weight_results': DataFrame with per-weight metrics
            - 'rank_matrix': Catalyst ranks across all weights
            - 'asci_matrix': ASCI scores across all weights
        """
        n_weights = len(weights)
        n_catalysts = len(self.calculator.data)

        # Storage
        rank_matrix = np.zeros((n_catalysts, n_weights), dtype=int)
        asci_matrix = np.zeros((n_catalysts, n_weights))
        weight_results = []

        # Get catalyst symbols for indexing
        symbols = None

        if verbose:
            print(f"\nRunning sensitivity analysis: {n_weights} weight combinations")

        for i, (w_a, w_s, w_c) in enumerate(weights):
            if verbose and (i % 20 == 0 or i == n_weights - 1):
                print(f"  Progress: {i+1}/{n_weights} ({100*(i+1)/n_weights:.0f}%)")

            # Calculate ASCI
            result = self.calculator.calculate_asci(w_a=w_a, w_s=w_s, w_c=w_c, show_progress=False)
            result = result.reset_index(drop=True)

            # Store symbols on first iteration
            if symbols is None:
                symbols = result['symbol'].values

            # Store ranks and scores
            result['rank'] = range(1, len(result) + 1)

            for j, row in result.iterrows():
                rank_matrix[j, i] = row['rank']
                asci_matrix[j, i] = row['ASCI']

            # Per-weight summary
            top = result.head(track_top_n)
            weight_results.append({
                'w_activity': w_a,
                'w_stability': w_s,
                'w_cost': w_c,
                'best_catalyst': top.iloc[0]['symbol'],
                'best_asci': top.iloc[0]['ASCI'],
                'top10_mean_asci': result.head(10)['ASCI'].mean(),
                'asci_std': result['ASCI'].std(),
                'top10_symbols': list(top.head(10)['symbol']),
            })

        return {
            'weight_results': pd.DataFrame(weight_results),
            'rank_matrix': rank_matrix,
            'asci_matrix': asci_matrix,
            'symbols': symbols,
            'weights': weights,
        }

    def compute_rank_statistics(self, sensitivity_results: Dict) -> pd.DataFrame:
        """
        Compute comprehensive rank statistics for each catalyst.

        Parameters
        ----------
        sensitivity_results : dict
            Output from run_full_sensitivity()

        Returns
        -------
        pd.DataFrame
            Statistics including mean, std, CI, robustness score
        """
        rank_matrix = sensitivity_results['rank_matrix']
        asci_matrix = sensitivity_results['asci_matrix']
        symbols = sensitivity_results['symbols']
        n_weights = rank_matrix.shape[1]
        n_catalysts = len(symbols)

        stats_list = []

        for i, symbol in enumerate(symbols):
            ranks = rank_matrix[i, :]
            ascis = asci_matrix[i, :]

            # Basic statistics
            mean_rank = np.mean(ranks)
            std_rank = np.std(ranks)
            median_rank = np.median(ranks)
            min_rank = np.min(ranks)
            max_rank = np.max(ranks)

            # Bootstrap confidence interval for mean rank
            bootstrap_means = []
            for _ in range(self.n_bootstrap):
                sample = np.random.choice(ranks, size=len(ranks), replace=True)
                bootstrap_means.append(np.mean(sample))
            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)

            # Frequency metrics
            top5_freq = np.mean(ranks <= 5)
            top10_freq = np.mean(ranks <= 10)
            top20_freq = np.mean(ranks <= 20)
            top50_freq = np.mean(ranks <= 50)

            # Coefficient of variation (lower = more stable)
            cv = std_rank / mean_rank if mean_rank > 0 else float('inf')

            # Interquartile range
            iqr = np.percentile(ranks, 75) - np.percentile(ranks, 25)

            # Robustness score (composite metric)
            robustness = self._compute_robustness_score(
                mean_rank, std_rank, top10_freq, top20_freq, n_catalysts
            )

            # ASCI statistics
            mean_asci = np.mean(ascis)
            std_asci = np.std(ascis)

            stats_list.append({
                'symbol': symbol,
                'mean_rank': mean_rank,
                'std_rank': std_rank,
                'median_rank': median_rank,
                'min_rank': min_rank,
                'max_rank': max_rank,
                'rank_range': max_rank - min_rank,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_width': ci_upper - ci_lower,
                'top5_frequency': top5_freq,
                'top10_frequency': top10_freq,
                'top20_frequency': top20_freq,
                'top50_frequency': top50_freq,
                'cv': cv,
                'iqr': iqr,
                'robustness_score': robustness,
                'mean_asci': mean_asci,
                'std_asci': std_asci,
            })

        df = pd.DataFrame(stats_list)
        df = df.sort_values('robustness_score', ascending=False).reset_index(drop=True)

        return df

    def _compute_robustness_score(self, mean_rank: float, std_rank: float,
                                  top10_freq: float, top20_freq: float,
                                  n_catalysts: int) -> float:
        """
        Compute composite robustness score.

        Combines multiple metrics into a single robustness indicator.
        Higher score = more robust performer.
        """
        # Normalize mean rank (0 = worst, 1 = best)
        rank_score = 1 - (mean_rank / n_catalysts)

        # Stability score based on CV (lower CV = higher score)
        cv = std_rank / mean_rank if mean_rank > 0 else 1.0
        stability_score = max(0, 1 - cv)

        # Weighted combination
        robustness = (
            0.35 * top10_freq +
            0.25 * top20_freq +
            0.25 * stability_score +
            0.15 * rank_score
        )

        return np.clip(robustness, 0, 1)

    def compute_sensitivity_indices(self, sensitivity_results: Dict) -> Dict:
        """
        Compute variance-based sensitivity indices for each weight.

        Quantifies how much each weight contributes to ranking variance.
        Similar to first-order Sobol indices.

        Parameters
        ----------
        sensitivity_results : dict
            Output from run_full_sensitivity()

        Returns
        -------
        dict
            Sensitivity indices for each weight
        """
        weights = sensitivity_results['weights']
        rank_matrix = sensitivity_results['rank_matrix']

        # Overall variance
        total_var = np.var(rank_matrix, axis=1).mean()

        if total_var == 0:
            return {'S_activity': 0.33, 'S_stability': 0.33, 'S_cost': 0.33}

        # Conditional variances for each weight
        indices = {}
        weight_names = ['activity', 'stability', 'cost']

        for w_idx, w_name in enumerate(weight_names):
            # Group by this weight value
            unique_weights = np.unique(np.round(weights[:, w_idx], 2))

            conditional_means = []
            for w_val in unique_weights:
                mask = np.abs(weights[:, w_idx] - w_val) < 0.05
                if mask.sum() > 0:
                    conditional_means.append(
                        np.mean(rank_matrix[:, mask], axis=1)
                    )

            if len(conditional_means) > 1:
                # Variance of conditional means
                var_cond_mean = np.var(np.array(conditional_means), axis=0).mean()
                indices[f'S_{w_name}'] = var_cond_mean / total_var
            else:
                indices[f'S_{w_name}'] = 0.0

        # Normalize to sum to ~1
        total = sum(indices.values())
        if total > 0:
            indices = {k: v/total for k, v in indices.items()}

        return indices

    def compute_rank_correlations(self, sensitivity_results: Dict,
                                  n_pairs: int = 50) -> pd.DataFrame:
        """
        Compute Spearman rank correlations between weight scenarios.

        Parameters
        ----------
        sensitivity_results : dict
            Output from run_full_sensitivity()
        n_pairs : int
            Number of random pairs to sample

        Returns
        -------
        pd.DataFrame
            Pairwise correlations with weight differences
        """
        rank_matrix = sensitivity_results['rank_matrix']
        weights = sensitivity_results['weights']
        n_weights = len(weights)

        correlations = []

        # Sample random pairs
        np.random.seed(self.random_state)
        pairs = []
        for _ in range(n_pairs):
            i, j = np.random.choice(n_weights, 2, replace=False)
            pairs.append((i, j))

        for i, j in pairs:
            ranks_i = rank_matrix[:, i]
            ranks_j = rank_matrix[:, j]

            # Spearman correlation
            rho, p_value = stats.spearmanr(ranks_i, ranks_j)

            # Weight distance
            weight_dist = np.linalg.norm(weights[i] - weights[j])

            correlations.append({
                'pair': f'{i}-{j}',
                'w1_activity': weights[i, 0],
                'w1_stability': weights[i, 1],
                'w1_cost': weights[i, 2],
                'w2_activity': weights[j, 0],
                'w2_stability': weights[j, 1],
                'w2_cost': weights[j, 2],
                'spearman_rho': rho,
                'p_value': p_value,
                'weight_distance': weight_dist,
            })

        return pd.DataFrame(correlations)

    def statistical_tests(self, sensitivity_results: Dict,
                         top_n: int = 20) -> Dict:
        """
        Perform statistical significance tests on rankings.

        Parameters
        ----------
        sensitivity_results : dict
            Output from run_full_sensitivity()
        top_n : int
            Number of top catalysts to test

        Returns
        -------
        dict
            Test results including Friedman test
        """
        rank_matrix = sensitivity_results['rank_matrix']
        symbols = sensitivity_results['symbols']

        # Get indices of top catalysts (by mean rank)
        mean_ranks = np.mean(rank_matrix, axis=1)
        top_indices = np.argsort(mean_ranks)[:top_n]

        # Prepare data for Friedman test
        # Transpose: rows = weight combinations, cols = catalysts
        rank_data = rank_matrix[top_indices, :].T

        # Friedman test (non-parametric test for ranking differences)
        try:
            stat, p_value = stats.friedmanchisquare(*[rank_data[:, i] for i in range(top_n)])
            friedman_result = {
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
            }
        except Exception as e:
            friedman_result = {
                'statistic': None,
                'p_value': None,
                'significant': None,
                'error': str(e)
            }

        # Kendall's W (coefficient of concordance)
        # Measures agreement across weight combinations
        n_weights = rank_matrix.shape[1]
        n_cats = top_n

        rank_sums = np.sum(rank_data, axis=0)
        mean_rank_sum = np.mean(rank_sums)
        ss = np.sum((rank_sums - mean_rank_sum) ** 2)

        kendall_w = 12 * ss / (n_weights**2 * (n_cats**3 - n_cats))

        return {
            'friedman_test': friedman_result,
            'kendall_w': kendall_w,
            'interpretation': self._interpret_kendall_w(kendall_w),
        }

    def _interpret_kendall_w(self, w: float) -> str:
        """Interpret Kendall's W coefficient."""
        if w < 0.1:
            return "Very low agreement - rankings highly weight-dependent"
        elif w < 0.3:
            return "Low agreement - moderate weight sensitivity"
        elif w < 0.5:
            return "Moderate agreement - some ranking stability"
        elif w < 0.7:
            return "Good agreement - reasonably robust rankings"
        else:
            return "Strong agreement - highly robust rankings"


class SensitivityVisualizer:
    """
    High-resolution visualizations for sensitivity analysis.

    Creates Comprehensive figures including:
    - Ternary diagrams (requires matplotlib-ternary or plotly)
    - Rank trajectory plots
    - Bootstrap confidence interval plots
    - Sensitivity index charts
    """

    def __init__(self, output_dir: str = 'results/sensitivity'):
        """
        Initialize visualizer.

        Parameters
        ----------
        output_dir : str
            Directory for saving figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Figure settings
        self.dpi = 600
        self.figsize_single = (6, 5)
        self.figsize_double = (12, 5)
        self.figsize_large = (10, 8)

        # Colors
        self.colors = {
            'activity': '#2CA02C',
            'stability': '#1F77B4',
            'cost': '#FF7F0E',
            'highlight': '#D62728',
            'neutral': '#7F7F7F',
        }

    def plot_ternary_heatmap(self, sensitivity_results: Dict,
                             metric: str = 'best_asci',
                             title: str = None,
                             save_name: str = 'ternary_sensitivity') -> None:
        """
        Create High-resolution ternary diagram showing metric across weight space.

        This is the proper visualization for 3-weight sensitivity,
        showing the full simplex with complete coverage.

        Parameters
        ----------
        sensitivity_results : dict
            Output from run_full_sensitivity()
        metric : str
            Column from weight_results to visualize
        title : str
            Figure title
        save_name : str
            Output filename (without extension)
        """
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri
        from matplotlib.colors import LinearSegmentedColormap

        weight_df = sensitivity_results['weight_results']
        weights = sensitivity_results['weights']

        # Get metric values
        if metric in weight_df.columns:
            values = weight_df[metric].values
        else:
            raise ValueError(f"Metric '{metric}' not found in results")

        # Create figure with High-resolution settings
        fig, ax = plt.subplots(figsize=(10, 9))

        # Transform to 2D ternary coordinates
        # Using standard ternary projection: w_a at right, w_s at top, w_c at left
        x = weights[:, 0] + 0.5 * weights[:, 1]
        y = (np.sqrt(3) / 2) * weights[:, 1]

        # Create triangulation with improved refinement
        triang = tri.Triangulation(x, y)

        # Use more levels for smoother gradients
        n_levels = 30
        levels = np.linspace(values.min(), values.max(), n_levels)

        # Plot filled contours with high-quality colormap
        contourf = ax.tricontourf(triang, values, levels=levels, cmap='plasma', extend='both')

        # Add subtle contour lines for readability
        contour = ax.tricontour(triang, values, levels=levels[::3],
                                colors='white', linewidths=0.3, alpha=0.6)

        # Draw elegant triangle border
        triangle = plt.Polygon(
            [[0, 0], [1, 0], [0.5, np.sqrt(3)/2]],
            fill=False, edgecolor='black', linewidth=2.5
        )
        ax.add_patch(triangle)

        # Add grid lines inside triangle for reference
        for w in [0.2, 0.4, 0.6, 0.8]:
            # Horizontal lines (constant w_s)
            x1 = 0.5 * w
            y1 = (np.sqrt(3)/2) * w
            x2 = 1 - 0.5 * w
            y2 = y1
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=0.3, alpha=0.3)

            # Lines parallel to left edge (constant w_a)
            x1 = w
            y1 = 0
            x2 = 0.5 + 0.5 * w
            y2 = (np.sqrt(3)/2) * (1 - w)
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=0.3, alpha=0.3)

            # Lines parallel to right edge (constant w_c)
            x1 = 1 - w
            y1 = 0
            x2 = 0.5 - 0.5 * w
            y2 = (np.sqrt(3)/2) * (1 - w)
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=0.3, alpha=0.3)

        # Corner labels with icons
        ax.text(0, -0.08, 'Cost\n(wc = 1)', ha='center', va='top',
                fontweight='bold', fontsize=12, color='#FF7F0E')
        ax.text(1, -0.08, 'Activity\n(wa = 1)', ha='center', va='top',
                fontweight='bold', fontsize=12, color='#2CA02C')
        ax.text(0.5, np.sqrt(3)/2 + 0.06, 'Stability\n(ws = 1)',
                ha='center', va='bottom', fontweight='bold', fontsize=12, color='#1F77B4')

        # Edge tick labels
        for w in [0.2, 0.4, 0.6, 0.8]:
            # Bottom edge (w_s = 0): wa varies
            ax.text(w, -0.03, f'{w:.1f}', ha='center', va='top', fontsize=9, color='gray')
            # Left edge (w_a = 0): ws varies
            x_left = 0.5 * w
            y_left = (np.sqrt(3)/2) * w
            ax.text(x_left - 0.04, y_left, f'{w:.1f}', ha='right', va='center',
                    fontsize=9, color='gray')
            # Right edge (w_c = 0): ws varies
            x_right = 1 - 0.5 * w
            y_right = (np.sqrt(3)/2) * w
            ax.text(x_right + 0.04, y_right, f'{w:.1f}', ha='left', va='center',
                    fontsize=9, color='gray')

        # Mark optimal region (highest values)
        opt_idx = np.argmax(values)
        opt_x = x[opt_idx]
        opt_y = y[opt_idx]
        ax.scatter([opt_x], [opt_y], marker='*', s=400, c='white',
                  edgecolors='black', linewidth=2, zorder=10)
        ax.annotate(f'Optimal\n({values[opt_idx]:.3f})',
                   (opt_x, opt_y), xytext=(15, 15), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                   arrowprops=dict(arrowstyle='->', color='black'))

        # Colorbar with better formatting
        cbar = plt.colorbar(contourf, ax=ax, shrink=0.75, aspect=25, pad=0.02)
        cbar.set_label(self._format_metric_name(metric), fontweight='bold', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        # Add statistics box
        stats_text = f'n = {len(values)} weight combinations\n'
        stats_text += f'Range: [{values.min():.3f}, {values.max():.3f}]\n'
        stats_text += f'Mean: {values.mean():.3f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)

        # Title
        if title is None:
            title = f'Weight Space Sensitivity Analysis\n{self._format_metric_name(metric)}'
        ax.set_title(title, fontweight='bold', fontsize=14, pad=20)

        # Clean up axes
        ax.set_xlim(-0.12, 1.12)
        ax.set_ylim(-0.15, np.sqrt(3)/2 + 0.15)
        ax.set_aspect('equal')
        ax.axis('off')

        plt.tight_layout()

        # Save
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.dpi, bbox_inches='tight')
        fig.savefig(self.output_dir / f'{save_name}.pdf', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_name}.png/pdf")

    def plot_rank_confidence_intervals(self, rank_stats: pd.DataFrame,
                                       n_show: int = 20,
                                       save_name: str = 'rank_confidence_intervals') -> None:
        """
        Plot rank distributions with bootstrap confidence intervals.

        Parameters
        ----------
        rank_stats : pd.DataFrame
            Output from compute_rank_statistics()
        n_show : int
            Number of top catalysts to show
        save_name : str
            Output filename
        """
        import matplotlib.pyplot as plt
        from .utils import generate_unique_labels

        fig, ax = plt.subplots(figsize=(12, 6))

        # Get top catalysts with unique labels
        top = rank_stats.head(n_show).copy()
        top = generate_unique_labels(top, label_col='display_label')

        positions = np.arange(len(top))

        # Plot confidence intervals as error bars
        for i, (_, row) in enumerate(top.iterrows()):
            # CI error bar
            ci_lower_err = row['mean_rank'] - row['ci_lower']
            ci_upper_err = row['ci_upper'] - row['mean_rank']

            # Full range (min-max) as lighter bar
            ax.plot([i, i], [row['min_rank'], row['max_rank']],
                   color='lightgray', linewidth=8, solid_capstyle='round', zorder=1)

            # IQR as medium bar
            q25 = row['mean_rank'] - row['iqr']/2
            q75 = row['mean_rank'] + row['iqr']/2
            ax.plot([i, i], [q25, q75],
                   color='steelblue', linewidth=4, solid_capstyle='round', zorder=2)

            # Mean with 95% CI
            ax.errorbar(i, row['mean_rank'],
                       yerr=[[ci_lower_err], [ci_upper_err]],
                       fmt='o', color='darkblue', markersize=8,
                       capsize=4, capthick=2, linewidth=2, zorder=3)

            # Median marker
            ax.scatter(i, row['median_rank'], marker='|', s=150,
                      color='red', linewidth=2, zorder=4)

        # Labels
        ax.set_xticks(positions)
        ax.set_xticklabels(top['display_label'], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Rank (across weight combinations)', fontweight='bold', fontsize=11)
        ax.set_xlabel('Catalyst', fontweight='bold', fontsize=11)

        # Invert y-axis (lower rank = better)
        ax.invert_yaxis()

        # Grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        # Legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgray', label='Min-Max Range'),
            Patch(facecolor='steelblue', label='Interquartile Range'),
            Line2D([0], [0], marker='o', color='darkblue', label='Mean (95% CI)',
                   markersize=8, linestyle='none'),
            Line2D([0], [0], marker='|', color='red', label='Median',
                   markersize=12, linestyle='none', linewidth=2),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

        ax.set_title('Ranking Robustness with Bootstrap Confidence Intervals',
                    fontweight='bold', fontsize=12)

        plt.tight_layout()

        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.dpi, bbox_inches='tight')
        fig.savefig(self.output_dir / f'{save_name}.pdf', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_name}.png/pdf")

    def plot_sensitivity_indices(self, indices: Dict,
                                 save_name: str = 'sensitivity_indices') -> None:
        """
        Plot variance-based sensitivity indices.

        Parameters
        ----------
        indices : dict
            Output from compute_sensitivity_indices()
        save_name : str
            Output filename
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))

        names = ['Activity\n(w_a)', 'Stability\n(w_s)', 'Cost\n(w_c)']
        values = [indices['S_activity'], indices['S_stability'], indices['S_cost']]
        colors = [self.colors['activity'], self.colors['stability'], self.colors['cost']]

        bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

        ax.set_ylabel('Sensitivity Index (S)', fontweight='bold', fontsize=11)
        ax.set_title('Weight Sensitivity Indices\n(Contribution to Ranking Variance)',
                    fontweight='bold', fontsize=12)

        ax.set_ylim(0, max(values) * 1.2)
        ax.axhline(y=0.33, color='gray', linestyle='--', alpha=0.5, label='Equal sensitivity')

        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

        plt.tight_layout()

        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.dpi, bbox_inches='tight')
        fig.savefig(self.output_dir / f'{save_name}.pdf', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_name}.png/pdf")

    def plot_correlation_vs_distance(self, correlations: pd.DataFrame,
                                     save_name: str = 'correlation_vs_distance') -> None:
        """
        Plot Spearman correlation vs weight distance.

        Shows how ranking correlation decreases with weight difference.

        Parameters
        ----------
        correlations : pd.DataFrame
            Output from compute_rank_correlations()
        save_name : str
            Output filename
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.scatter(correlations['weight_distance'], correlations['spearman_rho'],
                  alpha=0.6, s=50, c='steelblue', edgecolors='black', linewidth=0.5)

        # Fit trend line
        z = np.polyfit(correlations['weight_distance'], correlations['spearman_rho'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, correlations['weight_distance'].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.2f})')

        ax.set_xlabel('Weight Vector Distance', fontweight='bold', fontsize=11)
        ax.set_ylabel('Spearman Rank Correlation (rho)', fontweight='bold', fontsize=11)
        ax.set_title('Ranking Stability vs Weight Difference', fontweight='bold', fontsize=12)

        ax.axhline(y=0.9, color='green', linestyle=':', alpha=0.7, label='rho = 0.9 (high)')
        ax.axhline(y=0.7, color='orange', linestyle=':', alpha=0.7, label='rho = 0.7 (moderate)')

        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='lower left', fontsize=9)

        plt.tight_layout()

        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.dpi, bbox_inches='tight')
        fig.savefig(self.output_dir / f'{save_name}.pdf', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_name}.png/pdf")

    def plot_robustness_quadrant(self, rank_stats: pd.DataFrame,
                                 n_label: int = 15,
                                 save_name: str = 'robustness_quadrant') -> None:
        """
        Create performance vs robustness quadrant plot.

        Identifies catalysts that are both high-performing AND robust.

        Parameters
        ----------
        rank_stats : pd.DataFrame
            Output from compute_rank_statistics()
        n_label : int
            Number of top catalysts to label
        save_name : str
            Output filename
        """
        import matplotlib.pyplot as plt
        from .utils import generate_unique_labels

        fig, ax = plt.subplots(figsize=(9, 7))

        # Filter to relevant catalysts
        df = rank_stats[rank_stats['mean_rank'] <= 100].copy()
        df = generate_unique_labels(df, label_col='display_label')

        # Scatter plot
        scatter = ax.scatter(
            df['mean_rank'],
            df['robustness_score'],
            c=df['top10_frequency'],
            s=100,
            alpha=0.7,
            cmap='RdYlGn',
            edgecolors='black',
            linewidths=0.5
        )

        # Label top catalysts
        top_label = df.head(n_label)
        for _, row in top_label.iterrows():
            ax.annotate(
                row['display_label'],
                (row['mean_rank'], row['robustness_score']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.9
            )

        # Quadrant lines
        median_rank = df['mean_rank'].median()
        median_robust = df['robustness_score'].median()

        ax.axvline(x=median_rank, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=median_robust, color='gray', linestyle='--', alpha=0.5)

        # Quadrant labels
        ax.text(0.02, 0.98, 'HIGH PERFORMANCE\nHIGH ROBUSTNESS\n(Ideal)',
                transform=ax.transAxes, fontsize=9, fontweight='bold',
                color='darkgreen', va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        ax.text(0.98, 0.02, 'LOW PERFORMANCE\nLOW ROBUSTNESS\n(Avoid)',
                transform=ax.transAxes, fontsize=9, fontweight='bold',
                color='darkred', va='bottom', ha='right',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Top 10 Frequency', fontweight='bold')

        ax.set_xlabel('Mean Rank (lower = better performance)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Robustness Score (higher = more stable)', fontweight='bold', fontsize=11)
        ax.set_title('Performance vs Robustness Trade-off Analysis', fontweight='bold', fontsize=12)

        ax.invert_xaxis()  # Lower rank to the right
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()

        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.dpi, bbox_inches='tight')
        fig.savefig(self.output_dir / f'{save_name}.pdf', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_name}.png/pdf")

    def plot_comprehensive_summary(self, sensitivity_results: Dict,
                                   rank_stats: pd.DataFrame,
                                   indices: Dict,
                                   save_name: str = 'sensitivity_summary') -> None:
        """
        Create High-resolution comprehensive 4-panel sensitivity analysis figure.

        Parameters
        ----------
        sensitivity_results : dict
            Output from run_full_sensitivity()
        rank_stats : pd.DataFrame
            Output from compute_rank_statistics()
        indices : dict
            Output from compute_sensitivity_indices()
        save_name : str
            Output filename
        """
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri
        from .utils import generate_unique_labels

        fig = plt.figure(figsize=(16, 14))

        # Panel A: Ternary diagram (larger, more prominent)
        ax1 = fig.add_subplot(221)

        weights = sensitivity_results['weights']
        weight_df = sensitivity_results['weight_results']
        values = weight_df['best_asci'].values

        x = weights[:, 0] + 0.5 * weights[:, 1]
        y = (np.sqrt(3) / 2) * weights[:, 1]

        triang = tri.Triangulation(x, y)
        levels = np.linspace(values.min(), values.max(), 25)
        contourf = ax1.tricontourf(triang, values, levels=levels, cmap='plasma')

        # Add grid lines
        for w in [0.2, 0.4, 0.6, 0.8]:
            x1 = 0.5 * w
            y1 = (np.sqrt(3)/2) * w
            x2 = 1 - 0.5 * w
            ax1.plot([x1, x2], [y1, y1], 'w-', linewidth=0.3, alpha=0.4)

        triangle = plt.Polygon([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]],
                              fill=False, edgecolor='black', linewidth=2.5)
        ax1.add_patch(triangle)

        # Mark optimal point
        opt_idx = np.argmax(values)
        ax1.scatter([x[opt_idx]], [y[opt_idx]], marker='*', s=300, c='white',
                   edgecolors='black', linewidth=2, zorder=10)

        ax1.text(0, -0.06, 'Cost', ha='center', va='top', fontweight='bold', fontsize=11, color='#FF7F0E')
        ax1.text(1, -0.06, 'Activity', ha='center', va='top', fontweight='bold', fontsize=11, color='#2CA02C')
        ax1.text(0.5, np.sqrt(3)/2 + 0.04, 'Stability', ha='center', va='bottom', fontweight='bold', fontsize=11, color='#1F77B4')

        cbar1 = plt.colorbar(contourf, ax=ax1, shrink=0.75, aspect=20)
        cbar1.set_label('Best ASCI Score', fontweight='bold', fontsize=10)
        ax1.set_xlim(-0.08, 1.08)
        ax1.set_ylim(-0.12, np.sqrt(3)/2 + 0.1)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title('(A) Weight Space Sensitivity\n(Full Simplex Coverage)', fontweight='bold', fontsize=12)

        # Panel B: Sensitivity indices (enhanced)
        ax2 = fig.add_subplot(222)

        names = ['Activity\n(wa)', 'Stability\n(ws)', 'Cost\n(wc)']
        vals = [indices['S_activity'], indices['S_stability'], indices['S_cost']]
        colors = [self.colors['activity'], self.colors['stability'], self.colors['cost']]

        bars = ax2.bar(names, vals, color=colors, edgecolor='black', linewidth=2, width=0.6)
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

        ax2.axhline(y=0.33, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, label='Equal sensitivity')
        ax2.set_ylabel('Sensitivity Index (S)', fontweight='bold', fontsize=11)
        ax2.set_ylim(0, max(vals) * 1.3)
        ax2.set_title('(B) Weight Sensitivity Indices\n(Contribution to Ranking Variance)', fontweight='bold', fontsize=12)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax2.legend(loc='upper right', fontsize=9)

        # Panel C: Rank confidence intervals (top 12)
        ax3 = fig.add_subplot(223)

        top12 = rank_stats.head(12).copy()
        top12 = generate_unique_labels(top12, label_col='display_label')

        for i, (_, row) in enumerate(top12.iterrows()):
            # Full range
            ax3.plot([i, i], [row['min_rank'], row['max_rank']],
                    color='#E0E0E0', linewidth=10, solid_capstyle='round', zorder=1)
            # IQR
            q25 = row['mean_rank'] - row['iqr']/2
            q75 = row['mean_rank'] + row['iqr']/2
            ax3.plot([i, i], [q25, q75],
                    color='steelblue', linewidth=5, solid_capstyle='round', zorder=2)
            # Mean with CI
            ci_lower_err = row['mean_rank'] - row['ci_lower']
            ci_upper_err = row['ci_upper'] - row['mean_rank']
            ax3.errorbar(i, row['mean_rank'],
                        yerr=[[ci_lower_err], [ci_upper_err]],
                        fmt='o', color='#1a237e', markersize=10,
                        capsize=4, capthick=2, linewidth=2, zorder=3)
            # Median
            ax3.scatter(i, row['median_rank'], marker='|', s=200,
                       color='red', linewidth=2.5, zorder=4)

        ax3.set_xticks(range(len(top12)))
        ax3.set_xticklabels(top12['display_label'], rotation=45, ha='right', fontsize=10)
        ax3.set_ylabel('Rank (across weight combinations)', fontweight='bold', fontsize=11)
        ax3.invert_yaxis()
        ax3.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax3.set_title('(C) Ranking Robustness with Bootstrap 95% CI', fontweight='bold', fontsize=12)

        # Legend for panel C
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#E0E0E0', label='Min-Max Range'),
            Patch(facecolor='steelblue', label='IQR'),
            Line2D([0], [0], marker='o', color='#1a237e', label='Mean (95% CI)',
                   markersize=8, linestyle='none'),
            Line2D([0], [0], marker='|', color='red', label='Median',
                   markersize=10, linestyle='none', linewidth=2),
        ]
        ax3.legend(handles=legend_elements, loc='lower right', fontsize=8, ncol=2)

        # Panel D: Robustness scores (enhanced)
        ax4 = fig.add_subplot(224)

        top15 = rank_stats.head(15).copy()
        top15 = generate_unique_labels(top15, label_col='display_label')

        # Color by robustness score
        colors_robust = plt.cm.RdYlGn(top15['robustness_score'].values)
        bars = ax4.barh(range(len(top15)), top15['robustness_score'], color=colors_robust,
                       edgecolor='black', linewidth=0.8, height=0.7)

        # Add value labels
        for i, (_, row) in enumerate(top15.iterrows()):
            ax4.text(row['robustness_score'] + 0.02, i, f"{row['robustness_score']:.3f}",
                    va='center', ha='left', fontsize=9, fontweight='bold')

        ax4.set_yticks(range(len(top15)))
        ax4.set_yticklabels(top15['display_label'], fontsize=10)
        ax4.set_xlabel('Robustness Score (composite metric)', fontweight='bold', fontsize=11)
        ax4.invert_yaxis()
        ax4.set_xlim(0, 1.15)
        ax4.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold')
        ax4.grid(True, axis='x', linestyle='--', alpha=0.3)
        ax4.set_title('(D) Catalyst Robustness Ranking', fontweight='bold', fontsize=12)
        ax4.legend(loc='lower right', fontsize=9)

        plt.tight_layout()
        fig.suptitle('Comprehensive Sensitivity Analysis', fontweight='bold', fontsize=14, y=1.01)

        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.dpi, bbox_inches='tight')
        fig.savefig(self.output_dir / f'{save_name}.pdf', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_name}.png/pdf")

    def _format_metric_name(self, metric: str) -> str:
        """Format metric name for display."""
        replacements = {
            'best_asci': 'Best ASCI Score',
            'top10_mean_asci': 'Mean ASCI (Top 10)',
            'asci_std': 'ASCI Standard Deviation',
        }
        return replacements.get(metric, metric.replace('_', ' ').title())


def run_enhanced_sensitivity_analysis(calculator, output_dir: str = None,
                                      n_weight_points: int = 15,
                                      n_bootstrap: int = 100,
                                      verbose: bool = True) -> Dict:
    """
    Run complete enhanced sensitivity analysis.

    Convenience function to run full analysis pipeline.

    Parameters
    ----------
    calculator : ASCICalculator
        Initialized calculator with loaded data
    output_dir : str
        Output directory for figures
    n_weight_points : int
        Number of weight points per simplex edge
    n_bootstrap : int
        Bootstrap iterations for confidence intervals
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Complete analysis results
    """
    if output_dir is None:
        output_dir = 'results/enhanced_sensitivity'

    # Initialize analyzer
    analyzer = SensitivityAnalyzer(calculator, n_bootstrap=n_bootstrap)
    visualizer = SensitivityVisualizer(output_dir=output_dir)

    if verbose:
        print("=" * 70)
        print("Enhanced Sensitivity Analysis")
        print("=" * 70)

    # Generate weight grid
    weights = analyzer.generate_weight_grid(n_points=n_weight_points)
    if verbose:
        print(f"\nWeight combinations: {len(weights)}")

    # Run sensitivity analysis
    sensitivity_results = analyzer.run_full_sensitivity(weights, verbose=verbose)

    # Compute statistics
    if verbose:
        print("\nComputing rank statistics with bootstrap CIs...")
    rank_stats = analyzer.compute_rank_statistics(sensitivity_results)

    # Compute sensitivity indices
    if verbose:
        print("Computing sensitivity indices...")
    indices = analyzer.compute_sensitivity_indices(sensitivity_results)

    # Compute rank correlations
    if verbose:
        print("Computing rank correlations...")
    correlations = analyzer.compute_rank_correlations(sensitivity_results)

    # Statistical tests
    if verbose:
        print("Running statistical tests...")
    tests = analyzer.statistical_tests(sensitivity_results)

    # Generate figures
    if verbose:
        print("\n" + "=" * 70)
        print("Generating High-Resolution Figures")
        print("=" * 70)

    visualizer.plot_ternary_heatmap(sensitivity_results, metric='best_asci')
    visualizer.plot_rank_confidence_intervals(rank_stats)
    visualizer.plot_sensitivity_indices(indices)
    visualizer.plot_correlation_vs_distance(correlations)
    visualizer.plot_robustness_quadrant(rank_stats)
    visualizer.plot_comprehensive_summary(sensitivity_results, rank_stats, indices)

    # Save data
    rank_stats.to_csv(Path(output_dir) / 'rank_statistics.csv', index=False)
    sensitivity_results['weight_results'].to_csv(
        Path(output_dir) / 'weight_sensitivity_data.csv', index=False
    )

    if verbose:
        print("\n" + "=" * 70)
        print("SENSITIVITY ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"\nSensitivity Indices:")
        print(f"  Activity weight (S_a):  {indices['S_activity']:.3f}")
        print(f"  Stability weight (S_s): {indices['S_stability']:.3f}")
        print(f"  Cost weight (S_c):      {indices['S_cost']:.3f}")

        print(f"\nKendall's W (ranking concordance): {tests['kendall_w']:.3f}")
        print(f"  Interpretation: {tests['interpretation']}")

        print(f"\nMost Robust Catalysts:")
        for i, row in rank_stats.head(5).iterrows():
            print(f"  {i+1}. {row['symbol']}: robustness={row['robustness_score']:.3f}, "
                  f"mean_rank={row['mean_rank']:.1f}")

    return {
        'sensitivity_results': sensitivity_results,
        'rank_stats': rank_stats,
        'indices': indices,
        'correlations': correlations,
        'tests': tests,
    }
