"""
ASCICat Visualizer - Figure Generation
======================================

Generates figures for multi-objective catalyst screening.

Four key visualization panels:
- Panel A: 3D ASCI Component Space
- Panel B: ASCI Rank vs Adsorption Energy
- Panel C: Volcano Optimization Landscape
- Panel D: Top Performers Breakdown

For large datasets (>1000 catalysts), automatic stratified sampling
is applied to ensure clear figures.

Author: Nabil Khossossi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
from typing import Optional, Tuple, List
from pathlib import Path
import warnings

from .config import ReactionConfig
from .utils import generate_unique_labels

# Matplotlib settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.0,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Color palette
COLORS = {
    'activity': '#2CA02C',    # Green
    'stability': '#1F77B4',   # Blue
    'cost': '#FF7F0E',        # Orange
    'highlight': '#D62728',   # Red
    'optimal': '#D62728',     # Red
}


class Visualizer:
    """
    Visualization for ASCICat results.

    Generates four key figure panels:
    - Panel A: 3D ASCI component space
    - Panel B: Rank vs adsorption energy (with quadratic trend)
    - Panel C: Volcano optimization landscape (ASCI contours)
    - Panel D: Top performers breakdown

    For large datasets (>1000 catalysts), automatic stratified sampling
    is applied to ensure clear figures.

    CRITICAL AXIS RULES:
    - All axis limits derived ONLY from sampled DataFrame
    - Rank axis is strictly 1 to N_sampled (no global ranks)
    - Energy axes reflect actual data range (not full 50k dataset)
    - Score axes normalized to [0,1] based on sampled data

    Parameters
    ----------
    results : pd.DataFrame
        ASCI calculation results from ASCICalculator
    config : ReactionConfig
        Reaction configuration object
    auto_sample : bool
        Automatically sample large datasets (default: True)
    n_samples : int
        Number of samples for visualization (default: 1000)
    renormalize_scores : bool
        Re-normalize scores after sampling for better 3D distribution (default: True)

    Examples
    --------
    >>> calc = ASCICalculator('HER')
    >>> calc.load_data('data/HER_clean.csv')
    >>> results = calc.calculate_asci()
    >>> viz = Visualizer(results, calc.config)
    >>> viz.generate_publication_figures('figures/')
    """

    def __init__(self, results: pd.DataFrame, config: ReactionConfig,
                 auto_sample: bool = True, n_samples: int = 1000,
                 energy_window: float = 0.4, renormalize_scores: bool = True):
        # Store full results
        self.full_results = results.sort_values('ASCI', ascending=False).reset_index(drop=True)
        self.config = config
        self.reaction = config.name
        self.n_total = len(self.full_results)
        self.optimal_energy = config.optimal_energy

        # Apply 3D-diverse sampling for large datasets (better 3D visualization)
        if auto_sample and len(results) > 1000:
            from .sampling import sample_diverse_3d
            print(f"\nLarge dataset detected ({len(results):,} catalysts)")
            print(f"Optimal energy for {self.reaction}: ΔE_opt = {self.optimal_energy:.2f} eV")
            print(f"Applying 3D-diverse sampling for visualization...")

            self.results = sample_diverse_3d(
                self.full_results,
                n_samples=n_samples,
                n_top=min(150, n_samples // 5),
                n_bins_per_dim=5,
                random_state=42
            )
            self.is_sampled = True
            self.energy_window = energy_window
        else:
            self.results = self.full_results.copy()
            self.is_sampled = False
            self.energy_window = energy_window

        # CRITICAL: Re-normalize scores based on sampled data for better 3D distribution
        if renormalize_scores and self.is_sampled:
            self.results = self._renormalize_scores(self.results)
            print("  Scores re-normalized to [0,1] based on sampled data")

        # Ensure rank column exists - STRICTLY 1 to N_sampled
        self.results = self.results.sort_values('ASCI', ascending=False).reset_index(drop=True)
        self.results['rank'] = range(1, len(self.results) + 1)

    def _renormalize_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Re-normalize score columns to [0,1] based on sampled data."""
        df = df.copy()
        score_cols = ['activity_score', 'stability_score', 'cost_score']

        for col in score_cols:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[col] = 0.5

        return df

    def generate_publication_figures(self,
                                     output_dir: str = 'figures',
                                     dpi: int = 600,
                                     formats: List[str] = ['png', 'pdf']) -> None:
        """
        Generate all four figure panels.

        Parameters
        ----------
        output_dir : str
            Output directory for figures
        dpi : int
            Resolution (default: 600)
        formats : list
            Output formats (default: ['png', 'pdf'])
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating figures...")
        print(f"Output: {output_path}")
        print(f"Resolution: {dpi} DPI")
        if self.is_sampled:
            e_min = self.optimal_energy - self.energy_window
            e_max = self.optimal_energy + self.energy_window
            print(f"Data: {len(self.results)} samples from {self.n_total:,} total")
            print(f"Energy window: [{e_min:.2f}, {e_max:.2f}] eV (around ΔE_opt = {self.optimal_energy:.2f} eV)")
        else:
            print(f"Data: {len(self.results)} catalysts")

        # Panel A: 3D Pareto Space
        fig_a = self.plot_3d_pareto_space()
        self._save_figure(fig_a, output_path / 'panel_a_3d_pareto', dpi, formats)

        # Panel B: Rank vs Adsorption Energy
        fig_b = self.plot_rank_vs_adsorption()
        self._save_figure(fig_b, output_path / 'panel_b_rank_vs_adsorption', dpi, formats)

        # Panel C: Volcano Optimization
        fig_c = self.plot_volcano_optimization()
        self._save_figure(fig_c, output_path / 'panel_c_volcano_optimization', dpi, formats)

        # Panel D: Top Performers
        fig_d = self.plot_top_performers()
        self._save_figure(fig_d, output_path / 'panel_d_top_performers', dpi, formats)

        print(f"\nAll figures saved to {output_path}")

    def _save_figure(self, fig, filepath, dpi, formats):
        """Save figure in multiple formats."""
        for fmt in formats:
            full_path = f"{filepath}.{fmt}"
            fig.savefig(full_path, dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"  Saved: {Path(full_path).name}")
        plt.close(fig)

    def plot_3d_pareto_space(self,
                             figsize: Tuple[float, float] = (5.5, 5),
                             panel_label: str = 'a') -> plt.Figure:
        """
        Panel A: 3D visualization of ASCI component space.

        Shows the multi-objective optimization landscape with
        Activity, Stability, and Cost scores on three axes.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Main scatter plot
        scatter = ax.scatter(
            self.results['activity_score'],
            self.results['stability_score'],
            self.results['cost_score'],
            c=self.results['ASCI'],
            s=35,
            alpha=0.7,
            cmap='viridis',
            edgecolors='white',
            linewidths=0.2,
            vmin=0, vmax=1
        )

        # Highlight top 10 catalysts
        top10 = self.results.head(10)
        ax.scatter(
            top10['activity_score'],
            top10['stability_score'],
            top10['cost_score'],
            s=100,
            marker='*',
            facecolors='none',
            edgecolors=COLORS['highlight'],
            linewidths=1.5,
            label='Top 10 Catalysts'
        )

        # Labels
        ax.set_xlabel('Activity Score', fontweight='bold')
        ax.set_ylabel('Stability Score', fontweight='bold')
        ax.set_zlabel('Cost Score', fontweight='bold')

        # Limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

        # View angle
        ax.view_init(elev=20, azim=30)

        # Panel label
        ax.text2D(-0.05, 1.05, panel_label, transform=ax.transAxes,
                 fontsize=14, fontweight='bold', va='top', ha='right')

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, aspect=30, shrink=0.7)
        cbar.set_label('ASCI Score', fontweight='bold', fontsize=10)
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

        plt.tight_layout()
        return fig

    def plot_rank_vs_adsorption(self,
                                figsize: Tuple[float, float] = (5.5, 5),
                                panel_label: str = 'b') -> plt.Figure:
        """
        Panel B: ASCI rank vs adsorption energy (volcano-style).

        Shows the activity-performance relationship with the
        Sabatier volcano principle. Uses quadratic polynomial fit
        for the performance trend.
        """
        fig, ax = plt.subplots(figsize=figsize)

        df = self.results.copy()
        n_samples = len(df)

        # STRICT: Ranks must be 1 to n_samples only
        df['rank'] = range(1, n_samples + 1)

        # Main scatter
        scatter = ax.scatter(
            df['DFT_ads_E'],
            df['rank'],
            c=df['ASCI'],
            s=40,
            alpha=0.8,
            cmap='viridis',
            edgecolors='white',
            linewidths=0.3,
            vmin=0, vmax=1
        )

        # STRICT: Energy axis from sampled data ONLY
        e_min, e_max = df['DFT_ads_E'].min(), df['DFT_ads_E'].max()
        e_margin = (e_max - e_min) * 0.08

        # Activity window shading (if defined)
        if self.energy_window is not None:
            window_min = self.optimal_energy - self.energy_window
            window_max = self.optimal_energy + self.energy_window
            ax.axvspan(window_min, window_max,
                      alpha=0.1, color='green', zorder=0,
                      label=f'Activity Window')

        # Optimal energy line with annotation
        ax.axvline(x=self.config.optimal_energy, color=COLORS['optimal'],
                  linestyle='--', linewidth=1.5,
                  label=f'ΔE_opt = {self.config.optimal_energy:.2f} eV')

        # Highlight top 10
        top10 = df.head(10)
        ax.scatter(
            top10['DFT_ads_E'],
            top10['rank'],
            s=120,
            marker='*',
            facecolors='none',
            edgecolors=COLORS['highlight'],
            linewidths=1.5,
            label='Top 10 Catalysts'
        )

        # Performance trend (QUADRATIC FIT on high performers)
        high_score = df[df['ASCI'] > df['ASCI'].quantile(0.75)]
        if len(high_score) > 5:
            z = np.polyfit(high_score['DFT_ads_E'], high_score['rank'], 2)
            p = np.poly1d(z)
            x_range = np.linspace(e_min, e_max, 100)
            ax.plot(x_range, p(x_range), '--', color=COLORS['activity'],
                   linewidth=1.5, alpha=0.7, label='Trend')

        # STRICT: Axis limits from sampled data only
        ax.set_xlim(e_min - e_margin, e_max + e_margin)
        ax.set_ylim(1, n_samples)
        ax.invert_yaxis()  # Lower rank = better

        # Labels with explicit sample count
        ax.set_xlabel('Adsorption Energy, ΔE (eV)', fontweight='bold')
        ax.set_ylabel(f'ASCI Rank (1-{n_samples})', fontweight='bold')

        # Grid and legend
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

        # Panel label
        ax.text(-0.05, 1.05, panel_label, transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top', ha='right')

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('ASCI Score', fontweight='bold', fontsize=10)
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # Add data info annotation
        if self.is_sampled:
            info_text = f'n = {n_samples} (sampled from {self.n_total:,})'
        else:
            info_text = f'n = {n_samples}'
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
               fontsize=8, ha='right', va='bottom', color='gray',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        plt.tight_layout()
        return fig

    def plot_volcano_optimization(self,
                                  figsize: Tuple[float, float] = (5.5, 5),
                                  panel_label: str = 'c',
                                  n_contours: int = 20) -> plt.Figure:
        """
        Panel C: Volcano optimization landscape.

        Shows ASCI contours over (ΔE, log₁₀ Cost) space,
        revealing the multi-objective optimization landscape.

        CRITICAL: Computes ASCI using the formula on a grid,
        not interpolating raw values. Colorbar is 0-1 range.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Data
        x = self.results['DFT_ads_E'].values
        y = np.log10(self.results['Cost'].values)
        z = self.results['ASCI'].values

        # Grid for interpolation
        x_range = np.linspace(x.min() - 0.02, x.max() + 0.02, 100)
        y_range = np.linspace(y.min(), y.max(), 100)
        X, Y = np.meshgrid(x_range, y_range)

        # Calculate activity score for grid using ASCI formula
        max_deviation = self.config.activity_width
        activity_grid = 1 - np.abs(X - self.config.optimal_energy) / max_deviation
        activity_grid = np.clip(activity_grid, 0, 1)

        # Interpolate stability and cost scores
        try:
            points = np.vstack((x, y)).T
            stability_grid = griddata(points, self.results['stability_score'].values,
                                      (X, Y), method='linear', fill_value=0)
            cost_grid = griddata(points, self.results['cost_score'].values,
                                (X, Y), method='linear', fill_value=0)
            # Compute Z as weighted sum (ASCI formula on grid) - bounded 0-1
            Z = 0.33 * activity_grid + 0.33 * stability_grid + 0.34 * cost_grid
            Z = np.clip(Z, 0, 1)
        except Exception:
            Z = activity_grid

        # Contour plot with explicit 0-1 range
        levels = np.linspace(0, 1, n_contours + 1)
        contour = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7, vmin=0, vmax=1)

        # Contour lines with labels
        contour_levels = [0.3, 0.45, 0.6, 0.75]
        contour_lines = ax.contour(X, Y, Z, levels=contour_levels,
                                   colors='white', linewidths=0.7, alpha=0.8)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')

        # Data points
        ax.scatter(x, y, c='white', s=20, alpha=0.5,
                  edgecolors='black', linewidths=0.2)

        # Top 10 catalysts
        top10 = self.results.head(10)
        ax.scatter(top10['DFT_ads_E'], np.log10(top10['Cost']),
                  s=100, marker='*', facecolors='none',
                  edgecolors=COLORS['highlight'], linewidths=1.5,
                  label='Top 10 Catalysts')

        # Optimal energy line
        ax.axvline(x=self.config.optimal_energy, color=COLORS['optimal'],
                  linestyle='--', linewidth=1.5,
                  label=f'Optimal ΔE = {self.config.optimal_energy:.2f} eV')

        # Set axis limits based on sampled data
        x_margin = (x.max() - x.min()) * 0.05
        ax.set_xlim(x.min() - x_margin, x.max() + x_margin)

        # Labels
        ax.set_xlabel('Adsorption Energy, ΔE (eV)', fontweight='bold')
        ax.set_ylabel('log$_{10}$ Cost (USD/kg)', fontweight='bold')

        # Colorbar with explicit 0-1 range
        cbar = plt.colorbar(contour, ax=ax, pad=0.02)
        cbar.set_label('ASCI Score', fontweight='bold', fontsize=10)
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # Grid and legend
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

        # Panel label
        ax.text(-0.05, 1.05, panel_label, transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top', ha='right')

        plt.tight_layout()
        return fig

    def plot_top_performers(self,
                           figsize: Tuple[float, float] = (6, 5),
                           panel_label: str = 'd',
                           n_top: int = 8) -> plt.Figure:
        """
        Panel D: Top performers breakdown.

        Shows component score breakdown for top catalysts
        as grouped bar chart.
        """
        fig, ax = plt.subplots(figsize=figsize)

        top_cats = self.results.head(n_top).copy()
        top_cats = generate_unique_labels(top_cats)

        # Bar positions
        n = len(top_cats)
        bar_width = 0.2
        index = np.arange(n)

        # Component bars
        bars1 = ax.bar(index - bar_width, top_cats['activity_score'], bar_width,
                      label='Activity', color=COLORS['activity'],
                      edgecolor='black', linewidth=0.5)

        bars2 = ax.bar(index, top_cats['stability_score'], bar_width,
                      label='Stability', color=COLORS['stability'],
                      edgecolor='black', linewidth=0.5)

        bars3 = ax.bar(index + bar_width, top_cats['cost_score'], bar_width,
                      label='Cost', color=COLORS['cost'],
                      edgecolor='black', linewidth=0.5)

        # ASCI score markers
        ax.scatter(index, top_cats['ASCI'], marker='_', s=120,
                  color='black', label='ASCI Score', zorder=10, linewidth=2)

        # Labels - use unique display labels
        catalyst_labels = top_cats['display_label'].tolist()
        ax.set_xticks(index)
        ax.set_xticklabels(catalyst_labels, fontsize=9, rotation=45, ha='right')

        ax.set_ylabel('Score', fontweight='bold')
        ax.set_ylim(0, 1.05)

        # Add rank labels
        for i, (_, row) in enumerate(top_cats.iterrows()):
            ax.text(i, 0.03, f"#{i+1}", ha='center', va='bottom',
                   fontsize=8, color='dimgray')

        # Grid and legend
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9,
                 bbox_to_anchor=(1.0, 1.0))

        # Panel label
        ax.text(-0.05, 1.05, panel_label, transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top', ha='right')

        plt.tight_layout()
        return fig

    def plot_combined_figure(self,
                            figsize: Tuple[float, float] = (12, 10),
                            save_path: Optional[str] = None,
                            dpi: int = 600) -> plt.Figure:
        """
        Generate combined 4-panel figure.
        """
        fig = plt.figure(figsize=figsize)

        # Panel A: 3D
        ax_a = fig.add_subplot(2, 2, 1, projection='3d')
        self._draw_3d_panel(ax_a, 'a')

        # Panel B: Rank vs ΔE
        ax_b = fig.add_subplot(2, 2, 2)
        self._draw_rank_panel(ax_b, 'b')

        # Panel C: Volcano
        ax_c = fig.add_subplot(2, 2, 3)
        self._draw_volcano_panel(ax_c, 'c')

        # Panel D: Top performers
        ax_d = fig.add_subplot(2, 2, 4)
        self._draw_top_performers_panel(ax_d, 'd')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Combined figure saved: {save_path}")

        return fig

    def _draw_3d_panel(self, ax, label):
        """Draw 3D panel on given axes."""
        scatter = ax.scatter(
            self.results['activity_score'],
            self.results['stability_score'],
            self.results['cost_score'],
            c=self.results['ASCI'], s=25, alpha=0.7,
            cmap='viridis', edgecolors='white', linewidths=0.1,
            vmin=0, vmax=1
        )
        top10 = self.results.head(10)
        ax.scatter(top10['activity_score'], top10['stability_score'],
                  top10['cost_score'], s=80, marker='*',
                  facecolors='none', edgecolors='red', linewidths=1.2)
        ax.set_xlabel('Activity', fontsize=9)
        ax.set_ylabel('Stability', fontsize=9)
        ax.set_zlabel('Cost', fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.view_init(elev=20, azim=30)
        ax.text2D(-0.05, 1.02, label, transform=ax.transAxes,
                 fontsize=12, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('ASCI', fontsize=9)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    def _draw_rank_panel(self, ax, label):
        """Draw rank panel on given axes."""
        df = self.results.copy()
        df['rank'] = range(1, len(df) + 1)
        scatter = ax.scatter(df['DFT_ads_E'], df['rank'], c=df['ASCI'],
                           s=30, alpha=0.8, cmap='viridis', edgecolors='white', linewidths=0.2,
                           vmin=0, vmax=1)
        ax.axvline(x=self.config.optimal_energy, color='red', linestyle='--', linewidth=1.2)
        top10 = df.head(10)
        ax.scatter(top10['DFT_ads_E'], top10['rank'], s=80, marker='*',
                  facecolors='none', edgecolors='red', linewidths=1.2)

        # Quadratic trend
        high_score = df[df['ASCI'] > df['ASCI'].quantile(0.75)]
        if len(high_score) > 5:
            e_min, e_max = df['DFT_ads_E'].min(), df['DFT_ads_E'].max()
            z = np.polyfit(high_score['DFT_ads_E'], high_score['rank'], 2)
            p = np.poly1d(z)
            x_range = np.linspace(e_min, e_max, 100)
            ax.plot(x_range, p(x_range), '--', color=COLORS['activity'],
                   linewidth=1.5, alpha=0.7)

        ax.set_xlabel('ΔE (eV)', fontsize=10)
        ax.set_ylabel('ASCI Rank', fontsize=10)
        ax.invert_yaxis()
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.text(-0.05, 1.02, label, transform=ax.transAxes,
               fontsize=12, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('ASCI', fontsize=9)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    def _draw_volcano_panel(self, ax, label):
        """Draw volcano panel on given axes."""
        x = self.results['DFT_ads_E'].values
        y = np.log10(self.results['Cost'].values)

        x_range = np.linspace(x.min()-0.02, x.max()+0.02, 80)
        y_range = np.linspace(y.min(), y.max(), 80)
        X, Y = np.meshgrid(x_range, y_range)

        # Compute ASCI on grid
        activity_grid = np.clip(1 - np.abs(X - self.config.optimal_energy) / self.config.activity_width, 0, 1)
        try:
            points = np.vstack((x, y)).T
            stab = griddata(points, self.results['stability_score'].values, (X, Y), method='linear', fill_value=0)
            cost = griddata(points, self.results['cost_score'].values, (X, Y), method='linear', fill_value=0)
            Z = np.clip(0.33 * activity_grid + 0.33 * stab + 0.34 * cost, 0, 1)
        except:
            Z = activity_grid

        levels = np.linspace(0, 1, 16)
        contour = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7, vmin=0, vmax=1)
        ax.scatter(x, y, c='white', s=15, alpha=0.5, edgecolors='black', linewidths=0.1)
        top10 = self.results.head(10)
        ax.scatter(top10['DFT_ads_E'], np.log10(top10['Cost']), s=80, marker='*',
                  facecolors='none', edgecolors='red', linewidths=1.2)
        ax.axvline(x=self.config.optimal_energy, color='red', linestyle='--', linewidth=1.2)
        ax.set_xlabel('ΔE (eV)', fontsize=10)
        ax.set_ylabel('log₁₀ Cost', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.text(-0.05, 1.02, label, transform=ax.transAxes,
               fontsize=12, fontweight='bold')
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('ASCI', fontsize=9)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    def _draw_top_performers_panel(self, ax, label):
        """Draw top performers panel on given axes."""
        top8 = self.results.head(8).copy()
        top8 = generate_unique_labels(top8)
        n = len(top8)
        w = 0.2
        idx = np.arange(n)
        ax.bar(idx-w, top8['activity_score'], w, label='Activity', color=COLORS['activity'], edgecolor='black', linewidth=0.3)
        ax.bar(idx, top8['stability_score'], w, label='Stability', color=COLORS['stability'], edgecolor='black', linewidth=0.3)
        ax.bar(idx+w, top8['cost_score'], w, label='Cost', color=COLORS['cost'], edgecolor='black', linewidth=0.3)
        ax.scatter(idx, top8['ASCI'], marker='_', s=80, color='black', linewidth=1.5, zorder=10)
        ax.set_xticks(idx)
        ax.set_xticklabels(top8['display_label'].tolist(), fontsize=8, rotation=45, ha='right')
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.text(-0.05, 1.02, label, transform=ax.transAxes,
               fontsize=12, fontweight='bold')

    # =========================================================================
    # INTERACTIVE VISUALIZATION (Plotly)
    # =========================================================================

    def create_interactive_3d(self,
                              output_path: Optional[str] = None,
                              auto_open: bool = False) -> None:
        """
        Create interactive 3D visualization using Plotly.
        """
        try:
            import plotly.graph_objects as go
            from plotly.offline import plot
        except ImportError:
            print("Plotly not installed. Install with: pip install plotly")
            print("Skipping interactive visualization.")
            return

        print("\nGenerating interactive 3D visualization...")

        results_labeled = generate_unique_labels(self.results.copy())
        top10 = results_labeled.head(10)

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=results_labeled['activity_score'],
            y=results_labeled['stability_score'],
            z=results_labeled['cost_score'],
            mode='markers',
            marker=dict(
                size=4,
                color=results_labeled['ASCI'],
                colorscale='Viridis',
                opacity=0.7,
                cmin=0, cmax=1,
                colorbar=dict(title="ASCI Score", thickness=15)
            ),
            text=[f"<b>{row['display_label']}</b><br>"
                  f"ASCI: {row['ASCI']:.3f}<br>"
                  f"Rank: {row['rank']}<br>"
                  f"Activity: {row['activity_score']:.3f}<br>"
                  f"Stability: {row['stability_score']:.3f}<br>"
                  f"Cost: {row['cost_score']:.3f}"
                  for _, row in results_labeled.iterrows()],
            hoverinfo='text',
            name='All Catalysts'
        ))

        fig.add_trace(go.Scatter3d(
            x=top10['activity_score'],
            y=top10['stability_score'],
            z=top10['cost_score'],
            mode='markers+text',
            marker=dict(
                size=10,
                color='red',
                symbol='diamond',
                line=dict(color='black', width=1)
            ),
            text=[f"#{i+1}" for i in range(len(top10))],
            textposition='top center',
            textfont=dict(size=10, color='red'),
            hovertext=[f"<b>#{i+1}: {row['display_label']}</b><br>"
                       f"ASCI: {row['ASCI']:.3f}<br>"
                       f"Activity: {row['activity_score']:.3f}<br>"
                       f"Stability: {row['stability_score']:.3f}<br>"
                       f"Cost: {row['cost_score']:.3f}"
                       for i, (_, row) in enumerate(top10.iterrows())],
            hoverinfo='text',
            name='Top 10 Catalysts'
        ))

        fig.update_layout(
            title=dict(
                text=f'<b>ASCICat Interactive 3D - {self.reaction}</b>',
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title="Activity Score",
                yaxis_title="Stability Score",
                zaxis_title="Cost Score",
                xaxis=dict(range=[0, 1], gridcolor='lightgray'),
                yaxis=dict(range=[0, 1], gridcolor='lightgray'),
                zaxis=dict(range=[0, 1], gridcolor='lightgray'),
                bgcolor='white'
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255,255,255,0.8)'
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            scene_camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        )

        if output_path is None:
            output_path = 'interactive_3d_asci.html'

        plot(fig, filename=output_path, auto_open=auto_open)
        print(f"Interactive 3D saved: {output_path}")

    def generate_all_outputs(self,
                             output_dir: str = 'results',
                             dpi: int = 600,
                             include_interactive: bool = True) -> None:
        """
        Generate all outputs: static figures + interactive HTML.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate static figures
        self.generate_publication_figures(
            output_dir=str(output_path),
            dpi=dpi,
            formats=['png', 'pdf']
        )

        # Generate combined figure
        combined_path = output_path / 'figure_combined.png'
        self.plot_combined_figure(
            figsize=(12, 10),
            save_path=str(combined_path),
            dpi=dpi
        )

        # Generate interactive 3D
        if include_interactive:
            interactive_path = output_path / 'interactive_3d_asci.html'
            self.create_interactive_3d(output_path=str(interactive_path))

        print(f"\nAll outputs saved to: {output_path}/")
