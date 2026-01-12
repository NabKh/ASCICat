#!/usr/bin/env python3
"""
Example: Advanced Visualization and Pareto Analysis
====================================================

Advanced analysis techniques for multi-objective catalyst screening.

This script demonstrates:
1. Pareto front identification (non-dominated solutions)
2. Interactive 3D Pareto visualization
3. 2D Pareto projections for each objective pair
4. Weight sensitivity exploration
5. Statistical analysis of scores

Author: Nabil Khossossi
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ascicat import ASCICalculator
from ascicat.visualizer import Visualizer
from ascicat.sampling import sample_diverse_3d


# ============================================================================
# PARETO FRONT ANALYSIS
# ============================================================================

def identify_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify Pareto-optimal catalysts (non-dominated solutions).

    A catalyst is Pareto-optimal if no other catalyst is better in all
    three objectives simultaneously.
    """
    scores = df[['activity_score', 'stability_score', 'cost_score']].values
    n = len(scores)
    pareto_mask = np.ones(n, dtype=bool)

    for i in range(n):
        if not pareto_mask[i]:
            continue
        for j in range(n):
            if i != j and pareto_mask[j]:
                # Check if j dominates i
                if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                    pareto_mask[i] = False
                    break

    pareto_df = df[pareto_mask].copy()
    pareto_df['pareto_rank'] = range(1, len(pareto_df) + 1)
    return pareto_df


def plot_pareto_3d(results: pd.DataFrame, pareto: pd.DataFrame,
                   output_dir: Path, dpi=600):
    """
    High-resolution 3D Pareto front visualization.
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # All catalysts (faded)
    ax.scatter(
        results['activity_score'],
        results['stability_score'],
        results['cost_score'],
        c='lightgray', s=15, alpha=0.3,
        label='All catalysts'
    )

    # Pareto front (colored by ASCI)
    scatter = ax.scatter(
        pareto['activity_score'],
        pareto['stability_score'],
        pareto['cost_score'],
        c=pareto['ASCI'], cmap='viridis',
        s=60, edgecolors='black', linewidths=0.5,
        vmin=0, vmax=1,
        label=f'Pareto front ({len(pareto)} catalysts)'
    )

    # Top 5 from Pareto front (stars)
    top5 = pareto.nlargest(5, 'ASCI')
    ax.scatter(
        top5['activity_score'],
        top5['stability_score'],
        top5['cost_score'],
        s=200, marker='*', c='red',
        edgecolors='darkred', linewidths=1,
        label='Top 5 ASCI'
    )

    # Labels
    ax.set_xlabel('Activity Score', fontweight='bold', fontsize=11)
    ax.set_ylabel('Stability Score', fontweight='bold', fontsize=11)
    ax.set_zlabel('Cost Score', fontweight='bold', fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.view_init(elev=25, azim=45)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('ASCI Score', fontweight='bold')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / 'pareto_3d.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(output_dir / 'pareto_3d.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: pareto_3d.png/pdf")


def plot_pareto_2d_projections(results: pd.DataFrame, pareto: pd.DataFrame,
                               output_dir: Path, dpi=600):
    """
    2D projections of Pareto front for each objective pair.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    pairs = [
        ('activity_score', 'stability_score', 'Activity', 'Stability'),
        ('activity_score', 'cost_score', 'Activity', 'Cost'),
        ('stability_score', 'cost_score', 'Stability', 'Cost')
    ]

    for ax, (x_col, y_col, x_label, y_label) in zip(axes, pairs):
        # All catalysts
        ax.scatter(results[x_col], results[y_col],
                  c='lightgray', s=15, alpha=0.3)

        # Pareto front
        scatter = ax.scatter(pareto[x_col], pareto[y_col],
                            c=pareto['ASCI'], cmap='viridis',
                            s=40, edgecolors='black', linewidths=0.3,
                            vmin=0, vmax=1)

        # Top 3
        top3 = pareto.nlargest(3, 'ASCI')
        ax.scatter(top3[x_col], top3[y_col],
                  s=150, marker='*', c='red', edgecolors='darkred')

        ax.set_xlabel(f'{x_label} Score', fontweight='bold')
        ax.set_ylabel(f'{y_label} Score', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, linestyle='--')

    # Shared colorbar
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.8)
    cbar.set_label('ASCI Score', fontweight='bold')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    plt.tight_layout()
    fig.savefig(output_dir / 'pareto_2d_projections.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(output_dir / 'pareto_2d_projections.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: pareto_2d_projections.png/pdf")


def plot_score_distributions(results: pd.DataFrame, output_dir: Path, dpi=600):
    """
    Score distribution analysis with statistics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    scores = [
        ('activity_score', 'Activity', '#2CA02C'),
        ('stability_score', 'Stability', '#1F77B4'),
        ('cost_score', 'Cost', '#FF7F0E'),
        ('ASCI', 'ASCI', '#9467BD')
    ]

    for ax, (col, label, color) in zip(axes, scores):
        data = results[col]

        # Histogram
        n, bins, patches = ax.hist(data, bins=30, color=color, alpha=0.7,
                                   edgecolor='black', linewidth=0.5)

        # Statistics lines
        ax.axvline(data.mean(), color='red', linestyle='-', linewidth=2,
                  label=f'Mean = {data.mean():.3f}')
        ax.axvline(data.median(), color='blue', linestyle='--', linewidth=2,
                  label=f'Median = {data.median():.3f}')

        ax.set_xlabel(f'{label} Score', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        # Add statistics text
        stats_text = f'σ = {data.std():.3f}\nRange: [{data.min():.2f}, {data.max():.2f}]'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_dir / 'score_distributions.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(output_dir / 'score_distributions.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: score_distributions.png/pdf")


def main():
    """Run advanced analysis example."""

    print("="*70)
    print("ASCICat: Advanced Visualization and Pareto Analysis")
    print("="*70)

    # Setup
    data_path = Path(__file__).parent.parent / 'data' / 'HER_clean.csv'
    output_dir = Path(__file__).parent.parent / 'results' / 'advanced_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize and calculate
    calc = ASCICalculator(reaction='HER', verbose=True)
    calc.load_data(str(data_path))
    results = calc.calculate_asci(w_a=0.34, w_s=0.33, w_c=0.33)

    # Sample for visualization if large
    print("\nPreparing data for visualization...")
    if len(results) > 1000:
        sampled = sample_diverse_3d(results, n_samples=1000)
    else:
        sampled = results.copy()

    # =========================================================================
    # Pareto Front Analysis
    # =========================================================================
    print("\n" + "="*70)
    print("Pareto Front Analysis")
    print("="*70)

    pareto = identify_pareto_front(sampled)
    print(f"\nPareto-optimal catalysts: {len(pareto)} / {len(sampled)}")
    print(f"  ({100*len(pareto)/len(sampled):.1f}% of sampled data)")

    # Display top Pareto catalysts
    print(f"\nTop 10 Pareto-optimal catalysts (by ASCI):")
    print("-"*70)
    print(f"{'Rank':<6} {'Catalyst':<12} {'ASCI':<8} {'Activity':<10} "
          f"{'Stability':<10} {'Cost':<10}")
    print("-"*70)

    for i, (_, row) in enumerate(pareto.nlargest(10, 'ASCI').iterrows(), 1):
        print(f"{i:<6} {row['symbol']:<12} {row['ASCI']:.3f}    "
              f"{row['activity_score']:.3f}      {row['stability_score']:.3f}       "
              f"{row['cost_score']:.3f}")

    # =========================================================================
    # Generate Figures
    # =========================================================================
    print("\n" + "="*70)
    print("Generating Figures")
    print("="*70)

    # 3D Pareto
    plot_pareto_3d(sampled, pareto, output_dir)

    # 2D Projections
    plot_pareto_2d_projections(sampled, pareto, output_dir)

    # Score distributions
    plot_score_distributions(sampled, output_dir)

    # Standard 4-panel figure
    print("\n  Generating standard figures...")
    viz = Visualizer(results, calc.config)
    viz.generate_publication_figures(output_dir=str(output_dir), dpi=600)

    # =========================================================================
    # Export
    # =========================================================================
    print("\n" + "="*70)
    print("Exporting Results")
    print("="*70)

    pareto.to_csv(output_dir / 'pareto_front.csv', index=False)
    print(f"  Pareto front: pareto_front.csv ({len(pareto)} catalysts)")

    print("\n" + "="*70)
    print("Advanced Analysis Complete!")
    print(f"Results saved to: {output_dir}/")
    print("="*70)


if __name__ == '__main__':
    main()
