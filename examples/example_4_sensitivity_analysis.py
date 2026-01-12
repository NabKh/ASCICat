#!/usr/bin/env python3
"""
Example 4: Comprehensive Weight Sensitivity Analysis for HER
=============================================================

A critical concern for weighted-sum approaches involves potential arbitrariness
in weight selection. This example addresses this through comprehensive sensitivity
analysis examining ranking stability across the entire weight space.

Scientific Question:
    How sensitive are catalyst rankings to the choice of weights (w_a, w_s, w_c)?
    Which catalysts remain top performers regardless of weight choices?

Analysis Components:
    1. Weight Space Exploration - Systematic sweep across simplex
    2. Ternary Diagram - Best ASCI score across weight space
    3. Ranking Stability Analysis - How ranks change with weights
    4. Robustness Metrics - Identify weight-insensitive catalysts
    5. Kendall's Tau Correlation - Ranking consistency heatmap
    6. Critical Weight Regions - Where do rankings flip?

This analysis demonstrates that ASCICat provides deterministic, reproducible
rankings that can be validated through sensitivity testing.

Reaction: HER (Hydrogen Evolution Reaction)
Data: HER_clean.csv

Author: Nabil Khossossi
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import cm
from matplotlib.colors import Normalize
from collections import Counter
from scipy.stats import kendalltau

from ascicat import ASCICalculator


# High-resolution settings
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.0,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


# ============================================================================
# WEIGHT SPACE GENERATION
# ============================================================================

def generate_simplex_grid(n_points: int = 21, min_weight: float = 0.05) -> np.ndarray:
    """
    Generate weight combinations on the simplex (w_a + w_s + w_c = 1).

    The simplex constraint ensures all weights sum to unity while
    maintaining minimum weight requirements for each objective.
    """
    weights = []
    step = 1.0 / (n_points - 1)

    for i in range(n_points):
        for j in range(n_points - i):
            k = n_points - 1 - i - j
            w_a = i * step
            w_s = j * step
            w_c = k * step

            if w_a >= min_weight and w_s >= min_weight and w_c >= min_weight:
                weights.append([w_a, w_s, w_c])

    return np.array(weights)


def run_sensitivity_sweep(calc: ASCICalculator, weights: np.ndarray) -> dict:
    """
    Execute ASCI calculations across all weight combinations.

    Returns comprehensive results for sensitivity analysis.
    """
    print(f"\nRunning sensitivity sweep across {len(weights)} weight combinations...")

    results = {
        'weights': weights,
        'best_asci': [],
        'best_catalyst': [],
        'top10_catalysts': [],
        'top20_rankings': [],
        'all_asci_scores': []
    }

    for i, (w_a, w_s, w_c) in enumerate(weights):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(weights)}")

        df = calc.calculate_asci(w_a=w_a, w_s=w_s, w_c=w_c, show_progress=False)

        results['best_asci'].append(df['ASCI'].max())
        results['best_catalyst'].append(df.iloc[0]['symbol'])
        results['top10_catalysts'].extend(df.head(10)['symbol'].tolist())
        results['top20_rankings'].append(df.head(20)[['symbol', 'ASCI']].copy())
        results['all_asci_scores'].append(df[['symbol', 'ASCI']].copy())

    results['best_asci'] = np.array(results['best_asci'])
    print(f"  Completed!")

    return results


# ============================================================================
# ROBUSTNESS METRICS
# ============================================================================

def compute_robustness_metrics(results: dict) -> pd.DataFrame:
    """
    Compute comprehensive robustness metrics for each catalyst.

    A robust catalyst:
    - Appears frequently in top rankings
    - Has low rank variance across weight combinations
    - Maintains high average ASCI score
    """
    n_weights = len(results['weights'])

    # Count top-10 appearances
    top10_counts = Counter(results['top10_catalysts'])

    # Collect all ranks and ASCI scores per catalyst
    catalyst_data = {}
    for rankings_df in results['top20_rankings']:
        for rank, (_, row) in enumerate(rankings_df.iterrows(), 1):
            symbol = row['symbol']
            if symbol not in catalyst_data:
                catalyst_data[symbol] = {'ranks': [], 'scores': []}
            catalyst_data[symbol]['ranks'].append(rank)
            catalyst_data[symbol]['scores'].append(row['ASCI'])

    # Compute metrics
    robustness_data = []
    for symbol, data in catalyst_data.items():
        ranks = np.array(data['ranks'])
        scores = np.array(data['scores'])

        robustness_data.append({
            'symbol': symbol,
            'top10_count': top10_counts.get(symbol, 0),
            'top10_frequency': top10_counts.get(symbol, 0) / n_weights,
            'mean_rank': np.mean(ranks),
            'std_rank': np.std(ranks),
            'min_rank': np.min(ranks),
            'max_rank': np.max(ranks),
            'rank_range': np.max(ranks) - np.min(ranks),
            'mean_asci': np.mean(scores),
            'std_asci': np.std(scores),
            'cv_asci': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0
        })

    df = pd.DataFrame(robustness_data)

    # Compute normalized robustness score
    freq_norm = df['top10_frequency'] / df['top10_frequency'].max() if df['top10_frequency'].max() > 0 else 0
    std_max = df['std_rank'].max()
    stability_norm = 1 - (df['std_rank'] / std_max) if std_max > 0 else 1
    df['robustness_score'] = 0.5 * freq_norm + 0.5 * stability_norm

    return df.sort_values('robustness_score', ascending=False)


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_ternary_sensitivity(weights: np.ndarray, values: np.ndarray,
                             output_dir: Path, title: str = 'Best ASCI Score',
                             filename: str = 'fig1_ternary_asci', dpi: int = 600):
    """
    High-resolution ternary diagram showing ASCI landscape across weight space.

    This visualization directly addresses the weight arbitrariness concern
    by showing how the optimal ASCI varies across the entire feasible region.
    """
    print(f"\n  Creating ternary diagram: {title}...")

    fig, ax = plt.subplots(figsize=(8, 7))

    # Transform to ternary coordinates
    w_a, w_s, w_c = weights[:, 0], weights[:, 1], weights[:, 2]
    x = w_s + 0.5 * w_c
    y = np.sqrt(3) / 2 * w_c

    # Create triangulation and refine
    triang = tri.Triangulation(x, y)
    refiner = tri.UniformTriRefiner(triang)
    tri_refi, values_refi = refiner.refine_field(values, subdiv=3)

    # Contour plot
    levels = np.linspace(values.min(), values.max(), 20)
    contourf = ax.tricontourf(tri_refi, values_refi, levels=levels, cmap='viridis')

    # Contour lines
    contour = ax.tricontour(tri_refi, values_refi, levels=8, colors='white',
                           linewidths=0.5, alpha=0.7)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

    # Triangle boundary
    triangle = plt.Polygon(
        [[0, 0], [1, 0], [0.5, np.sqrt(3)/2]],
        fill=False, edgecolor='black', linewidth=2
    )
    ax.add_patch(triangle)

    # Vertex labels
    ax.text(-0.08, -0.05, 'Activity\n(w_a = 1)', fontsize=11,
            fontweight='bold', ha='center', va='top')
    ax.text(1.08, -0.05, 'Stability\n(w_s = 1)', fontsize=11,
            fontweight='bold', ha='center', va='top')
    ax.text(0.5, np.sqrt(3)/2 + 0.08, 'Cost\n(w_c = 1)', fontsize=11,
            fontweight='bold', ha='center', va='bottom')

    # Gridlines
    for frac in [0.25, 0.5, 0.75]:
        y_val = frac * np.sqrt(3) / 2
        x_start = frac / 2
        x_end = 1 - frac / 2
        ax.plot([x_start, x_end], [y_val, y_val], 'k-', alpha=0.2, linewidth=0.5)

    # Optimal point
    best_idx = np.argmax(values)
    ax.scatter([x[best_idx]], [y[best_idx]], s=200, c='red', marker='*',
              edgecolors='darkred', linewidths=1.5, zorder=10,
              label=f'Optimal: w_a={w_a[best_idx]:.2f}, w_s={w_s[best_idx]:.2f}, w_c={w_c[best_idx]:.2f}')

    # Equal weights point
    eq_x = 0.33 + 0.5 * 0.34
    eq_y = np.sqrt(3) / 2 * 0.34
    ax.scatter([eq_x], [eq_y], s=150, c='blue', marker='D',
              edgecolors='darkblue', linewidths=1.5, zorder=10,
              label='Equal weights (0.33, 0.33, 0.34)')

    cbar = plt.colorbar(contourf, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label(title, fontweight='bold', fontsize=11)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, np.sqrt(3)/2 + 0.15)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='lower center', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_dir / f'{filename}.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(output_dir / f'{filename}.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {filename}.png/pdf")


def plot_ranking_stability(results: dict, robustness_df: pd.DataFrame,
                          output_dir: Path, top_n: int = 8, dpi: int = 600):
    """
    ASCI Score Sensitivity Analysis.

    Standard sensitivity plot showing how ASCI scores change across
    weight configurations. Line crossings indicate rank reversals.
    """
    print(f"\n  Creating ASCI score sensitivity plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Get top catalysts
    top_catalysts = robustness_df.head(top_n)['symbol'].tolist()

    # Collect ASCI scores for each catalyst across all configurations
    n_configs = len(results['weights'])
    scores = {cat: [] for cat in top_catalysts}

    for df in results['all_asci_scores']:
        score_dict = dict(zip(df['symbol'], df['ASCI']))
        for cat in top_catalysts:
            scores[cat].append(score_dict.get(cat, 0))

    # Sort configurations by activity weight for x-axis
    sort_idx = np.argsort(results['weights'][:, 0])

    # Plot lines
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))
    for i, cat in enumerate(top_catalysts):
        sorted_scores = [scores[cat][j] for j in sort_idx]
        ax.plot(range(n_configs), sorted_scores, '-', color=colors[i],
               linewidth=2, label=cat, marker='o', markersize=3, markevery=10)

    ax.set_xlabel('Weight Configuration (sorted by activity weight, w_a)',
                 fontweight='bold', fontsize=12)
    ax.set_ylabel('ASCI Score', fontweight='bold', fontsize=12)
    ax.set_title('Sensitivity Analysis: ASCI Score Variation Across Weight Space',
                fontweight='bold', fontsize=13)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, n_configs - 1)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig2_score_sensitivity.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(output_dir / 'fig2_score_sensitivity.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: fig2_score_sensitivity.png/pdf")


def plot_robustness_analysis(robustness_df: pd.DataFrame, output_dir: Path,
                              top_n: int = 20, dpi: int = 600):
    """
    High-resolution robustness metrics visualization.

    Scientific rationale:
    - For multi-criteria sensitivity analysis, the key robustness metric is
      "how often does this catalyst appear in the top-N across all weight combinations?"
    - This is the Top-10 Frequency: percentage of weight configurations where
      the catalyst ranks in top 10
    - Secondary metric: Mean rank across all configurations (lower = better)

    Simple, clear bar chart is the appropriate visualization for comparing
    a single metric across multiple catalysts.
    """
    print(f"\n  Creating robustness analysis plot...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))

    top = robustness_df.head(top_n).copy()

    # =========================================================================
    # Panel A: Top-10 Frequency (primary robustness metric)
    # =========================================================================
    ax1 = axes[0]

    # Sort by frequency for this panel
    top_freq = top.sort_values('top10_frequency', ascending=True)

    y_pos = range(len(top_freq))
    frequencies = top_freq['top10_frequency'] * 100  # Convert to percentage

    # Color gradient based on frequency
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(top_freq)))

    bars1 = ax1.barh(y_pos, frequencies, color=colors, edgecolor='black',
                     linewidth=0.5, height=0.75)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_freq['symbol'], fontsize=10)
    ax1.set_xlabel('Top-10 Frequency (%)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Catalyst', fontweight='bold', fontsize=12)
    ax1.set_title('(A) Robustness: Top-10 Appearance Frequency',
                 fontweight='bold', fontsize=13)
    ax1.set_xlim(0, 105)
    ax1.grid(True, alpha=0.3, axis='x', linestyle='-', linewidth=0.5)

    # Add percentage labels
    for bar, freq in zip(bars1, frequencies):
        if freq > 5:
            ax1.text(freq - 2, bar.get_y() + bar.get_height()/2,
                    f'{freq:.0f}%', va='center', ha='right', fontsize=9,
                    fontweight='bold', color='white')
        else:
            ax1.text(freq + 1, bar.get_y() + bar.get_height()/2,
                    f'{freq:.0f}%', va='center', ha='left', fontsize=9)

    # =========================================================================
    # Panel B: Mean Rank (secondary metric)
    # =========================================================================
    ax2 = axes[1]

    # Sort by mean rank for this panel (ascending = best first)
    top_rank = top.sort_values('mean_rank', ascending=False)

    y_pos = range(len(top_rank))
    mean_ranks = top_rank['mean_rank']
    std_ranks = top_rank['std_rank']

    # Color by mean rank (green = better = lower rank)
    norm_ranks = (mean_ranks - mean_ranks.min()) / (mean_ranks.max() - mean_ranks.min())
    colors = plt.cm.RdYlGn_r(norm_ranks)

    bars2 = ax2.barh(y_pos, mean_ranks, xerr=std_ranks, color=colors,
                     edgecolor='black', linewidth=0.5, height=0.75,
                     capsize=3, error_kw={'linewidth': 1, 'capthick': 1})

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_rank['symbol'], fontsize=10)
    ax2.set_xlabel('Mean Rank ± Std Dev', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Catalyst', fontweight='bold', fontsize=12)
    ax2.set_title('(B) Ranking Consistency Across Weight Space',
                 fontweight='bold', fontsize=13)
    ax2.set_xlim(0, 22)
    ax2.axvline(x=10, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.text(10.5, len(top_rank) - 0.5, 'Top 10\nthreshold', fontsize=9,
            color='red', va='top')
    ax2.grid(True, alpha=0.3, axis='x', linestyle='-', linewidth=0.5)

    # Add rank labels
    for bar, (mean_r, std_r) in zip(bars2, zip(mean_ranks, std_ranks)):
        ax2.text(mean_r + std_r + 0.5, bar.get_y() + bar.get_height()/2,
                f'{mean_r:.1f}', va='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig3_robustness_analysis.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(output_dir / 'fig3_robustness_analysis.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: fig3_robustness_analysis.png/pdf")


def plot_weight_region_analysis(results: dict, output_dir: Path, dpi: int = 600):
    """
    High-resolution ternary diagram showing catalyst preference regions.

    Scientific rationale:
    - The weight space forms a simplex (w_a + w_s + w_c = 1)
    - Different regions of this space favor different catalysts
    - This visualization shows WHERE in weight space each catalyst excels
    - Key insight: If a catalyst dominates a large region, it's robust
      to weight selection within that regime

    Visualization approach:
    - Simple scatter plot with discrete colors (no interpolation artifacts)
    - Each point = one weight configuration
    - Color = best catalyst at that configuration
    - Clear spatial clustering reveals weight preference regimes
    """
    print(f"\n  Creating weight region analysis...")

    fig, ax = plt.subplots(figsize=(10, 9))

    weights = results['weights']
    best_cats = results['best_catalyst']

    # Count occurrences
    cat_counts = Counter(best_cats)

    # Transform to ternary coordinates
    w_a, w_s, w_c = weights[:, 0], weights[:, 1], weights[:, 2]
    x = w_s + 0.5 * w_c
    y = np.sqrt(3) / 2 * w_c

    # Get top catalysts for visualization
    top_cats = [cat for cat, _ in cat_counts.most_common(6)]

    # Use qualitative colormap for discrete categories
    cat_colors = {
        top_cats[0]: '#e41a1c',  # Red
        top_cats[1]: '#377eb8',  # Blue
        top_cats[2]: '#4daf4a',  # Green
        top_cats[3]: '#984ea3',  # Purple
        top_cats[4]: '#ff7f00',  # Orange
        top_cats[5]: '#a65628' if len(top_cats) > 5 else '#999999',  # Brown
    }

    # Plot each catalyst's region
    for cat in reversed(top_cats):  # Plot most common last (on top)
        mask = np.array([c == cat for c in best_cats])
        count = mask.sum()
        pct = 100 * count / len(best_cats)
        ax.scatter(x[mask], y[mask], c=cat_colors.get(cat, '#999999'),
                  s=120, edgecolors='white', linewidths=0.8,
                  label=f'{cat} ({pct:.1f}%)', zorder=5, alpha=0.85)

    # Plot "other" catalysts
    other_mask = np.array([c not in top_cats for c in best_cats])
    if other_mask.sum() > 0:
        pct = 100 * other_mask.sum() / len(best_cats)
        ax.scatter(x[other_mask], y[other_mask], c='#cccccc',
                  s=60, edgecolors='gray', linewidths=0.5,
                  label=f'Others ({pct:.1f}%)', zorder=4, alpha=0.6)

    # Triangle boundary
    triangle = plt.Polygon(
        [[0, 0], [1, 0], [0.5, np.sqrt(3)/2]],
        fill=False, edgecolor='black', linewidth=2.5, zorder=10
    )
    ax.add_patch(triangle)

    # Vertex labels
    ax.text(0, -0.08, 'Activity\n(w_a = 1)', fontsize=11, fontweight='bold',
           ha='center', va='top')
    ax.text(1, -0.08, 'Stability\n(w_s = 1)', fontsize=11, fontweight='bold',
           ha='center', va='top')
    ax.text(0.5, np.sqrt(3)/2 + 0.05, 'Cost\n(w_c = 1)', fontsize=11, fontweight='bold',
           ha='center', va='bottom')

    # Mark equal weights point
    eq_x = 1/3 + 0.5 * 1/3
    eq_y = np.sqrt(3) / 2 * 1/3
    ax.scatter([eq_x], [eq_y], s=250, c='white', marker='*',
              edgecolors='black', linewidths=2, zorder=11)
    ax.annotate('Equal\nweights', (eq_x, eq_y), xytext=(eq_x + 0.12, eq_y),
               fontsize=10, ha='left', va='center',
               arrowprops=dict(arrowstyle='->', color='black', lw=1))

    # Add gridlines for reference
    for frac in [0.25, 0.5, 0.75]:
        # Lines parallel to each edge
        y_val = frac * np.sqrt(3) / 2
        x_start = frac / 2
        x_end = 1 - frac / 2
        ax.plot([x_start, x_end], [y_val, y_val], 'k-', alpha=0.15, linewidth=0.5, zorder=1)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.12, np.sqrt(3)/2 + 0.1)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('Best Catalyst by Weight Configuration\n(Each point = one weight combination)',
                fontweight='bold', fontsize=14, pad=10)

    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95,
             title='Best Catalyst (%)', title_fontsize=11,
             markerscale=0.8)

    # Add summary text
    n_unique = len(set(best_cats))
    ax.text(0.98, 0.02, f'{len(best_cats)} weight configurations\n{n_unique} unique best catalysts',
           transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
           style='italic', color='gray',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_dir / 'fig4_weight_regions.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(output_dir / 'fig4_weight_regions.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: fig4_weight_regions.png/pdf")


def plot_kendall_tau_heatmap(results: dict, output_dir: Path,
                             n_samples: int = 10, dpi: int = 600):
    """
    Kendall's Tau correlation heatmap showing ranking consistency.

    This quantifies how similar rankings are between different weight choices.
    High correlation = rankings are robust to weight changes.
    """
    print(f"\n  Creating Kendall's Tau correlation heatmap...")

    # Sample weight indices evenly
    n_weights = len(results['weights'])
    sample_indices = np.linspace(0, n_weights-1, n_samples, dtype=int)

    # Compute pairwise Kendall's Tau
    tau_matrix = np.zeros((n_samples, n_samples))

    rankings_list = []
    for idx in sample_indices:
        df = results['all_asci_scores'][idx]
        rankings_list.append(df.set_index('symbol')['ASCI'].rank(ascending=False))

    for i in range(n_samples):
        for j in range(n_samples):
            common = rankings_list[i].index.intersection(rankings_list[j].index)
            if len(common) > 5:
                tau, _ = kendalltau(rankings_list[i].loc[common], rankings_list[j].loc[common])
                tau_matrix[i, j] = tau
            else:
                tau_matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(tau_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')

    # Labels showing weights
    weight_labels = [f"({results['weights'][i][0]:.2f},{results['weights'][i][1]:.2f},{results['weights'][i][2]:.2f})"
                    for i in sample_indices]
    ax.set_xticks(range(n_samples))
    ax.set_xticklabels(range(1, n_samples+1), fontsize=9)
    ax.set_yticks(range(n_samples))
    ax.set_yticklabels(range(1, n_samples+1), fontsize=9)

    # Add correlation values
    for i in range(n_samples):
        for j in range(n_samples):
            if not np.isnan(tau_matrix[i, j]):
                color = 'white' if tau_matrix[i, j] < 0.5 else 'black'
                ax.text(j, i, f'{tau_matrix[i, j]:.2f}', ha='center', va='center',
                       fontsize=8, color=color)

    ax.set_xlabel('Weight Configuration Index', fontweight='bold', fontsize=11)
    ax.set_ylabel('Weight Configuration Index', fontweight='bold', fontsize=11)
    ax.set_title("Kendall's τ Ranking Correlation Across Weight Space",
                fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Kendall's τ", fontweight='bold')

    # Add interpretation
    mean_tau = np.nanmean(tau_matrix[np.triu_indices(n_samples, k=1)])
    ax.text(0.5, -0.12, f'Mean τ = {mean_tau:.3f} (High = Rankings are consistent)',
           transform=ax.transAxes, ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig5_kendall_tau.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(output_dir / 'fig5_kendall_tau.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: fig5_kendall_tau.png/pdf")


def plot_top_catalyst_frequency(results: dict, output_dir: Path,
                                top_n: int = 15, dpi: int = 600):
    """
    Horizontal bar chart showing how often each catalyst is ranked #1.
    """
    print(f"\n  Creating top catalyst frequency plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    best_counts = Counter(results['best_catalyst'])
    top_best = best_counts.most_common(top_n)

    catalysts = [c for c, _ in top_best]
    frequencies = [c / len(results['weights']) * 100 for _, c in top_best]

    colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(catalysts)))
    bars = ax.barh(range(len(catalysts)), frequencies, color=colors, edgecolor='black')

    ax.set_yticks(range(len(catalysts)))
    ax.set_yticklabels(catalysts, fontsize=10)
    ax.set_xlabel('Frequency as Best Catalyst (%)', fontweight='bold', fontsize=11)
    ax.set_title('Top Catalyst Frequency Across Weight Combinations',
                fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for bar, freq in zip(bars, frequencies):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
               f'{freq:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig6_top_frequency.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(output_dir / 'fig6_top_frequency.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: fig6_top_frequency.png/pdf")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run comprehensive HER sensitivity analysis."""

    print("="*70)
    print("ASCICat: Comprehensive Weight Sensitivity Analysis")
    print("Reaction: HER (Hydrogen Evolution Reaction)")
    print("="*70)

    print("""
This analysis addresses the critical concern about weight arbitrariness
in weighted-sum approaches by examining ranking stability across the
entire feasible weight space (w_a + w_s + w_c = 1).
    """)

    # Setup
    data_path = Path(__file__).parent.parent / 'data' / 'HER_clean.csv'
    output_dir = Path(__file__).parent.parent / 'results' / 'sensitivity_analysis_HER'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize calculator
    calc = ASCICalculator(reaction='HER', verbose=True)
    calc.load_data(str(data_path))

    # Generate weight grid
    print("\n" + "="*70)
    print("1. Generating Weight Grid on Simplex")
    print("="*70)

    weights = generate_simplex_grid(n_points=21, min_weight=0.05)
    print(f"Generated {len(weights)} weight combinations")
    print(f"Constraint: w_a + w_s + w_c = 1, each ≥ 0.05")

    # Run sensitivity sweep
    print("\n" + "="*70)
    print("2. Running Sensitivity Sweep")
    print("="*70)

    results = run_sensitivity_sweep(calc, weights)

    # Compute robustness metrics
    print("\n" + "="*70)
    print("3. Computing Robustness Metrics")
    print("="*70)

    robustness = compute_robustness_metrics(results)

    print("\nTop 10 Most Robust Catalysts (HER):")
    print("-"*80)
    print(f"{'Rank':<6} {'Catalyst':<12} {'Robustness':<12} {'Top10 Freq':<12} "
          f"{'Mean Rank':<12} {'Rank σ':<10}")
    print("-"*80)

    for i, (_, row) in enumerate(robustness.head(10).iterrows(), 1):
        print(f"{i:<6} {row['symbol']:<12} {row['robustness_score']:.3f}        "
              f"{row['top10_frequency']:.2%}        {row['mean_rank']:.1f}          "
              f"{row['std_rank']:.2f}")

    # Generate all figures
    print("\n" + "="*70)
    print("4. Generating Figures")
    print("="*70)

    # Fig 1: Ternary diagram
    plot_ternary_sensitivity(weights, results['best_asci'], output_dir)

    # Fig 2: Ranking stability (trajectories + box plots)
    plot_ranking_stability(results, robustness, output_dir)

    # Fig 3: Robustness analysis (scatter + bar chart)
    plot_robustness_analysis(robustness, output_dir)

    # Fig 4: Weight region analysis (ternary contour + bar chart)
    plot_weight_region_analysis(results, output_dir)

    # Fig 5: Kendall's Tau correlation
    plot_kendall_tau_heatmap(results, output_dir)

    # Fig 6: Top catalyst frequency
    plot_top_catalyst_frequency(results, output_dir)

    # Export results
    print("\n" + "="*70)
    print("5. Exporting Results")
    print("="*70)

    robustness.to_csv(output_dir / 'robustness_metrics.csv', index=False)
    print(f"  Robustness metrics: robustness_metrics.csv")

    weight_df = pd.DataFrame(weights, columns=['w_activity', 'w_stability', 'w_cost'])
    weight_df['best_asci'] = results['best_asci']
    weight_df['best_catalyst'] = results['best_catalyst']
    weight_df.to_csv(output_dir / 'weight_sensitivity.csv', index=False)
    print(f"  Weight sensitivity: weight_sensitivity.csv")

    # Summary
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print(f"\nKey Findings:")
    print(f"  - Weight combinations analyzed: {len(weights)}")
    print(f"  - Unique #1 catalysts: {len(set(results['best_catalyst']))}")
    print(f"  - Most robust catalyst: {robustness.iloc[0]['symbol']}")
    print(f"  - Robustness score: {robustness.iloc[0]['robustness_score']:.3f}")

    # Interpretation
    print(f"\nInterpretation:")
    print(f"  Catalysts with high robustness scores maintain consistent rankings")
    print(f"  across the entire weight space, demonstrating that ASCICat provides")
    print(f"  reproducible, weight-insensitive recommendations for systematic")
    print(f"  catalyst development programs.")


if __name__ == '__main__':
    main()
