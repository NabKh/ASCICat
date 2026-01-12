#!/usr/bin/env python3
"""
Complete Sensitivity Analysis
====================================================

Scientific Question: How does catalyst selection change with weight choices?

Figures:
1. Ternary ASCI Landscape - Best achievable score across weight space
2. Catalyst Dominance Map - Which catalyst is optimal at each weight point
3. Top Catalyst Consistency - Catalysts that remain top-ranked regardless of weights
4. Rank Trajectories - How rankings shift along weight gradients

Author: Nabil Khossossi
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from collections import Counter
from scipy.stats import kendalltau
from ascicat import ASCICalculator

# Standard style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'pdf.fonttype': 42,
})

COLORS = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#f4a261', '#9c89b8']


def generate_weights(n_points=21, min_weight=0.05):
    """Generate simplex grid."""
    weights = []
    step = 1.0 / (n_points - 1)
    for i in range(n_points):
        for j in range(n_points - i):
            k = n_points - 1 - i - j
            w_a, w_s, w_c = i * step, j * step, k * step
            if w_a >= min_weight and w_s >= min_weight and w_c >= min_weight:
                weights.append([w_a, w_s, w_c])
    return np.array(weights)


def run_full_analysis(calc, weights):
    """Run ASCI for all weight combinations, collect full data."""
    print(f"  Analyzing {len(weights)} weight configurations...")

    results = {
        'weights': weights,
        'best_asci': [],
        'best_catalyst': [],
        'rankings': [],  # Full ranking for each config
    }

    for i, (w_a, w_s, w_c) in enumerate(weights):
        if (i + 1) % 50 == 0:
            print(f"    Progress: {i+1}/{len(weights)}")

        df = calc.calculate_asci(w_a=w_a, w_s=w_s, w_c=w_c, show_progress=False)

        results['best_asci'].append(df['ASCI'].max())
        results['best_catalyst'].append(df.iloc[0]['symbol'])

        # Store ranking (symbol -> rank)
        ranking = {row['symbol']: rank for rank, (_, row) in enumerate(df.iterrows(), 1)}
        results['rankings'].append(ranking)

    results['best_asci'] = np.array(results['best_asci'])
    return results


def to_ternary(weights):
    """Convert simplex coordinates to 2D ternary plot coordinates."""
    w_a, w_s, w_c = weights[:, 0], weights[:, 1], weights[:, 2]
    x = w_s + 0.5 * w_c
    y = np.sqrt(3) / 2 * w_c
    return x, y


def draw_triangle(ax):
    """Draw triangle boundary and labels."""
    triangle = Polygon([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]],
                       fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(triangle)

    # Labels
    ax.text(0, -0.07, 'Activity\n($w_a$=1)', fontsize=9, ha='center', va='top', fontweight='bold')
    ax.text(1, -0.07, 'Stability\n($w_s$=1)', fontsize=9, ha='center', va='top', fontweight='bold')
    ax.text(0.5, np.sqrt(3)/2 + 0.05, 'Cost\n($w_c$=1)', fontsize=9, ha='center', va='bottom', fontweight='bold')

    # Grid
    for frac in [0.25, 0.5, 0.75]:
        y_val = frac * np.sqrt(3) / 2
        ax.plot([frac/2, 1 - frac/2], [y_val, y_val], 'k-', alpha=0.15, lw=0.5)

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.12, np.sqrt(3)/2 + 0.1)
    ax.set_aspect('equal')
    ax.axis('off')


# =============================================================================
# FIGURE 1: TERNARY ASCI LANDSCAPE
# =============================================================================

def create_fig1_ternary_asci(results, output_dir, title_suffix=''):
    """Best ASCI score landscape across weight space."""
    print("\n  Creating Fig 1: Ternary ASCI Landscape...")

    fig, ax = plt.subplots(figsize=(7, 6))

    x, y = to_ternary(results['weights'])
    values = results['best_asci']

    # Triangulation with refinement
    triang = tri.Triangulation(x, y)
    refiner = tri.UniformTriRefiner(triang)
    tri_refi, val_refi = refiner.refine_field(values, subdiv=3)

    # Contour
    levels = np.linspace(values.min(), values.max(), 20)
    cf = ax.tricontourf(tri_refi, val_refi, levels=levels, cmap='viridis')
    cs = ax.tricontour(tri_refi, val_refi, levels=8, colors='white', linewidths=0.4, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')

    draw_triangle(ax)

    # Optimal point
    best_idx = np.argmax(values)
    w = results['weights'][best_idx]
    ax.scatter([x[best_idx]], [y[best_idx]], s=200, c='red', marker='*',
               edgecolors='darkred', linewidths=1.5, zorder=10,
               label=f'Optimal ({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f})')

    # Equal weights
    eq_x, eq_y = 0.5, np.sqrt(3)/6
    ax.scatter([eq_x], [eq_y], s=100, c='white', marker='o',
               edgecolors='black', linewidths=1.5, zorder=10, label='Equal weights')

    cbar = plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Best ASCI Score', fontweight='bold')

    ax.legend(loc='lower center', fontsize=8, framealpha=0.9)
    ax.set_title(f'ASCI Score Landscape{title_suffix}', fontweight='bold', pad=10)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig1_ternary_asci.png', dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'fig1_ternary_asci.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved fig1_ternary_asci")


# =============================================================================
# FIGURE 2: CATALYST DOMINANCE MAP
# =============================================================================

def create_fig2_catalyst_dominance(results, output_dir, title_suffix=''):
    """Which catalyst is optimal at each weight configuration."""
    print("\n  Creating Fig 2: Catalyst Dominance Map...")

    fig, ax = plt.subplots(figsize=(7, 6))

    x, y = to_ternary(results['weights'])
    best_cats = results['best_catalyst']

    # Get top catalysts by dominance
    cat_counts = Counter(best_cats)
    top_cats = [c for c, _ in cat_counts.most_common(6)]

    draw_triangle(ax)

    # Plot each catalyst's region
    for i, cat in enumerate(top_cats):
        mask = np.array([c == cat for c in best_cats])
        pct = 100 * mask.sum() / len(best_cats)
        ax.scatter(x[mask], y[mask], c=COLORS[i], s=80, edgecolors='white',
                   linewidths=0.5, label=f'{cat} ({pct:.1f}%)', zorder=5, alpha=0.85)

    # Others
    other_mask = ~np.isin(best_cats, top_cats)
    if other_mask.sum() > 0:
        ax.scatter(x[other_mask], y[other_mask], c='#cccccc', s=40,
                   edgecolors='gray', linewidths=0.3, label='Others', zorder=4, alpha=0.5)

    # Equal weights marker
    eq_x, eq_y = 0.5, np.sqrt(3)/6
    ax.scatter([eq_x], [eq_y], s=150, c='white', marker='*',
               edgecolors='black', linewidths=2, zorder=10)
    ax.annotate('Equal\nweights', (eq_x, eq_y), xytext=(eq_x+0.12, eq_y),
                fontsize=8, ha='left', arrowprops=dict(arrowstyle='->', lw=1))

    ax.legend(loc='upper left', fontsize=7, framealpha=0.95, title='Best Catalyst', title_fontsize=8)
    ax.set_title(f'Optimal Catalyst by Weight Configuration{title_suffix}', fontweight='bold', pad=10)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig2_catalyst_dominance.png', dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'fig2_catalyst_dominance.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved fig2_catalyst_dominance")


# =============================================================================
# FIGURE 3: TOP CATALYST CONSISTENCY
# =============================================================================

def create_fig3_consistency(results, output_dir, title_suffix=''):
    """Which catalysts consistently appear in top rankings."""
    print("\n  Creating Fig 3: Top Catalyst Consistency...")

    fig, ax = plt.subplots(figsize=(6, 5))

    n_configs = len(results['weights'])

    # Count appearances in top-3, top-5, top-10
    top3_counts = Counter()
    top5_counts = Counter()
    top10_counts = Counter()

    for ranking in results['rankings']:
        sorted_cats = sorted(ranking.keys(), key=lambda c: ranking[c])
        top3_counts.update(sorted_cats[:3])
        top5_counts.update(sorted_cats[:5])
        top10_counts.update(sorted_cats[:10])

    # Get catalysts that appear in top-10 at least 50% of the time
    consistent_cats = [c for c, count in top10_counts.most_common(10)
                       if count >= n_configs * 0.3][:8]

    # Data for plotting
    top3_freq = [top3_counts[c] / n_configs * 100 for c in consistent_cats]
    top5_only = [(top5_counts[c] - top3_counts[c]) / n_configs * 100 for c in consistent_cats]
    top10_only = [(top10_counts[c] - top5_counts[c]) / n_configs * 100 for c in consistent_cats]

    y_pos = np.arange(len(consistent_cats))

    # Stacked bars
    ax.barh(y_pos, top3_freq, height=0.6, color='#2166ac', edgecolor='white',
            linewidth=0.5, label='Top 3')
    ax.barh(y_pos, top5_only, height=0.6, left=top3_freq, color='#67a9cf',
            edgecolor='white', linewidth=0.5, label='Top 4-5')
    ax.barh(y_pos, top10_only, height=0.6, left=np.array(top3_freq)+np.array(top5_only),
            color='#d1e5f0', edgecolor='white', linewidth=0.5, label='Top 6-10')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(consistent_cats)
    ax.set_xlabel('Frequency Across Weight Configurations (%)')
    ax.set_xlim(0, 105)
    ax.invert_yaxis()

    # Add percentage labels
    for i, (t3, t5, t10) in enumerate(zip(top3_freq, top5_only, top10_only)):
        total = t3 + t5 + t10
        if total > 0:
            ax.text(total + 1, i, f'{total:.0f}%', va='center', fontsize=8)

    ax.legend(loc='lower right', fontsize=7)
    ax.set_title(f'Ranking Consistency Across Weight Space{title_suffix}', fontweight='bold')

    # Add note
    ax.text(0.02, 0.98, f'n = {n_configs} weight configurations',
            transform=ax.transAxes, fontsize=7, va='top', style='italic', color='gray')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig3_consistency.png', dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'fig3_consistency.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved fig3_consistency")


# =============================================================================
# FIGURE 4: RANK TRAJECTORIES ALONG WEIGHT GRADIENTS
# =============================================================================

def create_fig4_rank_trajectories(calc, output_dir, title_suffix=''):
    """How rankings change as we vary weights along specific paths."""
    print("\n  Creating Fig 4: Rank Trajectories...")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    n_points = 12
    weight_range = np.linspace(0.15, 0.7, n_points)

    # Get reference top catalysts at equal weights
    ref_df = calc.calculate_asci(w_a=0.33, w_s=0.33, w_c=0.34, show_progress=False)

    # Get unique top catalysts
    seen = set()
    top_cats = []
    for sym in ref_df['symbol'].tolist():
        if sym not in seen:
            seen.add(sym)
            top_cats.append(sym)
        if len(top_cats) == 5:
            break

    gradients = [
        ('Activity Weight ($w_a$)', lambda w: (w, (1-w)/2, (1-w)/2)),
        ('Stability Weight ($w_s$)', lambda w: ((1-w)/2, w, (1-w)/2)),
        ('Cost Weight ($w_c$)', lambda w: ((1-w)/2, (1-w)/2, w)),
    ]

    for ax_idx, (xlabel, weight_func) in enumerate(gradients):
        ax = axes[ax_idx]

        # Collect ranks for each catalyst
        cat_ranks = {cat: [] for cat in top_cats}

        for w in weight_range:
            w_a, w_s, w_c = weight_func(w)
            df = calc.calculate_asci(w_a=w_a, w_s=w_s, w_c=w_c, show_progress=False)

            # Get ranks - use unique symbols only
            seen_syms = set()
            rank_dict = {}
            rank = 1
            for _, row in df.iterrows():
                sym = row['symbol']
                if sym not in seen_syms:
                    seen_syms.add(sym)
                    rank_dict[sym] = rank
                    rank += 1

            for cat in top_cats:
                cat_ranks[cat].append(rank_dict.get(cat, 20))

        # Plot
        for i, cat in enumerate(top_cats):
            ax.plot(weight_range, cat_ranks[cat], '-o', color=COLORS[i],
                    linewidth=2, markersize=5, markevery=2,
                    label=cat if ax_idx == 2 else None)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Rank' if ax_idx == 0 else '')
        ax.set_xlim(0.15, 0.7)
        ax.set_ylim(0.5, 15)
        ax.invert_yaxis()  # Rank 1 at top
        ax.axhline(y=5, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.axhline(y=10, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)
        ax.set_yticks([1, 5, 10, 15])
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_title(f'{"abc"[ax_idx]}', loc='left', fontweight='bold')

    axes[2].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)

    fig.suptitle(f'Rank Sensitivity to Weight Changes{title_suffix}', fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'fig4_rank_trajectories.png', dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'fig4_rank_trajectories.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved fig4_rank_trajectories")


# =============================================================================
# MAIN
# =============================================================================

def run_sensitivity_for_reaction(reaction, data_file, subfolder_name, pathway=None):
    """Run complete sensitivity analysis for one reaction."""

    base_dir = Path(__file__).parent.parent
    data_path = base_dir / 'data' / data_file

    # Single parent folder with subfolders for each reaction
    output_dir = base_dir / 'results' / 'sensitivity_analysis' / subfolder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Sensitivity Analysis: {reaction}" + (f" ({pathway})" if pathway else ""))
    print("="*70)

    # Initialize
    if pathway:
        calc = ASCICalculator(reaction=reaction, pathway=pathway, verbose=False)
        title_suffix = f' - {reaction} ({pathway})'
    else:
        calc = ASCICalculator(reaction=reaction, verbose=False)
        title_suffix = f' - {reaction}'

    calc.load_data(str(data_path))

    # Generate weights
    weights = generate_weights(n_points=21, min_weight=0.05)
    print(f"  Weight configurations: {len(weights)}")

    # Run analysis
    results = run_full_analysis(calc, weights)

    # Create figures
    create_fig1_ternary_asci(results, output_dir, title_suffix)
    create_fig2_catalyst_dominance(results, output_dir, title_suffix)
    create_fig3_consistency(results, output_dir, title_suffix)
    create_fig4_rank_trajectories(calc, output_dir, title_suffix)

    print(f"\n  Output: {output_dir}")
    return results


def main():
    print("="*70)
    print("COMPLETE SENSITIVITY ANALYSIS")
    print("="*70)

    # HER
    run_sensitivity_for_reaction('HER', 'HER_clean.csv', 'HER')

    # CO2RR - all pathways
    for pathway in ['CO', 'CHO', 'COCOH']:
        run_sensitivity_for_reaction('CO2RR', f'CO2RR_{pathway}_clean.csv',
                                     f'CO2RR_{pathway}', pathway)

    print("\n" + "="*70)
    print("ALL SENSITIVITY ANALYSES COMPLETE")
    print("="*70)
    print("\nOutput structure:")
    print("  results/sensitivity_analysis/")
    print("    ├── HER/")
    print("    ├── CO2RR_CO/")
    print("    ├── CO2RR_CHO/")
    print("    └── CO2RR_COCOH/")
    print("\nFigures in each folder:")
    print("  - fig1_ternary_asci: ASCI score landscape")
    print("  - fig2_catalyst_dominance: Which catalyst wins at each weight point")
    print("  - fig3_consistency: Catalysts that stay top-ranked")
    print("  - fig4_rank_trajectories: How ranks change with weight gradients")


if __name__ == '__main__':
    main()
