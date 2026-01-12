#!/usr/bin/env python3
"""
Example 5: ASCICat vs Pareto Frontier Comparison
=================================================

Reaction: HER (Hydrogen Evolution Reaction)

This example demonstrates that ASCICat's weighted sum approach is COMPLEMENTARY
to Pareto frontier methods rather than contradictory.

Key Advantages of ASCICat's Deterministic Ranking:
1. Reproducible rankings across studies and research groups
2. Direct comparability for systematic catalyst development
3. Interpretable weight-based prioritization
4. Pareto-compatible: top ASCICat catalysts are typically Pareto-optimal

Figures Generated:
- Fig 1: Side-by-side 3D comparison (ASCICat vs Pareto)
- Fig 2: Overlay visualization showing complementarity
- Fig 3: Pareto membership analysis of top ASCICat catalysts
- Fig 4: Ranking reproducibility demonstration
- Fig 5: Combined figure (4 panels)

Author: Nabil Khossossi
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from ascicat import ASCICalculator
from ascicat.sampling import sample_diverse_3d


# ============================================================================
# PARETO FRONTIER IDENTIFICATION
# ============================================================================

def identify_pareto_front(df: pd.DataFrame) -> np.ndarray:
    """
    Identify Pareto-optimal solutions (non-dominated catalysts).

    A catalyst is Pareto-optimal if no other catalyst is strictly better
    in ALL three objectives simultaneously.

    Returns:
        Boolean mask indicating Pareto-optimal catalysts
    """
    scores = df[['activity_score', 'stability_score', 'cost_score']].values
    n = len(scores)
    pareto_mask = np.ones(n, dtype=bool)

    for i in range(n):
        if not pareto_mask[i]:
            continue
        for j in range(n):
            if i != j and pareto_mask[j]:
                # Check if j dominates i (j >= i in all, j > i in at least one)
                if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                    pareto_mask[i] = False
                    break

    return pareto_mask


def compute_pareto_statistics(df: pd.DataFrame, pareto_mask: np.ndarray) -> dict:
    """Compute statistics about Pareto front."""
    pareto_df = df[pareto_mask]
    non_pareto_df = df[~pareto_mask]

    return {
        'n_total': len(df),
        'n_pareto': pareto_mask.sum(),
        'pareto_fraction': pareto_mask.sum() / len(df),
        'pareto_mean_asci': pareto_df['ASCI'].mean(),
        'pareto_std_asci': pareto_df['ASCI'].std(),
        'non_pareto_mean_asci': non_pareto_df['ASCI'].mean() if len(non_pareto_df) > 0 else 0,
        'pareto_min_asci': pareto_df['ASCI'].min(),
        'pareto_max_asci': pareto_df['ASCI'].max(),
    }


# ============================================================================
# FIGURE 1: SIDE-BY-SIDE 3D COMPARISON
# ============================================================================

def plot_side_by_side_comparison(df: pd.DataFrame, pareto_mask: np.ndarray,
                                  output_dir: Path, dpi: int = 600):
    """
    Side-by-side 3D comparison: ASCICat ranking vs Pareto frontier.
    """
    fig = plt.figure(figsize=(14, 6))

    # Panel A: ASCICat Approach (color by ASCI score)
    ax1 = fig.add_subplot(121, projection='3d')

    scatter1 = ax1.scatter(
        df['activity_score'],
        df['stability_score'],
        df['cost_score'],
        c=df['ASCI'],
        cmap='viridis',
        s=25,
        alpha=0.7,
        edgecolors='none'
    )

    # Highlight top 10 ASCICat
    top10 = df.nlargest(10, 'ASCI')
    ax1.scatter(
        top10['activity_score'],
        top10['stability_score'],
        top10['cost_score'],
        s=150, marker='*', c='red',
        edgecolors='darkred', linewidths=1,
        label='Top 10 ASCI', zorder=10
    )

    ax1.set_xlabel('Activity Score', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Stability Score', fontweight='bold', fontsize=10)
    ax1.set_zlabel('Cost Score', fontweight='bold', fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 1)
    ax1.set_title('(A) ASCICat: Deterministic Ranking', fontweight='bold', fontsize=12)
    ax1.view_init(elev=25, azim=45)
    ax1.legend(loc='upper left', fontsize=9)

    cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.6, pad=0.1)
    cbar1.set_label('ASCI Score', fontweight='bold')
    cbar1.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    # Panel B: Pareto Frontier Approach
    ax2 = fig.add_subplot(122, projection='3d')

    # Non-Pareto points (gray)
    non_pareto = df[~pareto_mask]
    ax2.scatter(
        non_pareto['activity_score'],
        non_pareto['stability_score'],
        non_pareto['cost_score'],
        c='lightgray', s=15, alpha=0.3,
        label=f'Dominated ({len(non_pareto)})'
    )

    # Pareto front (colored by ASCI)
    pareto_df = df[pareto_mask]
    scatter2 = ax2.scatter(
        pareto_df['activity_score'],
        pareto_df['stability_score'],
        pareto_df['cost_score'],
        c=pareto_df['ASCI'],
        cmap='plasma',
        s=50,
        edgecolors='black',
        linewidths=0.3,
        label=f'Pareto Front ({len(pareto_df)})'
    )

    ax2.set_xlabel('Activity Score', fontweight='bold', fontsize=10)
    ax2.set_ylabel('Stability Score', fontweight='bold', fontsize=10)
    ax2.set_zlabel('Cost Score', fontweight='bold', fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_zlim(0, 1)
    ax2.set_title('(B) Pareto Frontier: Non-Dominated Set', fontweight='bold', fontsize=12)
    ax2.view_init(elev=25, azim=45)
    ax2.legend(loc='upper left', fontsize=9)

    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.6, pad=0.1)
    cbar2.set_label('ASCI Score', fontweight='bold')
    cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    plt.tight_layout()
    fig.savefig(output_dir / 'fig1_side_by_side_comparison.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(output_dir / 'fig1_side_by_side_comparison.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig1_side_by_side_comparison.png/pdf")


# ============================================================================
# FIGURE 2: OVERLAY VISUALIZATION - COMPLEMENTARITY
# ============================================================================

def plot_overlay_complementarity(df: pd.DataFrame, pareto_mask: np.ndarray,
                                  output_dir: Path, dpi: int = 600):
    """
    Overlay visualization showing how ASCICat and Pareto complement each other.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Classify catalysts into 4 categories
    top_n = 50  # Top N by ASCI
    top_asci_mask = df['ASCI'].rank(ascending=False) <= top_n

    # Category 1: Top ASCI AND Pareto (ideal candidates)
    ideal = top_asci_mask & pareto_mask
    # Category 2: Top ASCI but NOT Pareto (high scoring but dominated)
    high_asci_only = top_asci_mask & ~pareto_mask
    # Category 3: Pareto but NOT top ASCI (trade-off solutions)
    pareto_only = pareto_mask & ~top_asci_mask
    # Category 4: Neither (dominated, low ASCI)
    neither = ~pareto_mask & ~top_asci_mask

    # Plot each category
    if neither.sum() > 0:
        ax.scatter(
            df[neither]['activity_score'],
            df[neither]['stability_score'],
            df[neither]['cost_score'],
            c='lightgray', s=10, alpha=0.2,
            label=f'Other ({neither.sum()})'
        )

    if pareto_only.sum() > 0:
        ax.scatter(
            df[pareto_only]['activity_score'],
            df[pareto_only]['stability_score'],
            df[pareto_only]['cost_score'],
            c='#3498db', s=40, alpha=0.7,
            edgecolors='darkblue', linewidths=0.3,
            label=f'Pareto only ({pareto_only.sum()})'
        )

    if high_asci_only.sum() > 0:
        ax.scatter(
            df[high_asci_only]['activity_score'],
            df[high_asci_only]['stability_score'],
            df[high_asci_only]['cost_score'],
            c='#e67e22', s=50, alpha=0.8,
            edgecolors='darkorange', linewidths=0.3,
            marker='s',
            label=f'High ASCI only ({high_asci_only.sum()})'
        )

    if ideal.sum() > 0:
        ax.scatter(
            df[ideal]['activity_score'],
            df[ideal]['stability_score'],
            df[ideal]['cost_score'],
            c='#27ae60', s=100, alpha=0.9,
            edgecolors='darkgreen', linewidths=1,
            marker='*',
            label=f'Ideal: Top ASCI + Pareto ({ideal.sum()})'
        )

    ax.set_xlabel('Activity Score', fontweight='bold', fontsize=11)
    ax.set_ylabel('Stability Score', fontweight='bold', fontsize=11)
    ax.set_zlabel('Cost Score', fontweight='bold', fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_title('Complementarity: ASCICat + Pareto Frontier',
                fontweight='bold', fontsize=13)
    ax.view_init(elev=25, azim=45)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig2_overlay_complementarity.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(output_dir / 'fig2_overlay_complementarity.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig2_overlay_complementarity.png/pdf")

    # Return statistics
    return {
        'ideal': ideal.sum(),
        'high_asci_only': high_asci_only.sum(),
        'pareto_only': pareto_only.sum(),
        'neither': neither.sum(),
        'top_n': top_n
    }


# ============================================================================
# FIGURE 3: PARETO MEMBERSHIP ANALYSIS
# ============================================================================

def plot_pareto_membership_analysis(df: pd.DataFrame, pareto_mask: np.ndarray,
                                     output_dir: Path, dpi: int = 600):
    """
    Analyze what fraction of top ASCI catalysts are Pareto-optimal.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Pareto membership by ASCI rank
    top_ns = [5, 10, 20, 30, 50, 100]
    pareto_fractions = []

    for n in top_ns:
        top_n_mask = df['ASCI'].rank(ascending=False) <= n
        pareto_in_top = (top_n_mask & pareto_mask).sum()
        pareto_fractions.append(pareto_in_top / n * 100)

    ax1 = axes[0]
    bars = ax1.bar(range(len(top_ns)), pareto_fractions,
                   color='#27ae60', edgecolor='black', linewidth=0.5)
    ax1.set_xticks(range(len(top_ns)))
    ax1.set_xticklabels([f'Top {n}' for n in top_ns], fontsize=10)
    ax1.set_ylabel('Pareto-Optimal (%)', fontweight='bold', fontsize=11)
    ax1.set_xlabel('ASCI Ranking Tier', fontweight='bold', fontsize=11)
    ax1.set_title('(A) Top ASCI Catalysts in Pareto Front', fontweight='bold', fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100% Pareto')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, pareto_fractions):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel B: ASCI distribution for Pareto vs Non-Pareto
    ax2 = axes[1]

    pareto_asci = df[pareto_mask]['ASCI']
    non_pareto_asci = df[~pareto_mask]['ASCI']

    bins = np.linspace(0, 1, 25)
    ax2.hist(non_pareto_asci, bins=bins, alpha=0.6, color='gray',
             label=f'Dominated (n={len(non_pareto_asci)})', edgecolor='black', linewidth=0.3)
    ax2.hist(pareto_asci, bins=bins, alpha=0.8, color='#27ae60',
             label=f'Pareto-Optimal (n={len(pareto_asci)})', edgecolor='black', linewidth=0.3)

    ax2.axvline(pareto_asci.mean(), color='darkgreen', linestyle='-', linewidth=2,
               label=f'Pareto Mean = {pareto_asci.mean():.3f}')
    ax2.axvline(non_pareto_asci.mean(), color='dimgray', linestyle='--', linewidth=2,
               label=f'Dominated Mean = {non_pareto_asci.mean():.3f}')

    ax2.set_xlabel('ASCI Score', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Count', fontweight='bold', fontsize=11)
    ax2.set_title('(B) ASCI Distribution by Pareto Status', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig3_pareto_membership_analysis.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(output_dir / 'fig3_pareto_membership_analysis.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig3_pareto_membership_analysis.png/pdf")

    return pareto_fractions


# ============================================================================
# FIGURE 4: RANKING REPRODUCIBILITY DEMONSTRATION
# ============================================================================

def plot_ranking_reproducibility(df: pd.DataFrame, output_dir: Path, dpi: int = 600):
    """
    Demonstrate deterministic ranking advantage for reproducibility.
    Shows how ASCI provides unique, reproducible rankings vs Pareto's equal status.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Get top 20 catalysts
    top20 = df.nlargest(20, 'ASCI').copy()
    top20['rank'] = range(1, 21)

    # Panel A: Deterministic ranking with score differences
    ax1 = axes[0]

    colors = plt.cm.viridis(top20['ASCI'] / top20['ASCI'].max())
    bars = ax1.barh(range(20), top20['ASCI'], color=colors, edgecolor='black', linewidth=0.5)

    ax1.set_yticks(range(20))
    ax1.set_yticklabels([f"{r}. {s}" for r, s in zip(top20['rank'], top20['symbol'])],
                        fontsize=9)
    ax1.set_xlabel('ASCI Score', fontweight='bold', fontsize=11)
    ax1.set_title('(A) ASCICat: Deterministic Ranking\n(Reproducible across studies)',
                 fontweight='bold', fontsize=11)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.05)

    # Add score values
    for i, (bar, asci) in enumerate(zip(bars, top20['ASCI'])):
        ax1.text(asci + 0.02, i, f'{asci:.3f}', va='center', fontsize=8)

    ax1.grid(axis='x', alpha=0.3)

    # Panel B: Pareto comparison - shows all Pareto catalysts have "equal" status
    ax2 = axes[1]

    # Identify Pareto among top 20
    pareto_mask = identify_pareto_front(df)
    top20_pareto = pareto_mask[top20.index]

    # Create visual showing Pareto "ranking problem"
    y_pos = range(20)
    pareto_colors = ['#27ae60' if p else '#e74c3c' for p in top20_pareto]

    # All Pareto-optimal get same "rank" visually
    pareto_scores = [1.0 if p else 0.0 for p in top20_pareto]

    bars2 = ax2.barh(y_pos, pareto_scores, color=pareto_colors,
                     edgecolor='black', linewidth=0.5, alpha=0.7)

    ax2.set_yticks(range(20))
    ax2.set_yticklabels([f"{s}" for s in top20['symbol']], fontsize=9)
    ax2.set_xlabel('Pareto Status', fontweight='bold', fontsize=11)
    ax2.set_title('(B) Pareto: No Ranking Among Non-Dominated\n(All equally "optimal")',
                 fontweight='bold', fontsize=11)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.2)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Dominated', 'Pareto-Optimal'])

    # Add status labels
    for i, (is_pareto, symbol) in enumerate(zip(top20_pareto, top20['symbol'])):
        status = "Pareto" if is_pareto else "Dominated"
        ax2.text(1.05 if is_pareto else 0.05, i, status, va='center', fontsize=8,
                color='darkgreen' if is_pareto else 'darkred')

    # Add legend
    legend_elements = [
        Patch(facecolor='#27ae60', edgecolor='black', label='Pareto-Optimal'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Dominated')
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig4_ranking_reproducibility.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(output_dir / 'fig4_ranking_reproducibility.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig4_ranking_reproducibility.png/pdf")


# ============================================================================
# FIGURE 5: COMBINED FIGURE (4 PANELS)
# ============================================================================

def plot_combined_figure(df: pd.DataFrame, pareto_mask: np.ndarray,
                                      output_dir: Path, dpi: int = 600):
    """
    Combined 4-panel figure summarizing ASCICat vs Pareto comparison.
    """
    fig = plt.figure(figsize=(14, 12))

    # Panel A: 3D ASCICat view
    ax1 = fig.add_subplot(221, projection='3d')

    scatter1 = ax1.scatter(
        df['activity_score'],
        df['stability_score'],
        df['cost_score'],
        c=df['ASCI'],
        cmap='viridis',
        s=20,
        alpha=0.6
    )

    top5 = df.nlargest(5, 'ASCI')
    ax1.scatter(
        top5['activity_score'],
        top5['stability_score'],
        top5['cost_score'],
        s=200, marker='*', c='red',
        edgecolors='darkred', linewidths=1,
        zorder=10
    )

    ax1.set_xlabel('Activity', fontweight='bold', fontsize=9)
    ax1.set_ylabel('Stability', fontweight='bold', fontsize=9)
    ax1.set_zlabel('Cost', fontweight='bold', fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 1)
    ax1.set_title('(A) ASCICat 3D Score Space', fontweight='bold', fontsize=11)
    ax1.view_init(elev=25, azim=45)

    cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.5, pad=0.1)
    cbar1.set_label('ASCI', fontsize=9)

    # Panel B: 3D Pareto view
    ax2 = fig.add_subplot(222, projection='3d')

    non_pareto = df[~pareto_mask]
    pareto_df = df[pareto_mask]

    ax2.scatter(
        non_pareto['activity_score'],
        non_pareto['stability_score'],
        non_pareto['cost_score'],
        c='lightgray', s=10, alpha=0.2
    )

    scatter2 = ax2.scatter(
        pareto_df['activity_score'],
        pareto_df['stability_score'],
        pareto_df['cost_score'],
        c=pareto_df['ASCI'],
        cmap='plasma',
        s=40,
        edgecolors='black',
        linewidths=0.3
    )

    ax2.set_xlabel('Activity', fontweight='bold', fontsize=9)
    ax2.set_ylabel('Stability', fontweight='bold', fontsize=9)
    ax2.set_zlabel('Cost', fontweight='bold', fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_zlim(0, 1)
    ax2.set_title(f'(B) Pareto Front ({len(pareto_df)} catalysts)', fontweight='bold', fontsize=11)
    ax2.view_init(elev=25, azim=45)

    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.5, pad=0.1)
    cbar2.set_label('ASCI', fontsize=9)

    # Panel C: Pareto membership by ASCI rank
    ax3 = fig.add_subplot(223)

    top_ns = [5, 10, 20, 30, 50, 100]
    pareto_fractions = []
    for n in top_ns:
        top_n_mask = df['ASCI'].rank(ascending=False) <= n
        pareto_in_top = (top_n_mask & pareto_mask).sum()
        pareto_fractions.append(pareto_in_top / n * 100)

    bars = ax3.bar(range(len(top_ns)), pareto_fractions,
                   color='#27ae60', edgecolor='black', linewidth=0.5)
    ax3.set_xticks(range(len(top_ns)))
    ax3.set_xticklabels([f'Top {n}' for n in top_ns], fontsize=9)
    ax3.set_ylabel('Pareto-Optimal (%)', fontweight='bold', fontsize=10)
    ax3.set_xlabel('ASCI Ranking Tier', fontweight='bold', fontsize=10)
    ax3.set_title('(C) Top ASCI Catalysts in Pareto Front', fontweight='bold', fontsize=11)
    ax3.set_ylim(0, 110)
    ax3.axhline(y=100, color='red', linestyle='--', alpha=0.5)
    ax3.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, pareto_fractions):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Panel D: Top 15 ranking comparison
    ax4 = fig.add_subplot(224)

    top15 = df.nlargest(15, 'ASCI').copy()
    top15_pareto = pareto_mask[top15.index]

    y_pos = range(15)
    colors = ['#27ae60' if p else '#95a5a6' for p in top15_pareto]

    bars4 = ax4.barh(y_pos, top15['ASCI'], color=colors, edgecolor='black', linewidth=0.5)

    ax4.set_yticks(range(15))
    ax4.set_yticklabels([f"{i+1}. {s}" for i, s in enumerate(top15['symbol'])], fontsize=9)
    ax4.set_xlabel('ASCI Score', fontweight='bold', fontsize=10)
    ax4.set_title('(D) Top 15 with Pareto Status', fontweight='bold', fontsize=11)
    ax4.invert_yaxis()
    ax4.set_xlim(0, 1.1)
    ax4.grid(axis='x', alpha=0.3)

    # Add Pareto indicators
    for i, (is_pareto, asci) in enumerate(zip(top15_pareto, top15['ASCI'])):
        marker = "★" if is_pareto else ""
        ax4.text(asci + 0.02, i, f'{asci:.3f} {marker}', va='center', fontsize=8)

    legend_elements = [
        Patch(facecolor='#27ae60', edgecolor='black', label='Pareto-Optimal'),
        Patch(facecolor='#95a5a6', edgecolor='black', label='Dominated')
    ]
    ax4.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig5_combined.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(output_dir / 'fig5_combined.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig5_combined.png/pdf")


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def print_summary_statistics(df: pd.DataFrame, pareto_mask: np.ndarray,
                              complement_stats: dict, pareto_fractions: list):
    """Print comprehensive summary statistics."""

    pareto_stats = compute_pareto_statistics(df, pareto_mask)

    print("\n" + "="*70)
    print("SUMMARY: ASCICat vs Pareto Comparison")
    print("="*70)

    print("\n1. DATASET OVERVIEW")
    print("-"*40)
    print(f"   Total catalysts analyzed: {pareto_stats['n_total']}")
    print(f"   Pareto-optimal catalysts: {pareto_stats['n_pareto']} ({100*pareto_stats['pareto_fraction']:.1f}%)")

    print("\n2. ASCI SCORE STATISTICS")
    print("-"*40)
    print(f"   Pareto catalysts mean ASCI: {pareto_stats['pareto_mean_asci']:.3f} ± {pareto_stats['pareto_std_asci']:.3f}")
    print(f"   Dominated catalysts mean ASCI: {pareto_stats['non_pareto_mean_asci']:.3f}")
    print(f"   Pareto ASCI range: [{pareto_stats['pareto_min_asci']:.3f}, {pareto_stats['pareto_max_asci']:.3f}]")

    print("\n3. COMPLEMENTARITY ANALYSIS (Top 50 by ASCI)")
    print("-"*40)
    print(f"   Ideal candidates (Top ASCI + Pareto): {complement_stats['ideal']}")
    print(f"   High ASCI only (dominated): {complement_stats['high_asci_only']}")
    print(f"   Pareto only (not in top 50): {complement_stats['pareto_only']}")

    print("\n4. PARETO MEMBERSHIP BY ASCI TIER")
    print("-"*40)
    top_ns = [5, 10, 20, 30, 50, 100]
    for n, frac in zip(top_ns, pareto_fractions):
        print(f"   Top {n:3d} ASCI → {frac:5.1f}% Pareto-optimal")

    print("\n5. KEY INSIGHTS")
    print("-"*40)
    print("   ✓ Top ASCI catalysts are predominantly Pareto-optimal")
    print("   ✓ ASCICat provides deterministic ranking within Pareto set")
    print("   ✓ Methods are complementary, not contradictory")
    print("   ✓ Reproducible rankings enable cross-study comparisons")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Run ASCICat vs Pareto comparison analysis."""

    print("="*70)
    print("ASCICat vs Pareto Frontier Comparison")
    print("Reaction: HER (Hydrogen Evolution Reaction)")
    print("="*70)

    # Setup paths
    data_path = Path(__file__).parent.parent / 'data' / 'HER_clean.csv'
    output_dir = Path(__file__).parent.parent / 'results' / 'ascicat_vs_pareto'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize calculator
    print("\n" + "="*70)
    print("Loading Data and Calculating ASCI Scores")
    print("="*70)

    calc = ASCICalculator(reaction='HER', verbose=True)
    calc.load_data(str(data_path))

    # Calculate ASCI with equal weights
    results = calc.calculate_asci(w_a=0.33, w_s=0.33, w_c=0.34)

    # Sample for visualization if large dataset
    print("\nPreparing data for visualization...")
    if len(results) > 1000:
        sampled = sample_diverse_3d(results, n_samples=1000)
        print(f"  Sampled {len(sampled)} catalysts for visualization")
    else:
        sampled = results.copy()

    # Identify Pareto front
    print("\n" + "="*70)
    print("Identifying Pareto Frontier")
    print("="*70)

    pareto_mask = identify_pareto_front(sampled)
    n_pareto = pareto_mask.sum()
    print(f"\nPareto-optimal catalysts: {n_pareto} / {len(sampled)} ({100*n_pareto/len(sampled):.1f}%)")

    # Generate figures
    print("\n" + "="*70)
    print("Generating Figures")
    print("="*70)

    # Figure 1: Side-by-side comparison
    print("\nFigure 1: Side-by-side 3D comparison...")
    plot_side_by_side_comparison(sampled, pareto_mask, output_dir)

    # Figure 2: Overlay complementarity
    print("\nFigure 2: Overlay complementarity visualization...")
    complement_stats = plot_overlay_complementarity(sampled, pareto_mask, output_dir)

    # Figure 3: Pareto membership analysis
    print("\nFigure 3: Pareto membership analysis...")
    pareto_fractions = plot_pareto_membership_analysis(sampled, pareto_mask, output_dir)

    # Figure 4: Ranking reproducibility
    print("\nFigure 4: Ranking reproducibility demonstration...")
    plot_ranking_reproducibility(sampled, output_dir)

    # Figure 5: Combined figure
    print("\nFigure 5: Combined figure...")
    plot_combined_figure(sampled, pareto_mask, output_dir)

    # Print summary statistics
    print_summary_statistics(sampled, pareto_mask, complement_stats, pareto_fractions)

    # Export data
    print("\n" + "="*70)
    print("Exporting Results")
    print("="*70)

    # Save Pareto analysis results
    sampled_copy = sampled.copy()
    sampled_copy['is_pareto'] = pareto_mask
    sampled_copy.to_csv(output_dir / 'ascicat_pareto_comparison.csv', index=False)
    print(f"  Saved: ascicat_pareto_comparison.csv")

    # Save Pareto front only
    pareto_df = sampled[pareto_mask].sort_values('ASCI', ascending=False)
    pareto_df.to_csv(output_dir / 'pareto_front_catalysts.csv', index=False)
    print(f"  Saved: pareto_front_catalysts.csv ({len(pareto_df)} catalysts)")

    print("\n" + "="*70)
    print("Analysis Complete!")
    print(f"Results saved to: {output_dir}/")
    print("="*70)

    print("\n" + "="*70)
    print("KEY TAKEAWAY")
    print("="*70)
    print("""
ASCICat's weighted sum approach and Pareto frontier methods are
COMPLEMENTARY, not contradictory:

• Pareto identifies the set of non-dominated trade-off solutions
• ASCICat provides deterministic ranking WITHIN this set
• Top ASCI catalysts are predominantly Pareto-optimal
• Deterministic ranking enables reproducible cross-study comparisons

This complementarity is a critical advantage for systematic
catalyst development programs.
""")


if __name__ == '__main__':
    main()
