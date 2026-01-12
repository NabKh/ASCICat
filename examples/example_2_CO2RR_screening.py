#!/usr/bin/env python3
"""
Example: CO2RR (CO2 Reduction Reaction) Multi-Pathway Screening
================================================================

Complete example for screening CO2RR electrocatalysts across multiple pathways.

CO2RR Pathways:
- CO pathway:    CO2 -> CO + H2O      (ΔE_opt = -0.67 eV)
- CHO pathway:   CO2 -> CH3OH         (ΔE_opt = -0.48 eV)
- COCOH pathway: CO2 -> HCOOH         (ΔE_opt = -0.32 eV)

Data Files (separate per pathway):
- data/CO2RR_CO_clean.csv
- data/CO2RR_CHO_clean.csv
- data/CO2RR_COCOH_clean.csv

Author: Nabil Khossossi
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ascicat import ASCICalculator
from ascicat.visualizer import Visualizer


# CO2RR pathway configurations
PATHWAYS = {
    'CO': {
        'pathway': 'CO',
        'optimal_energy': -0.67,
        'product': 'Carbon monoxide',
        'description': 'CO2 → CO + H2O',
        'data_file': 'CO2RR_CO_clean.csv'
    },
    'CHO': {
        'pathway': 'CHO',
        'optimal_energy': -0.48,
        'product': 'Methanol',
        'description': 'CO2 → CH3OH',
        'data_file': 'CO2RR_CHO_clean.csv'
    },
    'COCOH': {
        'pathway': 'COCOH',
        'optimal_energy': -0.32,
        'product': 'Formic acid',
        'description': 'CO2 → HCOOH',
        'data_file': 'CO2RR_COCOH_clean.csv'
    }
}


def analyze_pathway(pathway_name: str, pathway_info: dict, data_dir: Path,
                   output_base: Path) -> pd.DataFrame:
    """
    Analyze a single CO2RR pathway.
    """
    print(f"\n{'='*70}")
    print(f"CO2RR {pathway_name} Pathway: {pathway_info['description']}")
    print(f"Optimal ΔE: {pathway_info['optimal_energy']:.2f} eV")
    print(f"{'='*70}")

    # Check data file
    data_file = data_dir / pathway_info['data_file']
    if not data_file.exists():
        print(f"  WARNING: Data file not found: {data_file}")
        return None

    # Initialize calculator with CO2RR reaction and specific pathway
    calc = ASCICalculator(
        reaction='CO2RR',
        pathway=pathway_info['pathway'],
        scoring_method='linear',
        verbose=True
    )

    # Load pathway-specific data
    calc.load_data(str(data_file))

    # Calculate ASCI scores
    results = calc.calculate_asci(
        w_a=0.33,
        w_s=0.33,
        w_c=0.34
    )

    # Display top 10
    print(f"\nTop 10 {pathway_name} Pathway Catalysts:")
    print("-"*70)
    print(f"{'Rank':<6} {'Catalyst':<12} {'ASCI':<8} {'Activity':<10} "
          f"{'Stability':<10} {'Cost':<10}")
    print("-"*70)

    top10 = calc.get_top_catalysts(n=10)
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"{i:<6} {row['symbol']:<12} {row['ASCI']:.3f}    "
              f"{row['activity_score']:.3f}      {row['stability_score']:.3f}       "
              f"{row['cost_score']:.3f}")

    # Generate figures
    output_dir = output_base / f'CO2RR_{pathway_name}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating figures for {pathway_name} pathway...")

    # Create visualizer (handles large dataset sampling automatically)
    viz = Visualizer(results, calc.config, n_samples=1000)

    # Generate all outputs
    viz.generate_all_outputs(
        output_dir=str(output_dir),
        dpi=600,
        include_interactive=True
    )

    # Export results
    results.to_csv(output_dir / f'CO2RR_{pathway_name}_results.csv', index=False)
    top10.to_csv(output_dir / f'CO2RR_{pathway_name}_top10.csv', index=False)

    return results


def cross_pathway_comparison(results_dict: dict, output_dir: Path):
    """
    Compare top catalysts across all CO2RR pathways.
    """
    print(f"\n{'='*70}")
    print("Cross-Pathway Comparison")
    print(f"{'='*70}")

    # Get top 30 from each pathway
    top_catalysts = {}
    for pathway, results in results_dict.items():
        if results is None:
            continue
        top30 = results.nlargest(30, 'ASCI')
        for _, row in top30.iterrows():
            symbol = row['symbol']
            if symbol not in top_catalysts:
                top_catalysts[symbol] = {}
            top_catalysts[symbol][pathway] = row['ASCI']

    # Find catalysts appearing in multiple pathways
    versatile = []
    for symbol, scores in top_catalysts.items():
        if len(scores) >= 2:
            avg_score = np.mean(list(scores.values()))
            versatile.append({
                'symbol': symbol,
                'pathways': len(scores),
                'avg_asci': avg_score,
                **{f'{p}_asci': scores.get(p, np.nan) for p in PATHWAYS.keys()}
            })

    if versatile:
        versatile_df = pd.DataFrame(versatile)
        versatile_df = versatile_df.sort_values('avg_asci', ascending=False)

        print("\nVersatile Catalysts (Top in 2+ pathways):")
        print("-"*80)
        print(f"{'Catalyst':<12} {'Pathways':<10} {'Avg ASCI':<10} "
              f"{'CO':<10} {'CHO':<10} {'COCOH':<10}")
        print("-"*80)

        for _, row in versatile_df.head(10).iterrows():
            co_val = row.get('CO_asci', np.nan)
            cho_val = row.get('CHO_asci', np.nan)
            cocoh_val = row.get('COCOH_asci', np.nan)
            print(f"{row['symbol']:<12} {row['pathways']:<10} {row['avg_asci']:.3f}      "
                  f"{co_val:.3f}      {cho_val:.3f}      {cocoh_val:.3f}")

        # Save versatile catalysts
        versatile_df.to_csv(output_dir / 'versatile_catalysts.csv', index=False)

        # Create comparison figure
        _plot_pathway_comparison(versatile_df, output_dir)

        return versatile_df

    return None


def _plot_pathway_comparison(versatile_df: pd.DataFrame, output_dir: Path):
    """Create pathway comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    top15 = versatile_df.head(15)
    x = np.arange(len(top15))
    width = 0.25

    colors = ['#2ecc71', '#3498db', '#e67e22']

    for i, (pathway, color) in enumerate(zip(['CO', 'CHO', 'COCOH'], colors)):
        col = f'{pathway}_asci'
        if col in top15.columns:
            values = top15[col].fillna(0)
            ax.bar(x + (i-1)*width, values, width, label=pathway,
                   color=color, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(top15['symbol'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('ASCI Score', fontsize=11, fontweight='bold')
    ax.set_xlabel('Catalyst', fontsize=11, fontweight='bold')
    ax.set_title('Versatile Catalysts: Performance Across CO2RR Pathways',
                fontsize=12, fontweight='bold')
    ax.legend(title='Pathway', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'pathway_comparison.png', dpi=600, bbox_inches='tight')
    fig.savefig(output_dir / 'pathway_comparison.pdf', dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: pathway_comparison.png/pdf")


def main():
    """Run CO2RR multi-pathway screening."""

    print("="*70)
    print("ASCICat: CO2RR Multi-Pathway Catalyst Screening")
    print("="*70)

    data_dir = Path(__file__).parent.parent / 'data'
    output_base = Path(__file__).parent.parent / 'results' / 'CO2RR_results'
    output_base.mkdir(parents=True, exist_ok=True)

    # Check available data files
    print("\nChecking data files:")
    for pathway_name, info in PATHWAYS.items():
        data_file = data_dir / info['data_file']
        status = "✓" if data_file.exists() else "✗"
        print(f"  {status} {info['data_file']}")

    # Analyze each pathway
    results_dict = {}
    for pathway_name, pathway_info in PATHWAYS.items():
        try:
            results = analyze_pathway(
                pathway_name=pathway_name,
                pathway_info=pathway_info,
                data_dir=data_dir,
                output_base=output_base
            )
            if results is not None:
                results_dict[pathway_name] = results
        except Exception as e:
            print(f"\nError analyzing {pathway_name} pathway: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Cross-pathway comparison
    if len(results_dict) > 1:
        cross_pathway_comparison(results_dict, output_base)

    # Summary
    print(f"\n{'='*70}")
    print("CO2RR SCREENING COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_base}/")
    print(f"\nPathways analyzed: {len(results_dict)}")
    for pathway, results in results_dict.items():
        best = results.iloc[0]
        print(f"  - {pathway}: Best = {best['symbol']} (ASCI={best['ASCI']:.3f})")


if __name__ == '__main__':
    main()
