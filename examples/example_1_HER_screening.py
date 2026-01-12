#!/usr/bin/env python3
"""
Example: HER (Hydrogen Evolution Reaction) Catalyst Screening
=============================================================

Complete example for screening HER electrocatalysts using ASCICat.

This script demonstrates:
1. Loading HER catalyst data
2. Calculating ASCI scores with equal weights
3. Generating high-resolution figures (4 panels)
4. Creating interactive 3D visualization
5. Exporting results

Optimal Energy: ΔE_opt = -0.27 eV (Sabatier principle)

Author: Nabil Khossossi
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ascicat import ASCICalculator
from ascicat.visualizer import Visualizer


def main():
    """Run HER catalyst screening analysis."""

    print("="*70)
    print("ASCICat: HER Catalyst Screening")
    print("="*70)

    # -------------------------------------------------------------------------
    # 1. Initialize Calculator
    # -------------------------------------------------------------------------
    calc = ASCICalculator(
        reaction='HER',
        scoring_method='linear',
        verbose=True
    )

    # -------------------------------------------------------------------------
    # 2. Load Data
    # -------------------------------------------------------------------------
    data_path = Path(__file__).parent.parent / 'data' / 'HER_clean.csv'
    calc.load_data(str(data_path))

    # -------------------------------------------------------------------------
    # 3. Calculate ASCI Scores
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("Calculating ASCI Scores")
    print("="*70)

    # Equal weights for unbiased screening
    results = calc.calculate_asci(
        w_a=0.33,   # Activity weight
        w_s=0.33,   # Stability weight
        w_c=0.34    # Cost weight
    )

    # -------------------------------------------------------------------------
    # 4. Display Top Catalysts
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("TOP 10 HER CATALYSTS")
    print("="*70)

    top10 = calc.get_top_catalysts(n=10)
    print(f"\n{'Rank':<6} {'Catalyst':<12} {'ASCI':<8} {'Activity':<10} "
          f"{'Stability':<10} {'Cost':<10} {'ΔE (eV)':<10}")
    print("-"*76)

    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"{i:<6} {row['symbol']:<12} {row['ASCI']:.3f}    "
              f"{row['activity_score']:.3f}      {row['stability_score']:.3f}       "
              f"{row['cost_score']:.3f}      {row['DFT_ads_E']:.3f}")

    # -------------------------------------------------------------------------
    # 5. Generate Figures
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("Generating Figures")
    print("="*70)

    output_dir = Path(__file__).parent.parent / 'results' / 'HER_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualizer (handles sampling automatically for large datasets)
    viz = Visualizer(results, calc.config)

    # Generate all outputs (static + interactive)
    viz.generate_all_outputs(
        output_dir=str(output_dir),
        dpi=600,
        include_interactive=True
    )

    # -------------------------------------------------------------------------
    # 6. Export Data
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("Exporting Results")
    print("="*70)

    # Full results
    results_file = output_dir / 'HER_asci_results.csv'
    results.to_csv(results_file, index=False)
    print(f"  Full results: {results_file}")

    # Top 50 catalysts
    top50_file = output_dir / 'HER_top50.csv'
    calc.get_top_catalysts(n=50).to_csv(top50_file, index=False)
    print(f"  Top 50: {top50_file}")

    # Summary statistics
    print(f"\n  Statistics:")
    print(f"    Total catalysts: {len(results)}")
    print(f"    ASCI range: [{results['ASCI'].min():.3f}, {results['ASCI'].max():.3f}]")
    print(f"    Mean ASCI: {results['ASCI'].mean():.3f}")
    print(f"    Best catalyst: {results.iloc[0]['symbol']} (ASCI={results.iloc[0]['ASCI']:.3f})")

    print("\n" + "="*70)
    print("HER Analysis Complete!")
    print(f"Results saved to: {output_dir}/")
    print("="*70)


if __name__ == '__main__':
    main()
