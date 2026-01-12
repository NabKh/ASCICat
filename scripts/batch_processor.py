#!/usr/bin/env python3
"""
Example 5: Advanced Analysis and Pareto Optimization
=====================================================

This example demonstrates sophisticated analysis techniques:
- Pareto front identification (multi-objective optimization)
- Correlation analysis between descriptors
- Statistical distributions and outlier detection
- Element-wise performance analysis
- Weight sensitivity mapping
- Export analysis

Perfect for: Deep insights, research, comprehensive understanding

Expected runtime: < 20 seconds
Output: Statistical reports, Pareto analysis, correlation studies
"""

from ascicat import ASCICalculator, Visualizer, Analyzer
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FILE = 'data/HER_clean.csv'
OUTPUT_DIR = 'results/example_5_advanced_analysis'

# Weights
W_ACTIVITY = 0.4
W_STABILITY = 0.3
W_COST = 0.3

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║            Example 5: Advanced Analysis & Pareto Optimization            ║
║                                                                          ║
║  Analysis Techniques:                                                    ║
║    • Pareto front identification                                        ║
║    • Correlation analysis                                               ║
║    • Statistical distributions                                          ║
║    • Element-wise performance                                           ║
║    • Weight sensitivity mapping                                         ║
║                                                                          ║
║  Goal: Deep understanding of catalyst space                             ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

# Create output directory
output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CALCULATE ASCI
# ============================================================================

print("\n" + "="*80)
print("Step 1: Calculate ASCI and Initialize Analysis")
print("="*80)

calc = ASCICalculator(reaction='HER', verbose=True)
calc.load_data(DATA_FILE, validate=True)

results = calc.calculate_asci(
    w_a=W_ACTIVITY,
    w_s=W_STABILITY,
    w_c=W_COST,
    show_progress=True
)

print(f"\n✓ Analysis initialized")
print(f"  Total catalysts: {len(results)}")

# Initialize analyzer
analyzer = Analyzer(results, calc.config)

# ============================================================================
# PARETO FRONT ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("Step 2: Pareto Front Identification")
print("="*80)

print(f"\n🎯 Identifying Pareto-optimal catalysts...")
print(f"   (Non-dominated solutions in multi-objective space)")

pareto_front = analyzer.get_pareto_optimal()

print(f"\n✓ Pareto analysis complete")
print(f"  Pareto-optimal catalysts: {len(pareto_front)}")
print(f"  Dominated catalysts: {len(results) - len(pareto_front)}")
print(f"  Pareto efficiency: {len(pareto_front)/len(results)*100:.2f}%")

# Save Pareto front
pareto_file = output_path / 'pareto_front.csv'
pareto_front.to_csv(pareto_file, index=False)
print(f"\n✓ Pareto front saved: {pareto_file.name}")

# Display Pareto catalysts
print(f"\n🏆 Pareto-Optimal Catalysts (Top 10 by ASCI):")
print(f"{'─'*80}")
pareto_top10 = pareto_front.nlargest(10, 'ASCI')
for idx, row in enumerate(pareto_top10.itertuples(), 1):
    print(f"{idx:2d}. {row.symbol:20s} ASCI: {row.ASCI:.4f}  "
          f"Act: {row.activity_score:.3f}  Stab: {row.stability_score:.3f}  Cost: {row.cost_score:.3f}")

# Visualize Pareto front
fig1, axes = plt.subplots(1, 3, figsize=(15, 5))

# Activity vs Stability
ax1 = axes[0]
ax1.scatter(results['activity_score'], results['stability_score'],
           c='lightgray', s=30, alpha=0.5, label='All catalysts')
ax1.scatter(pareto_front['activity_score'], pareto_front['stability_score'],
           c=pareto_front['ASCI'], cmap='viridis', s=60, edgecolors='black',
           linewidth=1, label='Pareto front')
ax1.set_xlabel('Activity Score', fontweight='bold')
ax1.set_ylabel('Stability Score', fontweight='bold')
ax1.set_title('Activity vs Stability', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Activity vs Cost
ax2 = axes[1]
ax2.scatter(results['activity_score'], results['cost_score'],
           c='lightgray', s=30, alpha=0.5, label='All catalysts')
ax2.scatter(pareto_front['activity_score'], pareto_front['cost_score'],
           c=pareto_front['ASCI'], cmap='viridis', s=60, edgecolors='black',
           linewidth=1, label='Pareto front')
ax2.set_xlabel('Activity Score', fontweight='bold')
ax2.set_ylabel('Cost Score', fontweight='bold')
ax2.set_title('Activity vs Cost', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Stability vs Cost
ax3 = axes[2]
scatter = ax3.scatter(results['stability_score'], results['cost_score'],
                     c='lightgray', s=30, alpha=0.5, label='All catalysts')
scatter = ax3.scatter(pareto_front['stability_score'], pareto_front['cost_score'],
                     c=pareto_front['ASCI'], cmap='viridis', s=60, edgecolors='black',
                     linewidth=1, label='Pareto front')
ax3.set_xlabel('Stability Score', fontweight='bold')
ax3.set_ylabel('Cost Score', fontweight='bold')
ax3.set_title('Stability vs Cost', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Colorbar
cbar = plt.colorbar(scatter, ax=axes, location='right', shrink=0.8)
cbar.set_label('ASCI Score', fontweight='bold')

plt.tight_layout()
pareto_fig = output_path / 'pareto_fronts.png'
plt.savefig(pareto_fig, dpi=300, bbox_inches='tight')
print(f"✓ Pareto visualization saved: {pareto_fig.name}")
plt.close()

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("Step 3: Correlation Analysis")
print("="*80)

print(f"\n📊 Analyzing correlations between descriptors and scores...")

corr_analysis = analyzer.get_correlation_analysis()

print(f"\n✓ Correlation matrix computed")
print(f"\n🔗 Key ASCI Correlations:")
asci_corr = corr_analysis['ASCI'].sort_values(ascending=False)
print(f"{'─'*60}")
for var, corr in asci_corr.items():
    if var != 'ASCI':
        strength = 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.4 else 'Weak'
        direction = 'positive' if corr > 0 else 'negative'
        print(f"  ASCI ↔ {var:20s}: {corr:+.3f}  ({strength} {direction})")

# Visualize correlation matrix
fig2, ax = plt.subplots(figsize=(10, 8))

# Select key columns for correlation
corr_cols = ['DFT_ads_E', 'surface_energy', 'Cost', 
             'activity_score', 'stability_score', 'cost_score', 'ASCI']
corr_matrix = results[corr_cols].corr()

# Heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
           center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
           ax=ax)
ax.set_title('Correlation Matrix: Descriptors and Scores', 
            fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
corr_fig = output_path / 'correlation_matrix.png'
plt.savefig(corr_fig, dpi=300, bbox_inches='tight')
print(f"\n✓ Correlation heatmap saved: {corr_fig.name}")
plt.close()

# Save correlation data
corr_file = output_path / 'correlations.csv'
corr_matrix.to_csv(corr_file)
print(f"✓ Correlation data saved: {corr_file.name}")

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("Step 4: Statistical Distribution Analysis")
print("="*80)

print(f"\n📈 Computing statistical distributions...")

stats = analyzer.get_statistics()

print(f"\n✓ Statistics computed")
print(f"\n📊 Distribution Summary:")
print(f"{'─'*60}")

for metric in ['activity_score', 'stability_score', 'cost_score', 'ASCI']:
    data = results[metric]
    print(f"\n{metric.replace('_', ' ').title()}:")
    print(f"  Mean:    {data.mean():.4f}")
    print(f"  Median:  {data.median():.4f}")
    print(f"  Std Dev: {data.std():.4f}")
    print(f"  Range:   [{data.min():.4f}, {data.max():.4f}]")
    print(f"  IQR:     [{data.quantile(0.25):.4f}, {data.quantile(0.75):.4f}]")

# Detect outliers (using IQR method)
print(f"\n🔍 Outlier Detection (IQR method):")
print(f"{'─'*60}")

for metric in ['activity_score', 'stability_score', 'cost_score', 'ASCI']:
    Q1 = results[metric].quantile(0.25)
    Q3 = results[metric].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = results[(results[metric] < lower_bound) | (results[metric] > upper_bound)]
    
    if len(outliers) > 0:
        print(f"\n{metric.replace('_', ' ').title()}: {len(outliers)} outliers")
        print(f"  Top 3:")
        for idx, row in enumerate(outliers.nlargest(3, metric).itertuples(), 1):
            print(f"    {idx}. {row.symbol:15s} {metric}: {getattr(row, metric):.4f}")
    else:
        print(f"\n{metric.replace('_', ' ').title()}: No outliers detected")

# Visualize distributions
fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

metrics = ['activity_score', 'stability_score', 'cost_score', 'ASCI']
colors = ['#E74C3C', '#3498DB', '#F39C12', '#9B59B6']
titles = ['Activity Score', 'Stability Score', 'Cost Score', 'ASCI Score']

for ax, metric, color, title in zip(axes, metrics, colors, titles):
    # Histogram
    ax.hist(results[metric], bins=30, color=color, alpha=0.7, 
           edgecolor='black', linewidth=1.2)
    
    # Add mean and median lines
    mean = results[metric].mean()
    median = results[metric].median()
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.3f}')
    ax.axvline(median, color='blue', linestyle=':', linewidth=2, label=f'Median: {median:.3f}')
    
    ax.set_xlabel('Score', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
dist_fig = output_path / 'score_distributions.png'
plt.savefig(dist_fig, dpi=300, bbox_inches='tight')
print(f"\n✓ Distribution plots saved: {dist_fig.name}")
plt.close()

# ============================================================================
# ELEMENT-WISE ANALYSIS (if composition data available)
# ============================================================================

print("\n" + "="*80)
print("Step 5: Element-Wise Performance Analysis")
print("="*80)

if 'Ametal' in results.columns:
    print(f"\n🧪 Analyzing performance by primary element...")
    
    element_stats = results.groupby('Ametal').agg({
        'ASCI': ['mean', 'std', 'max', 'count'],
        'activity_score': 'mean',
        'stability_score': 'mean',
        'cost_score': 'mean'
    }).round(4)
    
    # Flatten column names
    element_stats.columns = ['_'.join(col).strip() for col in element_stats.columns.values]
    element_stats = element_stats.sort_values('ASCI_mean', ascending=False)
    
    print(f"\n✓ Element analysis complete")
    print(f"  Elements analyzed: {len(element_stats)}")
    
    print(f"\n🏆 Top 10 Elements by Mean ASCI:")
    print(f"{'─'*80}")
    top_elements = element_stats.head(10)
    for elem, row in top_elements.iterrows():
        print(f"  {elem:3s}  Mean ASCI: {row['ASCI_mean']:.4f}  "
              f"Std: {row['ASCI_std']:.4f}  Max: {row['ASCI_max']:.4f}  "
              f"Count: {int(row['ASCI_count'])}")
    
    # Save element stats
    elem_file = output_path / 'element_performance.csv'
    element_stats.to_csv(elem_file)
    print(f"\n✓ Element statistics saved: {elem_file.name}")
    
    # Visualize top elements
    if len(element_stats) >= 10:
        fig4, ax = plt.subplots(figsize=(12, 6))
        
        top10_elem = element_stats.head(10)
        x = range(len(top10_elem))
        
        ax.bar(x, top10_elem['ASCI_mean'], yerr=top10_elem['ASCI_std'],
              color='#3498DB', alpha=0.7, edgecolor='black', linewidth=1.5,
              capsize=5, error_kw={'linewidth': 2})
        
        ax.set_xticks(x)
        ax.set_xticklabels(top10_elem.index, fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean ASCI Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Primary Element', fontsize=12, fontweight='bold')
        ax.set_title('Top 10 Elements by Mean ASCI Performance', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, top10_elem['ASCI_mean'].max() * 1.2)
        
        plt.tight_layout()
        elem_fig = output_path / 'element_performance_bar.png'
        plt.savefig(elem_fig, dpi=300, bbox_inches='tight')
        print(f"✓ Element performance plot saved: {elem_fig.name}")
        plt.close()
else:
    print(f"\n⚠️  Element composition data not available in dataset")

# ============================================================================
# WEIGHT SENSITIVITY MAPPING
# ============================================================================

print("\n" + "="*80)
print("Step 6: Weight Sensitivity Mapping")
print("="*80)

print(f"\n🔬 Generating weight sensitivity map...")
print(f"   (This may take a moment...)")

# Create weight grid
n_points = 20
weight_grid = []

for w_a in np.linspace(0.1, 0.8, n_points):
    for w_s in np.linspace(0.1, 0.8, n_points):
        w_c = 1.0 - w_a - w_s
        if 0.1 <= w_c <= 0.8:  # Valid weight
            weight_grid.append((w_a, w_s, w_c))

print(f"   Testing {len(weight_grid)} weight combinations...")

# Calculate ASCI for each weight combination
sensitivity_data = []

for w_a, w_s, w_c in weight_grid:
    temp_results = calc.calculate_asci(w_a=w_a, w_s=w_s, w_c=w_c, show_progress=False)
    top_cat = temp_results.iloc[0]
    
    sensitivity_data.append({
        'w_activity': w_a,
        'w_stability': w_s,
        'w_cost': w_c,
        'best_asci': top_cat['ASCI'],
        'mean_asci': temp_results['ASCI'].mean(),
        'top_catalyst': top_cat['symbol']
    })

sensitivity_df = pd.DataFrame(sensitivity_data)

print(f"\n✓ Sensitivity analysis complete")

# Save sensitivity data
sens_file = output_path / 'weight_sensitivity_data.csv'
sensitivity_df.to_csv(sens_file, index=False)
print(f"✓ Sensitivity data saved: {sens_file.name}")

# Visualize sensitivity (2D heatmap: activity vs stability weight, color = best ASCI)
fig5, ax = plt.subplots(figsize=(10, 8))

# Pivot for heatmap
pivot_data = sensitivity_df.pivot_table(
    values='best_asci',
    index='w_stability',
    columns='w_activity',
    aggfunc='mean'
)

sns.heatmap(pivot_data, cmap='viridis', annot=False, fmt='.3f',
           cbar_kws={'label': 'Best ASCI Score'}, ax=ax)
ax.set_xlabel('Activity Weight', fontsize=12, fontweight='bold')
ax.set_ylabel('Stability Weight', fontsize=12, fontweight='bold')
ax.set_title('Weight Sensitivity Map: Best ASCI Score\n(Cost weight = 1 - Activity - Stability)', 
            fontsize=13, fontweight='bold')

plt.tight_layout()
sens_fig = output_path / 'weight_sensitivity_map.png'
plt.savefig(sens_fig, dpi=300, bbox_inches='tight')
print(f"✓ Sensitivity map saved: {sens_fig.name}")
plt.close()

# Find optimal weights
optimal_idx = sensitivity_df['best_asci'].idxmax()
optimal_weights = sensitivity_df.loc[optimal_idx]

print(f"\n🎯 Optimal Weights (maximizing best ASCI):")
print(f"   Activity:  {optimal_weights['w_activity']:.3f}")
print(f"   Stability: {optimal_weights['w_stability']:.3f}")
print(f"   Cost:      {optimal_weights['w_cost']:.3f}")
print(f"   Best ASCI: {optimal_weights['best_asci']:.4f}")
print(f"   Top catalyst: {optimal_weights['top_catalyst']}")

# ============================================================================
# COMPREHENSIVE SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("Generating Comprehensive Analysis Report")
print("="*80)

report = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                  ASCICat Advanced Analysis Report                        ║
╚══════════════════════════════════════════════════════════════════════════╝

Dataset: {DATA_FILE}
Reaction: {calc.config.name}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
1. DATASET OVERVIEW
{'='*80}

Total Catalysts: {len(results):,}
Descriptor Ranges:
  • ΔE (Adsorption Energy): {results['DFT_ads_E'].min():.3f} to {results['DFT_ads_E'].max():.3f} eV
  • γ (Surface Energy):     {results['surface_energy'].min():.3f} to {results['surface_energy'].max():.3f} J/m²
  • Cost:                   ${results['Cost'].min():.2f} to ${results['Cost'].max():.2f}/kg

{'='*80}
2. ASCI SCORE ANALYSIS
{'='*80}

Weights Used:
  • Activity:  {W_ACTIVITY:.2f}
  • Stability: {W_STABILITY:.2f}
  • Cost:      {W_COST:.2f}

ASCI Statistics:
  • Mean:   {results['ASCI'].mean():.4f}
  • Median: {results['ASCI'].median():.4f}
  • Std:    {results['ASCI'].std():.4f}
  • Range:  [{results['ASCI'].min():.4f}, {results['ASCI'].max():.4f}]

Top 3 Catalysts:
"""

for idx, row in enumerate(results.head(3).itertuples(), 1):
    report += f"  {idx}. {row.symbol:20s} ASCI: {row.ASCI:.4f}\n"

report += f"""
{'='*80}
3. PARETO OPTIMIZATION
{'='*80}

Pareto-Optimal Catalysts: {len(pareto_front)} ({len(pareto_front)/len(results)*100:.2f}%)
Dominated Catalysts:      {len(results) - len(pareto_front)} ({(len(results)-len(pareto_front))/len(results)*100:.2f}%)

Interpretation:
  • {len(pareto_front)} catalysts lie on the Pareto front
  • These represent optimal trade-offs between objectives
  • No improvement in one objective without sacrificing another

{'='*80}
4. CORRELATION INSIGHTS
{'='*80}

Key Correlations with ASCI:
"""

for var, corr in list(asci_corr.items())[1:6]:  # Top 5
    report += f"  • {var:20s}: {corr:+.3f}\n"

report += f"""
{'='*80}
5. WEIGHT SENSITIVITY
{'='*80}

Optimal Weights (for maximizing best ASCI):
  • Activity:  {optimal_weights['w_activity']:.3f}
  • Stability: {optimal_weights['w_stability']:.3f}
  • Cost:      {optimal_weights['w_cost']:.3f}

Resulting Performance:
  • Best ASCI: {optimal_weights['best_asci']:.4f}
  • Champion:  {optimal_weights['top_catalyst']}

Sensitivity Range:
  • Best ASCI range across all weights: {sensitivity_df['best_asci'].min():.4f} - {sensitivity_df['best_asci'].max():.4f}
  • ΔRange:  {sensitivity_df['best_asci'].max() - sensitivity_df['best_asci'].min():.4f}

{'='*80}
6. RECOMMENDATIONS
{'='*80}

✓ Pareto-optimal catalysts offer best trade-offs
✓ Focus on top {len(pareto_front)} Pareto catalysts for experimental validation
✓ Weight selection significantly impacts ranking (sensitivity: {sensitivity_df['best_asci'].std():.4f})
✓ Consider application-specific weights based on priorities

{'='*80}
OUTPUT FILES
{'='*80}

Generated Files:
  • pareto_front.csv - Pareto-optimal catalysts
  • pareto_fronts.png - Pareto visualization
  • correlation_matrix.png - Correlation heatmap
  • correlations.csv - Full correlation data
  • score_distributions.png - Statistical distributions
  • weight_sensitivity_data.csv - Sensitivity analysis
  • weight_sensitivity_map.png - Weight optimization map
"""

if 'Ametal' in results.columns:
    report += f"  • element_performance.csv - Element statistics\n"
    report += f"  • element_performance_bar.png - Element comparison\n"

report += f"""
{'='*80}
END OF REPORT
{'='*80}
"""

# Save report
report_file = output_path / 'analysis_report.txt'
with open(report_file, 'w') as f:
    f.write(report)

print(f"\n✓ Comprehensive report saved: {report_file.name}")
print(report)

# ============================================================================
# CONCLUSION
# ============================================================================

print("\n" + "="*80)
print(" Example 5 Complete!")
print("="*80)

print(f"\n📁 Output directory: {OUTPUT_DIR}")
print(f" Analysis files generated:")
print(f"   • analysis_report.txt (comprehensive summary)")
print(f"   • pareto_front.csv")
print(f"   • pareto_fronts.png")
print(f"   • correlation_matrix.png")
print(f"   • correlations.csv")
print(f"   • score_distributions.png")
print(f"   • weight_sensitivity_data.csv")
print(f"   • weight_sensitivity_map.png")

if 'Ametal' in results.columns:
    print(f"   • element_performance.csv")
    print(f"   • element_performance_bar.png")

print(f"\n🎯 Key Findings:")
print(f"   • {len(pareto_front)} Pareto-optimal catalysts identified")
print(f"   • Optimal weights: Act={optimal_weights['w_activity']:.2f}, "
      f"Stab={optimal_weights['w_stability']:.2f}, Cost={optimal_weights['w_cost']:.2f}")
print(f"   • Champion catalyst: {optimal_weights['top_catalyst']}")
print(f"   • Weight sensitivity: {sensitivity_df['best_asci'].std():.4f} std deviation")

print(f"\n💡 Next Steps:")
print(f"   • Review Pareto front for experimental candidates")
print(f"   • Consider weight sensitivity in decision-making")
print(f"   • Use correlation insights for descriptor engineering")
print(f"   • Validate top catalysts experimentally")

print("\n" + "="*80 + "\n")