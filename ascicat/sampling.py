"""
ASCICat Data Sampling Module
============================

Scientifically accurate data sampling for large catalyst datasets.

When dealing with large datasets (10K+ catalysts), visualization becomes
cluttered. This module provides sampling strategies that:

1. Focus on the SCIENTIFICALLY RELEVANT energy window around ΔE_opt
2. Preserve top performers (essential for ranking)
3. Sample densely near the Sabatier optimum (volcano peak)
4. Enable clear figures

Key Principle:
    Catalysts with ΔE far from the optimal are scientifically irrelevant.
    For example, if ΔE_opt = -0.67 eV for CO2RR-CO, a catalyst with
    ΔE = +2.0 eV has zero activity and should not dominate the visualization.

Scientific Energy Windows:
    - HER: ΔE_opt = -0.27 eV, relevant window ≈ [-0.5, 0.0] eV
    - CO2RR-CO: ΔE_opt = -0.67 eV, relevant window ≈ [-1.0, -0.2] eV
    - CO2RR-CHO: ΔE_opt = -0.48 eV, relevant window ≈ [-0.8, 0.0] eV
    - CO2RR-COCOH: ΔE_opt = -0.32 eV, relevant window ≈ [-0.6, 0.0] eV

Author: Nabil Khossossi
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def sample_for_visualization(
    df: pd.DataFrame,
    optimal_energy: float,
    n_samples: int = 1000,
    energy_window: float = 0.4,
    n_top: int = 200,
    energy_col: str = 'DFT_ads_E',
    score_col: str = 'ASCI',
    n_bins: int = 25,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Scientifically-focused sampling for visualization of large catalyst datasets.

    This function creates a sample focused on the RELEVANT energy region
    around the Sabatier optimum, ensuring meaningful volcano plots.

    Sampling Strategy:
    1. FILTER: Keep only catalysts within [ΔE_opt - window, ΔE_opt + window]
    2. TOP PERFORMERS: Always include top N by ASCI score
    3. STRATIFY: Distribute remaining samples across energy bins
    4. WEIGHT: Within each bin, prioritize higher ASCI scores

    Parameters
    ----------
    df : pd.DataFrame
        Full results DataFrame with ASCI scores
    optimal_energy : float
        Sabatier optimal adsorption energy (eV) for the reaction
    n_samples : int
        Target number of samples (default: 1000)
    energy_window : float
        Energy window around optimal (eV). Only catalysts within
        [ΔE_opt - window, ΔE_opt + window] are considered (default: 0.4 eV)
    n_top : int
        Number of top performers to always include (default: 200)
    energy_col : str
        Column name for adsorption energy (default: 'DFT_ads_E')
    score_col : str
        Column name for ASCI score (default: 'ASCI')
    n_bins : int
        Number of energy bins for stratification (default: 25)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Sampled DataFrame with representative catalysts from the
        scientifically relevant energy region

    Examples
    --------
    >>> from ascicat.sampling import sample_for_visualization
    >>> # CO2RR-CO with 55K catalysts, ΔE_opt = -0.67 eV
    >>> sampled = sample_for_visualization(
    ...     results,
    ...     optimal_energy=-0.67,
    ...     n_samples=1000,
    ...     energy_window=0.4
    ... )
    >>> print(f"Sampled {len(sampled)} from relevant region")
    """
    np.random.seed(random_state)

    # Sort by ASCI score
    df_sorted = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    n_total = len(df_sorted)

    # =========================================================================
    # STEP 1: Filter to scientifically relevant energy window
    # =========================================================================
    e_min = optimal_energy - energy_window
    e_max = optimal_energy + energy_window

    mask_relevant = (df_sorted[energy_col] >= e_min) & (df_sorted[energy_col] <= e_max)
    df_relevant = df_sorted[mask_relevant].copy()
    n_relevant = len(df_relevant)

    print(f"Filtering to relevant energy window: [{e_min:.2f}, {e_max:.2f}] eV")
    print(f"  Catalysts in relevant window: {n_relevant:,} / {n_total:,} ({100*n_relevant/n_total:.1f}%)")

    # If relevant region is small enough, return all
    if n_relevant <= n_samples:
        print(f"  Using all {n_relevant} catalysts from relevant region")
        df_relevant['rank'] = range(1, len(df_relevant) + 1)
        return df_relevant

    # =========================================================================
    # STEP 2: Always include top performers (by ASCI within relevant region)
    # =========================================================================
    n_top_actual = min(n_top, n_samples // 2, n_relevant)
    top_df = df_relevant.head(n_top_actual)
    top_indices = set(top_df.index)

    # =========================================================================
    # STEP 3: Stratified sampling from remaining catalysts
    # =========================================================================
    n_remaining = n_samples - n_top_actual
    remaining_df = df_relevant[~df_relevant.index.isin(top_indices)].copy()

    if len(remaining_df) == 0 or n_remaining <= 0:
        result = top_df.copy()
    else:
        # Create energy bins within the relevant window
        bins = np.linspace(e_min, e_max, n_bins + 1)
        remaining_df['_energy_bin'] = pd.cut(
            remaining_df[energy_col],
            bins=bins,
            labels=range(n_bins),
            include_lowest=True
        )

        # Calculate samples per bin (proportional to population)
        bin_counts = remaining_df['_energy_bin'].value_counts().sort_index()
        total_in_bins = len(remaining_df)

        if total_in_bins > 0:
            samples_per_bin = (bin_counts / total_in_bins * n_remaining).astype(int)
            samples_per_bin = samples_per_bin.clip(lower=1)

            # Adjust to match target
            while samples_per_bin.sum() > n_remaining:
                max_bin = samples_per_bin.idxmax()
                if samples_per_bin[max_bin] > 1:
                    samples_per_bin[max_bin] -= 1
            while samples_per_bin.sum() < n_remaining:
                for bin_idx in bin_counts.index:
                    if samples_per_bin.sum() >= n_remaining:
                        break
                    if bin_counts[bin_idx] > samples_per_bin.get(bin_idx, 0):
                        samples_per_bin[bin_idx] = samples_per_bin.get(bin_idx, 0) + 1

        # Sample from each bin (weighted by ASCI score)
        sampled_indices = []
        for bin_idx in range(n_bins):
            bin_data = remaining_df[remaining_df['_energy_bin'] == bin_idx]
            if len(bin_data) == 0:
                continue

            n_to_sample = min(samples_per_bin.get(bin_idx, 0), len(bin_data))
            if n_to_sample > 0:
                # Weight by ASCI score
                weights = bin_data[score_col].values
                weights = weights - weights.min() + 0.01
                weights = weights / weights.sum()

                sampled = bin_data.sample(
                    n=n_to_sample,
                    weights=weights,
                    random_state=random_state + bin_idx,
                    replace=False
                )
                sampled_indices.extend(sampled.index.tolist())

        # Combine top performers and stratified samples
        result = pd.concat([
            top_df,
            df_relevant.loc[sampled_indices]
        ]).drop_duplicates()

    # Sort by ASCI and assign ranks
    result = result.sort_values(score_col, ascending=False).reset_index(drop=True)
    result['rank'] = range(1, len(result) + 1)

    # Print summary
    print(f"\nSampled {len(result)} catalysts from relevant region:")
    print(f"  - Top performers: {n_top_actual}")
    print(f"  - Stratified samples: {len(result) - n_top_actual}")
    print(f"  - Energy range: [{result[energy_col].min():.3f}, {result[energy_col].max():.3f}] eV")
    print(f"  - ASCI range: [{result[score_col].min():.3f}, {result[score_col].max():.3f}]")

    return result


def get_relevant_window(reaction: str, pathway: str = None) -> Tuple[float, float]:
    """
    Get the scientifically relevant energy window for a reaction.

    Based on Sabatier principle and literature values for optimal
    adsorption energies.

    Parameters
    ----------
    reaction : str
        Reaction type ('HER' or 'CO2RR')
    pathway : str, optional
        CO2RR pathway ('CO', 'CHO', 'COCOH')

    Returns
    -------
    tuple
        (optimal_energy, window_size) in eV
    """
    if reaction.upper() == 'HER':
        return -0.27, 0.3  # [-0.57, 0.03] eV

    elif reaction.upper() == 'CO2RR':
        pathways = {
            'CO': (-0.67, 0.4),      # [-1.07, -0.27] eV
            'CHO': (-0.48, 0.4),     # [-0.88, -0.08] eV
            'COCOH': (-0.32, 0.4),   # [-0.72, 0.08] eV
        }
        if pathway and pathway.upper() in pathways:
            return pathways[pathway.upper()]
        else:
            return -0.50, 0.5  # Default CO2RR window

    else:
        return 0.0, 0.5  # Default


def sample_around_optimum(
    df: pd.DataFrame,
    optimal_energy: float,
    energy_col: str = 'DFT_ads_E',
    score_col: str = 'ASCI',
    window: float = 0.4,
    n_samples: int = 1000,
    include_outliers: int = 0
) -> pd.DataFrame:
    """
    Sample catalysts focused around the Sabatier optimum.

    Creates a sample that emphasizes the volcano peak region.
    Unlike sample_for_visualization, this does NOT include outliers
    by default for cleaner scientific figures.

    Parameters
    ----------
    df : pd.DataFrame
        Full results DataFrame
    optimal_energy : float
        Sabatier optimal adsorption energy (eV)
    energy_col : str
        Column name for adsorption energy
    score_col : str
        Column name for ASCI score
    window : float
        Energy window around optimum (eV) for focused sampling
    n_samples : int
        Maximum number of samples from the focused region
    include_outliers : int
        Number of outlier samples (outside window) - default 0

    Returns
    -------
    pd.DataFrame
        Sampled DataFrame focused on volcano peak
    """
    return sample_for_visualization(
        df=df,
        optimal_energy=optimal_energy,
        n_samples=n_samples,
        energy_window=window,
        n_top=min(200, n_samples // 3),
        energy_col=energy_col,
        score_col=score_col
    )


def sample_diverse_3d(
    df: pd.DataFrame,
    n_samples: int = 1000,
    n_top: int = 100,
    n_bins_per_dim: int = 5,
    random_state: int = 42,
    min_activity: float = 0.01
) -> pd.DataFrame:
    """
    Sample to ensure diversity across all THREE score dimensions for 3D visualization.

    This creates a sample with good spread in activity, stability, AND cost scores,
    avoiding the clustering that occurs when sampling only by energy.

    IMPORTANT: Only samples from catalysts with non-zero activity (within Sabatier window).
    Catalysts with activity_score = 0 are scientifically irrelevant and excluded.

    Strategy:
    1. FILTER: Only keep catalysts with activity_score > min_activity
    2. Always include top N performers by ASCI
    3. Divide remaining space into 3D bins (activity x stability x cost)
    4. Sample uniformly from each occupied bin

    Parameters
    ----------
    df : pd.DataFrame
        Full results with activity_score, stability_score, cost_score columns
    n_samples : int
        Target number of samples
    n_top : int
        Number of top ASCI performers to always include
    n_bins_per_dim : int
        Number of bins per dimension (total bins = n_bins^3)
    random_state : int
        Random seed
    min_activity : float
        Minimum activity score to include (default: 0.01 to exclude zero-activity)

    Returns
    -------
    pd.DataFrame
        Sampled DataFrame with diverse 3D coverage from active catalysts only
    """
    np.random.seed(random_state)

    # =========================================================================
    # CRITICAL: Filter to only ACTIVE catalysts (within Sabatier window)
    # Catalysts with activity_score = 0 are outside the activity window
    # and are scientifically irrelevant for visualization
    # =========================================================================
    n_total = len(df)
    df_active = df[df['activity_score'] > min_activity].copy()
    n_active = len(df_active)

    print(f"\nFiltering to active catalysts (activity > {min_activity}):")
    print(f"  Active catalysts: {n_active:,} / {n_total:,} ({100*n_active/n_total:.1f}%)")

    if n_active == 0:
        print("  WARNING: No active catalysts found! Returning top by ASCI.")
        result = df.sort_values('ASCI', ascending=False).head(n_samples).copy()
        result['rank'] = range(1, len(result) + 1)
        return result

    df_sorted = df_active.sort_values('ASCI', ascending=False).reset_index(drop=True)

    if len(df_sorted) <= n_samples:
        df_sorted['rank'] = range(1, len(df_sorted) + 1)
        return df_sorted

    # Step 1: Always include top performers
    n_top_actual = min(n_top, n_samples // 4)
    top_df = df_sorted.head(n_top_actual)
    top_indices = set(top_df.index)

    # Step 2: Create 3D bins for remaining samples
    remaining_df = df_sorted[~df_sorted.index.isin(top_indices)].copy()
    n_remaining = n_samples - n_top_actual

    # Create bins for each dimension
    bins = np.linspace(0, 1, n_bins_per_dim + 1)

    remaining_df['_act_bin'] = pd.cut(
        remaining_df['activity_score'].clip(0, 1),
        bins=bins, labels=range(n_bins_per_dim), include_lowest=True
    )
    remaining_df['_stab_bin'] = pd.cut(
        remaining_df['stability_score'].clip(0, 1),
        bins=bins, labels=range(n_bins_per_dim), include_lowest=True
    )
    remaining_df['_cost_bin'] = pd.cut(
        remaining_df['cost_score'].clip(0, 1),
        bins=bins, labels=range(n_bins_per_dim), include_lowest=True
    )

    # Group by 3D bin
    remaining_df['_3d_bin'] = (
        remaining_df['_act_bin'].astype(str) + '_' +
        remaining_df['_stab_bin'].astype(str) + '_' +
        remaining_df['_cost_bin'].astype(str)
    )

    # Count catalysts per 3D bin
    bin_counts = remaining_df['_3d_bin'].value_counts()
    n_occupied_bins = len(bin_counts)

    # Distribute samples across bins
    samples_per_bin = max(1, n_remaining // n_occupied_bins)

    sampled_indices = []
    for bin_id in bin_counts.index:
        bin_data = remaining_df[remaining_df['_3d_bin'] == bin_id]
        n_to_sample = min(samples_per_bin, len(bin_data))

        if n_to_sample > 0 and len(sampled_indices) < n_remaining:
            # Sample with preference for higher ASCI
            weights = bin_data['ASCI'].values
            weights = weights - weights.min() + 0.01
            weights = weights / weights.sum()

            sampled = bin_data.sample(
                n=min(n_to_sample, n_remaining - len(sampled_indices)),
                weights=weights,
                random_state=random_state,
                replace=False
            )
            sampled_indices.extend(sampled.index.tolist())

    # Combine
    result = pd.concat([
        top_df,
        df_sorted.loc[sampled_indices]
    ]).drop_duplicates()

    # Clean up temp columns
    for col in ['_act_bin', '_stab_bin', '_cost_bin', '_3d_bin']:
        if col in result.columns:
            result = result.drop(columns=[col])

    result = result.sort_values('ASCI', ascending=False).reset_index(drop=True)
    result['rank'] = range(1, len(result) + 1)

    print(f"\n3D-Diverse Sampling: {len(result)} catalysts")
    print(f"  - Top performers: {n_top_actual}")
    print(f"  - From {n_occupied_bins} 3D bins: {len(result) - n_top_actual}")
    print(f"  - Activity range: [{result['activity_score'].min():.3f}, {result['activity_score'].max():.3f}]")
    print(f"  - Stability range: [{result['stability_score'].min():.3f}, {result['stability_score'].max():.3f}]")
    print(f"  - Cost range: [{result['cost_score'].min():.3f}, {result['cost_score'].max():.3f}]")

    return result


def get_representative_sample(
    df: pd.DataFrame,
    optimal_energy: float,
    n_samples: int = 1000,
    energy_window: float = 0.4,
    **kwargs
) -> pd.DataFrame:
    """
    Get a representative sample for visualization.

    Wrapper function for easy access to scientifically-focused sampling.

    Parameters
    ----------
    df : pd.DataFrame
        Full results DataFrame
    optimal_energy : float
        Sabatier optimal adsorption energy (eV)
    n_samples : int
        Target number of samples
    energy_window : float
        Energy window around optimal (eV)
    **kwargs
        Additional arguments for sample_for_visualization

    Returns
    -------
    pd.DataFrame
        Sampled DataFrame
    """
    if len(df) <= n_samples:
        result = df.copy()
        result['rank'] = range(1, len(result) + 1)
        return result

    return sample_for_visualization(
        df=df,
        optimal_energy=optimal_energy,
        n_samples=n_samples,
        energy_window=energy_window,
        **kwargs
    )
