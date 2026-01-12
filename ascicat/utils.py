"""
ascicat/utils.py
Utility functions for ASCICat package
Helper functions for data manipulation, validation, and formatting
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import json
from datetime import datetime


def format_catalyst_name(row: pd.Series) -> str:
    """
    Format catalyst name from composition
    
    Parameters
    ----------
    row : pd.Series
        Data row containing composition info
    
    Returns
    -------
    str
        Formatted catalyst name
    """
    if 'symbol' in row:
        return str(row['symbol'])
    elif 'Ametal' in row and 'Bmetal' in row:
        metals = [str(row['Ametal']), str(row['Bmetal'])]
        if 'Cmetal' in row and pd.notna(row['Cmetal']):
            metals.append(str(row['Cmetal']))
        return ''.join(metals)
    else:
        return 'Unknown'


def format_surface(row: pd.Series) -> str:
    """
    Format surface description
    
    Parameters
    ----------
    row : pd.Series
        Data row containing surface info
    
    Returns
    -------
    str
        Formatted surface description
    """
    if 'AandB' in row:
        return str(row['AandB'])
    elif 'slab_millers' in row:
        symbol = format_catalyst_name(row)
        return f"{symbol}_{row['slab_millers']}"
    else:
        return format_catalyst_name(row)


def calculate_distance_from_optimal(delta_E: Union[float, np.ndarray],
                                   optimal_E: float) -> Union[float, np.ndarray]:
    """
    Calculate deviation from optimal binding energy
    
    Parameters
    ----------
    delta_E : float or array-like
        Adsorption energy (eV)
    optimal_E : float
        Optimal binding energy (eV)
    
    Returns
    -------
    float or np.ndarray
        Absolute deviation from optimum (eV)
    """
    return np.abs(delta_E - optimal_E)


def normalize_scores(scores: pd.Series) -> pd.Series:
    """
    Min-max normalize scores to [0, 1]
    
    Parameters
    ----------
    scores : pd.Series
        Raw scores
    
    Returns
    -------
    pd.Series
        Normalized scores [0, 1]
    """
    min_score = scores.min()
    max_score = scores.max()
    
    if max_score == min_score:
        return pd.Series([0.5] * len(scores), index=scores.index)
    
    normalized = (scores - min_score) / (max_score - min_score)
    return normalized.clip(0, 1)


def rank_by_column(df: pd.DataFrame, 
                   column: str, 
                   ascending: bool = False) -> pd.DataFrame:
    """
    Rank DataFrame by specified column
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to rank
    column : str
        Column to rank by
    ascending : bool
        Rank in ascending order (default: False for descending)
    
    Returns
    -------
    pd.DataFrame
        Ranked DataFrame with 'rank' column
    """
    ranked = df.sort_values(column, ascending=ascending).copy()
    ranked['rank'] = range(1, len(ranked) + 1)
    return ranked.reset_index(drop=True)


def filter_by_threshold(df: pd.DataFrame,
                       column: str,
                       threshold: float,
                       greater_than: bool = True) -> pd.DataFrame:
    """
    Filter DataFrame by threshold value
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to filter
    column : str
        Column to filter on
    threshold : float
        Threshold value
    greater_than : bool
        If True, keep values > threshold, else < threshold
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    if greater_than:
        return df[df[column] > threshold].copy()
    else:
        return df[df[column] < threshold].copy()


def get_pareto_front(df: pd.DataFrame,
                    objectives: List[str],
                    maximize: List[bool]) -> pd.DataFrame:
    """
    Extract Pareto front from multi-objective data
    
    A solution is Pareto optimal if no other solution is better
    in all objectives simultaneously.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with objective columns
    objectives : List[str]
        List of objective column names
    maximize : List[bool]
        Whether to maximize each objective (True) or minimize (False)
    
    Returns
    -------
    pd.DataFrame
        Pareto optimal solutions
    
    Examples
    --------
    >>> pareto = get_pareto_front(
    ...     df, 
    ...     objectives=['activity_score', 'cost_score'],
    ...     maximize=[True, True]
    ... )
    """
    if len(objectives) != len(maximize):
        raise ValueError("objectives and maximize must have same length")
    
    # Create copy for manipulation
    data = df.copy()
    
    # Adjust for minimization (flip sign)
    for obj, max_flag in zip(objectives, maximize):
        if not max_flag:
            data[obj] = -data[obj]
    
    # Extract objective values
    values = data[objectives].values
    
    # Find Pareto front
    pareto_mask = np.ones(len(values), dtype=bool)
    
    for i, point in enumerate(values):
        if pareto_mask[i]:
            # Check if this point is dominated
            dominated = np.all(values >= point, axis=1) & np.any(values > point, axis=1)
            pareto_mask[i] = not np.any(dominated)
    
    return df[pareto_mask].copy()


def calculate_correlation_matrix(df: pd.DataFrame,
                                 columns: List[str]) -> pd.DataFrame:
    """
    Calculate correlation matrix for specified columns
    
    Parameters
    ----------
    df : pd.DataFrame
        Data
    columns : List[str]
        Columns to include in correlation
    
    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    return df[columns].corr()


def save_to_json(data: Dict, file_path: str) -> None:
    """
    Save dictionary to JSON file
    
    Parameters
    ----------
    data : dict
        Data to save
    file_path : str
        Output file path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    converted_data = convert_types(data)
    
    with open(file_path, 'w') as f:
        json.dump(converted_data, f, indent=2)


def load_from_json(file_path: str) -> Dict:
    """
    Load dictionary from JSON file
    
    Parameters
    ----------
    file_path : str
        Input file path
    
    Returns
    -------
    dict
        Loaded data
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def create_metadata(results_df: pd.DataFrame,
                   config: dict,
                   weights: Tuple[float, float, float]) -> Dict:
    """
    Create metadata for ASCI results
    
    Parameters
    ----------
    results_df : pd.DataFrame
        ASCI results
    config : dict
        Reaction configuration
    weights : tuple
        (w_a, w_s, w_c)
    
    Returns
    -------
    dict
        Metadata dictionary
    """
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'ascicat_version': '1.0.0',
        'reaction': config.get('name', 'Unknown'),
        'pathway': config.get('pathway', 'default'),
        'total_catalysts': len(results_df),
        'weights': {
            'activity': weights[0],
            'stability': weights[1],
            'cost': weights[2]
        },
        'configuration': {
            'optimal_energy': config.get('optimal_energy'),
            'activity_width': config.get('activity_width'),
            'stability_range': config.get('stability_range'),
            'cost_range': config.get('cost_range')
        },
        'statistics': {
            'asci': {
                'mean': float(results_df['ASCI'].mean()),
                'median': float(results_df['ASCI'].median()),
                'std': float(results_df['ASCI'].std()),
                'min': float(results_df['ASCI'].min()),
                'max': float(results_df['ASCI'].max())
            }
        }
    }
    
    return metadata


def format_number(value: float, decimals: int = 3) -> str:
    """
    Format number for display
    
    Parameters
    ----------
    value : float
        Number to format
    decimals : int
        Number of decimal places
    
    Returns
    -------
    str
        Formatted string
    """
    if abs(value) >= 1000:
        return f"{value:,.{decimals-2}f}"
    else:
        return f"{value:.{decimals}f}"


def print_table(df: pd.DataFrame,
               columns: Optional[List[str]] = None,
               max_rows: int = 20) -> None:
    """
    Print DataFrame as formatted table
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to print
    columns : List[str], optional
        Columns to include (default: all)
    max_rows : int
        Maximum rows to print
    """
    if columns is not None:
        df = df[columns]
    
    # Set display options
    with pd.option_context('display.max_rows', max_rows,
                          'display.max_columns', None,
                          'display.width', None,
                          'display.precision', 3):
        print(df)


def validate_file_path(file_path: str, must_exist: bool = False) -> Path:
    """
    Validate and convert file path
    
    Parameters
    ----------
    file_path : str
        File path to validate
    must_exist : bool
        If True, raise error if file doesn't exist
    
    Returns
    -------
    Path
        Validated Path object
    
    Raises
    ------
    FileNotFoundError
        If file doesn't exist and must_exist=True
    """
    path = Path(file_path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    return path


def create_output_directory(dir_path: str) -> Path:
    """
    Create output directory if it doesn't exist
    
    Parameters
    ----------
    dir_path : str
        Directory path
    
    Returns
    -------
    Path
        Created directory path
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    """
    Get current timestamp as string
    
    Returns
    -------
    str
        Timestamp in ISO format
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


# ============================================================================
# ADDITIONAL UTILITY FUNCTIONS FOR __init__.py IMPORTS
# ============================================================================

def load_catalyst_data(file_path: str) -> pd.DataFrame:
    """
    Load catalyst data from file.
    
    Convenience wrapper for loading CSV data.
    
    Parameters
    ----------
    file_path : str
        Path to data file
    
    Returns
    -------
    pd.DataFrame
        Loaded data
    
    Examples
    --------
    >>> data = load_catalyst_data('data/HER_clean.csv')
    >>> print(data.shape)
    (200, 10)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    return pd.read_csv(file_path)


def save_results(data: pd.DataFrame, file_path: str) -> None:
    """
    Save results to file.
    
    Parameters
    ----------
    data : pd.DataFrame
        Results data
    file_path : str
        Output file path
    
    Examples
    --------
    >>> save_results(results, 'output/HER_results.csv')
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(file_path, index=False)


def calculate_element_cost(element: str, database: Optional[Dict] = None) -> float:
    """
    Get cost for a single element.
    
    Parameters
    ----------
    element : str
        Element symbol (e.g., 'Pt', 'Cu', 'Ni')
    database : dict, optional
        Custom cost database. If None, uses default values.
    
    Returns
    -------
    float
        Cost in $/kg
    
    Notes
    -----
    Default costs based on USGS Commodity data (2024).
    Values are approximate and should be updated periodically.
    
    Examples
    --------
    >>> cost_pt = calculate_element_cost('Pt')
    >>> print(f"Platinum: ${cost_pt:,.0f}/kg")
    Platinum: $30,000/kg
    
    >>> cost_cu = calculate_element_cost('Cu')
    >>> print(f"Copper: ${cost_cu:.2f}/kg")
    Copper: $8.50/kg
    """
    # Default cost database (USGS 2024 data)
    default_costs = {
        # Light elements
        'H': 1.0, 'C': 5.0, 'N': 3.0, 'O': 2.0, 'S': 1.5,
        
        # Earth-abundant metals
        'Fe': 0.5, 'Al': 2.0, 'Ti': 10.0, 'Mn': 2.5, 'Cr': 15.0,
        
        # Common transition metals
        'Ni': 18.0, 'Cu': 8.5, 'Zn': 3.0, 'Co': 35.0,
        
        # Precious/expensive transition metals
        'Mo': 45.0, 'W': 40.0, 'V': 30.0, 'Nb': 50.0, 'Ta': 300.0,
        
        # Noble metals
        'Ag': 800.0, 'Au': 60000.0, 'Pt': 30000.0, 'Pd': 50000.0,
        
        # Platinum group metals
        'Ru': 12000.0, 'Rh': 150000.0, 'Ir': 165000.0, 'Os': 12000.0,
        
        # Other catalytically relevant elements
        'Re': 3000.0, 'Sn': 25.0, 'Pb': 2.0, 'Bi': 15.0,
        'Sb': 12.0, 'As': 3.0, 'Se': 100.0, 'Te': 80.0,
    }
    
    # Use provided database or default
    costs = database if database is not None else default_costs
    
    # Return cost or default value if element not in database
    return costs.get(element, 100.0)  # Default to $100/kg if unknown


def get_periodic_table_data() -> Dict:
    """
    Get periodic table data for common elements.
    
    Returns comprehensive element information including:
    - Atomic number
    - Atomic mass
    - Element name
    - Common oxidation states
    - Electronegativity
    
    Returns
    -------
    dict
        Dictionary with element symbols as keys, properties as values
    
    Examples
    --------
    >>> pt_data = get_periodic_table_data()
    >>> pt_info = pt_data['Pt']
    >>> print(f"{pt_info['name']}: Z={pt_info['number']}, M={pt_info['mass']:.3f}")
    Platinum: Z=78, M=195.084
    """
    periodic_table = {
        # Period 1
        'H': {
            'name': 'Hydrogen',
            'number': 1,
            'mass': 1.008,
            'electronegativity': 2.20,
            'oxidation_states': [-1, 1]
        },
        
        # Period 2
        'C': {
            'name': 'Carbon',
            'number': 6,
            'mass': 12.011,
            'electronegativity': 2.55,
            'oxidation_states': [-4, -3, -2, -1, 1, 2, 3, 4]
        },
        'N': {
            'name': 'Nitrogen',
            'number': 7,
            'mass': 14.007,
            'electronegativity': 3.04,
            'oxidation_states': [-3, -2, -1, 1, 2, 3, 4, 5]
        },
        'O': {
            'name': 'Oxygen',
            'number': 8,
            'mass': 15.999,
            'electronegativity': 3.44,
            'oxidation_states': [-2, -1, 1, 2]
        },
        
        # Period 3
        'Al': {
            'name': 'Aluminum',
            'number': 13,
            'mass': 26.982,
            'electronegativity': 1.61,
            'oxidation_states': [3]
        },
        
        # Period 4 - 3d transition metals
        'Ti': {
            'name': 'Titanium',
            'number': 22,
            'mass': 47.867,
            'electronegativity': 1.54,
            'oxidation_states': [2, 3, 4]
        },
        'V': {
            'name': 'Vanadium',
            'number': 23,
            'mass': 50.942,
            'electronegativity': 1.63,
            'oxidation_states': [2, 3, 4, 5]
        },
        'Cr': {
            'name': 'Chromium',
            'number': 24,
            'mass': 51.996,
            'electronegativity': 1.66,
            'oxidation_states': [2, 3, 6]
        },
        'Mn': {
            'name': 'Manganese',
            'number': 25,
            'mass': 54.938,
            'electronegativity': 1.55,
            'oxidation_states': [2, 3, 4, 6, 7]
        },
        'Fe': {
            'name': 'Iron',
            'number': 26,
            'mass': 55.845,
            'electronegativity': 1.83,
            'oxidation_states': [2, 3]
        },
        'Co': {
            'name': 'Cobalt',
            'number': 27,
            'mass': 58.933,
            'electronegativity': 1.88,
            'oxidation_states': [2, 3]
        },
        'Ni': {
            'name': 'Nickel',
            'number': 28,
            'mass': 58.693,
            'electronegativity': 1.91,
            'oxidation_states': [2, 3]
        },
        'Cu': {
            'name': 'Copper',
            'number': 29,
            'mass': 63.546,
            'electronegativity': 1.90,
            'oxidation_states': [1, 2]
        },
        'Zn': {
            'name': 'Zinc',
            'number': 30,
            'mass': 65.38,
            'electronegativity': 1.65,
            'oxidation_states': [2]
        },
        
        # Period 5 - 4d transition metals
        'Nb': {
            'name': 'Niobium',
            'number': 41,
            'mass': 92.906,
            'electronegativity': 1.6,
            'oxidation_states': [3, 5]
        },
        'Mo': {
            'name': 'Molybdenum',
            'number': 42,
            'mass': 95.95,
            'electronegativity': 2.16,
            'oxidation_states': [2, 3, 4, 5, 6]
        },
        'Ru': {
            'name': 'Ruthenium',
            'number': 44,
            'mass': 101.07,
            'electronegativity': 2.2,
            'oxidation_states': [2, 3, 4, 6, 8]
        },
        'Rh': {
            'name': 'Rhodium',
            'number': 45,
            'mass': 102.906,
            'electronegativity': 2.28,
            'oxidation_states': [3]
        },
        'Pd': {
            'name': 'Palladium',
            'number': 46,
            'mass': 106.42,
            'electronegativity': 2.20,
            'oxidation_states': [2, 4]
        },
        'Ag': {
            'name': 'Silver',
            'number': 47,
            'mass': 107.868,
            'electronegativity': 1.93,
            'oxidation_states': [1]
        },
        
        # Period 6 - 5d transition metals
        'W': {
            'name': 'Tungsten',
            'number': 74,
            'mass': 183.84,
            'electronegativity': 2.36,
            'oxidation_states': [2, 3, 4, 5, 6]
        },
        'Re': {
            'name': 'Rhenium',
            'number': 75,
            'mass': 186.207,
            'electronegativity': 1.9,
            'oxidation_states': [4, 6, 7]
        },
        'Os': {
            'name': 'Osmium',
            'number': 76,
            'mass': 190.23,
            'electronegativity': 2.2,
            'oxidation_states': [3, 4, 6, 8]
        },
        'Ir': {
            'name': 'Iridium',
            'number': 77,
            'mass': 192.217,
            'electronegativity': 2.20,
            'oxidation_states': [3, 4]
        },
        'Pt': {
            'name': 'Platinum',
            'number': 78,
            'mass': 195.084,
            'electronegativity': 2.28,
            'oxidation_states': [2, 4]
        },
        'Au': {
            'name': 'Gold',
            'number': 79,
            'mass': 196.967,
            'electronegativity': 2.54,
            'oxidation_states': [1, 3]
        },
        
        # Post-transition metals
        'Sn': {
            'name': 'Tin',
            'number': 50,
            'mass': 118.710,
            'electronegativity': 1.96,
            'oxidation_states': [2, 4]
        },
        'Pb': {
            'name': 'Lead',
            'number': 82,
            'mass': 207.2,
            'electronegativity': 2.33,
            'oxidation_states': [2, 4]
        },
        'Bi': {
            'name': 'Bismuth',
            'number': 83,
            'mass': 208.980,
            'electronegativity': 2.02,
            'oxidation_states': [3, 5]
        },
    }
    
    return periodic_table


def calculate_composition_cost(composition: Dict[str, float], 
                               cost_database: Optional[Dict] = None) -> float:
    """
    Calculate composition-weighted cost for alloys.
    
    Parameters
    ----------
    composition : dict
        Dictionary of element symbols to atomic fractions
        Example: {'Cu': 0.7, 'Ni': 0.3}
    cost_database : dict, optional
        Custom cost database. If None, uses default values.
    
    Returns
    -------
    float
        Composition-weighted cost in $/kg
    
    Examples
    --------
    >>> # CuNi alloy (70% Cu, 30% Ni)
    >>> cost = calculate_composition_cost({'Cu': 0.7, 'Ni': 0.3})
    >>> print(f"CuNi alloy: ${cost:.2f}/kg")
    CuNi alloy: $11.35/kg
    
    >>> # PtRu alloy (50% Pt, 50% Ru)
    >>> cost = calculate_composition_cost({'Pt': 0.5, 'Ru': 0.5})
    >>> print(f"PtRu alloy: ${cost:,.0f}/kg")
    PtRu alloy: $21,000/kg
    """
    total_cost = 0.0
    
    for element, fraction in composition.items():
        element_cost = calculate_element_cost(element, cost_database)
        total_cost += fraction * element_cost
    
    return total_cost


def generate_unique_labels(df: pd.DataFrame,
                          label_col: str = 'display_label') -> pd.DataFrame:
    """
    Generate unique display labels for catalysts in ranking plots.

    Creates unambiguous labels by combining chemical formula with surface facet.
    If duplicates still exist, adds a numerical suffix.

    Format: "CuZn(211)" or "CuZn(211)#2" if still not unique

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with catalyst data. Must contain 'symbol' column.
        Optionally contains 'slab_millers' for facet information.
    label_col : str
        Name of the new column for unique labels (default: 'display_label')

    Returns
    -------
    pd.DataFrame
        DataFrame with added unique label column

    Examples
    --------
    >>> df = generate_unique_labels(results)
    >>> print(df[['symbol', 'slab_millers', 'display_label']].head())
        symbol  slab_millers  display_label
    0     CuZn          211       CuZn(211)
    1     CuZn          111       CuZn(111)
    2     CuZn          100       CuZn(100)
    3   Nb2Pt6          110     Nb2Pt6(110)
    4   Nb2Pt6          110   Nb2Pt6(110)#2

    Notes
    -----
    This function is essential for ranking plots where
    the same chemical formula may appear multiple times with different
    surface facets or configurations.
    """
    df = df.copy()

    # Step 1: Create base label from symbol + facet
    if 'slab_millers' in df.columns:
        df['_base_label'] = df.apply(
            lambda row: f"{row['symbol']}({row['slab_millers']})", axis=1
        )
    else:
        df['_base_label'] = df['symbol'].astype(str)

    # Step 2: Handle remaining duplicates by adding numerical suffix
    label_counts = {}
    unique_labels = []

    for idx, row in df.iterrows():
        base_label = row['_base_label']

        if base_label not in label_counts:
            label_counts[base_label] = 0
            unique_labels.append(base_label)
        else:
            label_counts[base_label] += 1
            unique_labels.append(f"{base_label}#{label_counts[base_label] + 1}")

    df[label_col] = unique_labels

    # Clean up temporary column
    df = df.drop(columns=['_base_label'])

    return df


def get_display_labels(df: pd.DataFrame, n_top: int = 10) -> List[str]:
    """
    Get unique display labels for top N catalysts.

    Convenience function for visualization code.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with catalyst data (should be sorted by ranking)
    n_top : int
        Number of top catalysts to label

    Returns
    -------
    List[str]
        List of unique display labels

    Examples
    --------
    >>> labels = get_display_labels(results.head(10))
    >>> print(labels)
    ['CuZn(211)', 'AgAu(111)', 'PdZn(100)', ...]
    """
    top_df = df.head(n_top).copy()
    labeled_df = generate_unique_labels(top_df)
    return labeled_df['display_label'].tolist()


def format_scientific(value: float, precision: int = 3) -> str:
    """
    Format number in scientific notation.
    
    Parameters
    ----------
    value : float
        Number to format
    precision : int
        Number of significant figures
    
    Returns
    -------
    str
        Formatted scientific notation string
    
    Examples
    --------
    >>> format_scientific(0.000123)
    '1.23×10⁻⁴'
    >>> format_scientific(1234567)
    '1.23×10⁶'
    """
    if value == 0:
        return "0"
    
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    
    # Unicode superscripts for exponents
    superscripts = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    exp_str = str(exponent).translate(superscripts)
    
    return f"{mantissa:.{precision-1}f}×10{exp_str}"