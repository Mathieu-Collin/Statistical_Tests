"""Data processor module for expanding hospital datasets.

This module provides functions to transform frequency-based datasets
into expanded single-column datasets for statistical analysis.
"""

import numpy as np
import pandas as pd


def expand_dataset(df: pd.DataFrame) -> np.ndarray:
    """Expand a frequency-based dataset into a single-column array.

    Takes a DataFrame with 'door_to_needle' and 'n' columns, and expands it
    by repeating each door_to_needle value n times, creating a single-column
    array suitable for statistical testing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns 'door_to_needle' and 'n'.
        - 'door_to_needle': Integer values representing time measurements.
        - 'n': Integer frequency count for each door_to_needle value.

    Returns
    -------
    np.ndarray
        1D numpy array containing all expanded door_to_needle values.

    Raises
    ------
    ValueError
        If the DataFrame doesn't contain required columns.
    TypeError
        If input is not a pandas DataFrame.

    Notes
    -----
    The expansion creates a flat array where each door_to_needle value
    appears exactly 'n' times, allowing for proper statistical analysis.

    Examples
    --------
    >>> df = pd.DataFrame({'door_to_needle': [30, 45, 60], 'n': [2, 3, 1]})
    >>> expanded = expand_dataset(df)
    >>> expanded
    array([30, 30, 45, 45, 45, 60])
    >>> len(expanded)
    6
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    required_columns = {'door_to_needle', 'n'}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"DataFrame must contain columns: {required_columns}. "
            f"Found columns: {set(df.columns)}"
        )
    
    expanded_data = []
    for _, row in df.iterrows():
        door_to_needle_value = row['door_to_needle']
        frequency = row['n']
        expanded_data.extend([door_to_needle_value] * frequency)
    
    return np.array(expanded_data, dtype=np.int64)


def expand_both_datasets(
    df_hospital_1: pd.DataFrame,
    df_hospital_2: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Expand both hospital datasets for statistical comparison.

    Parameters
    ----------
    df_hospital_1 : pd.DataFrame
        First hospital dataset with 'door_to_needle' and 'n' columns.
    df_hospital_2 : pd.DataFrame
        Second hospital dataset with 'door_to_needle' and 'n' columns.

    Returns
    -------
    tuple of np.ndarray
        Two 1D numpy arrays: (expanded_hospital_1, expanded_hospital_2).

    Examples
    --------
    >>> df1 = pd.DataFrame({'door_to_needle': [30, 45], 'n': [2, 1]})
    >>> df2 = pd.DataFrame({'door_to_needle': [40, 50], 'n': [1, 3]})
    >>> exp1, exp2 = expand_both_datasets(df1, df2)
    >>> exp1
    array([30, 30, 45])
    >>> exp2
    array([40, 50, 50, 50])
    """
    expanded_hospital_1 = expand_dataset(df_hospital_1)
    expanded_hospital_2 = expand_dataset(df_hospital_2)
    
    return expanded_hospital_1, expanded_hospital_2
