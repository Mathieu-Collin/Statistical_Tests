"""Data generator module for creating synthetic hospital datasets.

This module provides functions to generate synthetic hospital datasets
with door-to-needle times and frequency counts.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_hospital_dataset(
    n_rows: int = 50,
    door_to_needle_min: int = 30,
    door_to_needle_max: int = 120,
    n_min: int = 1,
    n_max: int = 50,
    random_state: int = None
) -> pd.DataFrame:
    """Generate a synthetic hospital dataset with door-to-needle times.

    Parameters
    ----------
    n_rows : int, default=50
        Number of rows in the dataset.
    door_to_needle_min : int, default=30
        Minimum value for door_to_needle column.
    door_to_needle_max : int, default=120
        Maximum value for door_to_needle column.
    n_min : int, default=1
        Minimum value for n (frequency) column.
    n_max : int, default=50
        Maximum value for n (frequency) column.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with two columns: 'door_to_needle' and 'n'.

    Examples
    --------
    >>> df = generate_hospital_dataset(n_rows=10, random_state=42)
    >>> df.shape
    (10, 2)
    >>> list(df.columns)
    ['door_to_needle', 'n']
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    door_to_needle = np.random.randint(
        door_to_needle_min, 
        door_to_needle_max + 1, 
        size=n_rows
    )
    
    n = np.random.randint(
        n_min,
        n_max + 1,
        size=n_rows
    )
    
    df = pd.DataFrame({
        'door_to_needle': door_to_needle,
        'n': n
    })
    
    return df


def generate_both_hospital_datasets(
    n_rows: int = 50,
    door_to_needle_min: int = 30,
    door_to_needle_max: int = 120,
    n_min: int = 1,
    n_max: int = 50,
    random_state: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate both hospital datasets (df_hospital_1 and df_hospital_2).

    Parameters
    ----------
    n_rows : int, default=50
        Number of rows in each dataset.
    door_to_needle_min : int, default=30
        Minimum value for door_to_needle column.
    door_to_needle_max : int, default=120
        Maximum value for door_to_needle column.
    n_min : int, default=1
        Minimum value for n (frequency) column.
    n_max : int, default=50
        Maximum value for n (frequency) column.
    random_state : int, optional
        Random seed for reproducibility. If provided, hospital_2 will use
        random_state + 1 to ensure different data.

    Returns
    -------
    tuple of pd.DataFrame
        Two DataFrames: (df_hospital_1, df_hospital_2), each with columns
        'door_to_needle' and 'n'.

    Examples
    --------
    >>> df1, df2 = generate_both_hospital_datasets(n_rows=10, random_state=42)
    >>> df1.shape, df2.shape
    ((10, 2), (10, 2))
    """
    df_hospital_1 = generate_hospital_dataset(
        n_rows=n_rows,
        door_to_needle_min=door_to_needle_min,
        door_to_needle_max=door_to_needle_max,
        n_min=n_min,
        n_max=n_max,
        random_state=random_state
    )
    
    # Use different random state for second hospital if provided
    random_state_2 = random_state + 1 if random_state is not None else None
    
    df_hospital_2 = generate_hospital_dataset(
        n_rows=n_rows,
        door_to_needle_min=door_to_needle_min,
        door_to_needle_max=door_to_needle_max,
        n_min=n_min,
        n_max=n_max,
        random_state=random_state_2
    )
    
    return df_hospital_1, df_hospital_2
