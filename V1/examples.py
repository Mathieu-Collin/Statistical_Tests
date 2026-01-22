"""Example script demonstrating advanced usage of the statistical tests package.

This script shows various ways to use the package with custom configurations.
"""

from src.data_generator import generate_both_hospital_datasets
from src.data_processor import expand_both_datasets
from src.statistical_tests import run_statistical_pipeline
import numpy as np


def example_1_basic_usage():
    """Example 1: Basic usage with default parameters."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 70)
    
    # Generate datasets
    df1, df2 = generate_both_hospital_datasets(random_state=42)
    
    # Expand datasets
    exp1, exp2 = expand_both_datasets(df1, df2)
    
    # Run analysis
    result = run_statistical_pipeline(exp1, exp2)
    
    print(f"\nTest type: {result.test_type}")
    print(f"P-value: {result.test_result.p_value:.6f}")
    print()


def example_2_custom_dataset_size():
    """Example 2: Custom dataset size."""
    print("=" * 70)
    print("EXAMPLE 2: Custom Dataset Size (100 rows)")
    print("=" * 70)
    
    # Generate larger datasets
    df1, df2 = generate_both_hospital_datasets(n_rows=100, random_state=123)
    
    print(f"Dataset 1 shape: {df1.shape}")
    print(f"Dataset 2 shape: {df2.shape}")
    
    # Expand and analyze
    exp1, exp2 = expand_both_datasets(df1, df2)
    print(f"Expanded size 1: {len(exp1)}")
    print(f"Expanded size 2: {len(exp2)}")
    
    result = run_statistical_pipeline(exp1, exp2)
    print(f"Test type: {result.test_type}")
    print()


def example_3_custom_value_ranges():
    """Example 3: Custom value ranges."""
    print("=" * 70)
    print("EXAMPLE 3: Custom Value Ranges")
    print("=" * 70)
    
    # Generate with custom ranges
    from src.data_generator import generate_hospital_dataset
    
    df1 = generate_hospital_dataset(
        n_rows=50,
        door_to_needle_min=20,
        door_to_needle_max=100,
        n_min=5,
        n_max=30,
        random_state=42
    )
    
    df2 = generate_hospital_dataset(
        n_rows=50,
        door_to_needle_min=25,
        door_to_needle_max=110,
        n_min=5,
        n_max=30,
        random_state=43
    )
    
    print(f"Hospital 1 - door_to_needle range: {df1['door_to_needle'].min()}-{df1['door_to_needle'].max()}")
    print(f"Hospital 1 - n range: {df1['n'].min()}-{df1['n'].max()}")
    print(f"Hospital 2 - door_to_needle range: {df2['door_to_needle'].min()}-{df2['door_to_needle'].max()}")
    print(f"Hospital 2 - n range: {df2['n'].min()}-{df2['n'].max()}")
    print()


def example_4_detailed_results():
    """Example 4: Accessing detailed results."""
    print("=" * 70)
    print("EXAMPLE 4: Detailed Results Access")
    print("=" * 70)
    
    df1, df2 = generate_both_hospital_datasets(n_rows=50, random_state=999)
    exp1, exp2 = expand_both_datasets(df1, df2)
    result = run_statistical_pipeline(exp1, exp2)
    
    # Access individual components
    print("\nShapiro-Wilk Results:")
    print(f"  Hospital 1 - W: {result.shapiro_hospital_1.statistic:.4f}, "
          f"p: {result.shapiro_hospital_1.p_value:.4f}, "
          f"Normal: {result.shapiro_hospital_1.is_normal}")
    print(f"  Hospital 2 - W: {result.shapiro_hospital_2.statistic:.4f}, "
          f"p: {result.shapiro_hospital_2.p_value:.4f}, "
          f"Normal: {result.shapiro_hospital_2.is_normal}")
    
    print(f"\nComparison Test: {result.test_type.upper()}")
    if result.test_type == 't-test':
        print(f"  T-statistic: {result.test_result.t_statistic:.4f}")
        print(f"  P-value: {result.test_result.p_value:.4f}")
    else:
        print(f"  U-statistic: {result.test_result.statistic:.4f}")
        print(f"  P-value: {result.test_result.p_value:.4f}")
    
    print(f"\nSignificant difference: {result.test_result.p_value < 0.05}")
    print()


def example_5_comparing_real_data():
    """Example 5: Template for real data comparison."""
    print("=" * 70)
    print("EXAMPLE 5: Template for Real Data")
    print("=" * 70)
    
    print("""
# If you have real data in CSV format:

import pandas as pd
from src.data_processor import expand_both_datasets
from src.statistical_tests import run_statistical_pipeline

# Load your data
df_hospital_1 = pd.read_csv('data/hospital_1.csv')
df_hospital_2 = pd.read_csv('data/hospital_2.csv')

# Ensure correct column names: 'door_to_needle' and 'n'
# Your DataFrame should look like:
#    door_to_needle    n
#    45               3
#    60               2
#    ...

# Expand and analyze
expanded_1, expanded_2 = expand_both_datasets(df_hospital_1, df_hospital_2)
result = run_statistical_pipeline(expanded_1, expanded_2)

# Display results
print(f"Test: {result.test_type}")
print(f"P-value: {result.test_result.p_value:.6f}")
    """)
    print()


if __name__ == "__main__":
    example_1_basic_usage()
    example_2_custom_dataset_size()
    example_3_custom_value_ranges()
    example_4_detailed_results()
    example_5_comparing_real_data()
