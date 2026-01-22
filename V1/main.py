"""Main execution script for statistical analysis of hospital datasets.

This script demonstrates the complete statistical analysis pipeline for
comparing door-to-needle times between two hospitals.
"""

from datetime import datetime
from pathlib import Path
from src.data_generator import generate_both_hospital_datasets
from src.data_processor import expand_both_datasets
from src.statistical_tests import run_statistical_pipeline


def format_results(result, expanded_1_size, expanded_2_size):
    """Format statistical analysis results as a string.
    
    Parameters
    ----------
    result : PipelineResult
        The complete pipeline result to display.
    expanded_1_size : int
        Size of expanded hospital 1 dataset.
    expanded_2_size : int
        Size of expanded hospital 2 dataset.
    
    Returns
    -------
    str
        Formatted results as a string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("STATISTICAL ANALYSIS RESULTS: HOSPITAL COMPARISON")
    lines.append("=" * 70)
    lines.append(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nDataset Sizes:")
    lines.append(f"  Hospital 1: {expanded_1_size} observations")
    lines.append(f"  Hospital 2: {expanded_2_size} observations")
    
    # Normality test results
    lines.append("\n1. SHAPIRO-WILK NORMALITY TESTS")
    lines.append("-" * 70)
    lines.append(f"Hospital 1:")
    lines.append(f"  Statistic (W): {result.shapiro_hospital_1.statistic:.6f}")
    lines.append(f"  P-value:       {result.shapiro_hospital_1.p_value:.6f}")
    lines.append(f"  Normal:        {result.shapiro_hospital_1.is_normal}")
    
    lines.append(f"\nHospital 2:")
    lines.append(f"  Statistic (W): {result.shapiro_hospital_2.statistic:.6f}")
    lines.append(f"  P-value:       {result.shapiro_hospital_2.p_value:.6f}")
    lines.append(f"  Normal:        {result.shapiro_hospital_2.is_normal}")
    
    # Comparison test results
    lines.append(f"\n2. COMPARISON TEST: {result.test_type.upper()}")
    lines.append("-" * 70)
    
    if result.test_type == 't-test':
        lines.append(f"  T-statistic:   {result.test_result.t_statistic:.6f}")
        lines.append(f"  P-value:       {result.test_result.p_value:.6f}")
    else:  # wilcoxon
        lines.append(f"  U-statistic:   {result.test_result.statistic:.6f}")
        lines.append(f"  P-value:       {result.test_result.p_value:.6f}")
    
    # Interpretation
    lines.append("\n3. INTERPRETATION")
    lines.append("-" * 70)
    alpha = 0.05
    if result.test_result.p_value < alpha:
        lines.append(f"  Result: SIGNIFICANT difference (p < {alpha})")
        lines.append("  The two hospitals show statistically different door-to-needle times.")
    else:
        lines.append(f"  Result: NO significant difference (p >= {alpha})")
        lines.append("  The two hospitals show similar door-to-needle times.")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


def save_results(result, expanded_1_size, expanded_2_size, results_dir="results"):
    """Save statistical analysis results to a file.
    
    Parameters
    ----------
    result : PipelineResult
        The complete pipeline result to save.
    expanded_1_size : int
        Size of expanded hospital 1 dataset.
    expanded_2_size : int
        Size of expanded hospital 2 dataset.
    results_dir : str, default="results"
        Directory to save results in.
    
    Returns
    -------
    str
        Path to the saved results file.
    """
    # Create results directory relative to this script's location
    script_dir = Path(__file__).parent
    results_path = script_dir / results_dir
    results_path.mkdir(exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_results_{timestamp}.txt"
    filepath = results_path / filename
    
    # Format and save results
    formatted_results = format_results(result, expanded_1_size, expanded_2_size)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(formatted_results)
    
    return str(filepath)


def main():
    """Execute the complete statistical analysis pipeline."""
    # Configuration
    N_ROWS = 50  # Number of rows in each dataset
    RANDOM_STATE = 42  # For reproducibility
    
    print("Generating synthetic hospital datasets...")
    df_hospital_1, df_hospital_2 = generate_both_hospital_datasets(
        n_rows=N_ROWS,
        random_state=RANDOM_STATE
    )
    
    print(f"\nHospital 1 dataset shape: {df_hospital_1.shape}")
    print("Hospital 1 DataFrame:")
    print(df_hospital_1)
    
    print(f"\nHospital 2 dataset shape: {df_hospital_2.shape}")
    print("Hospital 2 DataFrame:")
    print(df_hospital_2)
    
    print("\nExpanding datasets based on frequency counts...")
    expanded_1, expanded_2 = expand_both_datasets(df_hospital_1, df_hospital_2)
    
    print(f"Hospital 1 expanded size: {len(expanded_1)} observations")
    print(f"Hospital 2 expanded size: {len(expanded_2)} observations")
    
    print("\nRunning statistical analysis pipeline...")
    result = run_statistical_pipeline(expanded_1, expanded_2)
    
    print("\nSaving results to file...")
    results_file = save_results(result, len(expanded_1), len(expanded_2))
    print(f"Results saved to: {results_file}")
    
    print("\n")
    formatted_output = format_results(result, len(expanded_1), len(expanded_2))
    print(formatted_output)


if __name__ == "__main__":
    main()
