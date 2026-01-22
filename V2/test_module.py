"""Simple test script for V2 monolithic module."""

from hospital_statistics import HospitalStatistics


def main():
    """Run a simple test of the module."""
    print("Testing V2 Monolithic Hospital Statistics Module\n")
    print("=" * 70)
    
    # Create analyzer
    stats = HospitalStatistics()
    
    # Run analysis
    print("Generating synthetic data and running analysis...")
    result = stats.run_analysis(n_rows=50, random_state=42)
    
    # Display results
    print("\n" + result.get_summary())
    
    # Save to file
    print("\nSaving results...")
    output_file = stats.save_result(result)
    print(f"Results saved to: {output_file}")
    
    # Show dictionary format
    print("\n" + "=" * 70)
    print("Result as dictionary (for API/Rasa integration):")
    print("=" * 70)
    result_dict = result.to_dict()
    
    import json
    print(json.dumps(result_dict, indent=2))


if __name__ == "__main__":
    main()
