"""Example usage of the monolithic hospital_statistics module.

This file demonstrates various ways to use the module in different contexts,
including integration scenarios for chatbots like Rasa.
"""

from hospital_statistics import (
    HospitalStatistics,
    quick_analysis,
    analyze_dataframes
)
import pandas as pd


def example_1_quick_usage():
    """Example 1: Quickest way to run an analysis."""
    print("=" * 70)
    print("EXAMPLE 1: Quick Analysis")
    print("=" * 70)
    
    result = quick_analysis(n_rows=50, random_state=42, save=True)
    print(result.get_summary())
    print()


def example_2_class_usage():
    """Example 2: Using the HospitalStatistics class."""
    print("=" * 70)
    print("EXAMPLE 2: Class-Based Usage")
    print("=" * 70)
    
    # Create analyzer instance
    stats = HospitalStatistics(alpha=0.05)
    
    # Run analysis
    result = stats.run_analysis(n_rows=100, random_state=123)
    
    # Access specific results
    print(f"Test type: {result.test_type}")
    print(f"P-value: {result.test_result.p_value:.6f}")
    print(f"Significant: {result.is_significant()}")
    print(f"Hospital 1 normal: {result.shapiro_hospital_1.is_normal}")
    print(f"Hospital 2 normal: {result.shapiro_hospital_2.is_normal}")
    print()


def example_3_custom_data():
    """Example 3: Using custom DataFrames."""
    print("=" * 70)
    print("EXAMPLE 3: Custom DataFrames")
    print("=" * 70)
    
    # Create custom data
    df1 = pd.DataFrame({
        'door_to_needle': [45, 50, 55, 60, 65],
        'n': [10, 15, 20, 15, 10]
    })
    
    df2 = pd.DataFrame({
        'door_to_needle': [50, 55, 60, 65, 70],
        'n': [12, 18, 22, 18, 12]
    })
    
    # Analyze
    result = analyze_dataframes(df1, df2, save=False)
    print(result.get_summary())
    print()


def example_4_dict_output():
    """Example 4: Getting results as dictionary (for APIs/chatbots)."""
    print("=" * 70)
    print("EXAMPLE 4: Dictionary Output for Integration")
    print("=" * 70)
    
    stats = HospitalStatistics()
    result = stats.run_analysis(n_rows=50, random_state=999)
    
    # Get as dictionary (useful for JSON APIs, Rasa responses, etc.)
    result_dict = result.to_dict()
    
    print("Result as dictionary:")
    import json
    print(json.dumps(result_dict, indent=2))
    print()


def example_5_step_by_step():
    """Example 5: Step-by-step analysis (maximum control)."""
    print("=" * 70)
    print("EXAMPLE 5: Step-by-Step Analysis")
    print("=" * 70)
    
    stats = HospitalStatistics()
    
    # Step 1: Generate data
    df1 = stats.generate_dataset(n_rows=50, random_state=42)
    df2 = stats.generate_dataset(n_rows=50, random_state=43)
    
    print(f"Generated datasets:")
    print(f"  Hospital 1: {df1.shape}")
    print(f"  Hospital 2: {df2.shape}")
    
    # Step 2: Expand data
    expanded_1 = stats.expand_dataset(df1)
    expanded_2 = stats.expand_dataset(df2)
    
    print(f"\nExpanded datasets:")
    print(f"  Hospital 1: {len(expanded_1)} observations")
    print(f"  Hospital 2: {len(expanded_2)} observations")
    
    # Step 3: Run tests
    result = stats.run_analysis_from_arrays(expanded_1, expanded_2)
    
    print(f"\nTest results:")
    print(f"  Test type: {result.test_type}")
    print(f"  Significant: {result.is_significant()}")
    print()


def example_6_rasa_integration_template():
    """Example 6: Template for Rasa integration."""
    print("=" * 70)
    print("EXAMPLE 6: Rasa Integration Template")
    print("=" * 70)
    
    print("""
# In your Rasa custom action (actions/actions.py):

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from hospital_statistics import HospitalStatistics
import pandas as pd

class ActionAnalyzeHospitals(Action):
    def name(self) -> str:
        return "action_analyze_hospitals"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> list:
        
        # Initialize statistics analyzer
        stats = HospitalStatistics()
        
        # Option 1: Analyze synthetic data
        result = stats.run_analysis(n_rows=50, random_state=42)
        
        # Option 2: Load real data from files/database
        # df1 = pd.read_csv('data/hospital1.csv')
        # df2 = pd.read_csv('data/hospital2.csv')
        # result = stats.run_analysis_from_dataframes(df1, df2)
        
        # Get results as dictionary
        result_dict = result.to_dict()
        
        # Create user-friendly message
        if result.is_significant():
            message = (
                f"L'analyse statistique montre une différence SIGNIFICATIVE "
                f"entre les deux hôpitaux (p = {result.test_result.p_value:.4f}). "
                f"Test utilisé: {result.test_type}."
            )
        else:
            message = (
                f"L'analyse statistique ne montre PAS de différence significative "
                f"entre les deux hôpitaux (p = {result.test_result.p_value:.4f}). "
                f"Test utilisé: {result.test_type}."
            )
        
        # Send message to user
        dispatcher.utter_message(text=message)
        
        # Optionally save detailed results
        # output_file = stats.save_result(result)
        
        return []
    """)
    print()


if __name__ == "__main__":
    example_1_quick_usage()
    example_2_class_usage()
    example_3_custom_data()
    example_4_dict_output()
    example_5_step_by_step()
    example_6_rasa_integration_template()
