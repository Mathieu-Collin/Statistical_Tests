"""Unit tests for data_generator module."""

import pytest
import numpy as np
import pandas as pd
from src.data_generator import generate_hospital_dataset, generate_both_hospital_datasets


class TestGenerateHospitalDataset:
    """Tests for generate_hospital_dataset function."""
    
    def test_default_parameters(self):
        """Test dataset generation with default parameters."""
        df = generate_hospital_dataset()
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (50, 2)
        assert list(df.columns) == ['door_to_needle', 'n']
    
    def test_custom_n_rows(self):
        """Test dataset generation with custom number of rows."""
        df = generate_hospital_dataset(n_rows=100)
        
        assert df.shape == (100, 2)
    
    def test_value_ranges(self):
        """Test that generated values are within specified ranges."""
        df = generate_hospital_dataset(
            n_rows=100,
            door_to_needle_min=30,
            door_to_needle_max=120,
            n_min=1,
            n_max=50,
            random_state=42
        )
        
        assert df['door_to_needle'].min() >= 30
        assert df['door_to_needle'].max() <= 120
        assert df['n'].min() >= 1
        assert df['n'].max() <= 50
    
    def test_data_types(self):
        """Test that columns have correct data types."""
        df = generate_hospital_dataset()
        
        assert pd.api.types.is_integer_dtype(df['door_to_needle'])
        assert pd.api.types.is_integer_dtype(df['n'])
    
    def test_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        df1 = generate_hospital_dataset(random_state=42)
        df2 = generate_hospital_dataset(random_state=42)
        
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_different_seeds_produce_different_data(self):
        """Test that different random states produce different data."""
        df1 = generate_hospital_dataset(random_state=42)
        df2 = generate_hospital_dataset(random_state=43)
        
        assert not df1.equals(df2)


class TestGenerateBothHospitalDatasets:
    """Tests for generate_both_hospital_datasets function."""
    
    def test_returns_two_dataframes(self):
        """Test that function returns two DataFrames."""
        df1, df2 = generate_both_hospital_datasets()
        
        assert isinstance(df1, pd.DataFrame)
        assert isinstance(df2, pd.DataFrame)
    
    def test_both_have_correct_shape(self):
        """Test that both DataFrames have correct shape."""
        df1, df2 = generate_both_hospital_datasets(n_rows=50)
        
        assert df1.shape == (50, 2)
        assert df2.shape == (50, 2)
    
    def test_both_have_correct_columns(self):
        """Test that both DataFrames have correct columns."""
        df1, df2 = generate_both_hospital_datasets()
        
        assert list(df1.columns) == ['door_to_needle', 'n']
        assert list(df2.columns) == ['door_to_needle', 'n']
    
    def test_datasets_are_different(self):
        """Test that the two generated datasets are different."""
        df1, df2 = generate_both_hospital_datasets(random_state=42)
        
        assert not df1.equals(df2)
    
    def test_reproducibility(self):
        """Test reproducibility with random_state."""
        df1_a, df2_a = generate_both_hospital_datasets(random_state=42)
        df1_b, df2_b = generate_both_hospital_datasets(random_state=42)
        
        pd.testing.assert_frame_equal(df1_a, df1_b)
        pd.testing.assert_frame_equal(df2_a, df2_b)
