"""Unit tests for data_processor module."""

import pytest
import numpy as np
import pandas as pd
from src.data_processor import expand_dataset, expand_both_datasets


class TestExpandDataset:
    """Tests for expand_dataset function."""
    
    def test_basic_expansion(self):
        """Test basic dataset expansion."""
        df = pd.DataFrame({
            'door_to_needle': [30, 45, 60],
            'n': [2, 3, 1]
        })
        
        result = expand_dataset(df)
        expected = np.array([30, 30, 45, 45, 45, 60])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_expansion_length(self):
        """Test that expanded array has correct total length."""
        df = pd.DataFrame({
            'door_to_needle': [30, 45, 60],
            'n': [2, 3, 1]
        })
        
        result = expand_dataset(df)
        expected_length = df['n'].sum()
        
        assert len(result) == expected_length
    
    def test_single_row(self):
        """Test expansion with single row."""
        df = pd.DataFrame({
            'door_to_needle': [50],
            'n': [5]
        })
        
        result = expand_dataset(df)
        expected = np.array([50, 50, 50, 50, 50])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_return_type(self):
        """Test that function returns numpy array."""
        df = pd.DataFrame({
            'door_to_needle': [30, 45],
            'n': [1, 1]
        })
        
        result = expand_dataset(df)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64
    
    def test_missing_columns_raises_error(self):
        """Test that missing columns raise ValueError."""
        df = pd.DataFrame({
            'door_to_needle': [30, 45]
        })
        
        with pytest.raises(ValueError, match="must contain columns"):
            expand_dataset(df)
    
    def test_wrong_column_names_raises_error(self):
        """Test that wrong column names raise ValueError."""
        df = pd.DataFrame({
            'time': [30, 45],
            'count': [1, 2]
        })
        
        with pytest.raises(ValueError, match="must contain columns"):
            expand_dataset(df)
    
    def test_invalid_input_type_raises_error(self):
        """Test that non-DataFrame input raises TypeError."""
        invalid_input = [[30, 2], [45, 3]]
        
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            expand_dataset(invalid_input)
    
    def test_zero_frequency(self):
        """Test handling of zero frequency values."""
        df = pd.DataFrame({
            'door_to_needle': [30, 45, 60],
            'n': [2, 0, 1]
        })
        
        result = expand_dataset(df)
        expected = np.array([30, 30, 60])
        
        np.testing.assert_array_equal(result, expected)


class TestExpandBothDatasets:
    """Tests for expand_both_datasets function."""
    
    def test_returns_two_arrays(self):
        """Test that function returns two numpy arrays."""
        df1 = pd.DataFrame({'door_to_needle': [30, 45], 'n': [2, 1]})
        df2 = pd.DataFrame({'door_to_needle': [40, 50], 'n': [1, 3]})
        
        result1, result2 = expand_both_datasets(df1, df2)
        
        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)
    
    def test_correct_expansion_both_datasets(self):
        """Test correct expansion of both datasets."""
        df1 = pd.DataFrame({'door_to_needle': [30, 45], 'n': [2, 1]})
        df2 = pd.DataFrame({'door_to_needle': [40, 50], 'n': [1, 3]})
        
        result1, result2 = expand_both_datasets(df1, df2)
        
        expected1 = np.array([30, 30, 45])
        expected2 = np.array([40, 50, 50, 50])
        
        np.testing.assert_array_equal(result1, expected1)
        np.testing.assert_array_equal(result2, expected2)
    
    def test_different_sizes(self):
        """Test expansion with datasets of different sizes."""
        df1 = pd.DataFrame({'door_to_needle': [30], 'n': [10]})
        df2 = pd.DataFrame({'door_to_needle': [40, 50], 'n': [2, 3]})
        
        result1, result2 = expand_both_datasets(df1, df2)
        
        assert len(result1) == 10
        assert len(result2) == 5
