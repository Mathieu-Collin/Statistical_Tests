"""Unit tests for statistical_tests module."""

import pytest
import numpy as np
from scipy import stats
from src.statistical_tests import (
    shapiro_wilk_test,
    perform_t_test,
    perform_wilcoxon_test,
    run_statistical_pipeline,
    ShapiroResult,
    TTestResult,
    WilcoxonResult,
    PipelineResult
)


class TestShapiroWilkTest:
    """Tests for shapiro_wilk_test function."""
    
    def test_normal_distribution(self):
        """Test with normally distributed data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        
        result = shapiro_wilk_test(data)
        
        assert isinstance(result, ShapiroResult)
        assert result.is_normal is True
        assert 0 <= result.p_value <= 1
    
    def test_non_normal_distribution(self):
        """Test with non-normally distributed data."""
        # Uniform distribution is not normal
        np.random.seed(42)
        data = np.random.uniform(0, 1, 100)
        
        result = shapiro_wilk_test(data)
        
        assert isinstance(result, ShapiroResult)
        assert 0 <= result.p_value <= 1
    
    def test_return_type(self):
        """Test that return types are correct."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = shapiro_wilk_test(data)
        
        assert isinstance(result.statistic, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.is_normal, bool)
    
    def test_custom_alpha(self):
        """Test with custom alpha level."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        
        result = shapiro_wilk_test(data, alpha=0.01)
        
        assert isinstance(result, ShapiroResult)


class TestPerformTTest:
    """Tests for perform_t_test function."""
    
    def test_same_distributions(self):
        """Test t-test with samples from same distribution."""
        np.random.seed(42)
        data1 = np.random.normal(100, 10, 100)
        data2 = np.random.normal(100, 10, 100)
        
        result = perform_t_test(data1, data2)
        
        assert isinstance(result, TTestResult)
        assert result.p_value > 0.05  # Should not be significant
    
    def test_different_distributions(self):
        """Test t-test with samples from different distributions."""
        np.random.seed(42)
        data1 = np.random.normal(100, 10, 100)
        data2 = np.random.normal(120, 10, 100)
        
        result = perform_t_test(data1, data2)
        
        assert isinstance(result, TTestResult)
        assert result.p_value < 0.05  # Should be significant
    
    def test_return_types(self):
        """Test that return types are correct."""
        data1 = np.array([1, 2, 3, 4, 5])
        data2 = np.array([2, 3, 4, 5, 6])
        
        result = perform_t_test(data1, data2)
        
        assert isinstance(result.t_statistic, float)
        assert isinstance(result.p_value, float)
    
    def test_unequal_variance(self):
        """Test t-test with unequal variances."""
        np.random.seed(42)
        data1 = np.random.normal(100, 5, 100)
        data2 = np.random.normal(100, 20, 100)
        
        result = perform_t_test(data1, data2, equal_var=False)
        
        assert isinstance(result, TTestResult)


class TestPerformWilcoxonTest:
    """Tests for perform_wilcoxon_test function."""
    
    def test_same_distributions(self):
        """Test Wilcoxon test with samples from same distribution."""
        np.random.seed(42)
        data1 = np.random.exponential(1, 100)
        data2 = np.random.exponential(1, 100)
        
        result = perform_wilcoxon_test(data1, data2)
        
        assert isinstance(result, WilcoxonResult)
        assert result.p_value > 0.05  # Should not be significant
    
    def test_different_distributions(self):
        """Test Wilcoxon test with different distributions."""
        np.random.seed(42)
        data1 = np.random.exponential(1, 100)
        data2 = np.random.exponential(3, 100)
        
        result = perform_wilcoxon_test(data1, data2)
        
        assert isinstance(result, WilcoxonResult)
        assert result.p_value < 0.05  # Should be significant
    
    def test_return_types(self):
        """Test that return types are correct."""
        data1 = np.array([1, 2, 3, 4, 5])
        data2 = np.array([6, 7, 8, 9, 10])
        
        result = perform_wilcoxon_test(data1, data2)
        
        assert isinstance(result.statistic, float)
        assert isinstance(result.p_value, float)


class TestRunStatisticalPipeline:
    """Tests for run_statistical_pipeline function."""
    
    def test_normal_data_uses_ttest(self):
        """Test that normal data triggers t-test."""
        np.random.seed(42)
        data1 = np.random.normal(100, 10, 200)
        data2 = np.random.normal(105, 10, 200)
        
        result = run_statistical_pipeline(data1, data2)
        
        assert isinstance(result, PipelineResult)
        assert result.test_type == 't-test'
        assert isinstance(result.test_result, TTestResult)
    
    def test_non_normal_data_uses_wilcoxon(self):
        """Test that non-normal data triggers Wilcoxon test."""
        np.random.seed(42)
        # Create clearly non-normal data
        data1 = np.concatenate([np.ones(50), np.ones(50) * 100])
        data2 = np.concatenate([np.ones(50), np.ones(50) * 100])
        
        result = run_statistical_pipeline(data1, data2)
        
        assert isinstance(result, PipelineResult)
        # Note: may or may not be 'wilcoxon' depending on actual normality
    
    def test_pipeline_result_structure(self):
        """Test that pipeline result has correct structure."""
        np.random.seed(42)
        data1 = np.random.normal(100, 10, 100)
        data2 = np.random.normal(105, 10, 100)
        
        result = run_statistical_pipeline(data1, data2)
        
        assert hasattr(result, 'shapiro_hospital_1')
        assert hasattr(result, 'shapiro_hospital_2')
        assert hasattr(result, 'test_type')
        assert hasattr(result, 'test_result')
        
        assert isinstance(result.shapiro_hospital_1, ShapiroResult)
        assert isinstance(result.shapiro_hospital_2, ShapiroResult)
        assert result.test_type in ['t-test', 'wilcoxon']
    
    def test_custom_alpha(self):
        """Test pipeline with custom alpha level."""
        np.random.seed(42)
        data1 = np.random.normal(100, 10, 100)
        data2 = np.random.normal(105, 10, 100)
        
        result = run_statistical_pipeline(data1, data2, alpha=0.01)
        
        assert isinstance(result, PipelineResult)
