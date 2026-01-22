"""Statistical tests module for hospital data analysis.

This module provides functions to perform normality tests (Shapiro-Wilk),
parametric tests (t-test), and non-parametric tests (Wilcoxon) on
hospital datasets.
"""

import numpy as np
from scipy import stats
from typing import NamedTuple, Literal


class ShapiroResult(NamedTuple):
    """Result of a Shapiro-Wilk normality test.
    
    Attributes
    ----------
    statistic : float
        The test statistic (W).
    p_value : float
        The p-value for the hypothesis test.
    is_normal : bool
        True if data appears to be normally distributed (p > 0.05).
    """
    statistic: float
    p_value: float
    is_normal: bool


class TTestResult(NamedTuple):
    """Result of an independent samples t-test.
    
    Attributes
    ----------
    t_statistic : float
        The t-test statistic.
    p_value : float
        The two-tailed p-value.
    """
    t_statistic: float
    p_value: float


class WilcoxonResult(NamedTuple):
    """Result of a Wilcoxon rank-sum (Mann-Whitney U) test.
    
    Attributes
    ----------
    statistic : float
        The test statistic (U-statistic).
    p_value : float
        The p-value for the hypothesis test.
    """
    statistic: float
    p_value: float


class PipelineResult(NamedTuple):
    """Complete result of the statistical analysis pipeline.
    
    Attributes
    ----------
    shapiro_hospital_1 : ShapiroResult
        Shapiro-Wilk test result for hospital 1 data.
    shapiro_hospital_2 : ShapiroResult
        Shapiro-Wilk test result for hospital 2 data.
    test_type : Literal['t-test', 'wilcoxon']
        Type of comparison test performed.
    test_result : TTestResult or WilcoxonResult
        Result of the comparison test (t-test or Wilcoxon).
    """
    shapiro_hospital_1: ShapiroResult
    shapiro_hospital_2: ShapiroResult
    test_type: Literal['t-test', 'wilcoxon']
    test_result: TTestResult | WilcoxonResult


def shapiro_wilk_test(data: np.ndarray, alpha: float = 0.05) -> ShapiroResult:
    """Perform Shapiro-Wilk test for normality.

    Parameters
    ----------
    data : np.ndarray
        1D array of sample data.
    alpha : float, default=0.05
        Significance level for normality decision.

    Returns
    -------
    ShapiroResult
        Named tuple containing test statistic, p-value, and normality decision.

    Examples
    --------
    >>> data = np.random.normal(0, 1, 100)
    >>> result = shapiro_wilk_test(data)
    >>> result.is_normal
    True
    """
    statistic, p_value = stats.shapiro(data)
    is_normal = p_value > alpha
    
    return ShapiroResult(
        statistic=float(statistic),
        p_value=float(p_value),
        is_normal=bool(is_normal)
    )


def perform_t_test(
    data1: np.ndarray,
    data2: np.ndarray,
    equal_var: bool = True
) -> TTestResult:
    """Perform independent samples t-test.

    Parameters
    ----------
    data1 : np.ndarray
        First sample data.
    data2 : np.ndarray
        Second sample data.
    equal_var : bool, default=True
        If True, perform standard independent t-test assuming equal variances.
        If False, perform Welch's t-test not assuming equal variances.

    Returns
    -------
    TTestResult
        Named tuple containing t-statistic and p-value.

    Examples
    --------
    >>> data1 = np.random.normal(100, 10, 50)
    >>> data2 = np.random.normal(105, 10, 50)
    >>> result = perform_t_test(data1, data2)
    >>> result.t_statistic
    -2.345...
    """
    statistic, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
    
    return TTestResult(
        t_statistic=float(statistic),
        p_value=float(p_value)
    )


def perform_wilcoxon_test(
    data1: np.ndarray,
    data2: np.ndarray
) -> WilcoxonResult:
    """Perform Wilcoxon rank-sum test (Mann-Whitney U test).

    This is the non-parametric alternative to the independent samples t-test,
    used when data is not normally distributed.

    Parameters
    ----------
    data1 : np.ndarray
        First sample data.
    data2 : np.ndarray
        Second sample data.

    Returns
    -------
    WilcoxonResult
        Named tuple containing U-statistic and p-value.

    Notes
    -----
    This function uses scipy.stats.mannwhitneyu, which is equivalent to
    the Wilcoxon rank-sum test. The statistic returned is the U-statistic.

    Examples
    --------
    >>> data1 = np.array([1, 2, 3, 4, 5])
    >>> data2 = np.array([6, 7, 8, 9, 10])
    >>> result = perform_wilcoxon_test(data1, data2)
    >>> result.p_value < 0.05
    True
    """
    statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    
    return WilcoxonResult(
        statistic=float(statistic),
        p_value=float(p_value)
    )


def run_statistical_pipeline(
    data_hospital_1: np.ndarray,
    data_hospital_2: np.ndarray,
    alpha: float = 0.05
) -> PipelineResult:
    """Run complete statistical analysis pipeline on two hospital datasets.

    Pipeline steps:
    1. Perform Shapiro-Wilk test on both datasets to check normality
    2. If both datasets are normal (p > alpha), perform parametric t-test
    3. Otherwise, perform non-parametric Wilcoxon rank-sum test

    Parameters
    ----------
    data_hospital_1 : np.ndarray
        Expanded data from hospital 1.
    data_hospital_2 : np.ndarray
        Expanded data from hospital 2.
    alpha : float, default=0.05
        Significance level for normality test.

    Returns
    -------
    PipelineResult
        Complete results including normality tests and comparison test.

    Notes
    -----
    The choice between parametric and non-parametric tests is automatic
    based on the normality assessment of both datasets.

    Examples
    --------
    >>> data1 = np.random.normal(100, 10, 100)
    >>> data2 = np.random.normal(105, 10, 100)
    >>> result = run_statistical_pipeline(data1, data2)
    >>> result.test_type
    't-test'
    >>> result.shapiro_hospital_1.is_normal
    True
    """
    shapiro_1 = shapiro_wilk_test(data_hospital_1, alpha=alpha)
    shapiro_2 = shapiro_wilk_test(data_hospital_2, alpha=alpha)
    
    if shapiro_1.is_normal and shapiro_2.is_normal:
        test_type: Literal['t-test', 'wilcoxon'] = 't-test'
        test_result = perform_t_test(data_hospital_1, data_hospital_2)
    else:
        test_type: Literal['t-test', 'wilcoxon'] = 'wilcoxon'
        test_result = perform_wilcoxon_test(data_hospital_1, data_hospital_2)
    
    return PipelineResult(
        shapiro_hospital_1=shapiro_1,
        shapiro_hospital_2=shapiro_2,
        test_type=test_type,
        test_result=test_result
    )
