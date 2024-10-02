import pytest
import pandas as pd
import numpy as np
from shap_select.select import binary_classifier_significance, regression_significance
import statsmodels.api as sm


@pytest.fixture
def shap_features_binary():
    """Generate sample SHAP values for binary classification."""
    np.random.seed(42)
    return pd.DataFrame(np.random.normal(size=(100, 5)), columns=[f"x{i}" for i in range(5)])


@pytest.fixture
def binary_target():
    """Generate binary target."""
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1], size=100))


def test_binary_classifier_significance(shap_features_binary, binary_target):
    """Test significance calculation for binary classification."""
    result_df = binary_classifier_significance(shap_features_binary, binary_target, alpha=1e-4)
    
    assert "feature name" in result_df.columns, "Result should contain feature names"
    assert "coefficient" in result_df.columns, "Result should contain coefficients"
    assert "stat.significance" in result_df.columns, "Result should contain statistical significance"
    assert result_df.shape[0] == shap_features_binary.shape[1], "Each feature should have a row in the output"
    assert (result_df["stat.significance"] > 0).all(), "All p-values should be non-negative"


@pytest.fixture
def shap_features_regression():
    """Generate sample SHAP values for regression."""
    np.random.seed(42)
    return pd.DataFrame(np.random.normal(size=(100, 5)), columns=[f"x{i}" for i in range(5)])


@pytest.fixture
def regression_target():
    """Generate regression target."""
    np.random.seed(42)
    return pd.Series(np.random.normal(size=100))


def test_regression_significance(shap_features_regression, regression_target):
    """Test significance calculation for regression."""
    result_df = regression_significance(shap_features_regression, regression_target, alpha=1e-6)

    assert "feature name" in result_df.columns, "Result should contain feature names"
    assert "coefficient" in result_df.columns, "Result should contain coefficients"
    assert "stat.significance" in result_df.columns, "Result should contain statistical significance"
    assert result_df.shape[0] == shap_features_regression.shape[1], "Each feature should have a row in the output"
    assert (result_df["stat.significance"] > 0).all(), "All p-values should be non-negative"
