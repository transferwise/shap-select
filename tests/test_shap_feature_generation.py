import pytest
import pandas as pd
import numpy as np
from shap_select import create_shap_features
import lightgbm as lgb


@pytest.fixture
def sample_data_binary():
    """Generate sample data for binary classification."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.normal(size=(100, 5)), columns=[f"x{i}" for i in range(5)])
    y = (X["x0"] > 0).astype(int)
    return X, y


@pytest.fixture
def sample_data_multiclass():
    """Generate sample data for multiclass classification."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.normal(size=(100, 5)), columns=[f"x{i}" for i in range(5)])
    y = np.random.choice([0, 1, 2], size=100)
    return X, y


def test_shap_feature_generation_binary(sample_data_binary):
    """Test SHAP feature generation for binary classification."""
    X, y = sample_data_binary

    model = lgb.LGBMClassifier()
    model.fit(X, y)

    shap_df = create_shap_features(model, X)
    assert isinstance(shap_df, pd.DataFrame), "SHAP output should be a DataFrame"
    assert shap_df.shape == X.shape, "SHAP output shape should match input data"
    assert shap_df.isnull().sum().sum() == 0, "No missing values expected in SHAP output"


def test_shap_feature_generation_multiclass(sample_data_multiclass):
    """Test SHAP feature generation for multiclass classification."""
    X, y = sample_data_multiclass

    model = lgb.LGBMClassifier(objective="multiclass", num_class=3)
    model.fit(X, y)

    shap_df = create_shap_features(model, X, classes=[0, 1, 2])
    assert isinstance(shap_df, dict), "SHAP output should be a dictionary for multiclass"
    assert all(isinstance(v, pd.DataFrame) for v in shap_df.values()), "Each class should have a DataFrame"
    assert shap_df[0].shape == X.shape, "SHAP output shape should match input data for each class"
