import pytest
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import train_test_split
from shap_select import score_features


@pytest.fixture
def generate_data_regression():
    np.random.seed(42)
    n_samples = 100000

    # Create 9 normally distributed features
    X = pd.DataFrame(
        {
            "x1": np.random.normal(size=n_samples),
            "x2": np.random.normal(size=n_samples),
            "x3": np.random.normal(size=n_samples),
            "x4": np.random.normal(size=n_samples),
            "x5": np.random.normal(size=n_samples),
            "x6": np.random.normal(size=n_samples),
            "x7": np.random.normal(size=n_samples),
            "x8": np.random.normal(size=n_samples),
            "x9": np.random.normal(size=n_samples),
        }
    )

    # Make all the features positive-ish
    X += 3

    # Define the target based on the formula y = x1 + x2*x3 + x4*x5*x6
    y = (
        3 * X["x1"]
        + X["x2"] * X["x3"]
        + X["x4"] * X["x5"] * X["x6"]
        + 10 * np.random.normal(size=n_samples)  # lots of noise
    )
    X["x6"] *= 0.1
    X["x6"] += np.random.normal(size=n_samples)

    # Split the dataset into training and validation sets (both with 10K rows)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    return X_train, X_val, y_train, y_val


@pytest.fixture
def generate_data_binary():
    np.random.seed(42)
    n_samples = 100000

    # Create 9 normally distributed features
    X = pd.DataFrame(
        {
            "x1": np.random.normal(size=n_samples),
            "x2": np.random.normal(size=n_samples),
            "x3": np.random.normal(size=n_samples),
            # "x4": np.random.normal(size=n_samples),
            # "x5": np.random.normal(size=n_samples),
            # "x6": np.random.normal(size=n_samples),
            "x7": np.random.normal(size=n_samples),
            "x8": np.random.normal(size=n_samples),
            "x9": np.random.normal(size=n_samples),
        }
    )

    # Make all the features positive-ish
    X += 3

    # Create a binary target based on a threshold
    y = (X["x1"] + X["x2"] * X["x3"] > 12).astype(int)

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    return X_train, X_val, y_train, y_val


@pytest.fixture
def generate_data_multiclass():
    np.random.seed(42)
    n_samples = 100000

    # Create 9 normally distributed features
    X = pd.DataFrame(
        {
            "x1": np.random.normal(size=n_samples),
            "x2": np.random.normal(size=n_samples),
            "x3": np.random.normal(size=n_samples),
            # "x4": np.random.normal(size=n_samples),
            # "x5": np.random.normal(size=n_samples),
            # "x6": np.random.normal(size=n_samples),
            "x7": np.random.normal(size=n_samples),
            "x8": np.random.normal(size=n_samples),
            "x9": np.random.normal(size=n_samples),
        }
    )

    # Make all the features positive-ish
    X += 3

    # Create a multiclass target with 3 classes
    y = pd.cut(
        X["x1"] + X["x2"] * X["x3"],  # + X["x4"] * X["x5"] * X["x6"],
        bins=3,
        labels=[0, 1, 2],
    ).astype(int)

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    return X_train, X_val, y_train, y_val


def train_lightgbm(X_train, X_val, y_train, y_val, task_type):
    """Train a LightGBM model based on the task type"""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    if task_type == "binary":
        params = {"objective": "binary", "metric": "binary_logloss", "verbose": -1}
    elif task_type == "regression":
        params = {"objective": "regression", "metric": "rmse", "verbose": -1}
    elif task_type == "multiclass":
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "verbose": -1,
        }
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
    )
    return model


def train_xgboost(X_train, X_val, y_train, y_val, task_type):
    """Train an XGBoost model based on the task type"""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    if task_type == "binary":
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "verbosity": 0,
        }
    elif task_type == "regression":
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "verbosity": 0,
        }
    elif task_type == "multiclass":
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "verbosity": 0,
        }
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    evals = [(dval, "valid")]
    model = xgb.train(
        params, dtrain, num_boost_round=1000, evals=evals, early_stopping_rounds=50
    )
    return model


def train_catboost(X_train, X_val, y_train, y_val, task_type):
    """Train a CatBoost model based on the task type"""
    if task_type == "binary":
        model = cb.CatBoostClassifier(
            iterations=1000,
            loss_function="Logloss",
            verbose=0,
            early_stopping_rounds=50,
        )
    elif task_type == "regression":
        model = cb.CatBoostRegressor(
            iterations=1000, loss_function="RMSE", verbose=0, early_stopping_rounds=50
        )
    elif task_type == "multiclass":
        model = cb.CatBoostClassifier(
            iterations=1000,
            loss_function="MultiClass",
            verbose=0,
            early_stopping_rounds=50,
        )
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    return model


@pytest.mark.parametrize(
    "model_type",
    ["lightgbm", "xgboost", "catboost"],
)
@pytest.mark.parametrize(
    "data_fixture, task_type",
    [
        ("generate_data_regression", "regression"),
        ("generate_data_binary", "binary"),
        ("generate_data_multiclass", "multiclass"),
    ],
)
def test_selected_column_values(model_type, data_fixture, task_type, request):
    """Parameterized test for regression, binary classification, and multiclass classification."""
    X_train, X_val, y_train, y_val = request.getfixturevalue(data_fixture)

    # Select the correct model to train
    if model_type == "lightgbm":
        model = train_lightgbm(X_train, X_val, y_train, y_val, task_type)
    elif model_type == "xgboost":
        model = train_xgboost(X_train, X_val, y_train, y_val, task_type)
    elif model_type == "catboost":
        model = train_catboost(X_train, X_val, y_train, y_val, task_type)
    else:
        raise ValueError("Unsupported model type")

    # Call the score_features function for the correct task (regression, binary, multiclass)
    selected_features_df = score_features(
        model, X_val, X_val.columns.tolist(), y_val, task=task_type
    )

    # Check feature significance for all task types
    selected_rows = selected_features_df[
        selected_features_df["feature name"].isin(["x7", "x8", "x9"])
    ]
    assert (
        selected_rows["Selected"] <= 0
    ).all(), (
        "The Selected column must have negative or zero values for features x7, x8, x9"
    )

    other_features_rows = selected_features_df[
        ~selected_features_df["feature name"].isin(["x7", "x8", "x9", "const"])
    ]
    assert (
        other_features_rows["Selected"] > 0
    ).all(), "The Selected column must have positive values for features other than x7, x8, x9"
