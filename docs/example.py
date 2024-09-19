import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split

from shap_select import score_features

# Generate a dataset with 8 normally distributed features and a target based on a given formula
np.random.seed(42)
n_samples = 100000

# Create 8 normally distributed features
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

# make all the features positive-ish
X += 3

# Define the target based on the formula y = x1 + x2*x3 + x4*x5*x6
y = (
    X["x1"]
    + X["x2"] * X["x3"]
    + X["x4"] * X["x5"] * X["x6"]
    + 10 * np.random.normal(size=n_samples)  # lots of noise
)
X["x6"] *= 0.1
X["x6"] += np.random.normal(size=n_samples)

# Split the dataset into training and validation sets (both with 10K rows)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

lightgbm = True
stopping_rounds = 50

if lightgbm:

    # Train a LightGBM model on the training data
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = {"objective": "regression", "metric": "rmse", "verbose": -1}
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,  # Max number of boosting rounds
        valid_sets=[train_data, val_data],  # Validation sets
        valid_names=["train", "valid"],  # Name the datasets
        callbacks=[
            lgb.early_stopping(stopping_rounds=stopping_rounds)
        ],  # Stop if validation score doesn't improve for 10 rounds
    )
else:
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Set parameters for XGBoost
    params = {
        "objective": "reg:squarederror",  # Regression task
        "eval_metric": "rmse",  # Metric to evaluate
        "verbosity": 0,  # Set to 0 to disable output
    }

    # Train the model with early stopping
    evals = [(dval, "valid")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,  # Max number of boosting rounds
        evals=evals,  # Evaluation set
        early_stopping_rounds=stopping_rounds,  # Stop if validation RMSE doesn't improve for 10 rounds
    )


# Call the select_features function
selected_features_df, shap_features = score_features(
    model, X_val, X.columns.tolist(), y_val
)

# Output the resulting DataFrame
print(selected_features_df.head())
