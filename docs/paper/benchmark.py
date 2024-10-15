from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from boruta import BorutaPy
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import time
from shap_select import shap_select
import hisel
from shap_selection import feature_selection
from skfeature.function.information_theoretical_based import MRMR

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Global XGBoost parameters for consistency
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "verbosity": 0,
    "seed": RANDOM_SEED,
    "nthread": 1,
}


# Define common XGBoost model
def train_xgboost(X_train, y_train):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    xgb_model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=100)
    return xgb_model


def predict_xgboost(xgb_model, X_val):
    dval = xgb.DMatrix(X_val)
    y_pred = (xgb_model.predict(dval) > 0.5).astype(int)
    return y_pred


# HISEL feature selection using MRMR
def hisel_feature_selection(xgb_model, X_train, X_val, y_train, y_val, n_features):
    return hisel.feature_selection.select_features(X_train, y_train)


def shap_selection(xgb_model, X_train, X_val, y_train, y_val, n_features) -> List[str]:
    selected_shap_selection, _ = feature_selection.shap_select(
        xgb_model, X_train, X_val, X_train.columns, agnostic=False
    )
    selected_shap_selection = selected_shap_selection[:n_features]  # Why 15?
    return selected_shap_selection


def shap_select_selection(
    xgb_model, X_train, X_val, y_train, y_val, n_features
) -> List[str]:
    shap_features, _ = shap_select(
        xgb_model,
        X_val,
        y_val,
        task="binary",
        alpha=1e-6,
        threshold=0.05,
        return_extended_data=True,
    )
    selected_features = shap_features[shap_features["selected"] == 1][
        "feature name"
    ].tolist()
    return selected_features


def no_selection(xgb_model, X_train, X_val, y_train, y_val, n_features) -> List[str]:
    return list(X_train.columns)


def rfe_selection(xgb_model, X_train, X_val, y_train, y_val, n_features) -> List[str]:
    rfe = RFE(
        xgb.XGBClassifier(**XGB_PARAMS, use_label_encoder=False),
        n_features_to_select=n_features,
    )
    rfe.fit(X_train, y_train)
    selected_rfe = X_train.columns[rfe.support_]
    return selected_rfe


def boruta_selection(
    xgb_model, X_train, X_val, y_train, y_val, n_features
) -> List[str]:
    rf_model = xgb.XGBClassifier(**XGB_PARAMS, use_label_encoder=False)
    boruta_selector = BorutaPy(rf_model, n_estimators=100, random_state=RANDOM_SEED)
    boruta_selector.fit(X_train.values, y_train.values)
    selected_boruta = X_train.columns[boruta_selector.support_].tolist()
    return selected_boruta


method_dict = {
    "No selection": no_selection,
    "shap-select": shap_select_selection,
    "shap-selection": shap_selection,
    "HISEL": hisel_feature_selection,
    "Boruta": boruta_selection,
    "RFE": rfe_selection,
}


# Run experiments with different feature selection methods and shap-select p-values
def run_experiments(X_train, X_val, X_test, y_train, y_val, y_test):
    results = []
    pretrained_model = None

    for name, fun in method_dict.items():
        print(f"\n--- {name} ---")
        start_time = time.time()
        selected = fun(pretrained_model, X_train, X_val, y_train, y_val, n_features=15)

        runtime = time.time() - start_time
        print(
            f"{name} completed in {runtime:.2f} seconds with {len(selected)} features."
        )

        this_model = train_xgboost(X_train[selected], y_train)

        if name == "No selection":
            pretrained_model = this_model

        y_pred = predict_xgboost(this_model, X_test[selected])
        results.append(
            {
                "Method": name,
                "Selected Features": selected,
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "Runtime (s)": runtime,
            }
        )

    #     assert set(X_train.columns) == set(selected_hisel), "Feature sets differ!"

    results_df = pd.DataFrame(results)
    print("\n--- Experiment Results ---")
    print(results_df)
    return results_df, pretrained_model


if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv("creditcard.csv")
    X = df.drop(columns=["Class"])
    y = df["Class"]
    # Perform a 60-20-20 split for train, validation, and test sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_SEED
    )

    results_df, trained_model = run_experiments(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    print(results_df)
    print("yay!")
