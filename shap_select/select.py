from typing import Any, Tuple, List, Dict

import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
import shap


def create_shap_features(
    tree_model: Any, validation_df: pd.DataFrame, classes: List | None = None
) -> pd.DataFrame | Dict[Any, pd.DataFrame]:
    """
    Generates SHAP (SHapley Additive exPlanations) values for a given tree-based model on a validation dataset.

    Parameters:
    - tree_model (Any): A trained tree-based model (e.g., XGBoost, LightGBM, or any model compatible with SHAP).
    - validation_df (pd.DataFrame): A DataFrame containing the validation data on which SHAP values will be computed.
      The DataFrame should contain the same feature columns used to train the `tree_model`.

    Returns:
    - pd.DataFrame: A DataFrame containing the SHAP values for each feature in the `validation_df`, where each column
      corresponds to the SHAP values of a feature, and the rows match the index of the `validation_df`.
    """
    explainer = shap.Explainer(tree_model, model_output="raw")(validation_df)
    shap_values = explainer.values

    if len(shap_values.shape) == 2:
        assert (
            classes is None
        ), "Don't specify classes for binary classification or regression"
        # Create a DataFrame with the SHAP values, with one column per feature
        return pd.DataFrame(
            shap_values, columns=validation_df.columns, index=validation_df.index
        )
    elif len(shap_values.shape) == 3:  # multiclass classification
        out = {}
        for i, c in enumerate(classes):
            out[i] = pd.DataFrame(
                shap_values[:, :, i],
                columns=validation_df.columns,
                index=validation_df.index,
            )
        return out


def binary_classifier_significance(
    shap_features: pd.DataFrame, target: pd.Series, alpha: float
) -> pd.DataFrame:
    """
    Fits a logistic regression model using the features from `shap_features` to predict the binary `target`.
    Returns a DataFrame containing feature names, coefficients, standard errors, and the significance (p-values).

    Parameters:
    shap_features (pd.DataFrame): A DataFrame containing the SHAP values or features used for prediction.
    target (pd.Series): A binary target series (0 or 1) to classify.

    Returns:
    pd.DataFrame: A DataFrame containing:
        - feature name: The names of the features.
        - coefficient: The logistic regression coefficients for each feature.
        - stderr: The standard error for each coefficient.
        - stat.significance: The p-value (statistical significance) for each feature.
    """

    # Add a constant to the features for the intercept in logistic regression
    shap_features_with_constant = sm.add_constant(shap_features)

    # Fit the logistic regression model that will generate confidence intervals
    logit_model = sm.Logit(target, shap_features_with_constant)
    result = logit_model.fit_regularized(disp=False, alpha=alpha)

    # Extract the results
    summary_frame = result.summary2().tables[1]

    # Create the DataFrame with the required columns
    result_df = pd.DataFrame(
        {
            "feature name": summary_frame.index,
            "coefficient": summary_frame["Coef."],
            "stderr": summary_frame["Std.Err."],
            "stat.significance": summary_frame["P>|z|"],
            "t-value": summary_frame["Coef."] / summary_frame["Std.Err."],
        }
    ).reset_index(drop=True)
    result_df["closeness to 1.0"] = (result_df["coefficient"] - 1.0).abs()
    return result_df.loc[~(result_df["feature name"] == "const"), :]


def multi_classifier_significance(
    shap_features: Dict[Any, pd.DataFrame],
    target: pd.Series,
    alpha: float,
    return_individual_significances: bool = False,
) -> (pd.DataFrame, list):
    """
    Fits a binary logistic regression model for each unique class in the target, comparing each class against all others (one-vs-all).
    Calls binary_classifier_significance for each binary classification.

    Parameters:
    shap_features (pd.DataFrame): A DataFrame containing the features used for classification.
    target (pd.Series): A target series containing more than two classes.

    Returns:
    - A DataFrame with feature names and their maximum significance values across all binary classifications.
    - A list of DataFrames, one for each binary classification, containing feature names, coefficients, standard errors, and statistical significance.
    """
    significance_dfs = []

    # Iterate through each class and perform binary classification (one-vs-all)
    for cls, feature_df in shap_features.items():
        binary_target = (target == cls).astype(int)
        significance_df = binary_classifier_significance(
            feature_df, binary_target, alpha
        )
        significance_dfs.append(significance_df)

    # Combine results into a single DataFrame with the max significance value for each feature
    combined_df = pd.concat(significance_dfs)
    max_significance_df = (
        combined_df.groupby("feature name", as_index=False)
        .agg(
            {
                "t-value": "max",
                "closeness to 1.0": "min",
                "coefficient": "max",
            }
        )
        .reset_index(drop=True)
    )

    # Len(shap_features) multiplier is the Bonferroni correction
    max_significance_df["stat.significance"] = max_significance_df["t-value"].apply(
        lambda x: len(shap_features) * (1 - stats.norm.cdf(x))
    )
    if return_individual_significances:
        return max_significance_df, significance_dfs
    else:
        return max_significance_df


def regression_significance(
    shap_features: pd.DataFrame, target: pd.Series, alpha: float
) -> pd.DataFrame:
    """
    Fits a linear regression model using the features from `shap_features` to predict the continuous `target`.
    Returns a DataFrame containing feature names, coefficients, standard errors, and the significance (p-values).

    Parameters:
    shap_features (pd.DataFrame): A DataFrame containing the features used for prediction.
    target (pd.Series): A continuous target series to predict.

    Returns:
    pd.DataFrame: A DataFrame containing:
        - feature name: The names of the features.
        - coefficient: The linear regression coefficients for each feature.
        - stderr: The standard error for each coefficient.
        - stat.significance: The p-value (statistical significance) for each feature.
    """
    # Fit the linear regression model that will generate confidence intervals
    ols_model = sm.OLS(target, shap_features)
    result = ols_model.fit_regularized(alpha=alpha, refit=True)

    # Extract the results
    summary_frame = result.summary2().tables[1]

    # Create the DataFrame with the required columns
    result_df = pd.DataFrame(
        {
            "feature name": summary_frame.index,
            "coefficient": summary_frame["Coef."],
            "stderr": summary_frame["Std.Err."],
            "stat.significance": summary_frame["P>|t|"],
            "t-value": summary_frame["Coef."] / summary_frame["Std.Err."],
        }
    ).reset_index(drop=True)
    result_df["closeness to 1.0"] = (result_df["coefficient"] - 1.0).abs()

    return result_df


def closeness_to_one(df: pd.DataFrame) -> pd.Series:
    return (df["coefficient"] - 1.0) / df["stderr"]


def shap_features_to_significance(
    shap_features: pd.DataFrame | List[pd.DataFrame],
    target: pd.Series,
    task: str,
    alpha: float,
) -> pd.DataFrame:
    """
    Determines the task (regression, binary, or multi-class classification) based on the target and calls the appropriate
    significance function. Returns a DataFrame with feature names and their significance values.

    Parameters:
    shap_features (pd.DataFrame): A DataFrame containing the features used for prediction.
    target (pd.Series): The target series for prediction (either continuous or categorical).
    task (str): The type of task to perform: "regression", "binary", or "multiclass".

    Returns:
    pd.DataFrame: A DataFrame containing:
        - feature name: The names of the features.
        - stat.significance: The p-value (statistical significance) for each feature.
        Sorted in descending order of significance (ascending p-value).
    """

    # Call the appropriate function based on the task
    if task == "regression":
        result_df = regression_significance(shap_features, target, alpha)
    elif task == "binary":
        result_df = binary_classifier_significance(shap_features, target, alpha)
    elif task == "multiclass":
        result_df = multi_classifier_significance(shap_features, target, alpha)
    else:
        raise ValueError("`task` must be 'regression', 'binary', 'multiclass' or None.")

    # Sort the result by statistical significance in ascending order (more significant features first)
    result_df_sorted = result_df.sort_values(by="t-value", ascending=False).reset_index(
        drop=True
    )

    return result_df_sorted


def iterative_shap_feature_reduction(
    shap_features: pd.DataFrame | List[pd.DataFrame],
    target: pd.Series,
    task: str,
    alpha: float = 1e-6,
) -> pd.DataFrame:
    collected_rows = []  # List to store the rows we collect during each iteration

    features_left = True
    while features_left:
        # Call the original shap_features_to_significance function
        significance_df = shap_features_to_significance(
            shap_features, target, task, alpha
        )

        # Find the feature with the lowest t-value
        min_t_value_row = significance_df.loc[significance_df["t-value"].idxmin()]

        # Remember this row (collect it in our list)
        collected_rows.append(min_t_value_row)

        # Drop the feature corresponding to the lowest t-value from shap_features
        feature_to_remove = min_t_value_row["feature name"]
        if isinstance(shap_features, pd.DataFrame):
            shap_features = shap_features.drop(columns=[feature_to_remove])
            features_left = len(shap_features.columns)
        else:
            shap_features = {
                k: v.drop(columns=[feature_to_remove]) for k, v in shap_features.items()
            }
            features_left = len(list(shap_features.values())[0].columns)

    # Convert collected rows back to a dataframe
    result_df = (
        pd.DataFrame(collected_rows)
        .sort_values(by="t-value", ascending=False)
        .reset_index()
    )

    return result_df


def shap_select(
    tree_model: Any,
    validation_df: pd.DataFrame,
    target: pd.Series | str,  # str is column name in validation_df
    feature_names: List[str] | None = None,
    task: str | None = None,
    threshold: float = 0.05,
    return_extended_data: bool = False,
    alpha: float = 1e-6,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select features based on their SHAP values and statistical significance.

    Parameters:
    - tree_model (Any): A trained tree-based model.
    - validation_df (pd.DataFrame): Validation dataset containing the features.
    - feature_names (List[str]): A list of feature names used by the model.
    - target (pd.Series | str): The target values, or the name of the target column in `validation_df`.
    - task (str | None): The task type ('regression', 'binary', or 'multiclass'). If None, it is inferred automatically.
    - threshold (float): Significance threshold to select features. Default is 0.05.
    - return_extended_data (bool): Whether to also return the shapley values dataframe(s) and some extra columns
    - alpha (float): Controls the regularization strength for the regression

    Returns:
    - pd.DataFrame: A DataFrame containing the feature names, statistical significance, and a 'Selected' column
      indicating whether the feature was selected based on the threshold.
    """
    # If target is a string (column name), extract the target series from validation_df
    if isinstance(target, str):
        target = validation_df[target]

    if feature_names is None:
        feature_names = validation_df.columns.tolist()

    # Infer the task if not provided
    if task is None:
        if pd.api.types.is_numeric_dtype(target) and target.nunique() > 10:
            task = "regression"
        elif target.nunique() == 2:
            task = "binary"
        else:
            task = "multiclass"

    if task == "multiclass":
        unique_classes = sorted(list(target.unique()))
        shap_features = create_shap_features(
            tree_model, validation_df[feature_names], unique_classes
        )
    else:
        shap_features = create_shap_features(tree_model, validation_df[feature_names])

    # Compute statistical significance of each feature, recursively ablating
    significance_df = iterative_shap_feature_reduction(
        shap_features, target, task, alpha
    )

    # Add 'Selected' column based on the threshold
    significance_df["selected"] = (
        significance_df["stat.significance"] < threshold
    ).astype(int)
    significance_df.loc[significance_df["t-value"] < 0, "selected"] = -1

    if return_extended_data:
        return significance_df, shap_features
    else:
        return significance_df[
            ["feature name", "t-value", "stat.significance", "coefficient", "selected"]
        ]
