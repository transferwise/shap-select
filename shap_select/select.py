from typing import Any, Tuple, List
import pandas as pd
import statsmodels.api as sm
import shap


def create_shap_features(tree_model: Any, validation_df: pd.DataFrame) -> pd.DataFrame:
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
    explainer = shap.TreeExplainer(tree_model, model_output="raw")
    shap_values = explainer(validation_df).values

    # Create a DataFrame with the SHAP values, with one column per feature
    return pd.DataFrame(
        shap_values, columns=validation_df.columns, index=validation_df.index
    )


def binary_classifier_significance(
    shap_features: pd.DataFrame, target: pd.Series
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
    shap_features_with_const = sm.add_constant(shap_features)

    # Fit the logistic regression model
    logit_model = sm.Logit(target, shap_features_with_const)
    result = logit_model.fit(disp=False)

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
    result_df["closeness to 1.0"] = closeness_to_one(result_df).abs()
    return result_df


def multi_classifier_significance(
    shap_features: pd.DataFrame, target: pd.Series
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
    unique_classes = target.unique()
    significance_dfs = []

    # Iterate through each class and perform binary classification (one-vs-all)
    for cls in unique_classes:
        binary_target = (target == cls).astype(int)
        significance_df = binary_classifier_significance(shap_features, binary_target)
        significance_dfs.append(significance_df)

    # Combine results into a single DataFrame with the max significance value for each feature
    combined_df = pd.concat(significance_dfs)
    max_significance_df = (
        combined_df.groupby("feature name", as_index=False)
        .agg({"stat.significance": "min", "t-value": "max", "closeness to 1.0": "min"})
        .reset_index(drop=True)
    )
    max_significance_df.columns = ["feature name", "max significance value"]

    return max_significance_df, significance_dfs


def regression_significance(
    shap_features: pd.DataFrame, target: pd.Series
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
    # Fit the linear regression model
    ols_model = sm.OLS(target, shap_features)
    result = ols_model.fit()

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
    result_df["closeness to 1.0"] = closeness_to_one(result_df).abs()

    return result_df


def closeness_to_one(df: pd.DataFrame) -> pd.Series:
    return (df["coefficient"] - 1.0) / df["stderr"]


def shap_features_to_significance(
    shap_features: pd.DataFrame, target: pd.Series, task: str | None = None
) -> pd.DataFrame:
    """
    Determines the task (regression, binary, or multi-class classification) based on the target and calls the appropriate
    significance function. Returns a DataFrame with feature names and their significance values.

    Parameters:
    shap_features (pd.DataFrame): A DataFrame containing the features used for prediction.
    target (pd.Series): The target series for prediction (either continuous or categorical).
    task (str | None): The type of task to perform. If None, the function will infer the task automatically.
                       The options are "regression", "binary", or "multi".

    Returns:
    pd.DataFrame: A DataFrame containing:
        - feature name: The names of the features.
        - stat.significance: The p-value (statistical significance) for each feature.
        Sorted in descending order of significance (ascending p-value).
    """

    # Infer the task if not provided
    if task is None:
        if pd.api.types.is_numeric_dtype(target) and target.nunique() > 10:
            task = "regression"
        elif target.nunique() == 2:
            task = "binary"
        else:
            task = "multi"

    # Call the appropriate function based on the task
    if task == "regression":
        result_df = regression_significance(shap_features, target)
    elif task == "binary":
        result_df = binary_classifier_significance(shap_features, target)
    elif task == "multi":
        max_significance_df, _ = multi_classifier_significance(shap_features, target)
        result_df = max_significance_df.rename(
            columns={"max significance value": "stat.significance"}
        )
    else:
        raise ValueError("`task` must be 'regression', 'binary', 'multi' or None.")

    # Sort the result by statistical significance in ascending order (more significant features first)
    result_df_sorted = result_df.sort_values(by="t-value", ascending=False).reset_index(
        drop=True
    )

    return result_df_sorted


def select_features(
    tree_model: Any,
    validation_df: pd.DataFrame,
    feature_names: List[str],
    target: pd.Series | str,  # str is column name in validation_df
    task: str | None = None,
    threshold: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select features based on their SHAP values and statistical significance.

    Parameters:
    - tree_model (Any): A trained tree-based model.
    - validation_df (pd.DataFrame): Validation dataset containing the features.
    - feature_names (List[str]): A list of feature names used by the model.
    - target (pd.Series | str): The target values, or the name of the target column in `validation_df`.
    - task (str | None): The task type ('regression', 'binary', or 'multi'). If None, it is inferred automatically.
    - threshold (float): Significance threshold to select features. Default is 0.05.

    Returns:
    - pd.DataFrame: A DataFrame containing the feature names, statistical significance, and a 'Selected' column
      indicating whether the feature was selected based on the threshold.
    """
    # If target is a string (column name), extract the target series from validation_df
    if isinstance(target, str):
        target = validation_df[target]

    # Generate SHAP values for the validation dataset
    shap_features = create_shap_features(tree_model, validation_df[feature_names])

    # Compute statistical significance of each feature
    significance_df = shap_features_to_significance(shap_features, target, task)

    # Add 'Selected' column based on the threshold
    significance_df["Selected"] = (
        significance_df["stat.significance"] < threshold
    ).astype(int)
    significance_df.loc[significance_df["coefficient"] < 0, "Selected"] = -1

    return significance_df, shap_features
