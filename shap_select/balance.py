import pandas as pd
from sklearn.utils import resample


def balance_dataset(X, y):
    """
    Balances an unbalanced dataset by oversampling the minority class.

    Parameters:
    X (pd.DataFrame): Feature DataFrame
    y (pd.Series): Target Series

    Returns:
    X_balanced (pd.DataFrame): Balanced features DataFrame
    y_balanced (pd.Series): Balanced target Series
    """
    # Combine features and target into a single DataFrame for easier manipulation
    df = pd.concat([X, y], axis=1)

    # Identify the name of the target column
    target_name = y.name

    # Separate the majority and minority classes
    class_counts = y.value_counts()
    majority_class_label = class_counts.idxmax()  # Label of the majority class
    minority_class_label = class_counts.idxmin()  # Label of the minority class

    majority_class = df[df[target_name] == majority_class_label]
    minority_class = df[df[target_name] == minority_class_label]

    # Calculate how many samples to add to balance the dataset
    n_majority = majority_class.shape[0]
    n_minority = minority_class.shape[0]
    n_to_add = n_majority - n_minority

    # Upsample the minority class (i.e., duplicate samples with replacement)
    minority_upsampled = resample(
        minority_class,
        replace=True,  # Sample with replacement
        n_samples=n_to_add,  # How many samples to add
        random_state=42,
    )  # Seed for reproducibility

    # Combine the majority class with the upsampled minority class
    df_balanced = pd.concat([majority_class, minority_class, minority_upsampled])

    # Shuffle the dataset to mix the new minority samples with the majority class
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Separate the features and target from the balanced DataFrame
    X_balanced = df_balanced.drop(columns=target_name)
    y_balanced = df_balanced[target_name]

    return X_balanced, y_balanced
