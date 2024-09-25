A library for fast feature selection for gradient boosting models using regression on feature Shapley values.
Unlike

## Overview
`shap-select` implements a heuristic to do fast feature selection for tabular regression and classification models. 

The basic idea is running a linear or logistic regression of the Shapley values on the target on the validation set,
discarding the features with negative coefficients, and ranking/filtering the rest according to their 
statistical significance. For details, see the [example notebook](https://github.com/transferwise/shap-select/blob/main/docs/Quick%20feature%20selection%20through%20regression%20on%20Shapley%20values.ipynb)

Earlier packages using Shapley values for feature selection exist, the advantages of this one are
* Regression on the **validation set** to combat overfitting
* A single pass regression calculation, not iterative
* A single intuitive hyperparameter for feature selection: statistical significance
* Bonferroni correction for multiclass classification
## Usage
```python
from shap_select import shap_select
# Here model is any model supported by the shap library, fitted on a different (train) dataset
selected_features_df = shap_select(model, X_val, y_val, task="multiclass", threshold=0.05)
```