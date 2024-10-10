## Overview
`shap-select` implements a heuristic for fast feature selection, for tabular regression and classification models. 

The basic idea is running a linear or logistic regression of the target on the Shapley values of 
the original features, on the validation set,
discarding the features with negative coefficients, and ranking/filtering the rest according to their 
statistical significance. For motivation and details, refer to our [research paper](https://arxiv.org/abs/2410.06815) see the [example notebook](https://github.com/transferwise/shap-select/blob/main/docs/Quick%20feature%20selection%20through%20regression%20on%20Shapley%20values.ipynb)

Earlier packages using Shapley values for feature selection exist, the advantages of this one are
* Regression on the **validation set** to combat overfitting
* Only a single fit of the original model needed
* A single intuitive hyperparameter for feature selection: statistical significance
* Bonferroni correction for multiclass classification
* Address collinearity of (Shapley value) features by repeated (linear/logistic) regression

## Usage
```python
from shap_select import shap_select
# Here model is any model supported by the shap library, fitted on a different (train) dataset
# Task can be regression, binary, or multiclass
selected_features_df = shap_select(model, X_val, y_val, task="multiclass", threshold=0.05)
```

<table id="T_694ab">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_694ab_level0_col0" class="col_heading level0 col0" >feature name</th>
      <th id="T_694ab_level0_col1" class="col_heading level0 col1" >t-value</th>
      <th id="T_694ab_level0_col2" class="col_heading level0 col2" >stat.significance</th>
      <th id="T_694ab_level0_col3" class="col_heading level0 col3" >coefficient</th>
      <th id="T_694ab_level0_col4" class="col_heading level0 col4" >selected</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_694ab_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_694ab_row0_col0" class="data row0 col0" >x5</td>
      <td id="T_694ab_row0_col1" class="data row0 col1" >20.211299</td>
      <td id="T_694ab_row0_col2" class="data row0 col2" >0.000000</td>
      <td id="T_694ab_row0_col3" class="data row0 col3" >1.052030</td>
      <td id="T_694ab_row0_col4" class="data row0 col4" >1</td>
    </tr>
    <tr>
      <th id="T_694ab_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_694ab_row1_col0" class="data row1 col0" >x4</td>
      <td id="T_694ab_row1_col1" class="data row1 col1" >18.315144</td>
      <td id="T_694ab_row1_col2" class="data row1 col2" >0.000000</td>
      <td id="T_694ab_row1_col3" class="data row1 col3" >0.952416</td>
      <td id="T_694ab_row1_col4" class="data row1 col4" >1</td>
    </tr>
    <tr>
      <th id="T_694ab_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_694ab_row2_col0" class="data row2 col0" >x3</td>
      <td id="T_694ab_row2_col1" class="data row2 col1" >6.835690</td>
      <td id="T_694ab_row2_col2" class="data row2 col2" >0.000000</td>
      <td id="T_694ab_row2_col3" class="data row2 col3" >1.098154</td>
      <td id="T_694ab_row2_col4" class="data row2 col4" >1</td>
    </tr>
    <tr>
      <th id="T_694ab_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_694ab_row3_col0" class="data row3 col0" >x2</td>
      <td id="T_694ab_row3_col1" class="data row3 col1" >6.457140</td>
      <td id="T_694ab_row3_col2" class="data row3 col2" >0.000000</td>
      <td id="T_694ab_row3_col3" class="data row3 col3" >1.044842</td>
      <td id="T_694ab_row3_col4" class="data row3 col4" >1</td>
    </tr>
    <tr>
      <th id="T_694ab_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_694ab_row4_col0" class="data row4 col0" >x1</td>
      <td id="T_694ab_row4_col1" class="data row4 col1" >5.530556</td>
      <td id="T_694ab_row4_col2" class="data row4 col2" >0.000000</td>
      <td id="T_694ab_row4_col3" class="data row4 col3" >0.917242</td>
      <td id="T_694ab_row4_col4" class="data row4 col4" >1</td>
    </tr>
    <tr>
      <th id="T_694ab_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_694ab_row5_col0" class="data row5 col0" >x6</td>
      <td id="T_694ab_row5_col1" class="data row5 col1" >2.390868</td>
      <td id="T_694ab_row5_col2" class="data row5 col2" >0.016827</td>
      <td id="T_694ab_row5_col3" class="data row5 col3" >1.497983</td>
      <td id="T_694ab_row5_col4" class="data row5 col4" >1</td>
    </tr>
    <tr>
      <th id="T_694ab_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_694ab_row6_col0" class="data row6 col0" >x7</td>
      <td id="T_694ab_row6_col1" class="data row6 col1" >0.901098</td>
      <td id="T_694ab_row6_col2" class="data row6 col2" >0.367558</td>
      <td id="T_694ab_row6_col3" class="data row6 col3" >2.865508</td>
      <td id="T_694ab_row6_col4" class="data row6 col4" >0</td>
    </tr>
    <tr>
      <th id="T_694ab_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_694ab_row7_col0" class="data row7 col0" >x8</td>
      <td id="T_694ab_row7_col1" class="data row7 col1" >0.563214</td>
      <td id="T_694ab_row7_col2" class="data row7 col2" >0.573302</td>
      <td id="T_694ab_row7_col3" class="data row7 col3" >1.933632</td>
      <td id="T_694ab_row7_col4" class="data row7 col4" >0</td>
    </tr>
    <tr>
      <th id="T_694ab_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_694ab_row8_col0" class="data row8 col0" >x9</td>
      <td id="T_694ab_row8_col1" class="data row8 col1" >-1.607814</td>
      <td id="T_694ab_row8_col2" class="data row8 col2" >0.107908</td>
      <td id="T_694ab_row8_col3" class="data row8 col3" >-4.537098</td>
      <td id="T_694ab_row8_col4" class="data row8 col4" >-1</td>
    </tr>
  </tbody>
</table>


## Citation

If you use `shap-select` in your research, please cite our paper:

```bibtex
@misc{kraev2024shapselectlightweightfeatureselection,
      title={Shap-Select: Lightweight Feature Selection Using SHAP Values and Regression}, 
      author={Egor Kraev and Baran Koseoglu and Luca Traverso and Mohammed Topiwalla},
      year={2024},
      eprint={2410.06815},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.06815}, 
}