# Comparative Evaluation of Linear Support Vector Machines and Random Forests for Multiclass Classification of Tabular Data with Missing Values

**Lavinia Maria Alexandra Skandali**

---

## Abstract

This project compares a linear Support Vector Machine (`LinearSVC`) and a Random Forest classifier (`RandomForestClassifier`) on a five-class tabular dataset. The dataset has 1,500 labelled observations, 25 numeric features and 3 nominal categorical features. Missing values occur in ten of these features, and the class distribution is moderately imbalanced.

Preprocessing is handled by a single scikit-learn `Pipeline` so that no information leaks from validation data into training. The pipeline applies:

- median imputation and standardisation to numeric features;
- mode imputation and one-hot encoding to categorical features.

Both models are tuned with stratified 5-fold cross-validation, using macro-averaged F1 as the selection criterion.

The Random Forest clearly outperforms the linear SVM:

| Model | Cross-validated macro F1 (mean ± SD) |
|---|---|
| Random Forest | 0.604 ± 0.024 |
| Linear SVM | 0.467 ± 0.021 |
| Majority-class baseline | 0.076 |

The difference is consistent across all folds. On a held-out stratified test set, the selected Random Forest achieves a macro F1 of 0.607 and an accuracy of 0.617, closely matching its cross-validated estimate. The results suggest that the class structure is substantially non-linear, which favours ensemble tree methods over linear decision boundaries.

---

## Research Question

> For a tabular multiclass problem with mixed feature types, missing data and class imbalance, does a non-linear ensemble method (Random Forest) outperform a linear margin-based classifier (Linear SVM), and under which hyperparameter configuration?

---

## Data

| Property | Value |
|---|---|
| Observations | 1,500 labelled samples (plus 1,500 unlabelled hidden test samples) |
| Features | 25 numeric, 3 categorical (nominal) |
| Target | 5 classes (labels 0–4) |
| Class counts | 277 / 214 / 347 / 352 / 310 |
| Imbalance ratio | ≈ 1.6 (largest class : smallest class) |
| Missing values | 8 numeric and 2 categorical columns (3.9%–16.1% per column) |

**Missing-value mechanism.** Exploratory analysis suggests the data are approximately missing completely at random (MCAR), for two reasons:

- the missing-value rates of the affected variables do not differ meaningfully across classes;
- whether one column is missing is essentially uncorrelated with whether another is (|r| < 0.1).

**Feature correlations.** Pairwise Pearson correlations between numeric features are weak (|r| ≤ 0.21), which indicates limited linear redundancy among the predictors.

---

## Methods

### 1. Preprocessing

All preprocessing is placed inside a `ColumnTransformer`, which is itself embedded in a `Pipeline`. This ensures every transformation is fitted only on training folds.

| Feature type | Imputation | Transformation |
|---|---|---|
| Numeric | Median | Standardisation (zero mean, unit variance) |
| Categorical | Most frequent category | One-hot encoding (`handle_unknown='ignore'`) |

Median imputation was preferred to mean imputation because several numeric features have wide ranges.

### 2. Validation protocol

- **Hold-out test set.** An 80/20 stratified split holds out 300 samples. This set is used only once, for the final evaluation.
- **Cross-validation.** All model selection uses stratified 5-fold cross-validation on the training set.
- **Primary metric.** Macro-averaged F1 is used because it weights every class equally, regardless of how often it occurs.
- **Class imbalance.** Both classifiers are trained with `class_weight='balanced'`.
- **Baseline.** A majority-class dummy classifier provides a reference level of performance.

### 3. Hyperparameter tuning

**Linear SVM.**
- A validation curve over the regularisation parameter C ∈ {10⁻³, …, 10²} was used first.
- A grid search over the same values followed.

**Random Forest.** Tuning followed a two-stage, budget-aware procedure:

1. A validation curve over `n_estimators` ∈ {50, …, 300} showed that performance levels off at around 150–200 trees. The number of trees was therefore fixed at 200.
2. A randomised search (40 iterations) then covered:
   - `max_depth` ∈ {10, 20, 30, None}
   - `min_samples_split` ∈ {2, 5, 10}
   - `min_samples_leaf` ∈ {1, 2, 4}
   - `max_features` ∈ {`sqrt`, `log2`, 0.3}

### 4. Diagnostics

The selected model was examined with:

- a per-class classification report;
- a confusion matrix;
- impurity-based feature importances;
- a learning curve.

---

## Results

### Model comparison (5-fold stratified CV, training set)

| Model | Selected hyperparameters | Macro F1 (mean ± SD) |
|---|---|---:|
| Majority-class baseline | — | 0.076 |
| Linear SVM | C = 0.01 | 0.467 ± 0.021 |
| **Random Forest** | n_estimators = 200, max_depth = 20, max_features = 0.3, min_samples_split = 2, min_samples_leaf = 1 | **0.604 ± 0.024** |

- **Linear SVM.** Performance was best under strong regularisation (small C) and declined as C increased.
- **Random Forest.** The fold-wise score distributions of the two models do not overlap.

### Final model: held-out test set (n = 300)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.600 | 0.536 | 0.566 | 56 |
| 1 | 0.704 | 0.442 | 0.543 | 43 |
| 2 | 0.588 | 0.681 | 0.631 | 69 |
| 3 | 0.598 | 0.700 | 0.645 | 70 |
| 4 | 0.656 | 0.645 | 0.650 | 62 |
| **Macro average** | 0.629 | 0.601 | **0.607** | 300 |

Overall accuracy: 0.617.

### Diagnostics

- **Minority class.** Class 1, the least frequent class, has high precision but low recall. The model rarely predicts class 1, but its class-1 predictions are usually correct.
- **Feature importances.** Importances are spread fairly evenly across the numeric features, ranging from about 0.02 to 0.05. `feature_25` and `feature_1` rank highest. None of the categorical features appear among the 20 most important.
- **Learning curve.**
  - Training performance is close to 1.0 throughout, as expected for deep trees.
  - Cross-validated performance rises steadily with training size and has not clearly levelled off. This suggests additional data would further improve generalisation.

---

## Discussion

The Random Forest's advantage over the linear SVM supports the hypothesis that the class structure is non-linear. The weak pairwise correlations and the evenly spread feature importances further suggest that predictive information comes from interactions among many features, rather than from a few dominant variables. A single separating hyperplane cannot represent this kind of structure well, whereas tree ensembles can.

The close agreement between the cross-validated score (0.604) and the held-out score (0.607) indicates that the model-selection procedure did not overfit the validation folds.

### Limitations

- **Missing-data assumption.** The MCAR assumption was checked only descriptively. More sophisticated imputation methods, such as KNN or iterative imputation, were not evaluated.
- **Search budget.** The hyperparameter search was deliberately limited in size, so a broader search might produce modest further gains.
- **Train–validation gap.** The gap between training and validation performance shows high model variance. Stronger regularisation, for example a smaller `max_depth` or a larger `min_samples_leaf`, might reduce it at some cost in bias.
- **Minority-class recall.** Recall for class 1 remains low. Resampling methods and class-specific decision thresholds were not explored.
- **Feature engineering.** No feature engineering or dimensionality reduction was attempted.

### Future work

- Compare imputation strategies under MCAR and MAR assumptions.
- Evaluate gradient-boosted tree ensembles as an additional non-linear benchmark.
- Explore resampling or threshold adjustment to improve minority-class recall.
- Use permutation importance to assess feature relevance without the bias of impurity-based importance.

---

## Repository Structure

```
├── ML Independent Project.ipynb           # Full analysis: EDA, preprocessing, tuning, evaluation
├── mldata_0003305682_predictions.txt      # Predicted labels for the hidden test set
└── README.md
```

**Data availability.** The dataset was provided as part of university coursework and is not included in this repository.

## Reproducing

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
jupyter notebook "ML Independent Project.ipynb"
```

All random processes use a fixed seed (`random_state = 42`).
