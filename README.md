# Asteroid Hazardous-Label Classifier

[![Project Tests](https://github.com/aditirkatare/asteroid-hazard-classifier/actions/workflows/tests.yml/badge.svg)](https://github.com/aditirkatare/asteroid-hazard-classifier/actions/workflows/tests.yml)

A machine-learning project investigating hazardous-label classification in a NASA near-Earth-object dataset, with a focus on target-label auditing, identity leakage, group-aware validation, threshold selection, and reproducible inference.

> **Scope:** This model predicts the dataset's `Hazardous` label. It does not estimate actual Earth-impact probability or replace professional orbital-risk assessment.

## Key Results

Final evaluation used a group-separated holdout set containing 948 records from 739 unique asteroid identifiers. No asteroid identifier appeared in both development and holdout partitions.

| Metric | Group-held-out holdout |
|---|---:|
| PR-AUC (average precision) | **0.6121** |
| ROC-AUC | **0.8953** |
| Positive-class precision | 0.3618 |
| Positive-class recall | **0.8819** |
| Positive-class F1 | 0.5131 |
| Accuracy | 0.7458 |

At the selected decision threshold of **0.20**, the record-level holdout confusion matrix was:

| Actual / predicted | Negative | Positive |
|---|---:|---:|
| Negative | 580 | 224 |
| Positive | 17 | 127 |

The threshold was selected using grouped out-of-fold predictions on the development set, targeting positive-class recall of at least 0.85 and then maximizing precision among qualifying thresholds. Holdout labels were not used to choose the threshold. At this threshold, recall is high but there are many false positives, so the threshold is a trade-off rather than a universal operating point.

## Dataset Audit and Label Investigation

The dataset contains 4,687 records and 3,692 unique asteroid identifiers.

- 751 identifiers occur in multiple records.
- No exact duplicate rows were found.
- The positive `Hazardous` label rate is approximately 16.1%.
- A diagnostic rule combining Minimum Orbit Intersection (MOID) and Absolute Magnitude agreed with 99.55% of row-level labels and 99.59% of labels after reducing to one record per identifier. Fifteen unique identifiers disagreed with that diagnostic rule.

This close agreement suggests the target label is strongly associated with underlying orbital criteria. It does **not** establish the exact process used to generate every label, and the disagreements remain relevant. This is a label-provenance finding, not proof that the rule is the definitive source of the target.

The rule-related fields, `Absolute Magnitude` and `Minimum Orbit Intersection`, were excluded from model features to avoid directly supplying this obvious shortcut to the classifier.

## Why Group-Aware Evaluation?

An initial random row split placed 287 asteroid identifiers in both training and test partitions. Since repeated records can represent the same asteroid, this can make a row-level test overly optimistic for generalization to unseen identifiers.

The final workflow used a development/holdout split with `StratifiedGroupKFold`, grouping by asteroid `Name`. Five-fold grouped cross-validation was used for model comparison on the development partition.

**Holdout-use note:** the holdout was not used to select the model or choose the decision threshold. Permutation feature importance was subsequently calculated on holdout records, however, so the holdout should not be described as untouched for every post-hoc analysis. Avoid using those holdout importance results to tune further model choices.

## Model and Validation

The selected model is a Random Forest pipeline with median imputation and balanced class weights, using 16 features. The decision threshold is stored in `metadata.json`.

Five-fold grouped cross-validation on the development set:

| Model | Mean PR-AUC | Std. PR-AUC | Mean ROC-AUC | Std. ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.3944 | 0.0242 | 0.8156 | 0.0072 |
| Random Forest | **0.5794** | 0.0433 | **0.8872** | 0.0193 |

The Random Forest had higher mean PR-AUC and ROC-AUC in this grouped cross-validation comparison. Standard deviations describe variation across the five folds.

## Feature Interpretation

Permutation importance on the group-held-out records ranked `Orbit Uncertainity`, `Inclination`, and `Perihelion Distance` as the three strongest features by decrease in average precision when permuted.

Feature importance describes model reliance under this evaluation setup. It does not establish causation, and importance rankings may change with the data, feature dependencies, and evaluation method. Because this analysis used holdout records, it is descriptive post-hoc analysis rather than evidence for additional tuning.


## Features and Usage

The exact 16-feature input schema is stored in `metadata.json`. The inference function restores the expected feature order, ignores extra columns, validates numeric inputs, and rejects infinite values or a feature that is entirely missing across the supplied batch. Individual missing numeric values can be handled by the fitted pipeline's imputer.

### 1. Set up the environment

Clone the repository and enter its directory:

```bash
git clone https://github.com/aditirkatare/asteroid-hazard-classifier.git
cd asteroid-hazard-classifier
```

Install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

Use a Python version compatible with the pinned dependencies in `requirements.txt`. A virtual environment is recommended to keep project dependencies isolated.

### 2. Run the automated tests

From the repository root, run:

```bash
python -m unittest discover -v
```

The tests cover the inference interface and consistency checks between the evaluation report and metadata. Synthetic inputs in inference tests verify software behavior; they are not scientific model-validation examples.

### 3. Run inference on a CSV file

Keep `inference.py`, `metadata.json`, and `asteroid_hazard_classifier.joblib` together in the repository directory.

Create `new_asteroid_records.csv` with all required feature columns listed in `metadata.json`, then run:

```python
import pandas as pd
from inference import predict_hazardous_label

new_records = pd.read_csv("new_asteroid_records.csv")
predictions = predict_hazardous_label(new_records)

print(predictions.head())
predictions.to_csv("predictions.csv", index=False)
```

The output contains the model's predictions for the supplied records. Check the inference function and metadata for the exact output format and threshold behavior.

### 4. Reproducibility notes

- The fitted model and its expected feature schema are stored in the repository.
- Dependency versions are pinned in `requirements.txt`; `environment.json` records the environment used for the project.
- Evaluation metrics and protocol details are recorded in `evaluation_report.json`.
- Re-running inference is different from reproducing model training and evaluation. Do not assume the training pipeline is reproducible from these instructions alone.
- Only load serialized model artifacts from trusted sources.


## Repository Contents

- `asteroid_hazard_classifier.joblib`: fitted model pipeline
- `metadata.json`: model configuration, feature schema, evaluation metadata, and limitations
- `inference.py`: reusable inference function with cached artifact loading
- `evaluation_report.json`: machine-readable evaluation protocol, metrics, counts, and caveats
- `test_inference.py`: inference interface tests
- `test_evaluation_report.py`: evaluation report consistency tests
- `.github/workflows/tests.yml`: automated tests on pushes to `main` and pull requests targeting `main`
- `requirements.txt`: pinned Python dependencies
- `environment.json`: recorded package versions
- `DATASET_PROVENANCE.md`: dataset source and licensing note

## Limitations

1. The target is the source dataset's `Hazardous` label, not actual impact risk.
2. The target closely aligns with a diagnostic combination of orbital criteria, with observed exceptions.
3. The 0.20 threshold prioritizes positive-class recall but produces many false positives.
4. Generalization to future observations or a different labeling process has not been established.
5. Model probabilities have not been established as calibrated real-world risk estimates.
6. Permutation feature importance was calculated on holdout data and should be treated as post-hoc descriptive analysis.
7. Serialized model artifacts should only be loaded from trusted sources.

## Dataset Source and Licensing

Source: [NASA: Asteroids Classification on Kaggle](https://www.kaggle.com/datasets/lovishbansal123/nasa-asteroids-classification) by `lovishbansal123`. The file used was `nasa.csv`.

See `DATASET_PROVENANCE.md`. Verify the license displayed on the Kaggle page before redistributing the original dataset or substantial portions of it; the applicable license has not been independently confirmed for this project.
