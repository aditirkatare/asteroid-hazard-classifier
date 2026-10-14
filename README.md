# Asteroid Hazardous-Label Classifier

A machine-learning project investigating hazardous-label classification
in a NASA near-Earth-object dataset, with a focus on target-label
auditing, identity leakage, group-aware validation, threshold selection,
and reproducible inference.

## Scope

This model predicts the dataset's Hazardous label. It
does not estimate actual Earth-impact probability or replace
professional orbital-risk assessment.

## Key Results

Final evaluation used a group-separated holdout set containing 948
records from 739 unique asteroid identifiers. No asteroid identifier
appeared in both the development and holdout sets.

Metric Group-held-out holdout

PR-AUC (average precision)                 0.6121
ROC-AUC                                    0.8953
Positive-class precision                   0.3618
Positive-class recall                      0.8819
Positive-class F1                          0.5131
Accuracy                                   0.7458
At the selected decision threshold of 0.20, the holdout confusion
matrix was:
Actual / predicted Negative Positive

Negative                    580        224
Positive                     17        127
The threshold was selected using grouped out-of-fold predictions on the
development set, targeting recall of at least 0.85 and then maximizing
precision among qualifying thresholds. It was not tuned on holdout
labels. The resulting recall comes with many false positives, so the
threshold represents a trade-off rather than a universal operating
point.

## Dataset Audit and Label Investigation

The dataset contains 4,687 records and 3,692 unique asteroid
identifiers.
- 751 identifiers occur in multiple records.
- No exact duplicate rows were found.
- The positive Hazardous label rate is approximately 16.1%.
- A diagnostic rule combining Minimum Orbit Intersection (MOID) and
  Absolute Magnitude agreed with 99.55% of row-level labels and 99.59%
  of labels after reducing to one record per identifier. Fifteen
  unique identifiers disagreed with that diagnostic rule.
This close agreement suggests the target label is strongly associated
with underlying orbital criteria. It does not establish the exact
process used to generate every label, and the disagreements remain
relevant. The project treats this as a label-provenance finding, not
proof that the rule is the definitive source of the target.
The rule-related fields, Absolute Magnitude and
Minimum Orbit Intersection, were excluded from the model features to
avoid directly supplying this obvious shortcut to the classifier.

## Why Group-Aware Evaluation?

An initial random row split placed 287 asteroid identifiers in both the
training and test partitions. Since repeated records can represent the
same asteroid, this can make a row-level test overly optimistic for
generalization to unseen identifiers.
The final workflow instead used a development/holdout split with
StratifiedGroupKFold, grouping by asteroid Name. Five-fold grouped
cross-validation was used for model comparison on the development
partition. The final holdout was reserved for final evaluation.

## Model and Validation

The selected model is a Random Forest pipeline with median imputation
and balanced class weights, using 16 features. The classification
threshold is stored in metadata.json.
Five-fold grouped cross-validation on the development set:
Model Mean PR-AUC Std. PR-AUC Mean ROC-AUC Std. ROC-AUC

## Features

The exact required feature list is stored in `metadata.json`. Supply a
pandas DataFrame containing all listed features. Extra columns are ignored,
and required columns are reordered automatically.

Logistic Regression          0.3944        0.0242         0.8156         0.0072
Random Forest                0.5794        0.0433         0.8872         0.0193
The Random Forest was the stronger candidate under these grouped
cross-validation results. Standard deviations describe variation across
the five folds.

## Feature Interpretation

Permutation importance on the group-held-out records ranked
Orbit Uncertainity, Inclination, and Perihelion Distance as the
three strongest features by decrease in average precision when permuted.
Feature importance describes model reliance under this evaluation setup.
It does not establish causation, and importance rankings may change with
the data, feature dependencies, and evaluation method.

## Features and Usage

The exact 16-feature schema is stored in metadata.json. The inference
function requires every listed feature, restores the expected feature
order, ignores extra columns, validates numeric inputs, and rejects
infinite values or a feature that is entirely missing in the supplied
batch. Individual missing numeric values can be handled by the fitted
pipeline's imputer.

Install the pinned dependencies:
```Bash
pip install -r requirements.txt
```
Place inference.py, metadata.json, and
asteroid_hazard_classifier.joblib together. Then supply a CSV
containing the required feature columns:

```python
import pandas as pd
from inference import predict_hazardous_label

new_records = pd.read_csv("new_asteroid_records.csv")
predictions = predict_hazardous_label(new_records)

print(predictions.head())
predictions.to_csv("predictions.csv", index=True)
```
Run inference tests from the repository directory:

```Bash
python -m unittest test_inference -v
```
Synthetic inputs in these tests verify the software interface; they are
not scientific model-validation examples.

## Repository Contents

- asteroid_hazard_classifier.joblib: fitted model pipeline
- metadata.json: model configuration, feature schema, evaluation
- metadata, and limitations
- inference.py: reusable inference function
- evaluation_report.json: machine-readable evaluation results
- test_inference.py: automated inference tests
- requirements.txt: pinned Python dependencies
- environment.json: recorded package versions
- DATASET_PROVENANCE.md: dataset source and licensing note

## Limitations

1. The target is the source dataset's Hazardous label, not actual
impact risk.
2. The target closely aligns with a diagnostic combination of orbital
criteria, with observed exceptions.
3. The low decision threshold prioritizes recall and produces false
positives.
4. Generalization to future observations or a different labeling
process has not been established by this evaluation.
5. Model probabilities are not established as calibrated real-world
risk estimates.
6. Serialized model artifacts should only be loaded from trusted
sources.

## Dataset source and licensing

Source: [NASA: Asteroids Classification on Kaggle](https://www.kaggle.com/datasets/lovishbansal123/nasa-asteroids-classification)
by `lovishbansal123`.

The file used was `nasa.csv`. Check the Kaggle page for the applicable
license before redistributing the original dataset. The license has not
been independently confirmed for this project.

See `DATASET_PROVENANCE.md` for details. 
