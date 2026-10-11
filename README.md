# Asteroid Hazardous-Label Classifier

A machine-learning project that classifies the `Hazardous` label in a
NASA near-Earth-object dataset using a Random Forest pipeline.

## Scope

This model predicts the dataset's existing classification label. It does
not estimate collision probability, predict an impact date, or replace
orbital calculations performed by astronomers.

The dataset's labels closely align with a simple combination of orbital
criteria, although exceptions were observed. This suggests that the target
may be strongly influenced by the underlying labeling criteria.

## Dataset audit

- Total records: 4,687
- Unique asteroid identifiers: 3,692
- Repeated identifiers: 751
- Exact duplicate rows: 0
- Positive label rate: approximately 16.1%

Repeated records were handled carefully during evaluation to prevent the
same asteroid identifier from appearing in both development and holdout
sets.

## Model

- Algorithm: Random Forest classifier
- Input features: 16
- Preprocessing: median imputation
- Class weighting: balanced
- Decision threshold: 0.20

## Grouped cross-validation

Five-fold StratifiedGroupKFold evaluation on the development set:

- Mean PR-AUC: 0.5794
- Standard deviation of PR-AUC: 0.0433
- Mean ROC-AUC: 0.8872
- Standard deviation of ROC-AUC: 0.0193

The final model was fitted on the development set and evaluated on a
separate holdout set with no overlapping asteroid identifiers.

## Group-held-out holdout results

- Records: 948
- Unique asteroid identifiers: 739
- PR-AUC: 0.6121
- ROC-AUC: 0.8953
- Positive-class precision at the selected threshold: 0.3618
- Positive-class recall at the selected threshold: 0.8819
- Positive-class F1: 0.5131
- Accuracy: 0.7458

The threshold was selected using grouped out-of-fold predictions on the
development data to target recall of at least 0.85. It was not selected
using holdout labels.

At this threshold, false positives are expected. Precision and recall
depend on the dataset, labeling process, and class distribution.

## Features

The exact required feature list is stored in `metadata.json`. Supply a
pandas DataFrame containing all listed features. Extra columns are ignored,
and required columns are reordered automatically.

## Usage

Place your input CSV alongside this module and the model artifacts.

```python
import pandas as pd
from inference import predict_hazardous_label

new_records = pd.read_csv("new_asteroid_records.csv")
predictions = predict_hazardous_label(new_records)

print(predictions.head())
predictions.to_csv("predictions.csv", index=True)
```

The input CSV must contain all required feature columns. Missing numeric
values can be imputed by the fitted pipeline, but a feature that is
entirely missing in the supplied batch is rejected by the inference
validation.

## Files

- `asteroid_hazard_classifier.joblib`: fitted model pipeline
- `metadata.json`: model configuration, feature schema, metrics, limitations
- `inference.py`: reusable inference function
- `requirements.txt`: Python package dependencies
- `README.md`: project documentation

## Limitations

1. The target is the source dataset's `Hazardous` label, not impact risk.
2. The target closely aligns with orbital criteria, with observed exceptions.
3. The selected low threshold prioritizes recall and increases false positives.
4. Generalization to future observations or a different labeling process
   has not been established by this evaluation.
5. Model artifacts should only be loaded from trusted sources.

## Dataset source and licensing

Source: [NASA: Asteroids Classification on Kaggle](https://www.kaggle.com/datasets/lovishbansal123/nasa-asteroids-classification)
by `lovishbansal123`.

The file used was `nasa.csv`. Check the Kaggle page for the applicable
license before redistributing the original dataset. The license has not
been independently confirmed for this project.

See `DATASET_PROVENANCE.md` for details.
