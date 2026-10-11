# Dataset Provenance

- Dataset title: NASA: Asteroids Classification
- Kaggle uploader: lovishbansal123
- Dataset page: https://www.kaggle.com/datasets/lovishbansal123/nasa-asteroids-classification
- File used: nasa.csv

## Usage and licensing

The Kaggle dataset page is the source used for this project. Verify the
license displayed on that page before redistributing the original dataset
or any substantial portion of it.

This repository's trained model and evaluation results were created from
the downloaded dataset. Their presence here does not establish or expand
the license of the original data.

## Target interpretation

The model predicts the `Hazardous` label supplied by the dataset. It is
not an impact-probability model and should not be used for operational
planetary-defense decisions.

## Reproducibility note

The dataset contains multiple records for some asteroid identifiers.
Model evaluation therefore used group-separated development and holdout
sets so that an identifier did not occur in both sets.
