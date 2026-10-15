import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ARTIFACT_DIR / "asteroid_hazard_classifier.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"


@lru_cache(maxsize=1)
def load_artifacts():
    """Load and cache the model pipeline and metadata for this process.

    Restart the Python process after replacing the model or metadata file.
    Only load trusted artifacts because joblib uses pickle-based deserialization.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

    model = joblib.load(MODEL_PATH)
    with METADATA_PATH.open(encoding="utf-8") as file:
        metadata = json.load(file)
    return model, metadata


def predict_hazardous_label(input_df):
    """Predict the source dataset's Hazardous label, not impact probability."""
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if input_df.empty:
        raise ValueError("Input DataFrame contains no rows.")
    if not input_df.columns.is_unique:
        raise ValueError("Input contains duplicate column names.")

    model, metadata = load_artifacts()
    features = metadata["features"]
    threshold = float(metadata["prediction_threshold"])

    missing = [column for column in features if column not in input_df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = input_df.loc[:, features].copy()
    for column in features:
        X[column] = pd.to_numeric(X[column], errors="raise")
        values = X[column].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any():
            raise ValueError(f"Feature '{column}' contains infinite values.")
        if np.isnan(values).all():
            raise ValueError(
                f"Feature '{column}' contains no usable numeric values."
            )

    probabilities = model.predict_proba(X)[:, 1]
    labels = (probabilities >= threshold).astype(int)
    return pd.DataFrame(
        {
            "Predicted probability": probabilities,
            "Predicted label": labels,
        },
        index=input_df.index,
    )
