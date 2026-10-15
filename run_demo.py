"""Run a small end-to-end demo of the asteroid label classifier.

The bundled CSV contains illustrative synthetic feature values, not observed
asteroid records. Predictions are for the source dataset's Hazardous label,
not estimates of real-world impact probability.
"""
import argparse
from pathlib import Path

import pandas as pd

from inference import predict_hazardous_label


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "examples" / "sample_input.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "demo_predictions.csv"
IDENTIFIER_COLUMN = "Example ID"


def resolve_path(path: Path) -> Path:
    """Resolve relative paths against the repository root."""
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="CSV with the required model features (default: examples/sample_input.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the prediction CSV (default: demo_predictions.csv)",
    )
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    records = pd.read_csv(input_path)
    if IDENTIFIER_COLUMN not in records.columns:
        raise ValueError(
            f"Input CSV must include an '{IDENTIFIER_COLUMN}' column "
            "so example rows can be identified in the output."
        )

    example_ids = records[IDENTIFIER_COLUMN].astype(str).reset_index(drop=True)
    feature_frame = records.drop(columns=[IDENTIFIER_COLUMN])
    predictions = predict_hazardous_label(feature_frame).reset_index(drop=True)

    result = pd.concat(
        [example_ids.rename(IDENTIFIER_COLUMN), predictions],
        axis=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Input records: {len(records)}")
    print(f"Predictions written to: {output_path}")
    print("Reminder: predictions refer to the dataset label, not real-world impact risk.")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
