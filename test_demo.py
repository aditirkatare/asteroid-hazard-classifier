import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "examples" / "sample_input.csv"
METADATA_PATH = ROOT / "metadata.json"
DEMO_PATH = ROOT / "run_demo.py"


class TestDemo(unittest.TestCase):
    def test_sample_input_contains_required_features(self):
        with METADATA_PATH.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        sample = pd.read_csv(INPUT_PATH)

        self.assertIn("Example ID", sample.columns)
        self.assertTrue(set(metadata["features"]).issubset(sample.columns))
        self.assertEqual(len(sample), 2)
        self.assertTrue(sample["Example ID"].is_unique)

    def test_demo_writes_prediction_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "demo_predictions.csv"
            result = subprocess.run(
                [
                    sys.executable,
                    str(DEMO_PATH),
                    "--input",
                    str(INPUT_PATH),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue(output_path.is_file(), result.stdout + result.stderr)
            predictions = pd.read_csv(output_path)
            self.assertEqual(len(predictions), 2)
            self.assertEqual(
                list(predictions.columns),
                [
                    "Example ID",
                    "Predicted probability",
                    "Predicted label",
                ],
            )
            self.assertTrue(
                predictions["Predicted probability"].between(0, 1).all()
            )
            self.assertTrue(
                set(predictions["Predicted label"].unique()).issubset({0, 1})
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
