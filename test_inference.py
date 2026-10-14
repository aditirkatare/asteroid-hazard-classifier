import unittest
import numpy as np
import pandas as pd

from inference import load_artifacts, predict_hazardous_label


class TestStandaloneInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.metadata = load_artifacts()
        cls.features = cls.metadata["features"]
        cls.threshold = float(cls.metadata["prediction_threshold"])

        # Synthetic numeric input for interface tests only. These rows are not
        # realistic asteroid examples and do not provide scientific evidence.
        cls.sample = pd.DataFrame(
            np.zeros((3, len(cls.features))),
            columns=cls.features,
            index=[10, 20, 30],
        )

    def test_artifacts_load(self):
        self.assertEqual(len(self.features), 16)
        self.assertAlmostEqual(self.threshold, 0.20)

    def test_metadata_feature_schema_matches_model(self):
        self.assertEqual(self.metadata["feature_count"], len(self.features))
        self.assertEqual(len(set(self.features)), len(self.features))
        self.assertEqual(self.model.n_features_in_, len(self.features))

    def test_output_shape_and_probability_range(self):
        result = predict_hazardous_label(self.sample)
        self.assertEqual(result.shape, (3, 2))
        self.assertTrue(
            result["Predicted probability"].between(0, 1).all()
        )
        self.assertTrue(
            set(result["Predicted label"].unique()).issubset({0, 1})
        )
        expected_labels = (
            result["Predicted probability"] >= self.threshold
        ).astype(int)
        np.testing.assert_array_equal(result["Predicted label"], expected_labels)

    def test_index_is_preserved(self):
        result = predict_hazardous_label(self.sample)
        self.assertEqual(list(result.index), [10, 20, 30])

    def test_column_order_does_not_matter(self):
        first = predict_hazardous_label(self.sample)
        second = predict_hazardous_label(self.sample.iloc[:, ::-1])
        np.testing.assert_allclose(
            first["Predicted probability"],
            second["Predicted probability"],
        )

    def test_extra_columns_are_ignored(self):
        extra = self.sample.copy()
        extra["unrelated"] = "ignored"
        first = predict_hazardous_label(self.sample)
        second = predict_hazardous_label(extra)
        np.testing.assert_allclose(
            first["Predicted probability"],
            second["Predicted probability"],
        )

    def test_individual_missing_values_are_imputed(self):
        with_missing = self.sample.copy()
        with_missing.iloc[0, 0] = np.nan

        result = predict_hazardous_label(with_missing)

        self.assertEqual(result.shape, (3, 2))
        self.assertTrue(np.isfinite(result["Predicted probability"]).all())

    def test_entirely_missing_feature_column_raises_error(self):
        bad = self.sample.copy()
        bad[self.features[0]] = np.nan

        with self.assertRaisesRegex(ValueError, "no usable numeric values"):
            predict_hazardous_label(bad)

    def test_missing_feature_raises_error(self):
        bad = self.sample.drop(columns=[self.features[0]])
        with self.assertRaisesRegex(ValueError, "Missing required features"):
            predict_hazardous_label(bad)

    def test_duplicate_column_names_raise_error(self):
        bad = self.sample.copy()
        bad.columns = [self.features[0]] + list(self.features[1:])
        bad.insert(len(bad.columns), self.features[0], 0.0, allow_duplicates=True)

        with self.assertRaisesRegex(ValueError, "duplicate column names"):
            predict_hazardous_label(bad)

    def test_infinite_value_raises_error(self):
        bad = self.sample.copy()
        bad.iloc[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "infinite"):
            predict_hazardous_label(bad)

    def test_empty_input_raises_error(self):
        with self.assertRaisesRegex(ValueError, "no rows"):
            predict_hazardous_label(self.sample.iloc[:0])

    def test_non_numeric_value_raises_error(self):
        bad = self.sample.astype(object).copy()
        bad.iloc[0, 0] = "not-a-number"
        with self.assertRaises((ValueError, TypeError)):
            predict_hazardous_label(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
