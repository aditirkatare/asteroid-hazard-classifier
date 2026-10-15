import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "evaluation_report.json"
METADATA_PATH = ROOT / "metadata.json"


class TestEvaluationReportConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with REPORT_PATH.open(encoding="utf-8") as handle:
            cls.report = json.load(handle)
        with METADATA_PATH.open(encoding="utf-8") as handle:
            cls.metadata = json.load(handle)

    def test_report_and_metadata_feature_schema_match(self):
        report_model = self.report["model"]
        self.assertEqual(report_model["feature_count"], len(self.metadata["features"]))
        self.assertEqual(report_model["features"], self.metadata["features"])
        self.assertEqual(self.metadata["feature_count"], len(self.metadata["features"]))

    def test_threshold_matches(self):
        # In the expanded report, threshold is stored under evaluation_protocol.
        report_threshold = self.report["evaluation_protocol"]["decision_threshold"]
        metadata_threshold = self.metadata["prediction_threshold"]
        self.assertAlmostEqual(report_threshold, metadata_threshold, places=12)

    def test_holdout_scores_match_metadata(self):
        report_metrics = self.report["holdout_record_level"]["metrics"]
        metadata_metrics = self.metadata["evaluation"]
        mappings = {
            "pr_auc_average_precision": "pr_auc_average_precision",
            "roc_auc": "roc_auc",
            "precision_positive": "precision_positive",
            "recall_positive": "recall_positive",
            "f1_positive": "f1_positive",
            "accuracy": "accuracy",
        }
        for report_key, metadata_key in mappings.items():
            with self.subTest(metric=report_key):
                self.assertAlmostEqual(
                    report_metrics[report_key],
                    metadata_metrics[metadata_key],
                    places=12,
                )

    def test_record_level_counts_match_confusion_matrix(self):
        holdout = self.report["holdout_record_level"]
        matrix = holdout["confusion_matrix"]["values"]
        support = holdout["actual_class_support"]
        predicted = holdout["predicted_class_count"]

        self.assertEqual(len(matrix), 2)
        self.assertTrue(all(len(row) == 2 for row in matrix))
        self.assertEqual(sum(support.values()), holdout["sample_count"])
        self.assertEqual(sum(map(sum, matrix)), holdout["sample_count"])
        self.assertEqual(sum(matrix[0]), support["negative"])
        self.assertEqual(sum(matrix[1]), support["positive"])
        self.assertEqual(matrix[0][0] + matrix[1][0], predicted["negative"])
        self.assertEqual(matrix[0][1] + matrix[1][1], predicted["positive"])

        self.assertEqual(matrix[0][0], holdout["confusion_matrix"]["true_negative"])
        self.assertEqual(matrix[0][1], holdout["confusion_matrix"]["false_positive"])
        self.assertEqual(matrix[1][0], holdout["confusion_matrix"]["false_negative"])
        self.assertEqual(matrix[1][1], holdout["confusion_matrix"]["true_positive"])

    def test_holdout_sizes_match_metadata(self):
        protocol = self.report["evaluation_protocol"]
        evaluation = self.metadata["evaluation"]
        self.assertEqual(
            protocol["holdout_rows"],
            self.report["holdout_record_level"]["sample_count"],
        )
        self.assertEqual(protocol["holdout_rows"], evaluation["holdout_rows"])
        self.assertEqual(
            protocol["holdout_unique_asteroids"],
            evaluation["holdout_unique_asteroids"],
        )
        self.assertEqual(protocol["development_holdout_identifier_overlap"], 0)

    def test_cross_validation_fold_mean_is_consistent(self):
        cv = self.report["development_cross_validation"]["random_forest"]
        mean_from_rounded_folds = sum(cv["pr_auc_per_fold_rounded"]) / len(
            cv["pr_auc_per_fold_rounded"]
        )
        # Fold scores are rounded to 4 decimals, so allow a small rounding tolerance.
        self.assertAlmostEqual(
            mean_from_rounded_folds, cv["pr_auc_mean"], delta=0.0001
        )

    def test_secondary_asteroid_confusion_matrix_has_expected_size(self):
        secondary = self.report["holdout_asteroid_level_secondary"]
        matrix = secondary["confusion_matrix"]["values"]
        self.assertEqual(sum(map(sum, matrix)), secondary["sample_count"])
        self.assertEqual(secondary["sample_count"], 739)


if __name__ == "__main__":
    unittest.main(verbosity=2)
