
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
        features = self.metadata["features"]

        self.assertEqual(report_model["feature_count"], len(features))
        self.assertEqual(report_model["features"], features)
        self.assertEqual(self.metadata["feature_count"], len(features))
        self.assertEqual(len(features), 16)
        self.assertNotIn("Absolute Magnitude", features)
        self.assertNotIn("Minimum Orbit Intersection", features)

    def test_threshold_matches_and_is_valid(self):
        threshold = self.report["evaluation_protocol"]["decision_threshold"]

        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)
        self.assertAlmostEqual(
            threshold, self.metadata["prediction_threshold"], places=12
        )
        self.assertFalse(
            self.report["evaluation_protocol"]["threshold_selection"][
                "holdout_labels_used_for_threshold_selection"
            ]
        )

    def test_holdout_metrics_match_confusion_matrix(self):
        holdout = self.report["holdout_record_level"]
        matrix = holdout["confusion_matrix"]["values"]

        tn, fp = matrix[0]
        fn, tp = matrix[1]
        total = tn + fp + fn + tp

        expected = {
            "precision_positive": tp / (tp + fp),
            "recall_positive": tp / (tp + fn),
            "f1_positive": 2 * tp / (2 * tp + fp + fn),
            "accuracy": (tp + tn) / total,
            "specificity": tn / (tn + fp),
            "false_positive_rate": fp / (fp + tn),
            "negative_predictive_value": tn / (tn + fn),
            "balanced_accuracy": (
                tp / (tp + fn) + tn / (tn + fp)
            ) / 2,
        }

        metrics = holdout["metrics"]
        for name, value in expected.items():
            with self.subTest(metric=name):
                self.assertAlmostEqual(metrics[name], value, places=12)

    def test_holdout_scores_match_metadata(self):
        report_metrics = self.report["holdout_record_level"]["metrics"]
        metadata_metrics = self.metadata["evaluation"]

        for key in (
            "pr_auc_average_precision",
            "roc_auc",
            "precision_positive",
            "recall_positive",
            "f1_positive",
            "accuracy",
        ):
            with self.subTest(metric=key):
                self.assertAlmostEqual(
                    report_metrics[key], metadata_metrics[key], places=12
                )

    def test_record_level_counts_are_consistent(self):
        holdout = self.report["holdout_record_level"]
        matrix = holdout["confusion_matrix"]["values"]
        support = holdout["actual_class_support"]
        predicted = holdout["predicted_class_count"]

        self.assertEqual(len(matrix), 2)
        self.assertTrue(all(len(row) == 2 for row in matrix))

        tn, fp = matrix[0]
        fn, tp = matrix[1]

        self.assertEqual(sum(support.values()), holdout["sample_count"])
        self.assertEqual(tn + fp + fn + tp, holdout["sample_count"])
        self.assertEqual(tn + fp, support["negative"])
        self.assertEqual(fn + tp, support["positive"])
        self.assertEqual(tn + fn, predicted["negative"])
        self.assertEqual(fp + tp, predicted["positive"])

        named_counts = holdout["confusion_matrix"]
        self.assertEqual(tn, named_counts["true_negative"])
        self.assertEqual(fp, named_counts["false_positive"])
        self.assertEqual(fn, named_counts["false_negative"])
        self.assertEqual(tp, named_counts["true_positive"])

    def test_holdout_protocol_matches_metadata(self):
        protocol = self.report["evaluation_protocol"]
        evaluation = self.metadata["evaluation"]

        self.assertEqual(protocol["holdout_rows"], 948)
        self.assertEqual(protocol["holdout_rows"],
                         self.report["holdout_record_level"]["sample_count"])
        self.assertEqual(protocol["holdout_rows"], evaluation["holdout_rows"])
        self.assertEqual(
            protocol["holdout_unique_asteroids"],
            evaluation["holdout_unique_asteroids"],
        )
        self.assertEqual(protocol["development_holdout_identifier_overlap"], 0)
        self.assertEqual(
            protocol["holdout_positive_rows"]
            + protocol["holdout_negative_rows"],
            protocol["holdout_rows"],
        )

    def test_cross_validation_fold_mean_is_consistent(self):
        cv = self.report["development_cross_validation"]["random_forest"]
        scores = cv["pr_auc_per_fold_rounded"]

        self.assertEqual(len(scores), 5)
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))

        mean_from_rounded_folds = sum(scores) / len(scores)
        self.assertAlmostEqual(
            mean_from_rounded_folds, cv["pr_auc_mean"], delta=0.0001
        )

    def test_asteroid_level_confusion_matrix_is_consistent(self):
        secondary = self.report["holdout_asteroid_level_secondary"]
        matrix = secondary["confusion_matrix"]["values"]

        tn, fp = matrix[0]
        fn, tp = matrix[1]
        total = tn + fp + fn + tp

        self.assertEqual(total, secondary["sample_count"])
        self.assertEqual(total, 739)

        metrics = secondary["metrics"]
        self.assertAlmostEqual(
            metrics["precision_positive"], tp / (tp + fp), places=12
        )
        self.assertAlmostEqual(
            metrics["recall_positive"], tp / (tp + fn), places=12
        )
        self.assertAlmostEqual(
            metrics["accuracy"], (tp + tn) / total, places=12
        )

    def test_holdout_feature_importance_is_marked_post_hoc(self):
        importance = self.report["post_hoc_feature_importance"]
        protocol = self.report["evaluation_protocol"]

        self.assertIn("holdout", importance["data"].lower())
        self.assertIn("post-hoc", importance["caveat"].lower())
        self.assertIn("holdout", protocol["holdout_caveat"].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
