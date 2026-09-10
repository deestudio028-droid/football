"""Tests for src/models/evaluate.py -- metric functions, all against
hand-computed small examples, plus probability-input validation.
"""
import _pathfix  # noqa: F401
import math
import unittest

import numpy as np

from models.evaluate import (
    CLASS_ORDER,
    MetricInputError,
    accuracy,
    balanced_accuracy,
    brier_score,
    confusion_matrix,
    encode_labels,
    evaluate,
    log_loss,
    macro_f1,
    validate_probabilities,
)


class TestValidateProbabilities(unittest.TestCase):
    def test_valid_matrix_passes(self):
        P = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
        validate_probabilities(P)  # should not raise

    def test_wrong_number_of_columns_raises(self):
        P = np.array([[0.5, 0.5]])
        with self.assertRaises(MetricInputError):
            validate_probabilities(P)

    def test_nan_raises(self):
        P = np.array([[0.5, np.nan, 0.5]])
        with self.assertRaises(MetricInputError):
            validate_probabilities(P)

    def test_inf_raises(self):
        P = np.array([[0.5, np.inf, -0.5]])
        with self.assertRaises(MetricInputError):
            validate_probabilities(P)

    def test_negative_probability_raises(self):
        P = np.array([[1.2, -0.1, -0.1]])
        with self.assertRaises(MetricInputError):
            validate_probabilities(P)

    def test_row_not_summing_to_one_raises(self):
        P = np.array([[0.5, 0.3, 0.3]])  # sums to 1.1
        with self.assertRaises(MetricInputError):
            validate_probabilities(P)

    def test_row_summing_to_one_within_tolerance_passes(self):
        P = np.array([[0.3333333, 0.3333333, 0.3333334]])
        validate_probabilities(P)  # should not raise


class TestEncodeLabels(unittest.TestCase):
    def test_maps_to_class_order_indices(self):
        self.assertEqual(CLASS_ORDER, ["H", "D", "A"])
        idx = encode_labels(["H", "D", "A", "H"])
        np.testing.assert_array_equal(idx, [0, 1, 2, 0])

    def test_unexpected_label_raises(self):
        with self.assertRaises(MetricInputError):
            encode_labels(["H", "WIN"])


class TestLogLoss(unittest.TestCase):
    def test_hand_computed_single_row(self):
        # true class H, P(H)=0.5 -> -log(0.5)
        ll = log_loss(["H"], [[0.5, 0.3, 0.2]])
        self.assertAlmostEqual(ll, -math.log(0.5), places=10)

    def test_hand_computed_two_rows(self):
        y = ["H", "A"]
        P = [[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]]
        expected = -(math.log(0.5) + math.log(0.8)) / 2
        self.assertAlmostEqual(log_loss(y, P), expected, places=10)

    def test_perfect_prediction_near_zero_loss(self):
        y = ["H", "D", "A"]
        P = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        self.assertAlmostEqual(log_loss(y, P), 0.0, places=10)

    def test_confident_wrong_prediction_is_clipped_not_infinite(self):
        y = ["H"]
        P = [[0.0, 0.0, 1.0]]  # predicted A with certainty, true was H
        ll = log_loss(y, P)
        self.assertTrue(np.isfinite(ll))
        self.assertGreater(ll, 30)  # clipped at eps=1e-15 -> -log(1e-15) ~= 34.5


class TestBrierScore(unittest.TestCase):
    def test_hand_computed_single_row(self):
        # true=H -> one-hot [1,0,0]; P=[0.5,0.3,0.2]
        # (0.5-1)^2 + (0.3-0)^2 + (0.2-0)^2 = 0.25 + 0.09 + 0.04 = 0.38
        b = brier_score(["H"], [[0.5, 0.3, 0.2]])
        self.assertAlmostEqual(b, 0.38, places=10)

    def test_perfect_prediction_is_zero(self):
        b = brier_score(["D"], [[0.0, 1.0, 0.0]])
        self.assertAlmostEqual(b, 0.0, places=10)

    def test_hand_computed_two_rows_averaged(self):
        y = ["H", "D"]
        P = [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2]]
        row1 = 0.25 + 0.09 + 0.04  # 0.38
        row2 = (0.2 - 0) ** 2 + (0.6 - 1) ** 2 + (0.2 - 0) ** 2  # 0.04+0.16+0.04=0.24
        self.assertAlmostEqual(brier_score(y, P), (row1 + row2) / 2, places=10)


class TestAccuracyAndConfusionMatrix(unittest.TestCase):
    def test_hand_computed_accuracy(self):
        y = ["H", "D", "A", "H"]
        P = [
            [0.9, 0.05, 0.05],  # predicts H, correct
            [0.1, 0.1, 0.8],    # predicts A, wrong (true D)
            [0.0, 0.0, 1.0],    # predicts A, correct
            [0.4, 0.4, 0.2],    # predicts H (first max), correct
        ]
        self.assertAlmostEqual(accuracy(y, P), 3 / 4, places=10)

    def test_hand_computed_confusion_matrix(self):
        y = ["H", "D", "A", "H"]
        P = [
            [0.9, 0.05, 0.05],  # true H, pred H
            [0.1, 0.1, 0.8],    # true D, pred A
            [0.0, 0.0, 1.0],    # true A, pred A
            [0.4, 0.4, 0.2],    # true H, pred H
        ]
        cm = confusion_matrix(y, P)
        expected = np.array([
            [2, 0, 0],  # true H: 2 predicted H
            [0, 0, 1],  # true D: 1 predicted A
            [0, 0, 1],  # true A: 1 predicted A
        ])
        np.testing.assert_array_equal(cm, expected)


class TestMacroF1AndBalancedAccuracy(unittest.TestCase):
    def test_perfect_predictions_give_one(self):
        y = ["H", "D", "A", "H", "D", "A"]
        P = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ]
        self.assertAlmostEqual(macro_f1(y, P), 1.0, places=10)
        self.assertAlmostEqual(balanced_accuracy(y, P), 1.0, places=10)

    def test_hand_computed_macro_f1(self):
        # Confusion matrix (true rows, pred cols), classes H,D,A:
        # H: 2 correct, 0 elsewhere
        # D: 1 predicted as A (0 correct)
        # A: 1 correct
        y = ["H", "H", "D", "A"]
        P = [
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05],
            [0.1, 0.1, 0.8],  # true D, predicted A
            [0.0, 0.0, 1.0],  # true A, predicted A
        ]
        # H: tp=2, fp=0, fn=0 -> precision=1, recall=1, f1=1
        # D: tp=0, fp=0, fn=1 -> precision=0 (0/0->0), recall=0 -> f1=0
        # A: tp=1, fp=1 (the D row), fn=0 -> precision=0.5, recall=1, f1=2*0.5*1/1.5=0.6667
        expected = (1.0 + 0.0 + (2 * 0.5 * 1 / 1.5)) / 3
        self.assertAlmostEqual(macro_f1(y, P), expected, places=6)

    def test_hand_computed_balanced_accuracy(self):
        y = ["H", "H", "D", "A"]
        P = [
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05],
            [0.1, 0.1, 0.8],
            [0.0, 0.0, 1.0],
        ]
        # recall H = 2/2 = 1.0, recall D = 0/1 = 0.0, recall A = 1/1 = 1.0
        expected = (1.0 + 0.0 + 1.0) / 3
        self.assertAlmostEqual(balanced_accuracy(y, P), expected, places=10)


class TestEvaluateOrchestrator(unittest.TestCase):
    def test_evaluate_returns_consistent_result(self):
        y = ["H", "D", "A"]
        P = [[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]]
        result = evaluate(y, P)
        self.assertEqual(result.n, 3)
        self.assertAlmostEqual(result.log_loss, log_loss(y, P))
        self.assertAlmostEqual(result.brier, brier_score(y, P))
        self.assertAlmostEqual(result.macro_f1, macro_f1(y, P))
        self.assertAlmostEqual(result.balanced_accuracy, balanced_accuracy(y, P))
        self.assertAlmostEqual(result.accuracy, accuracy(y, P))
        np.testing.assert_array_equal(result.confusion_matrix, confusion_matrix(y, P))

    def test_evaluate_rejects_invalid_probabilities(self):
        with self.assertRaises(MetricInputError):
            evaluate(["H"], [[0.5, 0.5, 0.5]])

    def test_as_dict_is_json_serializable_shape(self):
        y = ["H", "D"]
        P = [[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]]
        d = evaluate(y, P).as_dict()
        self.assertIsInstance(d["confusion_matrix"], list)
        self.assertEqual(set(d.keys()), {
            "log_loss", "brier", "macro_f1", "balanced_accuracy",
            "accuracy", "confusion_matrix", "n",
        })


if __name__ == "__main__":
    unittest.main()
