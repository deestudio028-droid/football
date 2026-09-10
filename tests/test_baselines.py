"""Tests for src/models/baselines.py -- majority-class and frequency
baselines. Hand-computed examples throughout, per the Step 2 spec.
"""
import _pathfix  # noqa: F401
import unittest

import numpy as np
import pandas as pd

from models.baselines import (
    CLASS_ORDER,
    BaselineError,
    FrequencyBaseline,
    MajorityClassBaseline,
    class_frequencies,
)


class TestClassFrequencies(unittest.TestCase):
    def test_hand_computed_frequencies(self):
        # 5 H, 3 D, 2 A -> [0.5, 0.3, 0.2] in H, D, A order
        y = pd.Series(["H"] * 5 + ["D"] * 3 + ["A"] * 2)
        freqs = class_frequencies(y)
        self.assertEqual(CLASS_ORDER, ["H", "D", "A"])
        np.testing.assert_allclose(freqs, [0.5, 0.3, 0.2])

    def test_frequencies_sum_to_one(self):
        y = pd.Series(["H", "H", "D", "A", "A", "A", "A"])
        freqs = class_frequencies(y)
        self.assertAlmostEqual(freqs.sum(), 1.0)

    def test_missing_class_gets_zero_frequency(self):
        # No draws at all in training data -- must not error, must yield 0.
        y = pd.Series(["H", "H", "A"])
        freqs = class_frequencies(y)
        np.testing.assert_allclose(freqs, [2 / 3, 0.0, 1 / 3])

    def test_unexpected_label_raises(self):
        y = pd.Series(["H", "WIN"])
        with self.assertRaises(BaselineError):
            class_frequencies(y)

    def test_empty_training_labels_raises(self):
        y = pd.Series([], dtype=object)
        with self.assertRaises(BaselineError):
            class_frequencies(y)


class TestFrequencyBaseline(unittest.TestCase):
    def test_predict_proba_repeats_training_frequencies_for_every_row(self):
        y_train = pd.Series(["H"] * 4 + ["D"] * 4 + ["A"] * 2)  # 0.4, 0.4, 0.2
        baseline = FrequencyBaseline().fit(y_train)
        P = baseline.predict_proba(n_rows=3)
        self.assertEqual(P.shape, (3, 3))
        for row in P:
            np.testing.assert_allclose(row, [0.4, 0.4, 0.2])

    def test_rows_sum_to_one(self):
        y_train = pd.Series(["H", "D", "D", "A"])
        baseline = FrequencyBaseline().fit(y_train)
        P = baseline.predict_proba(n_rows=10)
        np.testing.assert_allclose(P.sum(axis=1), np.ones(10))

    def test_predict_proba_before_fit_raises(self):
        with self.assertRaises(BaselineError):
            FrequencyBaseline().predict_proba(n_rows=1)

    def test_only_training_labels_influence_output_not_row_count_or_features(self):
        # Predicting for a different n_rows must not change the per-class
        # probabilities -- proves predict_proba has no access to anything
        # but the frequencies computed at fit() time (i.e. no leakage path
        # for validation-set information to enter via n_rows/features).
        y_train = pd.Series(["H"] * 7 + ["A"] * 3)
        baseline = FrequencyBaseline().fit(y_train)
        p_small = baseline.predict_proba(n_rows=1)[0]
        p_large = baseline.predict_proba(n_rows=500)[0]
        np.testing.assert_allclose(p_small, p_large)


class TestMajorityClassBaseline(unittest.TestCase):
    def test_predicts_majority_class_with_probability_one(self):
        y_train = pd.Series(["H"] * 6 + ["D"] * 3 + ["A"] * 1)
        baseline = MajorityClassBaseline().fit(y_train)
        P = baseline.predict_proba(n_rows=2)
        expected_row = [1.0, 0.0, 0.0]  # H is majority
        for row in P:
            np.testing.assert_allclose(row, expected_row)

    def test_draw_can_be_majority(self):
        y_train = pd.Series(["H", "D", "D", "D", "A"])
        baseline = MajorityClassBaseline().fit(y_train)
        P = baseline.predict_proba(n_rows=1)
        np.testing.assert_allclose(P[0], [0.0, 1.0, 0.0])

    def test_rows_sum_to_one(self):
        y_train = pd.Series(["A", "A", "H"])
        baseline = MajorityClassBaseline().fit(y_train)
        P = baseline.predict_proba(n_rows=5)
        np.testing.assert_allclose(P.sum(axis=1), np.ones(5))

    def test_predict_proba_before_fit_raises(self):
        with self.assertRaises(BaselineError):
            MajorityClassBaseline().predict_proba(n_rows=1)


if __name__ == "__main__":
    unittest.main()
