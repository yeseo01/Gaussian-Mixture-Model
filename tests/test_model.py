"""Regression tests for the from-scratch Gaussian Mixture Model."""

import io
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from src.model import GaussianMixture


class GaussianMixtureTests(unittest.TestCase):
    @staticmethod
    def _fit_silently(model, X):
        with redirect_stdout(io.StringIO()):
            model.fit(X)
        return model

    @staticmethod
    def _two_cluster_data():
        return np.array(
            [
                [-3.0, -3.1],
                [-2.9, -3.0],
                [-3.1, -2.9],
                [-2.8, -3.2],
                [3.0, 3.1],
                [2.9, 3.0],
                [3.1, 2.9],
                [2.8, 3.2],
            ],
            dtype=float,
        )

    def test_same_random_state_is_reproducible(self):
        X = self._two_cluster_data()

        first = self._fit_silently(
            GaussianMixture(2, random_state=7),
            X,
        )
        second = self._fit_silently(
            GaussianMixture(2, random_state=7),
            X,
        )

        np.testing.assert_array_equal(first.predict(X), second.predict(X))
        np.testing.assert_array_equal(first.phi, second.phi)
        np.testing.assert_array_equal(first.means, second.means)
        np.testing.assert_array_equal(
            np.asarray(first.covariance),
            np.asarray(second.covariance),
        )

    def test_fit_does_not_modify_global_numpy_rng(self):
        X = self._two_cluster_data()

        np.random.seed(12345)
        expected = np.random.random(5)

        np.random.seed(12345)

        self._fit_silently(
            GaussianMixture(2, random_state=7),
            X,
        )

        actual = np.random.random(5)

        np.testing.assert_array_equal(actual, expected)

    def test_dataframe_and_ndarray_inputs_match(self):
        X = self._two_cluster_data()
        frame = pd.DataFrame(X, columns=["x1", "x2"])

        array_model = self._fit_silently(
            GaussianMixture(2),
            X,
        )
        frame_model = self._fit_silently(
            GaussianMixture(2),
            frame,
        )

        np.testing.assert_array_equal(
            array_model.predict(X),
            frame_model.predict(frame),
        )
        np.testing.assert_array_equal(
            array_model.means,
            frame_model.means,
        )
        np.testing.assert_array_equal(
            np.asarray(array_model.covariance),
            np.asarray(frame_model.covariance),
        )

    def test_wrong_feature_count_is_rejected(self):
        X = self._two_cluster_data()

        model = self._fit_silently(
            GaussianMixture(2),
            X,
        )

        with self.assertRaisesRegex(
            ValueError,
            "fitted with 2 features",
        ):
            model.predict(X[:, :1])

        with self.assertRaisesRegex(
            ValueError,
            "fitted with 2 features",
        ):
            model.log_likelihood(X[:, :1])

        with self.assertRaisesRegex(
            ValueError,
            "fitted with 2 features",
        ):
            model.BIC(X[:, :1])

    def test_covariance_regularization_handles_degenerate_data(self):
        X = np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

        reg_covar = 1e-6

        model = self._fit_silently(
            GaussianMixture(
                1,
                reg_covar=reg_covar,
            ),
            X,
        )

        covariance = np.asarray(model.covariance[0])
        eigenvalues = np.linalg.eigvalsh(covariance)

        self.assertTrue(np.all(np.isfinite(covariance)))
        self.assertTrue(
            np.all(eigenvalues >= reg_covar * (1 - 1e-9))
        )
        self.assertTrue(np.isfinite(model.log_likelihood(X)))

    def test_log_space_estep_handles_far_away_samples(self):
        X = self._two_cluster_data()

        model = self._fit_silently(
            GaussianMixture(2),
            X,
        )

        far_away = np.array(
            [
                [10000.0, 10000.0],
                [-10000.0, -10000.0],
            ]
        )

        labels = model.predict(far_away)

        self.assertEqual(labels.shape, (2,))
        self.assertTrue(
            np.all(np.isfinite(model.responsibility))
        )
        np.testing.assert_allclose(
            model.responsibility.sum(axis=1),
            np.ones(2),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertTrue(
            np.isfinite(model.log_likelihood(far_away))
        )

    def test_bic_matches_full_covariance_parameter_count(self):
        X = self._two_cluster_data()

        model = self._fit_silently(
            GaussianMixture(2),
            X,
        )

        n_samples, n_features = X.shape

        parameter_count = (
            (model.K - 1)
            + model.K * n_features
            + model.K * n_features * (n_features + 1) / 2
        )

        expected = (
            -2.0 * model.log_likelihood(X)
            + parameter_count * np.log(n_samples)
        )

        self.assertAlmostEqual(
            model.BIC(X),
            expected,
            places=12,
        )

    def test_inference_before_fit_is_rejected(self):
        X = self._two_cluster_data()

        model = GaussianMixture(2)

        with self.assertRaisesRegex(
            RuntimeError,
            "must be fitted",
        ):
            model.predict(X)

        with self.assertRaisesRegex(
            RuntimeError,
            "must be fitted",
        ):
            model.log_likelihood(X)

        with self.assertRaisesRegex(
            RuntimeError,
            "must be fitted",
        ):
            model.BIC(X)


if __name__ == "__main__":
    unittest.main()
