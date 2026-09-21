"""Gaussian Mixture Model implemented from scratch using EM."""

import numpy as np
import pandas as pd


class GaussianMixture:
    def __init__(
        self,
        n_components,
        max_iter=100,
        reg_covar=1e-6,
        random_state=42,
        tol=0.1,
    ):
        if (
            not isinstance(n_components, (int, np.integer))
            or isinstance(n_components, bool)
            or n_components < 1
        ):
            raise ValueError("n_components must be a positive integer.")

        if (
            not isinstance(max_iter, (int, np.integer))
            or isinstance(max_iter, bool)
            or max_iter < 1
        ):
            raise ValueError("max_iter must be a positive integer.")

        if (
            not isinstance(reg_covar, (int, float, np.integer, np.floating))
            or not np.isfinite(reg_covar)
            or reg_covar < 0
        ):
            raise ValueError("reg_covar must be a finite non-negative number.")

        if (
            not isinstance(tol, (int, float, np.integer, np.floating))
            or not np.isfinite(tol)
            or tol <= 0
        ):
            raise ValueError("tol must be a finite positive number.")

        self.K = n_components
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.tol = tol
        self.phi = None  # (K,)
        self.means = None  # (K, d)
        self.covariance = None  # (K, d, d)
        self.weights = None
        self.responsibility = None
        self.n_features_in_ = None

    def _check_is_fitted(self):
        if (
            self.phi is None
            or self.means is None
            or self.covariance is None
        ):
            raise RuntimeError(
                "Model must be fitted before inference or evaluation."
            )

    @staticmethod
    def _validate_input(X):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")

        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("X must contain numeric values.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")

        return X

    def _validate_feature_count(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model "
                f"was fitted with {self.n_features_in_} features."
            )

    # Initialize model parameters
    def initialize(self, X):
        if self.K > X.shape[0]:
            raise ValueError(
                "n_components cannot exceed the number of samples."
            )

        rng = np.random.RandomState(self.random_state)

        self.phi = np.ones((self.K)) / self.K  # (K,)
        indices = rng.choice(X.shape[0], self.K, replace=False)  # (K,)
        self.means = X[indices]  # Sampled data points as initial means, (K, d)
        self.covariance = [
            np.eye(X.shape[1]) for _ in range(self.K)
        ]  # (K, d, d)

    # Compute the log-density of a multivariate Gaussian
    def log_multivariate_normal(self, x, mean, covariance):
        d = len(mean)
        diff = x - mean

        sign, log_det = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise np.linalg.LinAlgError(
                "Covariance matrix must be positive definite."
            )

        quadratic = diff.T @ np.linalg.solve(covariance, diff)

        return -0.5 * (
            d * np.log(2 * np.pi)
            + log_det
            + quadratic
        )

    # Compute the density of a multivariate Gaussian
    def multivariate_normal(self, x, mean, covariance):
        return np.exp(
            self.log_multivariate_normal(x, mean, covariance)
        )

    @staticmethod
    def _logsumexp(values, axis=None):
        values = np.asarray(values)

        max_values = np.max(
            values,
            axis=axis,
            keepdims=True,
        )

        result = max_values + np.log(
            np.sum(
                np.exp(values - max_values),
                axis=axis,
                keepdims=True,
            )
        )

        if axis is None:
            return float(result.squeeze())

        return np.squeeze(result, axis=axis)

    # Compute log-likelihood
    def log_likelihood(self, X):
        self._check_is_fitted()
        X = self._validate_input(X)
        self._validate_feature_count(X)
        log_likelihood = 0.0

        for i in range(X.shape[0]):
            log_components = np.empty(self.K)

            for k in range(self.K):
                log_components[k] = (
                    np.log(self.phi[k])
                    + self.log_multivariate_normal(
                        X[i],
                        self.means[k],
                        self.covariance[k],
                    )
                )

            log_likelihood += self._logsumexp(log_components)

        return log_likelihood

    def BIC(self, X):
        self._check_is_fitted()
        X = self._validate_input(X)
        self._validate_feature_count(X)
        N, d = X.shape  # Number of samples and features
        # Parameters: mixing(K-1) + means(K*d) + covariances(K*d*(d+1)/2)
        p = (self.K - 1) + self.K * d + self.K * d * (d + 1) / 2
        ll = self.log_likelihood(X)

        return -2.0 * ll + p * np.log(N)

    # Expectation step
    def e_step(self, X):
        """Compute posterior responsibilities for each sample and component.

        The responsibility matrix has shape (N, K) and is used by the
        M-step to update mixture weights, means, and covariances.
        """
        # Log joint probability of sample x_i and component k
        log_weights = np.empty((self.K, X.shape[0]))

        for i in range(X.shape[0]):
            for k in range(self.K):
                log_weights[k, i] = (
                    np.log(self.phi[k])
                    + self.log_multivariate_normal(
                        X[i],
                        self.means[k],
                        self.covariance[k],
                    )
                )

        # Preserve the weights attribute in probability space
        self.weights = np.exp(log_weights)

        # Normalize in log space to avoid numerical underflow
        log_normalizer = self._logsumexp(
            log_weights,
            axis=0,
        )

        self.responsibility = np.exp(
            log_weights - log_normalizer
        ).T

    # Maximization step
    def m_step(self, X):
        """Update parameters using the current responsibilities."""
        # Effective number of samples assigned to each component
        N_k = self.responsibility.sum(axis=0)  # (K,)

        # Update mixture weights
        self.phi = N_k / len(X)  # (K,)

        # Update component means using responsibility-weighted averages
        self.means = (self.responsibility.T @ X) / N_k[:, None]  # (K, d)

        # Update full covariance matrices using the same responsibilities
        for k in range(self.K):
            diff = (X - self.means[k])  # (N, d) - (d,) => (N, d)
            self.covariance[k] = (
                (self.responsibility[:, k][:, None] * diff).T @ diff
            ) / N_k[k]
            self.covariance[k] += self.reg_covar * np.eye(X.shape[1])

    # Fit the model with the EM algorithm
    def fit(self, x):
        x = self._validate_input(x)
        self.n_features_in_ = x.shape[1]

        # Initialize parameters
        self.initialize(x)

        prev_log_likelihood = -np.inf

        # Repeat E- and M-steps until convergence or max_iter
        for i in range(self.max_iter):
            self.e_step(x)
            self.m_step(x)

            # Evaluate the updated log-likelihood
            log_likelihood = self.log_likelihood(x)

            # Stop when the absolute log-likelihood change falls below tolerance
            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                print(f"Iteration {i + 1}")
                break

            prev_log_likelihood = log_likelihood

    # Assign each sample to the component with the highest responsibility
    def predict(self, x):
        self._check_is_fitted()
        x = self._validate_input(x)
        self._validate_feature_count(x)

        self.e_step(x)
        # Select the component with the largest posterior responsibility
        clusters = np.argmax(self.responsibility, axis=1)  # (N,)

        return clusters
