from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture


class ShrinkageGaussianRatio:
    def fit(self, positive: np.ndarray, negative: np.ndarray):
        self.models_ = [self._fit_class(negative), self._fit_class(positive)]
        return self

    @staticmethod
    def _fit_class(values):
        model = LedoitWolf().fit(values)
        sign, logdet = np.linalg.slogdet(model.covariance_)
        if sign <= 0:
            raise FloatingPointError("non-positive covariance determinant")
        return model.location_, model.precision_, float(logdet)

    @staticmethod
    def _logp(values, model):
        center, precision, logdet = model
        delta = values - center
        return -0.5 * (
            np.einsum("bi,ij,bj->b", delta, precision, delta) + logdet
        )

    def decision_function(self, values):
        return self._logp(values, self.models_[1]) - self._logp(values, self.models_[0])


class DiagonalGMMRatio:
    def __init__(self, seed: int):
        kwargs = dict(
            n_components=2, covariance_type="diag", n_init=3,
            reg_covar=1e-4, random_state=seed, max_iter=300,
        )
        self.negative_ = GaussianMixture(**kwargs)
        self.positive_ = GaussianMixture(**kwargs)

    def fit(self, positive: np.ndarray, negative: np.ndarray):
        self.positive_.fit(positive)
        self.negative_.fit(negative)
        return self

    def decision_function(self, values):
        return self.positive_.score_samples(values) - self.negative_.score_samples(values)

