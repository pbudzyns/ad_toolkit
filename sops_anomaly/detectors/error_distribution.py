import functools
from typing import Tuple

import numpy as np
import scipy.stats


class ErrorDistribution:

    def __init__(self, n_dims: int, l_preds: int) -> None:
        self._d_size: int = n_dims
        self._l_preds: int = l_preds
        self._dist: scipy.stats.multivariate_normal = None

    def __call__(self, errors: np.ndarray) -> np.ndarray:
        return -self._dist(errors)

    def get_errors(
            self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Computes reconstruction error of a point x(t) over consecutive
        reconstructions. All self._l_preds reconstructions are needed to
        construct error vector hence only a range of outputs contribute
        to the result.

        Ex. outputs = [
            [x2_1, x3_1, x4_1],
            [x3_2, x4_2, x5_2],
            [x4_3, x5_3, x6_3],
        ]
        Then error can be only computed for x4 using [x4_1, x4_2, x4_3].

        :param output:
        :param target:
        :return:
        """
        errors = []
        for i in range(self._l_preds-1):
            errors += [output[:, i:-self._l_preds+i+1, :, self._l_preds-1-i]]
        errors += [output[:, self._l_preds-1:, :, 0]]
        errors = np.stack(errors, axis=3)
        errors = errors - target[..., np.newaxis]
        return errors.squeeze(axis=0)

    def fit(self, outputs: np.ndarray, targets: np.ndarray) -> None:
        self._fit_error_distribution(outputs, targets)

    def _fit_error_distribution(
            self, outputs: np.ndarray, eval_targets: np.ndarray) -> None:
        # Shape: (batch_size, time_steps, d, l)
        # outputs = self._reshape_output(outputs)
        # Shape: (time_steps, d, l)
        errors = self.get_errors(outputs, eval_targets)
        # Shape: (time_steps, d*l)
        errors = errors.reshape((errors.shape[0], self._l_preds * self._d_size))
        means, cov = self._fit_multivariate_gauss(errors)
        self._dist = functools.partial(
            scipy.stats.multivariate_normal.pdf,
            mean=means,
            cov=cov,
            allow_singular=True,
        )

    @classmethod
    def _fit_multivariate_gauss(
            cls, sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit multivariate gaussian distribution to a given sample using
        maximum likelihood estimation method.

        Source:
          https://stackoverflow.com/questions/27230824/fit-multivariate-gaussian-distribution-to-a-given-dataset

        :param sample:
        :return:
        """
        mean = np.mean(sample, axis=0)
        cov = np.cov(sample, rowvar=0)
        return mean, cov
