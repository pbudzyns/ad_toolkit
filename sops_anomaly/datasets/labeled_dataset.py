import abc
from typing import Optional, Tuple

import pandas as pd

from sops_anomaly.datasets.dataset import BaseDataset


class LabeledDataset(BaseDataset, abc.ABC):

    @abc.abstractmethod
    def load(self) -> None:
        """To be implemented."""
        pass

    @abc.abstractmethod
    def _load(self):
        """To be implemented."""
        pass

    def get_train_samples(
            self, n_samples: Optional[int] = None) -> pd.DataFrame:
        """Generate a set of `n_samples` samples for training.

        :param n_samples: Number of samples to return.
        :return: Samples
        """
        x_train, _, _, _ = self.data

        if n_samples is None or n_samples > len(x_train):
            return x_train

        return x_train.sample(n=n_samples)

    def get_test_samples(
        self,
        n_samples: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate a set of `n_samples` samples for testing where 50% are
        anomalous samples if possible.

        :param n_samples: Number of samples to return.
        :return: Samples and labels.
        """
        _, _, x_test, y_test = self.data

        if n_samples is None or n_samples > len(x_test):
            return x_test, y_test

        x_test_sample = x_test.sample(n=n_samples)
        y_test_sample = y_test[x_test_sample.index]
        return x_test_sample, y_test_sample
