import abc
from typing import Optional, Tuple

import pandas as pd

from sops_anomaly.datasets.dataset import BaseDataset


class LabeledDataset(BaseDataset, abc.ABC):

    @abc.abstractmethod
    def load(self) -> None:
        pass

    @abc.abstractmethod
    def _load(self):
        pass

    def get_train_samples(
            self, n_samples: Optional[int] = None) -> pd.DataFrame:
        x_train, _, _, _ = self.data

        if n_samples is None or n_samples > len(x_train):
            return x_train

        return x_train.sample(n=n_samples)

    def get_test_samples(
        self,
        n_samples: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate a set of `n_samples` samples for testing there 10% are
        anomalous samples.

        :param n_samples:
        :return:
        """
        _, _, x_test, y_test = self.data

        if n_samples is None or n_samples > len(x_test):
            return x_test, y_test

        x_normal, x_anomaly = (
            x_test[y_test == 0],
            x_test[y_test == 1],
        )

        if len(x_anomaly) < int(0.5 * n_samples):
            n_anomaly = len(x_anomaly)
            n_normal = int(n_samples - n_anomaly)
        else:
            n_anomaly = int(0.5 * n_samples)
            n_normal = int(0.5 * n_samples)

        x_normal = x_normal.sample(n=n_normal)
        x_anomaly = x_anomaly.sample(n=n_anomaly)
        y_normal = y_test[x_normal.index]
        y_anomaly = y_test[x_anomaly.index]

        return (
            pd.concat((x_normal, x_anomaly)),
            pd.concat((y_normal, y_anomaly)),
        )
