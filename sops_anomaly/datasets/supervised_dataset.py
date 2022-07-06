from typing import Tuple

import pandas as pd

from sops_anomaly.datasets.dataset import BaseDataset


class SupervisedDataset(BaseDataset):

    def __init__(self, dataset: BaseDataset, anomaly_class: int = 1) -> None:
        """Dataset wrapper that filters out anomalous samples from
        training set.

        :param dataset:
        :param anomaly_class:
        """
        super(SupervisedDataset, self).__init__()
        self._dataset = dataset
        self._anomaly_class = anomaly_class

    def _load(self) -> None:
        self._dataset._load()

    def get_train_samples(self, n_samples: int) -> pd.DataFrame:
        x_train, y_train, _, _ = self._dataset.data
        x_train = x_train.loc[y_train != self._anomaly_class]

        return x_train.sample(n=n_samples)

    def get_test_samples(
        self, n_samples: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._dataset.get_test_samples(n_samples=n_samples)
