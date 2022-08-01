from typing import Any, Optional, Tuple

import pandas as pd

from ad_toolkit.datasets.dataset import BaseDataset
from ad_toolkit.datasets.labeled_dataset import LabeledDataset


class SupervisedDataset(BaseDataset):

    def __init__(self, dataset: LabeledDataset, anomaly_class: Any = 1) -> None:
        """Dataset wrapper that filters out anomalous samples from
        training set or includes a desired percentage of anomalies in the
        training samples.

        Parameters
        ----------
        dataset
            Dataset containing the data.
        anomaly_class
            Label corresponding to the anomaly class. Will be used to filter
            out anomalies based on dataset labels.
        """
        super(SupervisedDataset, self).__init__()
        self._dataset = dataset
        self._data = None
        self._anomaly_class = anomaly_class

    def load(self) -> None:
        """Loads the data."""
        self._load()

    @property
    def data(self):
        """Returns raw data."""
        if self._data is None:
            self._load()
        return self._data

    def _load(self) -> None:
        """Load the dataset."""
        self._dataset.load()
        self._data = self._dataset.data

    def get_train_samples(
        self,
        n_samples: Optional[int] = None,
        anomaly_percentage: Optional[float] = None,
    ) -> pd.DataFrame:
        """Create train set with respect to provided parameters. If `n_samples`
        is specified returns a total of `n_samples` samples. In case of
        `anomaly_percentage` provided the set contains given percentage
        of anomalous samples. Otherwise, it consists of normal data only.

        Parameters
        ----------
        n_samples
            Number of samples to return.
        anomaly_percentage
            Percentage of anomalous samples to include in the training set.

        Returns
        -------
        pd.DataFrame
            Training samples either without anomalies or with requested
            percentage of anomalous samples.
        """
        x_train, y_train, _, _ = self.data
        if anomaly_percentage is None:
            return self._get_normal_dataset(x_train, y_train, n_samples)

        return self._get_mixed_dataset(
            x_train, y_train, n_samples, anomaly_percentage)

    def _get_normal_dataset(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        n_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """Create a dataset composed of only normal samples."""
        normal_train = x_train.loc[y_train != self._anomaly_class]
        if n_samples is None or n_samples > len(normal_train):
            return normal_train
        return normal_train.sample(n=n_samples)

    def _get_mixed_dataset(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        n_samples: Optional[int],
        ap: float,
    ) -> pd.DataFrame:
        """Create a dataset composed of normal and anomalous samples."""
        normal_train = x_train[y_train != self._anomaly_class]
        anomaly_train = x_train[y_train == self._anomaly_class]
        if n_samples is None:
            n_anomaly = int((len(normal_train)/(1-ap)) * ap)
            x_normal = normal_train
            x_anomaly = anomaly_train.sample(n=n_anomaly)
        else:
            n_anomaly = int(ap * n_samples)
            x_normal = normal_train.sample(
                n=min(len(normal_train), n_samples-n_anomaly))
            x_anomaly = anomaly_train.sample(
                n=min(len(anomaly_train), n_anomaly))

        return pd.concat((x_normal, x_anomaly))

    def get_test_samples(
        self, n_samples: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns test set containing samples and corresponding labels.

        Parameters
        ----------
        n_samples
            Number of test samples to return.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Training samples and corresponding labels.
        """
        return self._dataset.get_test_samples(n_samples=n_samples)
