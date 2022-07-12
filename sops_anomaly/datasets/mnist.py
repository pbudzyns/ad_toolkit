from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sops_anomaly.datasets.dataset import BaseDataset


class MNIST(BaseDataset):

    def __init__(self, anomaly_class: int = 0) -> None:
        super(MNIST, self).__init__()
        self._anomaly_class = anomaly_class
        self._data: Optional[Tuple[
            pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]] = None

    @classmethod
    def sample_size(cls) -> int:
        return 784

    def load(self) -> None:
        self._load()

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

    def _load(self) -> None:
        # Download the dataset.
        (x_train, y_train), (x_test, y_test) = (
            tf.keras.datasets.mnist.load_data())
        # Transform the labels.
        y_train, y_test = (
            (y_train == self._anomaly_class), (y_test == self._anomaly_class))
        # Normalize the data and reshape samples to 1D vectors.
        x_train, x_test = x_train / 255, x_test / 255
        x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 784)
        self._test_mask = y_test
        self._data = (
            pd.DataFrame(x_train),
            pd.Series(y_train),
            pd.DataFrame(x_test),
            pd.Series(y_test),
        )

    @classmethod
    def plot_sample(cls, sample: np.ndarray) -> None:
        sample = sample.reshape((28, 28))
        plt.imshow(sample, cmap='gray_r')
        plt.show()
