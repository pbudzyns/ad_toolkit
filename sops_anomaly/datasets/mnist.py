from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf

from sops_anomaly.datasets.dataset import BaseDataset


class MNIST(BaseDataset):

    def __init__(self, anomaly_class: int = 0):
        super(MNIST, self).__init__()
        self._anomaly_class = anomaly_class
        self._data: Optional[Tuple[
            pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = None

    @classmethod
    def sample_size(cls) -> int:
        return 784

    def get_train_samples(self, n_samples: int) -> pd.DataFrame:
        x_train, _, _, _ = self.data
        return x_train.sample(n=n_samples)

    def get_test_samples(
        self,
        n_samples: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate a set of `n_samples` samples for testing there 10% are
        anomalous samples.

        :param n_samples:
        :return:
        """
        _, _, x_test, y_test = self.data
        print("having xtest ytest")
        x_normal, x_anomaly = (
            x_test.loc[~self._test_mask],
            x_test.loc[self._test_mask],
        )
        print("having xnomal, xanmoaly")

        x_normal = x_normal.sample(n=int(0.5*n_samples))
        x_anomaly = x_normal.sample(n=int(0.5*n_samples))
        y_normal = np.zeros((len(x_normal), 1))
        y_anomaly = np.ones((len(x_anomaly), 1))
        return (
            pd.concat((x_normal, x_anomaly)),
            pd.concat((pd.DataFrame(y_normal), pd.DataFrame(y_anomaly))),
        )

    def _load(self) -> None:
        # Download the dataset.
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # Transform the labels.
        y_train, y_test = (
            (y_train == self._anomaly_class), (y_test == self._anomaly_class))
        # Remove anomalies from th training data.
        x_train = x_train[~y_train]
        y_train = y_train[~y_train]
        # Normalize the data and reshape samples to 1D vectors.
        x_train, x_test = x_train / 255, x_test / 255
        x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 784)
        self._test_mask = y_test
        self._data = tuple(
            pd.DataFrame(data=data)
            for data
            in (x_train, y_train, x_test, y_test)
        )

    @classmethod
    def plot_sample(cls, sample: np.ndarray) -> None:
        sample = sample.reshape((28, 28))
        plt.imshow(sample, cmap='gray_r')
        plt.show()
