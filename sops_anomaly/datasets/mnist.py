from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf


class MNIST:

    def __init__(self, anomaly_class: int = 0):
        self._anomaly_class = anomaly_class
        self._data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None

    @classmethod
    def sample_size(cls) -> int:
        return 784

    @property
    def data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._data is None:
            self._load()
        return self._data

    def get_train_samples(self, n_samples: int) -> np.ndarray:
        x_train, _, _, _ = self.data
        return shuffle(x_train, n_samples=n_samples)

    def get_test_samples(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a set of `n_samples` samples for testing there 10% are
        anomalous samples.

        :param n_samples:
        :return:
        """
        _, _, x_test, y_test = self.data
        x_normal, x_anomaly = (
            x_test[~y_test],
            x_test[y_test],
        )

        x_normal = shuffle(x_normal, n_samples=int(0.9*n_samples))
        x_anomaly = shuffle(x_anomaly, n_samples=int(0.1*n_samples))
        return (
            np.concatenate((x_normal, x_anomaly)),
            np.concatenate(
                (np.zeros((len(x_normal),)), np.ones((len(x_anomaly),)))
            ),
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
        self._data = (x_train, y_train, x_test, y_test)

    @classmethod
    def plot_sample(cls, sample: np.ndarray) -> None:
        sample = sample.reshape((28, 28))
        plt.imshow(sample, cmap='gray_r')
        plt.show()
