from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sops_anomaly.datasets.labeled_dataset import LabeledDataset


class MNIST(LabeledDataset):

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
