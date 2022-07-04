import abc
from typing import List, Union

import numpy as np


class BaseDetector(abc.ABC):
    """Abstract class for the base model for anomaly detection.
    Defines methods that have to be implemented by specific models.
    """

    @abc.abstractmethod
    def train(self, train_data: np.ndarray, epochs: int):
        """Train the model using `train_data`. Model takes raw data
        vector representing entire time series and have to perform
        the data transformation independently, including normalization
        and constructing sliding window.

        :param train_data: Time series data.
        :param epochs:
        :return: training history
        """
        pass

    @abc.abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return raw model's output for the given `data`.

        :param data: Time series data.
        :return: Model's output.
        """
        pass

    @abc.abstractmethod
    def detect(self, data: np.ndarray) -> Union[List[int], np.ndarray]:
        """Detect anomalies in the provided data using trained model.

        :param data: Time series data.
        :return: Array of anomalous indices.
        """
        pass
