import abc

import pandas as pd
import numpy as np


class BaseDetector(abc.ABC):
    """Abstract class for the base model for anomaly detection.
    Defines methods that have to be implemented by specific detectors.
    """

    @abc.abstractmethod
    def train(self, train_data: pd.DataFrame) -> None:
        """Train the model using `train_data`. Model takes raw data
        vector representing entire time series and have to perform
        the data transformation independently, including normalization
        and constructing sliding window.

        Parameters
        ----------
        train_data
            Train data.
        """
        pass

    @abc.abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Return raw model's output for the given `data`.

        Parameters
        ----------
        data
            Data to make prediction.
        """
        pass
