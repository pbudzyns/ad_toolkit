import functools
from typing import Callable, List, Union

import numpy as np
import pandas as pd
import torch

from sops_anomaly.detectors.autoencoder import AutoEncoder
from sops_anomaly.detectors.base_detector import BaseDetector


class AutoEncoderTSS(BaseDetector):

    def __init__(self, window_size: int):
        """Auto-encoder using time series segmentation for anomaly detection.

        :param window_size:
        """
        super(AutoEncoderTSS, self).__init__()
        self.ae = AutoEncoder(window_size=window_size)
        self._window_size: int = window_size
        self.l2_norm: Callable[
            [np.ndarray], float] = functools.partial(np.linalg.norm, ord=2, axis=1)

    def distance(
        self,
        x: Union[np.ndarray, List[np.ndarray]],
        y: Union[np.ndarray, List[np.ndarray]],
    ) -> float:
        """Distance measure from Equation (10) in the reference paper.

        :param x:
        :param y:
        :return:
        """
        return self.l2_norm(x - y) / np.sqrt(self.l2_norm(x) - self.l2_norm(y))

    def train(self, train_data: pd.DataFrame) -> None:
        self.ae.train(train_data)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self._window_size > 1:
            input_data = self.ae._transform_data(data)
            input_data = self.ae._data_to_tensors(input_data)
        else:
            input_data = self.ae._data_to_tensors(data)

        encoded_values = []
        self.ae.model.eval()
        with torch.no_grad():
            for sample in input_data:
                encoded = self.ae.model.encoder(sample)
                encoded_values.append(encoded)

        distances = self.distance(encoded_values[:-1], encoded_values[1:])
        print(distances)

    def detect(self, data: pd.DataFrame) -> np.ndarray:
        pass

