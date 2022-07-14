"""
Time-series segmentation with auto-encoder.

References:
    - Lee, Wei-Han, et al. "Time series segmentation through automatic feature
      learning."
    - Boumghar, Redouane, et al. "Behaviour-based anomaly detection in
      spacecraft using deep learning."
    - https://gitlab.com/librespacefoundation/polaris/betsi

"""
import functools
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch

from sops_anomaly.detectors.autoencoder import AutoEncoder
from sops_anomaly.detectors.base_detector import BaseDetector


class AutoEncoderTSS(BaseDetector):

    def __init__(
        self,
        window_size: int,
        latent_size: int = 10,
        threshold: float = 1.1,
    ) -> None:
        """Auto-encoder using time series segmentation for anomaly detection.

        :param window_size:
        """
        super(AutoEncoderTSS, self).__init__()
        self.ae = AutoEncoder(window_size=window_size, latent_size=latent_size)
        self._window_size: int = window_size
        self._threshold: float = threshold
        self.l2_norm: Callable[[np.ndarray], float] = (
            functools.partial(np.linalg.norm, ord=2, axis=1))

    def distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Distance measure from Equation (10) in the reference paper.

        :param x:
        :param y:
        :return:
        """
        return self.l2_norm(x - y) / np.sqrt(self.l2_norm(x) - self.l2_norm(y))

    def train(self, train_data: pd.DataFrame, epochs: int = 30) -> None:
        self.ae.train(train_data, epochs=epochs)

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
                encoded_values.append(encoded.detach().numpy())
        encoded_values = np.array(encoded_values)

        distances = [0] * self._window_size
        distances += list(self.distance(encoded_values[:-1], encoded_values[1:]))
        return np.array(distances)

    def detect(
        self, data: pd.DataFrame, threshold: Optional[float] = None,
    ) -> np.ndarray:
        if threshold is None:
            threshold = self._threshold

        distance_list = self.predict(data)
        events_at = self._get_local_extrema(distance_list, threshold)

        anomalies = np.zeros_like(distance_list)
        anomalies[events_at] = 1
        return anomalies

    @classmethod
    def _get_local_extrema(
            cls, distance_list: np.ndarray, threshold: float) -> List[int]:
        events_at = []
        prev_distance = distance_list[0]
        curr_distance = distance_list[1]
        curr_sum = curr_distance
        sum_dict = {}
        for index in range(2, len(distance_list)):
            next_distance = distance_list[index]

            # To find extremum, events on both sides should have lower distance
            if (next_distance <= curr_distance
                    and prev_distance <= curr_distance):
                # The index is for the next distance, so -1
                sum_dict[index - 1] = curr_sum
                curr_sum = abs(curr_distance)
            else:
                curr_sum = curr_sum + abs(curr_distance)

            prev_distance = curr_distance
            curr_distance = next_distance
        average = sum(sum_dict.values()) / len(sum_dict)
        threshold = (threshold / 100 + 1) * average
        for index, sum_dist in sum_dict.items():
            if sum_dist >= threshold:
                events_at.append(index)
        return events_at

