"""
Time-series segmentation with auto-encoder.

References:

    [1] Lee, Wei-Han, et al. "Time series segmentation through automatic feature
        learning."

    [2] Boumghar, Redouane, et al. "Behaviour-based anomaly detection in
        spacecraft using deep learning."

    [3] https://gitlab.com/librespacefoundation/polaris/betsi

"""
import functools
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn

from ad_toolkit.detectors.autoencoder import AutoEncoder
from ad_toolkit.detectors.base_detector import BaseDetector


class AutoEncoderTSS(BaseDetector):

    def __init__(
        self,
        window_size: int,
        layers: Union[List[int], Tuple[int]] = (500, 200),
        latent_size: int = 10,
    ) -> None:
        """Auto-encoder based time series segmentation for anomaly detection.

        Parameters
        ----------
        window_size
            Size of the window if multiple time steps should be used as
            an input. If `window_size` > 1 then samples from consecutive
            time steps will be concatenated together.
        latent_size
            Size of the latent space for auto-encoder model.
        layers
            Sizes of hidden layer of auto-encoder model.
        """
        super(AutoEncoderTSS, self).__init__()
        self.ae = AutoEncoder(
            window_size=window_size, layers=layers, latent_size=latent_size)
        self.model: Optional[nn.Module] = None
        self._window_size: int = window_size
        self._latent_size: int = latent_size
        self.l2_norm: Callable[[np.ndarray], float] = (
            functools.partial(np.linalg.norm, ord=2, axis=1))

    def distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Distance measure from Equation (10) in ref. [1].

        Parameters
        ----------
        x
            The first data vector.
        y
            The second data vector.

        Returns
        -------
        np.ndarray
            An array with distance measures.
        """
        return self.l2_norm(x - y) / np.sqrt(self.l2_norm(x) - self.l2_norm(y))

    def train(self, train_data: pd.DataFrame, epochs: int = 30) -> None:
        """Trains underlying encoder-decoder model to fit the data.

        Parameters
        ----------
        train_data
            ``pandas.DataFrame`` containing samples as rows. Features should
            correspond to columns.
        epochs
            Number of epochs to use during the training.
        Returns
        -------
        None
        """
        self.ae.train(train_data, epochs=epochs)
        self.model = self.ae.model

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Returns prediction score for each data point. The predictions are
        meant to detect local extreme points which should correspond to
        breakpoints. A places where significant change in behaviour happened.

        Parameters
        ----------
        data
            ``pandas.DataFrame`` containing data samples.

        Returns
        -------
        np.ndarray
            Distance measures between consecutive time steps.
        """
        all_data_tensors = self.ae.prepare_data(data)

        encoded_values = []
        self.ae.model.eval()
        with torch.no_grad():
            for sample in all_data_tensors:
                encoded = self.ae.model.encoder(sample)
                encoded_values.append(encoded.detach().numpy())
        encoded_values = np.array(encoded_values)

        distances = [0] * self._window_size
        distances += list(
            self.distance(encoded_values[:-1], encoded_values[1:]))
        return np.array(distances)

    def detect(
        self, data: pd.DataFrame, threshold: float = 0.9,
    ) -> np.ndarray:
        """Applies a given threshold as a percent of average distance score
        got from the prediction to detect anomalous data points.

        Parameters
        ----------
        data
            ``pandas.DataFrame`` containing data samples.
        threshold
            A percent of average distance to use for anomaly detection.

        Returns
        -------
        np.ndarray
            Labels marking anomalous data point.
        """
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

            # To find extreme, events on both sides should have lower distance
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
