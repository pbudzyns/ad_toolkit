from typing import List, Union

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from sops_anomaly.models.model import BaseDetector


class AutoEncoder(BaseDetector):

    def __init__(
        self,
        input_size: int = 100,
        activation: str = "relu",
        threshold: float = 0.8,
    ):
        """Base auto-encoder anomaly detection model detecting anomalous
        examples by reconstruction error threshold.

        Implementation based on:
            - https://keras.io/examples/timeseries/timeseries_anomaly_detection

        :param input_size:
        :param activation:
        :param threshold:
        """
        self.model = None
        self._activation = activation
        self._input_size = input_size
        self._threshold = threshold
        self._real_threshold = None

    def _create_model(self) -> keras.Model:
        """Construct auto-encoder model using keras backend. This is a sample
        model that does not allow for much parametrization.

        :return: auto-encoder model
        """
        model = keras.Sequential(
            [
                layers.Input(shape=(self._input_size, 1)),
                layers.Conv1D(
                    filters=32, kernel_size=7, padding="same", strides=2,
                    activation=self._activation,
                ),
                layers.Dropout(rate=0.2),
                layers.Conv1D(
                    filters=16, kernel_size=7, padding="same", strides=2,
                    activation=self._activation,
                ),
                layers.Conv1DTranspose(
                    filters=16, kernel_size=7, padding="same", strides=2,
                    activation=self._activation,
                ),
                layers.Dropout(rate=0.2),
                layers.Conv1DTranspose(
                    filters=32, kernel_size=7, padding="same", strides=2,
                    activation=self._activation,
                ),
                layers.Conv1DTranspose(filters=1, kernel_size=7,
                                       padding="same"),
            ]
        )

        return model

    def _transform_data(self, data: np.ndarray) -> np.ndarray:
        # normalize the data.
        data = (data - np.mean(data)) / np.std(data)
        # construct sliding windows.
        output = []
        for i in range(len(data) - self._input_size + 1):
            output.append(data[i: (i + self._input_size)])
        return np.stack(output)

    def train(self, train_data: np.ndarray, epochs: int = 50):
        # train_data = self._transform_data(data)
        self.model = self._create_model()
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
        )
        history = self.model.fit(
            train_data,
            train_data,
            epochs=epochs,
            batch_size=128,
            validation_split=0.1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, mode="min")
            ],
        )

        reconstructed = self.model.predict(train_data)
        _data = train_data.reshape(reconstructed.shape)

        reconstruction_error = np.mean(np.abs(_data - reconstructed), axis=1)
        self._real_threshold = np.max(reconstruction_error) * self._threshold

        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        # _data = self._transform_data(x)
        _data = x
        reconstructed = self.model.predict(_data)

        _data = _data.reshape(reconstructed.shape)
        reconstruction_error = np.mean(np.abs(_data - reconstructed), axis=1)

        # anomalies = reconstruction_error.reshape((-1)) / _real_threshold
        scores = reconstruction_error.reshape((-1))
        scores[scores > self._real_threshold] = self._real_threshold
        scores /= self._real_threshold

        # results = np.zeros_like(x.flatten())
        # results[:len(results) - self._input_size + 1] = scores
        return scores

    def detect(self, data: np.ndarray) -> Union[List[int], np.ndarray]:
        anomalies = self.predict(data)
        anomalies = anomalies >= 0.9

        # anomalous_data_indices = []
        # for data_idx in range(self._input_size - 1, len(data) - self._input_size + 1):
        #     if np.all(anomalies[data_idx - self._input_size + 1: data_idx]):
        #         anomalous_data_indices.append(data_idx)

        return anomalies
