"""
Time-series segmentation with auto-encoder.

"""
from typing import List, Union

import betsi.models
import betsi.predictors
import betsi.preprocessors
from nab.detectors.base import AnomalyDetector
import numpy as np

from sops_anomaly.models.model import BaseDetector


class Betsi(BaseDetector):

    def __init__(
        self,
        layer_dims: List[int],
        activations: List = None,
        optimizer: str = 'adam',
        loss: str = 'mean_squared_error',
        metrics: List[str] = None,
        threshold: float = 0.8,
    ):
        """
        Anomaly detection model based on Behaviour Extraction for Time-Series
        Investigation (BETSI). Auto-encoder model trained with backpropagation.

        Source: https://gitlab.com/librespacefoundation/polaris/betsi

        :param layer_dims:
        :param activations:
        :param optimizer:
        :param loss:
        :param metrics:
        """
        super(Betsi, self).__init__()

        self._window_size = layer_dims[0]
        self._threshold = threshold
        self._t_mean, self._t_std = None, None

        metrics = metrics if metrics is not None else ['MSE']

        if activations is not None:
            self.ae_model, self.en_model, self.de_model = (
                betsi.models.custom_autoencoder(layer_dims, activations=activations)
            )
        else:
            self.ae_model, self.en_model, self.de_model = (
                betsi.models.custom_autoencoder(layer_dims)
            )

        self.ae_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.en_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.de_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def _transform_data(self, data: np.ndarray):
        if self._t_mean is None and self._t_std is None:
            self._t_mean = np.mean(data)
            self._t_std = np.std(data)
        # normalize the data.
        data = (data - self._t_mean) / self._t_std
        # construct sliding windows.
        output = []
        for i in range(len(data) - self._window_size + 1):
            output.append(data[i: (i + self._window_size)])
        return np.stack(output)

    def train(
        self,
        train_data,
        epochs: int = 50,
        batch_size: int = 128,
        validation_split: float = 0.1,
    ):
        """
        Train the model and return it. Uses given batch_size and number of epochs.

        :param train_data:
        :param batch_size:
        :param epochs:
        :param validation_split:
        :return:
        """

        train_data = self._transform_data(train_data)
        history = self.ae_model.fit(
            train_data,
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
        )

        return history

    def predict(self, x: np.ndarray) -> Union[List[float], np.ndarray]:
        """

        :param x:
        :return:
        """
        data = self._transform_data(x)
        data_reproduced = self.en_model.predict(data)

        distances = []
        for row in range(data_reproduced.shape[0] - 1):
            distances.append(
                betsi.predictors.distance_measure(
                    data_reproduced[row], data_reproduced[row+1])
            )

        return distances

    def detect(self, data: np.ndarray) -> Union[List[int], np.ndarray]:
        distances = self.predict(data)
        noise_margin_per = int(self._threshold * 100)
        return betsi.predictors.get_events(
            distances, threshold=noise_margin_per)


class BetsiNab(AnomalyDetector):

    # def initialize(self):
    #     self.window_size = 20
    #     self.model = Betsi([self.window_size, self.window_size // 2, ])
    #     self.previous = []
    #     self.normalizer = None
    #
    #     train_data = self.prepare_train_data(self.dataSet)
    #     self.model.train(train_data)
    #
    # def prepare_train_data(self, data_file: DataFile):
    #     data = np.array(data_file.data['value'])
    #     windowed_data = np.lib.stride_tricks.sliding_window_view(
    #         data,
    #         window_shape=self.window_size,
    #     )
    #
    #     normalized, normalizer = betsi.preprocessors.normalize_all_data(windowed_data)
    #     self.normalizer = normalizer
    #
    #     return normalized

    def handleRecord(self, inputData):

        # value = inputData['value']
        # if len(self.previous) < self.window_size:
        #     self.previous.append(value)
        #     return [0.0]
        #
        # previous = self.previous
        # current = self.previous[1:] + [value]
        # self.previous = current
        #
        # input_data = np.array([previous, current])
        # input_data = self.normalizer.transform(input_data)
        #
        # prediction = self.model.predict(input_data)
        # return prediction
        pass

    # @classmethod
    # def thresholding(cls, distances, threshold):
    #     distances = np.array(np.abs(distances))
    #     probas = distances.copy()
    #     mean = np.mean(distances)
    #     threshold_value = mean * threshold
    #     probas[distances > threshold_value] = 1.0
    #     probas[distances < threshold_value] /= threshold_value
    #     return list(probas)

    def local_maxima_score(self, distance_list):
        prev_distance = distance_list[0]
        curr_distance = distance_list[1]

        curr_sum = curr_distance
        sum_dict = {}
        for index in range(2, len(distance_list)):
            next_distance = distance_list[index]

            # To find extremum, events on both sides should have lower distance
            if next_distance <= curr_distance and prev_distance <= curr_distance:
                # The index is for the next distance, so -1
                sum_dict[index - 1] = curr_sum
                curr_sum = abs(curr_distance)
            else:
                curr_sum = curr_sum + abs(curr_distance)

            prev_distance = curr_distance
            curr_distance = next_distance

        scores = [0] * len(distance_list)
        for idx, score in sum_dict.items():
            scores[idx] = score

        return list(np.abs(scores) / np.max(np.abs(scores)))

    def format_results(self, probas, window_size):
        # headers = self.getHeader()
        # ans = pandas.DataFrame(rows, columns=headers)
        ans = self.dataSet.data.copy()
        ans['anomaly_score'] = probas
        return ans

    def run(self):
        window_size = 10
        data = np.array(self.dataSet.data['value'])
        data = np.array(([data[0]]*window_size) + list(data))
        normalized = data / np.max(data)
        # normalized, normalizer = betsi.preprocessors.normalize_all_data(
        #     np.array([data]))
        windowed_data = np.lib.stride_tricks.sliding_window_view(
            normalized,
            window_shape=window_size,
        )

        model = Betsi([window_size, window_size // 2, ])
        model.train(windowed_data, epochs=30)

        scores = model.predict(windowed_data)
        # scores = self.thresholding(scores, 1.5)
        anomalies = betsi.predictors.get_events(scores, 160)
        predictions = np.zeros((len(scores),))
        predictions[anomalies] = 1
        # predictions = np.abs(scores) / np.max(np.abs(scores))
        # predictions = self.local_maxima_score(scores)
        return self.format_results(list(predictions), window_size)


if __name__ == '__main__':
    import pathlib

    from nab.corpus import Corpus

    data_root = pathlib.Path('.').absolute().parents[1] / "nab_files" / "data"
    data_file = list(data_root.iterdir())[0]

    corpus = Corpus(str(data_file))
    data = list(corpus.dataFiles.values())[0]
    # normalized, _ = betsi.preprocessors.normalize_all_data()
    # print(normalized)
    windowed_data = np.lib.stride_tricks.sliding_window_view(
        np.array(data.data['value']),
        window_shape=20,
    )
    # print(data.data['value'])
    print(windowed_data.shape)

    model = Betsi([20, 10])
    model.train(windowed_data)
    results = model.predict(windowed_data[:200])
    # print(euc_dist(np.array([1,2,3]), np.array([1,2,3])))
    print(results)
