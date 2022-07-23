from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sops_anomaly.detectors.base_detector import BaseDetector


class Visualizer:

    def __init__(
        self,
        dataset: pd.DataFrame,
        anomaly_detector: BaseDetector,
    ):
        """Simple visualizer for validation of anomaly detector capabilities.

        :param dataset: time-series data.
        :param anomaly_detector: anomaly detector instance.
        """
        self._dataset = dataset
        data = np.array(dataset["value"])
        self._data = data.reshape((len(dataset["value"]), 1))
        self._ad = anomaly_detector

    def evaluate(self) -> None:
        """Fit the given model into the dataset and plots detected anomalies.

        :return: None.
        """
        self._ad.train(self._data)
        predictions = self._ad.predict(self._data)
        anomalies = self._ad.detect(self._data)
        print("Predict on: ", self._data.shape)
        print("Predictions: ", predictions.shape,
              "range", np.min(predictions),
              "-", np.max(predictions))
        print("Anomalies: ", len(anomalies))

        self._plot(anomalies)

    def _plot(self, anomalies: Union[List[int], np.ndarray]) -> None:
        _, ax = plt.subplots()
        self._dataset.plot(legend=False, ax=ax)
        if np.any(anomalies):
            data_anomaly = self._dataset.iloc[anomalies]
            data_anomaly.plot(
                legend=False, ax=ax, color='r', marker='.', linewidth=0)
        plt.show()


if __name__ == '__main__':
    master_url_root = (
        "https://raw.githubusercontent.com/numenta/NAB/master/data/")

    df_small_noise_url_suffix = "artificialNoAnomaly/art_daily_small_noise.csv"
    df_small_noise_url = master_url_root + df_small_noise_url_suffix
    df_small_noise = pd.read_csv(
        df_small_noise_url, parse_dates=True, index_col="timestamp"
    )

    df_daily_jumpsup_url_suffix = "artificialWithAnomaly/art_daily_nojump.csv"
    df_daily_jumpsup_url = master_url_root + df_daily_jumpsup_url_suffix
    df_daily_jumpsup = pd.read_csv(
        df_daily_jumpsup_url, parse_dates=True, index_col="timestamp"
    )

    from sops_anomaly.detectors import AutoEncoder, RandomDetector, Betsi

    autoencoder = AutoEncoder(
        input_size=180,
        threshold=0.55,
    )
    random_detector = RandomDetector()
    betsi = Betsi(layer_dims=[100, 50], threshold=0.9)
    visualizer = Visualizer(
        dataset=df_daily_jumpsup,
        anomaly_detector=autoencoder,
    )

    visualizer.evaluate()
