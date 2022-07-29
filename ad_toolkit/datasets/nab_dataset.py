from collections import defaultdict
import json
from typing import Any, Dict, Optional, Tuple
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ad_toolkit.datasets.dataset import BaseDataset
from ad_toolkit.evaluation import TimeSeriesPlot

URL_ROOT = "https://raw.githubusercontent.com/numenta/NAB/master/"


class NabDataset(BaseDataset):

    _datasets: Optional[Dict] = None

    def __init__(
        self,
        dataset: str = "artificialWithAnomaly",
        file: str = "art_daily_jumpsup.csv",
    ):
        super(NabDataset, self).__init__()
        self._dataset: str = dataset
        self._file: str = file
        self._data_url: str = f"{URL_ROOT}/data/{dataset}/{file}"
        self._all_labels: Optional[Dict] = None

    @classmethod
    def datasets(cls):
        if cls._datasets is None:
            cls._load_datasets()
        return cls._datasets

    @classmethod
    def _load_datasets(cls):
        data = cls._download_labels_file()
        datasets = defaultdict(list)
        for key in data:
            dataset, file = key.split('/')
            datasets[dataset].append(file)
        cls._datasets = datasets

    @classmethod
    def _download_labels_file(cls):
        r = urllib.request.urlopen(URL_ROOT + "/labels/combined_windows.json")
        data = json.loads(r.read().decode())
        return data

    def load(self) -> None:
        self._load()

    def _load(self):
        data = pd.read_csv(
            self._data_url, parse_dates=True, index_col="timestamp"
        )
        if self._all_labels is None:
            self._all_labels = self._download_labels_file()
        label_key = f"{self._dataset}/{self._file}"
        anomaly_windows = self._all_labels[label_key]

        labels = pd.Series(data=np.zeros((len(data))), index=data.index)
        for (start, stop) in anomaly_windows:
            labels[start:stop] = 1

        self._data = (data, labels)

    def get_train_samples(
        self, normalize: bool = True,
    ) -> pd.DataFrame:
        data = self._normalized_data() if normalize else self.data[0]
        return data

    def _normalized_data(self):
        normalized = self.data[0].copy()
        normalized = (normalized - normalized.mean()) / normalized.std()
        return normalized

    def get_test_samples(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self.data

    def plot(
        self,
        anomalies: Optional[Dict[str, np.ndarray]] = None,
        vertical_margin: int = 10,
        show_legend: bool = False,
        anomaly_style_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        x, labels = self.data
        TimeSeriesPlot.plot(
            x, labels=labels, anomalies=anomalies,
            vertical_margin=vertical_margin, show_legend=show_legend,
            anomaly_style_kwargs=anomaly_style_kwargs,
        )


if __name__ == '__main__':
    nab = NabDataset()
    _, y = nab.data
    anomalies = np.zeros_like(y)
    anomalies2 = anomalies.copy()
    step = int(len(y) / 10)
    anomalies[2*step] = 1
    anomalies[9*step] = 1
    anomalies[8*step] = 1
    anomalies2[4 * step] = 1
    anomalies2[5 * step] = 1
    anomalies2[7 * step] = 1

    nab.plot(anomalies={'anomalies': anomalies, 'anomalies2': anomalies2})
    plt.show()
