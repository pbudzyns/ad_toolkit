from collections import defaultdict
import json
from typing import Dict, Optional, Tuple
import urllib.request

import numpy as np
import pandas as pd

from sops_anomaly.datasets.dataset import BaseDataset

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

    def get_train_samples(self) -> pd.DataFrame:
        return self._data[0]

    def get_test_samples(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self._data


if __name__ == '__main__':
    nab = NabDataset(dataset="artificialWithAnomaly",
                     file="art_increase_spike_density.csv")
    x, y = nab.data
    import matplotlib.pyplot as plt
    x.plot()
    y.plot()
    plt.show()

