from collections import defaultdict
import json
from typing import Any, Dict, Optional, Tuple
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sops_anomaly.datasets.dataset import BaseDataset

URL_ROOT = "https://raw.githubusercontent.com/numenta/NAB/master/"


class NabDataset(BaseDataset):

    _datasets: Optional[Dict] = None
    _anomaly_colors = (
        'red', 'magenta', 'green', 'navy', 'dodgerblue', 'orange', 'brown')

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
        show_legend: bool = True,
        anomaly_style_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        ax = plt.gca()
        fig = plt.gcf()

        x, labels = self.data
        x_lim, y_lim = self._get_x_y_lim(vertical_margin, x)
        self._plot_known_anomalies(labels, y_lim)
        if anomalies is not None:
            self._plot_predicted_anomalies(
                anomalies, labels, y_lim, anomaly_style_kwargs)
        if show_legend:
            plt.legend()

        plt.plot(x)
        ax.set_ylim(y_lim)
        ax.set_xlim(x_lim)
        fig.set_size_inches(10, 5)

    @classmethod
    def _plot_predicted_anomalies(
            cls, anomalies, labels, y_lim, anomaly_style_kwargs) -> None:
        style = {'ls': '-.', 'lw': 0.5, 'alpha': 0.1}
        if anomaly_style_kwargs is not None:
            style.update(anomaly_style_kwargs)
        for i, (name, points) in enumerate(anomalies.items()):
            anomalies_idx = labels.index[points.astype(bool)]
            color = cls._anomaly_colors[i % len(cls._anomaly_colors)]
            plt.vlines(
                x=anomalies_idx,
                ymin=y_lim[0],
                ymax=y_lim[1],
                colors=color,
                label=name,
                **style,
            )

    @classmethod
    def _get_x_y_lim(cls, vertical_margin, x):
        y_lim = (
            float(np.min(x) - vertical_margin),
            float(np.max(x) + vertical_margin),
        )
        x_lim = (
            np.min(x.index),
            np.max(x.index),
        )
        return x_lim, y_lim

    @classmethod
    def _plot_known_anomalies(cls, labels, y_lim):
        anomalous_labels = labels[labels == 1].index
        plt.vlines(
            x=anomalous_labels,
            ymin=y_lim[0],
            ymax=y_lim[1],
            colors='#eda8a6',
            ls='-',
            lw=2,
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
