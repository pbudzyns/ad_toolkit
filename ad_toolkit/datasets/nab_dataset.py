from collections import defaultdict
import json
from typing import Any, Dict, List, Optional, Tuple
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
    ) -> None:
        """Downloads time series data contained in NAB Benchmark. The labeled
        anomalous windows are transformed into labels.

        To get a full list of datasets and files run `NabDataset.datasets()`.

        Parameters
        ----------
        dataset
            Dataset name from the NAB Repository.
        file
            File name from the NAB Repository.
        """
        super(NabDataset, self).__init__()
        self._dataset: str = dataset
        self._file: str = file
        self._data_url: str = f"{URL_ROOT}/data/{dataset}/{file}"
        self._all_labels: Optional[Dict] = None

    @classmethod
    def datasets(cls) -> Dict[str, List[str]]:
        """Returns a dictionary containing datasets and their files."""
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
        self, standardize: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Get train samples and their labels.

        Parameters
        ----------
        standardize
            Whether to standardize the data.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Time series and labels marking anomalous windows.
        """
        x, y = self.data
        data = self._standardize_data(x) if standardize else x
        return data, y

    @classmethod
    def _standardize_data(cls, data):
        standardised = (data - data.mean())
        std = standardised.std().value
        if std > 0:
            standardised /= std
        return standardised

    def get_test_samples(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns raw time series.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
        """
        return self.data

    def plot(
        self, anomalies: Optional[Dict[str, np.ndarray]] = None,
        vertical_margin: int = 10, show_legend: bool = False,
        anomaly_style_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot time series, mark anomalous windows and visualize detected
        anomalies if provided.

        Parameters
        ----------
        anomalies
            Dictionary with detector names as keys and anomaly labels returned
            by them as values.
        vertical_margin
            Vertical margin of the plot.
        show_legend
            Controls legend appearance.
        anomaly_style_kwargs
            Kwargs to style vertical lines representing anomalies.

        Returns
        -------
        None
        """
        x, labels = self.data
        fig, ax = TimeSeriesPlot.plot(
            x, labels=labels, anomalies=anomalies,
            vertical_margin=vertical_margin, show_legend=show_legend,
            anomaly_style_kwargs=anomaly_style_kwargs,
        )

        return fig, ax
