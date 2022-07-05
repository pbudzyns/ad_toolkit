import numpy as np
import pandas as pd

from sops_anomaly.datasets.dataset import BaseDataset

URL_ROOT = "https://raw.githubusercontent.com/numenta/NAB/master/data/"


class NabDataset(BaseDataset):

    def __init__(
        self,
        dataset: str = "artificialWithAnomaly",
        file: str = "art_daily_jumpsup.csv",
    ):
        super(NabDataset, self).__init__()
        self._data_url = f"{URL_ROOT}/{dataset}/{file}"

    def _load(self):
        data = pd.read_csv(
            self._data_url, parse_dates=True, index_col="timestamp"
        )
        self._data = data
