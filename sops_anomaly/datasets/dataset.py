import abc
from typing import Tuple

import pandas as pd


class BaseDataset(abc.ABC):

    def __init__(self):
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._load()
        return self._data

    @abc.abstractmethod
    def get_train_samples(self, n_samples: int) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def get_test_samples(
        self, n_samples: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass

    @abc.abstractmethod
    def _load(self):
        pass
