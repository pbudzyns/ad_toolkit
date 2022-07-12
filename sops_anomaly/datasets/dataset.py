import abc
from typing import Optional, Tuple

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
    def load(self) -> None:
        pass

    @abc.abstractmethod
    def get_train_samples(
            self, n_samples: Optional[int] = None) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def get_test_samples(
        self, n_samples: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        pass

    @abc.abstractmethod
    def _load(self):
        pass
