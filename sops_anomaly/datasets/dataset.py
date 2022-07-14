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
    def load(self) -> None:
        """To be implemented."""
        pass

    @abc.abstractmethod
    def get_train_samples(self) -> pd.DataFrame:
        """To be implemented."""
        pass

    @abc.abstractmethod
    def get_test_samples(self) -> Tuple[pd.DataFrame, pd.Series]:
        """To be implemented."""
        pass

    @abc.abstractmethod
    def _load(self):
        """To be implemented."""
        pass
