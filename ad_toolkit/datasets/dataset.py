import abc
from typing import Tuple

import pandas as pd


class BaseDataset(abc.ABC):

    def __init__(self):
        """Base dataset abstract class. A dataset should be able to return
        data containing training samples, testing samples and testing labels.
        It should be possible to get access to training labels from `data`
        attribute.
        """
        self._data = None

    @property
    def data(self):
        """Returns unprocessed data underlying the dataset."""
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
