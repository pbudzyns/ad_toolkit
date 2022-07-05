import abc


class BaseDataset(abc.ABC):

    def __init__(self):
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._load()
        return self._data

    @abc.abstractmethod
    def _load(self):
        pass
