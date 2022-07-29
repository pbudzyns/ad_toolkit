import pathlib
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from ad_toolkit.datasets.labeled_dataset import LabeledDataset


class KddCup(LabeledDataset):

    def __init__(
        self,
        full_dataset: bool = False,
    ) -> None:
        """KDD Cup 1999 dataset containing network intrusion events.

        Reference:
         - http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

        :param full_dataset:
        """

        super(KddCup, self).__init__()

        if full_dataset:
            self._gz_filename = "kddcup.data.gz"
        else:
            self._gz_filename = "kddcup.data_10_percent.gz"

        url_root: str = "http://kdd.ics.uci.edu/databases/kddcup99/"
        self._names_filename: str = "kddcup.names"
        self._label_name: str = "status"
        self._data_url: str = url_root + self._gz_filename
        self._names_url: str = url_root + self._names_filename
        self._data_file: Optional[pathlib.Path] = None

    def load(self) -> None:
        self._load()

    def _load(self) -> None:
        data, types = self._load_data_from_url()
        self._mark_attacks(data)
        self._encode_categorical(data, types)

        labels = data.pop('status')
        x_train, x_test, y_train, y_test = train_test_split(
            data.index, labels, train_size=0.8)
        self._data = (data.loc[x_train], y_train, data.loc[x_test], y_test)

    def _load_data_from_url(self) -> Tuple[pd.DataFrame, List[str]]:
        """Uses temporal directory to download the data files and load them
        into the memory.

        :return:
        """
        with tempfile.TemporaryDirectory() as tempdir:
            data_filename, names_filename = self._download_dataset(tempdir)
            names, types = self._extract_names(names_filename)
            data = pd.read_csv(data_filename, header=0, names=names)

        data[self._label_name] = (
            data[self._label_name].map(lambda x: x.rstrip('.')))
        return data, types

    def _mark_attacks(self, data: pd.DataFrame) -> None:
        """Simplify types of anomalous events as classifying specific types of
        attacks is not in the main area of interest.

        :param data:
        :return:
        """
        new_labels = []
        for row in data[self._label_name]:
            if row != "normal":
                new_labels.append(1)
            else:
                new_labels.append(0)
        data[self._label_name] = new_labels

    def _encode_categorical(self, data: pd.DataFrame, types: List[str]) -> None:
        """Encodes symbolic variables into categorical values.

        :param data:
        :param types:
        :return:
        """
        to_drop = []
        for i, (name, col) in enumerate(data.items()):
            if types[i] == "continuous":
                data[name] = np.nan_to_num(
                    (data[name] - data[name].mean())/data[name].std())
                continue
            if types[i] != "symbolic" or name == self._label_name:
                continue
            to_drop.append(name)
            encoder = OneHotEncoder().fit(col.values.reshape(-1, 1))
            encoded = encoder.transform(
                col.values.reshape(-1, 1))
            for xx in range(encoded.shape[1]):
                data[f"{name}_x{xx}"] = encoded[:, xx].toarray()
        for name in to_drop:
            data.drop(name, axis=1, inplace=True)

    def _extract_names(
        self,
        names_filename: pathlib.Path,
    ) -> Tuple[List[str], List[str]]:
        """Extracts column names and their types from the names file.

        :param names_filename:
        :return:
        """
        names = tuple(name.split(": ")
                      for name
                      in names_filename.read_text().split('\n')[1:-1])
        col_names = [name[0] for name in names] + [self._label_name]
        types = [name[1].rstrip('.') for name in names] + ["symbolic"]
        return col_names, types

    def _download_dataset(
        self,
        tempdir: str,
    ) -> Tuple[pathlib.Path, pathlib.Path]:
        """Downloads kdd cup dataset into the provided directory.

        :param tempdir:
        :return:
        """
        dir_path = pathlib.Path(tempdir)
        data_filename = dir_path / self._gz_filename
        names_filename = dir_path / self._names_filename
        self._download_file(data_filename, self._data_url)
        self._download_file(names_filename, self._names_url)
        return data_filename, names_filename

    @classmethod
    def _download_file(cls, data_filename: pathlib.Path, data_url: str) -> None:
        """Downloads content from the `data_url` into a file under
        `data_filename`.

        :param data_filename:
        :param data_url:
        :return:
        """
        with data_filename.open("wb") as f:
            r = requests.get(data_url)
            f.write(r.content)
