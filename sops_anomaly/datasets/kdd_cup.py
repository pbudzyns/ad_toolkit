import pathlib
import requests
import tempfile
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sops_anomaly.datasets.dataset import BaseDataset


class KddCup(BaseDataset):

    def __init__(
        self,
        full_dataset: bool = False,
        semi_supervised: bool = False,
    ) -> None:
        """KDD Cup 1999 dataset containing network intrusion events.

        Reference:
         - http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

        :param full_dataset:
        :param semi_supervised:
        """

        super(KddCup, self).__init__()

        if full_dataset:
            self._gz_filename = "kddcup.data.gz"
        else:
            self._gz_filename = "kddcup.data_10_percent.gz"

        url_root: str = "http://kdd.ics.uci.edu/databases/kddcup99/"
        self._semi_supervised: bool = semi_supervised
        self._names_filename: str = "kddcup.names"
        self._label_name: str = "status"
        self._data_url: str = url_root + self._gz_filename
        self._names_url: str = url_root + self._names_filename
        self._data_file: Optional[pathlib.Path] = None

    def _load(self) -> None:
        data, types = self._load_data_from_url()
        self._mark_attacks(data)
        self._encode_categorical(data, types)

        # TODO: prepare filtering for semi-supervised mode
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
                new_labels.append("anomaly")
            else:
                new_labels.append(row)
        data[self._label_name] = new_labels

    @classmethod
    def _encode_categorical(cls, data: pd.DataFrame, types: List[str]) -> None:
        """Encodes symbolic variables into categorical values.

        :param data:
        :param types:
        :return:
        """
        for i, (name, col) in enumerate(data.items()):
            if types[i] != "symbolic":
                continue
            data[name] = LabelEncoder().fit_transform(col)

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


if __name__ == '__main__':
    kdd = KddCup()
    a = kdd.data
    print(a)
