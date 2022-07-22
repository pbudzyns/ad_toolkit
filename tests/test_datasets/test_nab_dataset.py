import math

import numpy as np
import pandas as pd
import pytest

from sops_anomaly.datasets import NabDataset


@pytest.fixture(scope="session")
def nab() -> NabDataset:
    nab = NabDataset()
    nab.load()
    return nab


def test_nab_list_datasets():
    d = NabDataset.datasets()
    assert isinstance(d, dict)
    assert len(d.keys()) > 0
    assert all(len(d[k]) > 0 for k in d)


def test_nab_load_dataset(nab):
    x, y = nab.data
    assert isinstance(x, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(x) == len(y)
    assert set(y) == {0, 1}


@pytest.mark.parametrize("normalize", (False, True))
def test_nab_get_train_data(nab, normalize):
    x = nab.get_train_samples(normalize=normalize)
    assert isinstance(x, pd.DataFrame)
    if normalize:
        assert -0.1 < float(x.mean()) < 0.1


def test_nab_get_test_data(nab):
    x, y = nab.get_test_samples()
    assert isinstance(x, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(x) == len(y)


@pytest.mark.skip("Downloading all datasets is slow.")
def test_nab_download_all_datasets():
    datasets = NabDataset.datasets()

    for dataset in datasets:
        for file in datasets[dataset]:
            nab = NabDataset(dataset=dataset, file=file)
            nab.load()

            x, y = nab.data
            assert isinstance(x, pd.DataFrame)
            assert isinstance(y, pd.Series)
            assert len(x) == len(y)
