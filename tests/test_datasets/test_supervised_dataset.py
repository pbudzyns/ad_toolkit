from typing import Optional

import pandas as pd
import pytest

from ad_toolkit.datasets import KddCup
from ad_toolkit.datasets import SupervisedDataset


@pytest.fixture(scope="session")
def dataset() -> SupervisedDataset:
    # dataset = MNIST()
    dataset = KddCup()
    dataset = SupervisedDataset(dataset=dataset)
    return dataset


def test_sup_dataset_returns_data(dataset: SupervisedDataset):
    data = dataset.data
    assert data is not None


def test_sup_dataset_get_test_data(dataset: SupervisedDataset):
    x_test, y_test = dataset.get_test_samples()
    assert isinstance(x_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)
    assert len(x_test) == len(y_test)


@pytest.mark.parametrize("n_samples", (None, 10, 100, 1000, 10e9))
def test_sup_dataset_get_test_data_fix_len(
        dataset: SupervisedDataset, n_samples: Optional[int]):

    total_size = len(dataset.data[2])
    x_test, y_test = dataset.get_test_samples(n_samples=n_samples)
    assert len(x_test) == len(y_test)
    if n_samples is None or n_samples > total_size:
        assert len(x_test) == total_size
    else:
        assert len(x_test) == n_samples


def test_sup_dataset_get_normal_train_data(dataset: SupervisedDataset):
    x_train = dataset.get_train_samples()
    y_train = dataset.data[1][x_train.index]
    assert len(x_train) > 0
    assert set(y_train.values.flatten()) == {0}


@pytest.mark.parametrize("ap", (0.01, 0.03, 0.05))
@pytest.mark.parametrize("size", (100, 1000, 5000, None))
def test_sup_dataset_get_train_data_w_anomaly(
        dataset: SupervisedDataset, ap: float, size: Optional[int]):

    x_train = dataset.get_train_samples(anomaly_percentage=ap, n_samples=size)
    y_train = dataset.data[1][x_train.index]
    assert y_train.sum() == int(ap * len(y_train))


@pytest.mark.parametrize("ap", (0.01, 0.03, 0.05))
@pytest.mark.parametrize("n_samples", (None, 100, 500, 1000, 10e9))
def test_sup_dataset_get_train_data_w_anomaly_fix_len(
        dataset: SupervisedDataset, ap: float, n_samples: Optional[int]):

    max_size = len(dataset.data[0])
    normal_size = len(dataset.data[1][dataset.data[1] == 0])
    anomaly_size = len(dataset.data[1][dataset.data[1] == 1])
    total_size = normal_size + int(ap*normal_size/(1-ap))
    x_train = dataset.get_train_samples(n_samples, ap)

    if n_samples is not None and n_samples > max_size:
        assert len(x_train) == max_size
    elif n_samples is None or n_samples > total_size:
        assert len(x_train) == total_size
    else:
        assert len(x_train) == n_samples

    y_train = dataset.data[1][x_train.index]
    y_anomaly = y_train[y_train == 1]
    if n_samples is not None and n_samples > max_size:
        assert len(y_anomaly) == anomaly_size
    else:
        assert len(y_anomaly) == int(ap * len(x_train))
