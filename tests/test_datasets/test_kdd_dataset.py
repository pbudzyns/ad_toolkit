import pandas as pd
import pytest

from ad_toolkit.datasets import KddCup


@pytest.fixture(scope="session")
def kdd() -> KddCup:
    kdd = KddCup()
    kdd.load()
    return kdd


def test_kdd_loads_data(kdd: KddCup):
    assert kdd.data is not None


def test_kdd_data_returns_dataframes(kdd: KddCup):
    x_train, y_train, x_test, y_test = kdd.data
    assert isinstance(x_train, pd.DataFrame)
    assert isinstance(x_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)


def test_kdd_returns_labeled_dataset(kdd: KddCup):
    _, y_train, _, y_test = kdd.data
    assert set(y_train) == {0, 1}
    assert set(y_test) == {0, 1}


@pytest.mark.parametrize("n_samples", (None, 10, 100, 1000, 10e9))
def test_mnist_get_train_samples(kdd: KddCup, n_samples):
    total_samples = len(kdd.data[0])
    x_train = kdd.get_train_samples(n_samples=n_samples)
    if n_samples is None or n_samples > total_samples:
        assert len(x_train) == total_samples
    else:
        assert len(x_train) == n_samples


@pytest.mark.parametrize("n_samples", (None, 10, 100, 1000, 10e9))
def test_mnist_get_test_samples(kdd: KddCup, n_samples):
    total_samples = len(kdd.data[2])
    x_test, y_test = kdd.get_test_samples(n_samples=n_samples)
    assert len(x_test) == len(y_test)
    if n_samples is None or n_samples > total_samples:
        assert len(x_test) == total_samples
    else:
        assert len(x_test) == n_samples
