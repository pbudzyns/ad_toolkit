import pandas as pd
import pytest

from ad_toolkit.datasets import MNIST


@pytest.fixture(scope="session")
def mnist():
    mnist = MNIST()
    mnist.load()
    return mnist


def test_mnist_loads_data(mnist: MNIST):
    data = mnist.data
    assert data is not None


def test_mnist_data_returns_dataframes(mnist: MNIST):
    x_train, y_train, x_test, y_test = mnist.data
    assert isinstance(x_train, pd.DataFrame)
    assert isinstance(x_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)


@pytest.mark.parametrize("cls", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
def test_mnist_anomaly_classes(cls):
    mnist = MNIST(anomaly_class=cls)
    x_train, y_train, x_test, y_test = mnist.data
    assert len(x_train) > 0 and len(y_train) > 0
    assert len(x_test) > 0 and len(y_test) > 0
    assert set(y_test.values.flatten()) == {0, 1}
    assert set(y_train.values.flatten()) == {0, 1}


@pytest.mark.parametrize("n_samples", (None, 10, 100, 1000, 10e9))
def test_mnist_get_train_samples(mnist: MNIST, n_samples):
    total_samples = len(mnist.data[0])
    x_train = mnist.get_train_samples(n_samples=n_samples)
    if n_samples is None or n_samples > total_samples:
        assert len(x_train) == total_samples
    else:
        assert len(x_train) == n_samples


@pytest.mark.parametrize("n_samples", (None, 10, 100, 1000, 10e9))
def test_mnist_get_test_samples(mnist: MNIST, n_samples):
    total_samples = len(mnist.data[2])
    x_test, y_test = mnist.get_test_samples(n_samples=n_samples)
    assert len(x_test) == len(y_test)
    if n_samples is None or n_samples > total_samples:
        assert len(x_test) == total_samples
    else:
        assert len(x_test) == n_samples
