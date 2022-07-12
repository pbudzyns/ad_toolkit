import numpy as np
import pandas as pd
import pytest

from sops_anomaly.detectors import VariationalAutoEncoder

datasets = (
    pd.DataFrame(np.random.random((10, 1))),
    pd.DataFrame(np.random.random((10, 10))),
    pd.DataFrame(np.random.random((5, 200))),
    pd.DataFrame(np.random.random((200, 5))),
)


@pytest.mark.parametrize("data", datasets)
def test_train_vae(data):
    ae = VariationalAutoEncoder(window_size=3, latent_size=10)
    ae.train(data, epochs=2)


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 3, 5))
@pytest.mark.parametrize("latent_size", (10, 50, 100))
def test_train_predict_vae(data, window_size, latent_size):
    ae = VariationalAutoEncoder(window_size=window_size,
                                latent_size=latent_size)
    ae.train(data, epochs=2)

    p = ae.predict(data)
    assert len(p) == len(data)
    assert np.all(p >= 0) and np.all(p <= 1)


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 3, 5))
def test_train_detect_vae(data, window_size):
    ae = VariationalAutoEncoder(window_size=window_size, latent_size=10)
    ae.train(data, epochs=2)

    p = ae.detect(data)
    assert len(p) == len(data)
    assert all(pp in (0, 1) for pp in p)
