import numpy as np
import pandas as pd
import pytest

from sops_anomaly.detectors import AutoEncoder

datasets = (
    pd.DataFrame(np.random.random((10, 1))),
    pd.DataFrame(np.random.random((10, 10))),
    pd.DataFrame(np.random.random((5, 200))),
    pd.DataFrame(np.random.random((200, 5))),
)


@pytest.mark.parametrize("data", datasets)
def test_train_auto_encoder(data):
    ae = AutoEncoder(window_size=3)
    ae.train(data, epochs=2)


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 3, 5))
def test_train_predict_auto_encoder(data, window_size):
    ae = AutoEncoder(window_size=window_size)
    ae.train(data, epochs=2)

    p = ae.predict(data)
    assert len(p) == len(data)
    assert np.all(p >= 0) and np.all(p <= 1)


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 3, 5))
def test_train_detect_auto_encoder(data, window_size):
    ae = AutoEncoder(window_size=window_size)
    ae.train(data, epochs=2)

    p = ae.detect(data)
    assert len(p) == len(data)
    assert all(pp in (0, 1) for pp in p)
