import numpy as np
import pandas as pd
import pytest

from sops_anomaly.detectors import Donut

datasets = (
    pd.DataFrame(np.random.random((500, 1))),
    pd.DataFrame(np.random.random((500, 10))),
)


@pytest.mark.skip
@pytest.mark.parametrize("data", datasets)
def test_train_donut(data):
    donut = Donut()
    donut.train(data, epochs=3)


@pytest.mark.skip
@pytest.mark.parametrize("data", datasets)
def test_train_predict_donut(data):
    donut = Donut()
    donut.train(data, epochs=3)

    p = donut.predict(data)
    assert len(p) == len(data)
    assert np.all(p >= 0) and np.all(p <= 1)


@pytest.mark.skip
@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 3))
def test_train_detect_donut(data, window_size):
    ae = Donut(x_dim=window_size)
    ae.train(data, epochs=3)

    p = ae.detect(data)
    assert len(p) == len(data)
    assert all(pp in (0, 1) for pp in p)
