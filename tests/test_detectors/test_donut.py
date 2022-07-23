import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from sops_anomaly.detectors import Donut


@pytest.yield_fixture(autouse=True, scope='function')
def isolate_tf_session():
    with tf.Graph().as_default():
        yield


datasets = (
    pd.DataFrame(np.random.random((500, 1))),
    pd.DataFrame(np.random.random((900, 1))),
)


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("layers", ((300, 200), (100, 50), (100, 100)))
def test_train_donut(data, layers):
    donut = Donut(layers=layers)
    donut.train(data, epochs=3)


@pytest.mark.parametrize("data", datasets)
def test_train_predict_donut(data):
    donut = Donut()
    donut.train(data, epochs=3)
    p = donut.predict(data)
    assert len(p) == len(data)


@pytest.mark.skip
@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 3))
def test_train_detect_donut(data, window_size):
    ae = Donut(window_size=window_size)
    ae.train(data, epochs=3)
    p = ae.detect(data)
    assert len(p) == len(data)
    assert all(pp in (0, 1) for pp in p)
