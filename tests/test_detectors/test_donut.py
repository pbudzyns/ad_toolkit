import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from ad_toolkit.detectors import Donut


@pytest.fixture(autouse=True, scope='function')
def isolate_tf_session():
    with tf.Graph().as_default():
        yield


@pytest.fixture
def use_gpu():
    with tf.device('/gpu:0'):
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


@pytest.mark.skipif(not tf.test.is_gpu_available(), reason='no cuda device')
@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("layers", ((300, 200), (100, 50), (100, 100)))
def test_train_donut_gpu(data, layers, use_gpu):
    donut = Donut(layers=layers)
    donut.train(data, epochs=3)


@pytest.mark.parametrize("data", datasets)
def test_train_predict_donut(data):
    donut = Donut()
    donut.train(data, epochs=3)
    p = donut.predict(data)
    assert len(p) == len(data)


@pytest.mark.skipif(not tf.test.is_gpu_available(), reason='no cuda device')
@pytest.mark.parametrize("data", datasets)
def test_train_predict_donut_gpu(data, use_gpu):
    donut = Donut()
    donut.train(data, epochs=3)
    p = donut.predict(data)
    assert len(p) == len(data)


@pytest.mark.skip('not implemented')
@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 3))
def test_train_detect_donut(data, window_size):
    ae = Donut(window_size=window_size)
    ae.train(data, epochs=3)
    p = ae.detect(data)
    assert len(p) == len(data)
    assert all(pp in (0, 1) for pp in p)
