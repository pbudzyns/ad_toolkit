import numpy as np
import pandas as pd
import pytest

# Donut model requires extra dependencies that do not work with
# recent python versions. To enable testing on newer versions
# of python conditional import is used.
donut_ad = pytest.importorskip('ad_toolkit.detectors.donut_ad')
Donut = donut_ad.Donut

import tensorflow as tf  # noqa: E402


@pytest.fixture(autouse=True, scope='function')
def isolate_tf_session():
    with tf.Graph().as_default():
        yield


@pytest.fixture
def use_gpu():
    with tf.device('/gpu:0'):
        yield


datasets = (
    pd.DataFrame(np.random.random((1500, 1))),
    pd.DataFrame(np.random.random((5000, 1))),
)


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("include_labels", (False, True))
@pytest.mark.parametrize("layers", ((300, 200), (100, 50), (100, 100)))
@pytest.mark.parametrize("valid_portion", (0.1, 0.2))
def test_train_donut(data, include_labels, layers, valid_portion):
    donut = Donut(layers=layers)
    labels = np.random.randint(2, size=len(data)) if include_labels else None
    donut.train(data, labels=labels, epochs=3, valid_portion=valid_portion)


@pytest.mark.skipif(not tf.test.is_gpu_available(), reason='no cuda device')
@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("include_labels", (False, True))
@pytest.mark.parametrize("layers", ((300, 200), (100, 50), (100, 100)))
@pytest.mark.parametrize("valid_portion", (0.1, 0.2))
def test_train_donut_gpu(data, include_labels, layers, valid_portion, use_gpu):
    donut = Donut(layers=layers)
    labels = np.random.randint(2, size=len(data)) if include_labels else None
    donut.train(data, labels=labels, epochs=3, valid_portion=valid_portion)


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
