import numpy as np
import pandas as pd
import pytest
import torch

from ad_toolkit.detectors import AutoEncoderTSS

datasets = (
    pd.DataFrame(np.random.random((10, 1))),
    pd.DataFrame(np.random.random((10, 10))),
    pd.DataFrame(np.random.random((50, 200))),
    pd.DataFrame(np.random.random((200, 5))),
)


@pytest.mark.parametrize("data", datasets)
def test_train_ae_tss(data):
    ae = AutoEncoderTSS(window_size=3, latent_size=10)
    ae.train(data, epochs=2)


def test_retrain_ae_doesnt_replace_model():
    data = pd.DataFrame(np.random.random((1000, 10)))
    ae = AutoEncoderTSS(window_size=3)
    ae.train(data)
    prev_model = ae.model
    ae.train(data)

    assert ae.model == prev_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda device')
@pytest.mark.parametrize("data", datasets)
def test_train_ae_tss_gpu(data):
    ae = AutoEncoderTSS(window_size=3, latent_size=10, use_gpu=True)
    ae.train(data, epochs=2)


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 3, 5))
@pytest.mark.parametrize("latent_size", (5, 10))
def test_train_predict_ae_tss(data, window_size, latent_size):
    ae = AutoEncoderTSS(window_size=window_size, latent_size=latent_size)
    ae.train(data, epochs=2)

    p = ae.predict(data)
    assert len(p) == len(data)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda device')
@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (3, 5))
def test_train_detect_ae_tss_gpu(data, window_size):
    ae = AutoEncoderTSS(window_size=window_size, latent_size=10, use_gpu=True)
    ae.train(data, epochs=2)

    p = ae.detect(data)
    assert len(p) == len(data)
    assert all(pp in (0, 1) for pp in p)
