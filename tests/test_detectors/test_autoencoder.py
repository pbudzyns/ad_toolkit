import numpy as np
import pandas as pd
import pytest
import torch

from ad_toolkit.detectors import AutoEncoder

datasets = (
    pd.DataFrame(np.random.random((100, 1))),
    pd.DataFrame(np.random.random((100, 10))),
    pd.DataFrame(np.random.random((100, 200))),
    pd.DataFrame(np.random.random((200, 5))),
)


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 3, 5, 10))
@pytest.mark.parametrize("latent_size", (10, 20, 50))
@pytest.mark.parametrize("layers", ((50, 20), (50,),))
def test_train_auto_encoder(data, window_size, layers, latent_size):
    ae = AutoEncoder(window_size=window_size, layers=layers,
                     latent_size=latent_size, use_gpu=False)
    ae.train(data, epochs=2, verbose=False)


def test_retrain_ae_doesnt_replace_model():
    data = pd.DataFrame(np.random.random((1000, 10)))
    ae = AutoEncoder()
    ae.train(data)
    prev_model = ae.model
    ae.train(data)

    assert ae.model == prev_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda device')
@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 3, 5, 10))
@pytest.mark.parametrize("latent_size", (10, 20, 50))
@pytest.mark.parametrize("layers", ((50, 20), (50,),))
def test_train_auto_encoder_gpu(data, window_size, layers, latent_size):
    ae = AutoEncoder(window_size=window_size, layers=layers,
                     latent_size=latent_size, use_gpu=True)
    ae.train(data, epochs=2, verbose=False)


@pytest.mark.parametrize("layers", (
    (), (100, ), (500, 200), (500, 300, 200, 100),
))
def test_build_custom_network_auto_encoder(layers):
    ae = AutoEncoder(window_size=2, latent_size=50, layers=layers)
    ae.train(pd.DataFrame([1, 2, 3, 4, 5, 6, 7]), epochs=1)
    expected_enc_sizes = (ae._input_size, *layers, ae._latent_size)
    expected_dec_sizes = tuple(reversed(expected_enc_sizes))
    encoder_layers = [ae.model.encoder[i]
                      for i
                      in range(0, len(ae.model.encoder), 2)]
    decoder_layers = [ae.model.decoder[i]
                      for i
                      in range(0, len(ae.model.decoder), 2)]
    for i, layer in enumerate(encoder_layers):
        assert layer.in_features == expected_enc_sizes[i]
        assert layer.out_features == expected_enc_sizes[i+1]

    for i, layer in enumerate(decoder_layers):
        assert layer.in_features == expected_dec_sizes[i]
        assert layer.out_features == expected_dec_sizes[i+1]


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 3, 5))
def test_train_predict_auto_encoder(data, window_size):
    ae = AutoEncoder(window_size=window_size, use_gpu=False)
    ae.train(data, epochs=2)

    p = ae.predict(data)
    assert len(p) == len(data)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda device')
@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 3, 5))
def test_train_predict_auto_encoder_gpu(data, window_size):
    ae = AutoEncoder(window_size=window_size, use_gpu=True)
    ae.train(data, epochs=2)

    p = ae.predict(data)
    assert len(p) == len(data)


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 3, 5))
def test_train_predict_raw_errors_auto_encoder(data, window_size):
    ae = AutoEncoder(window_size=window_size)
    ae.train(data, epochs=2)

    p = ae.predict(data, raw_errors=True)
    assert p.shape == data.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda device')
@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 3, 5))
def test_train_predict_raw_errors_auto_encoder_gpu(data, window_size):
    ae = AutoEncoder(window_size=window_size)
    ae.train(data, epochs=2)

    p = ae.predict(data, raw_errors=True)
    assert p.shape == data.shape
