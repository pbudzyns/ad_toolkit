import numpy as np
import pandas as pd
import pytest
import torch

from ad_toolkit.detectors import VariationalAutoEncoder

datasets = (
    pd.DataFrame(np.random.random((100, 1))),
    pd.DataFrame(np.random.random((100, 10))),
    pd.DataFrame(np.random.random((100, 200))),
    pd.DataFrame(np.random.random((200, 5))),
)


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 5, 10))
@pytest.mark.parametrize("layers", ((50, 20), (50,)))
@pytest.mark.parametrize("latent_size", (10, 20, 50))
def test_train_vae(data, window_size, layers, latent_size):
    vae = VariationalAutoEncoder(window_size=window_size, layers=layers,
                                 latent_size=latent_size, use_gpu=False)
    vae.train(data, epochs=2)


def test_retrain_vae_doesnt_replace_model():
    data = pd.DataFrame(np.random.random((1000, 10)))
    vae = VariationalAutoEncoder()
    vae.train(data)
    prev_model = vae.model
    vae.train(data)

    assert vae.model == prev_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda device')
@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 5, 10))
@pytest.mark.parametrize("layers", ((50, 20), (50,)))
@pytest.mark.parametrize("latent_size", (10, 20, 50))
def test_train_vae_gpu(data, window_size, layers, latent_size):
    vae = VariationalAutoEncoder(window_size=window_size, layers=layers,
                                 latent_size=latent_size, use_gpu=True)
    vae.train(data, epochs=2)


@pytest.mark.parametrize("layers", (
    (), (100, ), (500, 200), (500, 300, 200, 100),
))
def test_build_custom_network_auto_encoder(layers):
    vae = VariationalAutoEncoder(window_size=2, latent_size=50, layers=layers)
    vae.train(pd.DataFrame([1, 2, 3, 4, 5, 6, 7]), epochs=1)
    expected_enc_sizes = (vae._input_size, *layers, vae._latent_size)
    expected_dec_sizes = tuple(reversed(expected_enc_sizes))
    encoder_layers = [vae.model.encoder[i]
                      for i
                      in range(0, len(vae.model.encoder), 2)]
    decoder_layers = [vae.model.decoder[i]
                      for i
                      in range(0, len(vae.model.decoder), 2)]
    for i, layer in enumerate(encoder_layers):
        assert layer.in_features == expected_enc_sizes[i]
        assert layer.out_features == expected_enc_sizes[i+1]

    for i, layer in enumerate(decoder_layers):
        assert layer.in_features == expected_dec_sizes[i]
        assert layer.out_features == expected_dec_sizes[i+1]


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 3, 5))
@pytest.mark.parametrize("latent_size", (10, 50, 100))
def test_train_predict_vae(data, window_size, latent_size):
    vae = VariationalAutoEncoder(window_size=window_size, use_gpu=False,
                                 latent_size=latent_size)
    vae.train(data, epochs=2)

    p = vae.predict(data)
    assert len(p) == len(data)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda device')
@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 3, 5))
@pytest.mark.parametrize("latent_size", (10, 50, 100))
def test_train_predict_vae_gpu(data, window_size, latent_size):
    vae = VariationalAutoEncoder(window_size=window_size, use_gpu=True,
                                 latent_size=latent_size)
    vae.train(data, epochs=2)

    p = vae.predict(data)
    assert len(p) == len(data)


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 3, 5))
def test_train_predict_raw_errors_vae(data, window_size):
    vae = VariationalAutoEncoder(window_size=window_size)
    vae.train(data, epochs=2)

    p = vae.predict(data, raw_errors=True)
    assert p.shape == data.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda device')
@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 3, 5))
def test_train_predict_raw_errors_vae_gpu(data, window_size):
    vae = VariationalAutoEncoder(window_size=window_size, use_gpu=True)
    vae.train(data, epochs=2)

    p = vae.predict(data, raw_errors=True)
    assert p.shape == data.shape
