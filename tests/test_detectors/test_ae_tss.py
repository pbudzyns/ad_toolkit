import numpy as np
import pandas as pd
import pytest

from sops_anomaly.detectors import AutoEncoderTSS

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


@pytest.mark.parametrize("layers", (
    (), (100, ), (500, 200), (500, 300, 200, 100),
))
def test_build_custom_network_auto_encoder(layers):
    ae = AutoEncoderTSS(window_size=2, latent_size=50, layers=layers)
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
@pytest.mark.parametrize("latent_size", (5, 10))
def test_train_predict_ae_tss(data, window_size, latent_size):
    ae = AutoEncoderTSS(window_size=window_size, latent_size=latent_size)
    ae.train(data, epochs=2)

    p = ae.predict(data)
    assert len(p) == len(data)


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (3, 5))
def test_train_detect_ae_tss(data, window_size):
    ae = AutoEncoderTSS(window_size=window_size, latent_size=10)
    ae.train(data, epochs=2)

    p = ae.detect(data)
    assert len(p) == len(data)
    assert all(pp in (0, 1) for pp in p)
