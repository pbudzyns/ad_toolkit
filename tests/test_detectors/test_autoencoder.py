import numpy as np
import pandas as pd
import pytest

from sops_anomaly.detectors import AutoEncoder

datasets = (
    pd.DataFrame(np.random.random((10, 1))),
    pd.DataFrame(np.random.random((10, 10))),
    pd.DataFrame(np.random.random((10, 200))),
    pd.DataFrame(np.random.random((200, 5))),
)


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("use_gpu", (False, True))
def test_train_auto_encoder(data, use_gpu):
    ae = AutoEncoder(window_size=3, use_gpu=use_gpu)
    ae.train(data, epochs=2)


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
    ae = AutoEncoder(window_size=window_size)
    ae.train(data, epochs=2)

    p = ae.predict(data)
    assert len(p) == len(data)


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 3, 5))
def test_train_detect_auto_encoder(data, window_size):
    ae = AutoEncoder(window_size=window_size)
    ae.train(data, epochs=2)

    p = ae.detect(data)
    assert len(p) == len(data)
    assert all(pp in (0, 1) for pp in p)
