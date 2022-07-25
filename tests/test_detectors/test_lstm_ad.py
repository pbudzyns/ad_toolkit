import numpy as np
import pandas as pd
import pytest

from sops_anomaly.detectors import LSTM_AD

datasets = (
    pd.DataFrame(np.random.random((100, 1))),
    pd.DataFrame(np.random.random((100, 10))),
    pd.DataFrame(np.random.random((200, 5))),
    pd.DataFrame(np.random.random((1000, 50)))
)


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 5, 10))
@pytest.mark.parametrize("hidden_size", (10, 20, 30))
@pytest.mark.parametrize("use_gpu", (True, False))
def test_train_lstm(data, window_size, hidden_size, use_gpu):
    lstm = LSTM_AD(window_size=window_size, hidden_size=hidden_size,
                   use_gpu=use_gpu)
    lstm.train(data, epochs=2)


@pytest.mark.parametrize("data", datasets)
def test_train_lstm_w_validation(data):
    lstm = LSTM_AD()
    validation = (
        pd.DataFrame(np.random.random(data.shape)),
        pd.Series((np.random.random((len(data))) > 0.5).astype(np.int32)),
    )
    lstm.train(data, epochs=2, validation_data=validation)


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("use_gpu", (True, False))
def test_train_predict_lstm(data, use_gpu):
    lstm = LSTM_AD(use_gpu=use_gpu)
    lstm.train(data, epochs=2)

    p = lstm.predict(data)
    assert len(p) == len(data)
    # assert np.all(p >= 0) and np.all(p <= 1)


@pytest.mark.parametrize("data", datasets)
def test_train_detect_lstm(data):
    lstm = LSTM_AD()
    lstm.train(data, epochs=2)

    p = lstm.detect(data)
    assert len(p) == len(data)
    assert all(pp in (0, 1) for pp in p)
