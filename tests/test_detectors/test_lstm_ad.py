import numpy as np
import pandas as pd
import pytest

from ad_toolkit.detectors import LSTM_AD

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


@pytest.mark.parametrize("window_size", (5, 10))
@pytest.mark.parametrize("hidden_size", (10, 20))
@pytest.mark.parametrize("use_gpu", (True, False))
def test_train_lstm_with_slices(window_size, hidden_size, use_gpu):
    data = pd.DataFrame(np.random.random((5000, 5)))
    lstm = LSTM_AD(window_size=window_size, hidden_size=hidden_size,
                   use_gpu=use_gpu)
    lstm.train_with_slices(data, slice_len=100, epochs=3)
    scores = lstm.predict(data)
    assert len(scores) == len(data)


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


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 5, 10))
def test_train_predict_raw_errors_lstm(data, window_size):
    lstm = LSTM_AD(window_size=window_size)
    lstm.train(data, epochs=2)

    p = lstm.predict(data, raw_errors=True)
    assert p.shape == data.shape


@pytest.mark.parametrize("data", datasets)
def test_train_detect_lstm(data):
    lstm = LSTM_AD()
    lstm.train(data, epochs=2)

    p = lstm.detect(data)
    assert len(p) == len(data)
    assert all(pp in (0, 1) for pp in p)
