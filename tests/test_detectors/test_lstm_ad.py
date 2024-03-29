import numpy as np
import pandas as pd
import pytest
import torch

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
def test_train_lstm(data, window_size, hidden_size):
    lstm = LSTM_AD(window_size=window_size, hidden_size=hidden_size,
                   use_gpu=False)
    lstm.train(data, epochs=2)


def test_retrain_lstm_doesnt_replace_model():
    data = pd.DataFrame(np.random.random((100, 1)))
    lstm = LSTM_AD()
    lstm.train(data)
    prev_model = lstm.model
    lstm.train(data)

    assert lstm.model == prev_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda device')
@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 5, 10))
@pytest.mark.parametrize("hidden_size", (10, 20, 30))
def test_train_lstm_gpu(data, window_size, hidden_size):
    lstm = LSTM_AD(window_size=window_size, hidden_size=hidden_size,
                   use_gpu=True)
    lstm.train(data, epochs=2)


@pytest.mark.parametrize("window_size", (5, 10))
@pytest.mark.parametrize("hidden_size", (10, 20))
def test_train_lstm_with_slices(window_size, hidden_size):
    data = pd.DataFrame(np.random.random((5000, 5)))
    lstm = LSTM_AD(window_size=window_size, hidden_size=hidden_size,
                   use_gpu=False)
    lstm.train_with_slices(data, slice_len=100, epochs=3)
    scores = lstm.predict(data)
    assert len(scores) == len(data)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda device')
@pytest.mark.parametrize("window_size", (5, 10))
@pytest.mark.parametrize("hidden_size", (10, 20))
def test_train_lstm_with_slices_gpu(window_size, hidden_size):
    data = pd.DataFrame(np.random.random((5000, 5)))
    lstm = LSTM_AD(window_size=window_size, hidden_size=hidden_size,
                   use_gpu=True)
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
def test_train_predict_lstm(data):
    lstm = LSTM_AD(use_gpu=False)
    lstm.train(data, epochs=2)

    p = lstm.predict(data)
    assert len(p) == len(data)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda device')
@pytest.mark.parametrize("data", datasets)
def test_train_predict_lstm_gpu(data):
    lstm = LSTM_AD(use_gpu=True)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda device')
@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("window_size", (1, 5, 10))
def test_train_predict_raw_errors_lstm_gpu(data, window_size):
    lstm = LSTM_AD(window_size=window_size, use_gpu=True)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda device')
@pytest.mark.parametrize("data", datasets)
def test_train_detect_lstm_gpu(data):
    lstm = LSTM_AD(use_gpu=True)
    lstm.train(data, epochs=2)

    p = lstm.detect(data)
    assert len(p) == len(data)
    assert all(pp in (0, 1) for pp in p)
