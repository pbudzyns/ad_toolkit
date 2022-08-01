import numpy as np
import pandas as pd
import pytest
import torch

from ad_toolkit.detectors import LSTM_ED

datasets = (
    pd.DataFrame(np.random.random((100, 1))),
    pd.DataFrame(np.random.random((100, 10))),
    pd.DataFrame(np.random.random((200, 5))),
)


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("hidden_size", (32, 16))
@pytest.mark.parametrize("sequence_len", (20, 10, 5))
@pytest.mark.parametrize("stride", (1, 5, 10))
def test_train_lstm_ed(data, hidden_size, sequence_len, stride):
    lstm = LSTM_ED(sequence_len=sequence_len, stride=stride,
                   hidden_size=hidden_size, use_gpu=False)
    lstm.train(data, epochs=2)


@pytest.mark.skipif(not torch.cuda.is_available, reason='no cuda device')
@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("hidden_size", (32, 16))
@pytest.mark.parametrize("sequence_len", (20, 10, 5))
@pytest.mark.parametrize("stride", (1, 5, 10))
def test_train_lstm_ed_gpu(data, hidden_size, sequence_len, stride):
    lstm = LSTM_ED(sequence_len=sequence_len, stride=stride,
                   hidden_size=hidden_size, use_gpu=True)
    lstm.train(data, epochs=2)


@pytest.mark.parametrize("data", datasets)
def test_train_lstm_ed_w_validation(data):
    lstm = LSTM_ED(hidden_size=200)
    validation = (
        pd.DataFrame(np.random.random(data.shape)),
        pd.Series((np.random.random((len(data))) > 0.5).astype(np.int32)),
    )
    lstm.train(data, epochs=2, validation_data=validation)


@pytest.mark.skipif(not torch.cuda.is_available, reason='no cuda device')
@pytest.mark.parametrize("data", datasets)
def test_train_lstm_ed_w_validation_gpu(data):
    lstm = LSTM_ED(hidden_size=200, use_gpu=True)
    validation = (
        pd.DataFrame(np.random.random(data.shape)),
        pd.Series((np.random.random((len(data))) > 0.5).astype(np.int32)),
    )
    lstm.train(data, epochs=2, validation_data=validation)


@pytest.mark.parametrize("data", datasets)
def test_train_predict_lstm_ed(data):
    lstm = LSTM_ED(use_gpu=False, stride=5)
    lstm.train(data, epochs=2)

    p = lstm.predict(data)
    assert len(p) == len(data)


@pytest.mark.skipif(not torch.cuda.is_available, reason='no cuda device')
@pytest.mark.parametrize("data", datasets)
def test_train_predict_lstm_ed_gpu(data):
    lstm = LSTM_ED(use_gpu=True, stride=5)
    lstm.train(data, epochs=2)

    p = lstm.predict(data)
    assert len(p) == len(data)


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("sequence_len", (20, 10, 5))
def test_train_predict_raw_errors_lstm_ed(data, sequence_len):
    lstm = LSTM_ED(sequence_len=sequence_len, use_gpu=False)
    lstm.train(data, epochs=2)

    p = lstm.predict(data, raw_errors=True)
    assert p.shape == data.shape


@pytest.mark.skipif(not torch.cuda.is_available, reason='no cuda device')
@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("sequence_len", (20, 10, 5))
def test_train_predict_raw_errors_lstm_ed_gpu(data, sequence_len):
    lstm = LSTM_ED(sequence_len=sequence_len, use_gpu=True)
    lstm.train(data, epochs=2)

    p = lstm.predict(data, raw_errors=True)
    assert p.shape == data.shape


@pytest.mark.parametrize("data", datasets)
def test_train_detect_lstm_ed(data):
    lstm = LSTM_ED(use_gpu=False)
    lstm.train(data, epochs=2)

    p = lstm.detect(data)
    assert len(p) == len(data)
    assert all(pp in (0, 1) for pp in p)


@pytest.mark.skipif(not torch.cuda.is_available, reason='no cuda device')
@pytest.mark.parametrize("data", datasets)
def test_train_detect_lstm_ed_gpu(data):
    lstm = LSTM_ED(use_gpu=True)
    lstm.train(data, epochs=2)

    p = lstm.detect(data)
    assert len(p) == len(data)
    assert all(pp in (0, 1) for pp in p)
