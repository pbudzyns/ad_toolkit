import numpy as np
import pandas as pd
import pytest

from sops_anomaly.detectors import LSTM_ED

datasets = (
    pd.DataFrame(np.random.random((100, 1))),
    pd.DataFrame(np.random.random((100, 10))),
    pd.DataFrame(np.random.random((200, 5))),
)


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("hidden_size", (32, 16))
@pytest.mark.parametrize("sequence_len", (20, 10, 5))
@pytest.mark.parametrize("stride", (1, 5, 10))
@pytest.mark.parametrize("use_gpu", (True, False))
def test_train_lstm_ed(data, hidden_size, sequence_len, stride, use_gpu):
    lstm = LSTM_ED(sequence_len=sequence_len, stride=stride,
                   hidden_size=hidden_size, use_gpu=use_gpu)
    lstm.train(data, epochs=2)


@pytest.mark.parametrize("data", datasets)
def test_train_lstm_ed_w_validation(data):
    lstm = LSTM_ED(hidden_size=200)
    validation = (
        pd.DataFrame(np.random.random(data.shape)),
        pd.Series((np.random.random((len(data))) > 0.5).astype(np.int32)),
    )
    lstm.train(data, epochs=2, validation_data=validation)


@pytest.mark.parametrize("data", datasets)
@pytest.mark.parametrize("use_gpu", (True, False))
def test_train_predict_lstm_ed(data, use_gpu):
    lstm = LSTM_ED(use_gpu=use_gpu, stride=5)
    lstm.train(data, epochs=2)

    p = lstm.predict(data)
    assert len(p) == len(data)
    # assert np.all(p >= 0) and np.all(p <= 1)


# @pytest.mark.skip("Not implemented yet")
@pytest.mark.parametrize("data", datasets)
def test_train_detect_lstm_ed(data):
    lstm = LSTM_ED()
    lstm.train(data, epochs=2)

    p = lstm.detect(data)
    assert len(p) == len(data)
    assert all(pp in (0, 1) for pp in p)
