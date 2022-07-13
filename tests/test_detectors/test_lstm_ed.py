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
def test_train_lstm_ed(data):
    lstm = LSTM_ED(hidden_size=200)
    lstm.train(data, epochs=2)


@pytest.mark.skip("Not implemented yet")
@pytest.mark.parametrize("data", datasets)
def test_train_predict_lstm_ed(data):
    lstm = LSTM_ED()
    lstm.train(data, epochs=2)

    p = lstm.predict(data)
    assert len(p) == len(data)
    # assert np.all(p >= 0) and np.all(p <= 1)


@pytest.mark.skip("Not implemented yet")
@pytest.mark.parametrize("data", datasets)
def test_train_detect_lstm_ed(data):
    lstm = LSTM_ED()
    lstm.train(data, epochs=2)

    p = lstm.detect(data)
    assert len(p) == len(data)
    assert all(pp in (0, 1) for pp in p)
