import numpy as np
import pandas as pd
import pytest

from ad_toolkit.detectors import RandomDetector

data = pd.DataFrame(data=np.random.random((10, 10)))


@pytest.fixture(scope="session")
def random_detector() -> RandomDetector:
    return RandomDetector()


def test_detector_train(random_detector: RandomDetector):
    random_detector.train(data)


def test_detector_predict(random_detector: RandomDetector):
    p = random_detector.predict(data)
    assert len(p) == len(data)
    assert np.all(0 <= p) and np.all(p <= 1)


def test_detector_detect(random_detector: RandomDetector):
    p = random_detector.detect(data)
    assert len(p) == len(data)
    assert set(p) == {0, 1}
