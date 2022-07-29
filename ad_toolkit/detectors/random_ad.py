import numpy as np
import pandas as pd

from ad_toolkit.detectors.base_detector import BaseDetector


class RandomDetector(BaseDetector):

    def __init__(self) -> None:
        """Dummy anomaly detector that returns random results.
        """
        super(RandomDetector, self).__init__()

    def train(self, train_data: pd.DataFrame) -> None:
        pass

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return np.random.random(len(data))

    def detect(self, data: pd.DataFrame) -> np.ndarray:
        return (self.predict(data) > 0.5).astype(np.int32)
