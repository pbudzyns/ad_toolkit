"""
LSTM with dynamic non-parametric thresholding.

"""
import numpy as np
import pandas as pd

from sops_anomaly.detectors.base_detector import BaseDetector


class LSTM(BaseDetector):

    def __init__(self):
        super(LSTM, self).__init__()

    def train(self, train_data: pd.DataFrame, epochs: int):
        pass

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        pass

    def detect(self, data: pd.DataFrame) -> np.ndarray:
        pass
