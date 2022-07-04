from typing import List, Union

import numpy as np
from nab.detectors.base import AnomalyDetector

from sops_anomaly.models.model import BaseDetector


class RandomDetector(BaseDetector):

    def detect(self, data: np.ndarray) -> Union[List[int], np.ndarray]:
        return self.predict(data)

    def train(self, train_data: np.ndarray, epochs: int = 0):
        pass

    def predict(self, data: np.ndarray):
        return np.random.random(data.shape)


class RandomDetectorNab(RandomDetector, AnomalyDetector):

    def initialize(self):
        pass

    def handleRecord(self, inputData):
        return [self.predict(inputData)]
