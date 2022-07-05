from typing import List, Union

import numpy as np
import pandas as pd
# from nab.detectors.base import AnomalyDetector

from sops_anomaly.models.base_model import BaseDetector


class RandomDetector(BaseDetector):

    def detect(self, data: pd.DataFrame) -> np.ndarray:
        return self.predict(data)

    def train(self, train_data: pd.DataFrame) -> None:
        pass

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return np.random.random(len(data))


# class RandomDetectorNab(RandomDetector, AnomalyDetector):
#
#     def initialize(self):
#         pass
#
#     def handleRecord(self, inputData):
#         return [self.predict(inputData)]


if __name__ == '__main__':
    data = pd.DataFrame(data=np.zeros((300, 20)))
    print(data)
    model = RandomDetector()
    model.train(data)
    print(model.predict(data))
