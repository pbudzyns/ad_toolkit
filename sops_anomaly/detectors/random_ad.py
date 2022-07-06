import numpy as np
import pandas as pd

from sops_anomaly.detectors.base_detector import BaseDetector


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


if __name__ == '__main__':
    data = pd.DataFrame(data=np.zeros((300, 20)))
    print(data)
    model = RandomDetector()
    model.train(data)
    print(model.predict(data))
