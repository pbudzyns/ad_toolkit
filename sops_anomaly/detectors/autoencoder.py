from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from sops_anomaly.detectors.base_detector import BaseDetector
from sops_anomaly.utils import window_data


class _AEModel(nn.Module):
    def __init__(self, input_size: int, latent_size: int) -> None:
        super().__init__()
        self.encoder: nn.Module = self._get_encoder(input_size, latent_size)
        self.decoder: nn.Module = self._get_decoder(latent_size, input_size)

    @classmethod
    def _get_encoder(cls, input_size: int, output_size: int) -> nn.Module:
        encoder = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, output_size)
        )
        return encoder

    @classmethod
    def _get_decoder(cls, input_size: int, output_size: int) -> nn.Module:
        decoder = nn.Sequential(
            nn.Linear(input_size, 200),
            nn.ReLU(),
            nn.Linear(200, 500),
            nn.ReLU(),
            nn.Linear(500, output_size)
        )
        return decoder

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoder(BaseDetector):

    def __init__(
        self,
        window_size: int,
        latent_size: int = 100,
        threshold: float = 0.8,
    ) -> None:
        self.model: Optional[nn.Module] = None
        self._window_size: int = window_size
        self._latent_size: int = latent_size
        self._threshold: float = threshold
        self._max_error: float = 0.0

    def _transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return window_data(data, self._window_size)

    @classmethod
    def _data_to_tensors(cls, data: pd.DataFrame) -> List[torch.Tensor]:
        tensors = [torch.Tensor(row) for _, row in data.iterrows()]
        return tensors

    def _compute_threshold(self, data: List[torch.Tensor]) -> float:
        scores = []
        self.model.eval()
        with torch.no_grad():
            for sample in data:
                rec = self.model.forward(sample)
                scores.append(F.mse_loss(rec, sample).item())
        scores = np.array(scores)
        return np.max(scores)

    def train(self, train_data: pd.DataFrame, epochs: int = 20) -> None:
        if self._window_size > 1:
            train_data = self._transform_data(train_data)
        input_size = len(train_data.iloc[0])
        train_data = self._data_to_tensors(train_data)

        self.model = _AEModel(input_size=input_size,
                              latent_size=self._latent_size)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        for epoch in range(epochs):
            epoch_loss = 0
            for sample in train_data:
                optimizer.zero_grad()
                reconstructed = self.model.forward(sample)
                loss = F.mse_loss(reconstructed, sample)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch} loss: {epoch_loss/len(train_data)}")

        self._max_error = self._compute_threshold(train_data)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self._window_size > 1:
            input_data = self._transform_data(data)
            input_data = self._data_to_tensors(input_data)
        else:
            input_data = self._data_to_tensors(data)

        # Zero padding to match input length.
        scores = [0] * (self._window_size - 1)
        self.model.eval()
        with torch.no_grad():
            for sample in input_data:
                rec = self.model.forward(sample)
                scores.append(F.mse_loss(rec, sample).item())
        return np.array(scores)

    def detect(self, data: pd.DataFrame) -> np.ndarray:
        scores = self.predict(data)
        return (scores >= self._threshold * self._max_error).astype(np.int32)


if __name__ == '__main__':
    # data1 = pd.DataFrame(data=[
    #     np.arange(0, 10), np.arange(10, 20), np.arange(20, 30),
    # ])
    #
    # data2 = pd.DataFrame(data=np.arange(30).reshape((30,1)))
    # print(data1)
    # print(data2)
    # print(np.array(AutoEncoder(window_size=1)._transform_data(data1)))
    # print(AutoEncoder(window_size=5)._transform_data(data2))

    from sops_anomaly.datasets import MNIST
    mnist = MNIST()
    x = mnist.get_train_samples(n_samples=100)
    test_x, test_y = mnist.get_test_samples(n_samples=50)
    print(test_x, test_y)

    ae = AutoEncoder(window_size=1)
    ae.train(x, epochs=10)

    pred_y = ae.detect(test_x)
    print(pred_y, test_y)
    from sops_anomaly.evaluation import Result
    res = Result(np.array(pred_y), np.array(test_y))
    print(res.accuracy, res.f1)

