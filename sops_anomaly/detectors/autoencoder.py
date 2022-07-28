"""
Auto-encoder anomaly detector.

References:
    - "Variational auto-encoder based anomaly detection using reconstruction
     probability" J.An, S.Cho.

"""
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

from sops_anomaly.detectors.base_detector import BaseDetector
from sops_anomaly.utils.torch_utils import build_layers, build_network
from sops_anomaly.utils import window_data


class _AEModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        layers: Union[List[int], Tuple[int]],
        latent_size: int,
    ) -> None:

        super().__init__()
        self.encoder: nn.Module = self._get_encoder(
            input_size, layers, latent_size)
        self.decoder: nn.Module = self._get_decoder(
            latent_size, list(reversed(layers)), input_size)

    @classmethod
    def _get_encoder(
        cls,
        input_size: int,
        layers: Union[List[int], Tuple[int]],
        output_size: int,
    ) -> nn.Module:
        nn_layers = build_layers(input_size, layers, output_size)
        encoder = build_network(nn_layers)
        return encoder

    @classmethod
    def _get_decoder(
        cls,
        input_size: int,
        layers: Union[List[int], Tuple[int]],
        output_size: int,
    ) -> nn.Module:
        nn_layers = build_layers(input_size, layers, output_size)
        decoder = build_network(nn_layers)
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
        layers: Union[List[int], Tuple[int]] = (500, 200),
        threshold: float = 0.8,
        use_gpu: bool = False,
    ) -> None:
        """

        :param window_size:
        :param latent_size:
        :param layers:
        :param threshold:
        """
        self.model: Optional[nn.Module] = None
        self._layers: Union[List[int], Tuple[int]] = layers
        self._window_size: int = window_size
        self._input_size: Optional[int] = None
        self._latent_size: int = latent_size
        self._threshold: float = threshold
        self._device: torch.device = torch.device(
            'cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.max_error: float = 0.0

    def train(
        self,
        train_data: pd.DataFrame,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        verbose: bool = False,
    ) -> None:
        if self._window_size > 1:
            all_data = self._transform_data(train_data)
        else:
            all_data = train_data

        if self._input_size is None:
            self._input_size = len(all_data.iloc[0])

        all_data_tensors = self._data_to_tensors(all_data)

        split = int(0.8 * len(all_data_tensors))
        train_data = all_data_tensors[:split]
        eval_data = all_data_tensors[split:]

        if self.model is None:
            self._init_model_if_needed()
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        indices = np.random.permutation(len(train_data))
        data_loader = DataLoader(
            dataset=train_data,
            batch_size=min(len(train_data), batch_size),
            drop_last=True,
            sampler=SubsetRandomSampler(indices),
            # pin_memory=True,
        )

        for epoch in range(epochs):
            epoch_loss = 0
            for batch in data_loader:
                optimizer.zero_grad()
                reconstructed = self.model.forward(batch)
                loss = F.mse_loss(reconstructed, batch)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            if verbose:
                print(f"Epoch {epoch} loss: {epoch_loss/len(data_loader)}")

        self.max_error = self._compute_threshold(eval_data)

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

    def detect(
        self, data: pd.DataFrame, threshold: Optional[float] = None,
    ) -> np.ndarray:

        if threshold is None:
            threshold = self._threshold
        scores = self.predict(data)
        return (scores >= threshold * self.max_error).astype(np.int32)

    def _transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return window_data(data, self._window_size)

    def _data_to_tensors(self, data: pd.DataFrame) -> List[torch.Tensor]:
        tensors = [torch.Tensor(row).to(self._device)
                   for _, row
                   in data.iterrows()]
        return tensors

    def _init_model_if_needed(self) -> None:
        if self.model is not None:
            return
        self.model = _AEModel(
            input_size=self._input_size,
            layers=self._layers,
            latent_size=self._latent_size,
        ).to(self._device)

    def _compute_threshold(self, data: List[torch.Tensor]) -> float:
        scores = []
        self.model.eval()
        with torch.no_grad():
            for sample in data:
                rec = self.model.forward(sample)
                scores.append(F.mse_loss(rec, sample).item())
        scores = np.array(scores)
        return np.max(scores)
