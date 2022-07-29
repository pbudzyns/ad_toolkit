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
from torch.utils.data import DataLoader

from ad_toolkit.detectors.base_detector import BaseDetector
from ad_toolkit.utils.torch_utils import (
    build_layers, build_network, get_data_loader, train_valid_split)
from ad_toolkit.utils import window_data


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
        cls, input_size: int, layers: Union[List[int], Tuple[int]],
        output_size: int,
    ) -> nn.Module:
        nn_layers = build_layers(input_size, layers, output_size)
        encoder = build_network(nn_layers)
        return encoder

    @classmethod
    def _get_decoder(
        cls, input_size: int, layers: Union[List[int], Tuple[int]],
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
        use_gpu: bool = False,
    ) -> None:
        """

        :param window_size:
        :param latent_size:
        :param layers:
        """
        self.model: Optional[nn.Module] = None
        self._input_size: Optional[int] = None
        self._window_size: int = window_size
        self._latent_size: int = latent_size
        self._layers: Union[List[int], Tuple[int]] = layers
        self._device: torch.device = torch.device(
            'cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

    def train(
        self, train_data: pd.DataFrame, epochs: int = 20, batch_size: int = 32,
        learning_rate: float = 1e-4, validation_portion: float = 1e-2,
        verbose: bool = False,
    ) -> None:

        all_data = self._prepare_data(train_data)
        self._init_detector(all_data)

        all_data_tensors = self._data_to_tensors(all_data)
        train_data_loader, valid_data_loader = train_valid_split(
            all_data_tensors, validation_portion, batch_size)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            train_loss = self._train_model(train_data_loader, optimizer)
            valid_loss = self._validate_model(valid_data_loader)
            if verbose:
                print(f"Epoch {epoch} train_loss: {train_loss}, "
                      f"valid_loss: {valid_loss}")

    def predict(
        self, data: pd.DataFrame, batch_size: int = 32,
        raw_errors: bool = False,
    ) -> np.ndarray:
        if self._window_size > 1:
            input_data = self._transform_data(data)
            input_data = self._data_to_tensors(input_data)
        else:
            input_data = self._data_to_tensors(data)

        data_loader = get_data_loader(input_data, batch_size, test=True)
        scores = []
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                rec = self.model.forward(batch)
                errors = F.mse_loss(rec, batch, reduction='none')
                if not raw_errors:
                    errors = errors.mean(1)
                scores.append(errors.cpu().numpy())

        if raw_errors:
            results = np.zeros((len(data), self._input_size))
        else:
            results = np.zeros((len(data),))
        results[self._window_size-1:] = np.concatenate(scores)

        return (results if not raw_errors
                else self._errors_to_reconstruction_error(results))

    def _errors_to_reconstruction_error(self, errors):
        d = int(self._input_size / self._window_size)
        rec_errors = np.zeros((len(errors), d))
        for i in range(self._window_size):
            rec_errors += errors[:, i*d:(i+1)*d]
        return rec_errors / self._window_size

    def _init_detector(self, all_data: pd.DataFrame) -> None:
        if self._input_size is None:
            self._input_size = len(all_data.iloc[0])
        self._init_model_if_needed()

    def _prepare_data(self, train_data: pd.DataFrame) -> pd.DataFrame:
        if self._window_size > 1:
            all_data = self._transform_data(train_data)
        else:
            all_data = train_data
        return all_data

    def _train_model(
        self, train_data_loader: DataLoader, optimizer: torch.optim.Optimizer,
    ) -> float:
        epoch_loss = 0
        self.model.train()
        for batch in train_data_loader:
            optimizer.zero_grad()
            reconstructed = self.model.forward(batch)
            loss = F.mse_loss(reconstructed, batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        return epoch_loss / len(train_data_loader)

    def _validate_model(self, valid_data_loader: DataLoader) -> float:
        epoch_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch in valid_data_loader:
                reconstructed = self.model.forward(batch)
                loss = F.mse_loss(reconstructed, batch)
                epoch_loss += loss.item()

        return epoch_loss / len(valid_data_loader)

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
