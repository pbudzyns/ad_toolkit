"""
Variational auto-encoder anomaly detector.

References:
    - An, Jinwon, and Sungzoon Cho. "Variational auto-encoder based anomaly
      detection using reconstruction probability."

"""
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from torch import nn
from torch.nn import functional as F
import torch.optim
import torch.distributions
from torch.utils.data import DataLoader


from ad_toolkit.detectors.base_detector import BaseDetector
from ad_toolkit.utils.torch_utils import (
    build_layers, build_network, train_valid_split)
from ad_toolkit.utils import window_data


class _VAE(nn.Module):
    """Vanilla Variational Auto Encoder implementation with ELBO training.
    """
    def __init__(
        self,
        input_size: int,
        layers: Union[List[int], Tuple[int]],
        latent_size: int,
    ) -> None:
        super(_VAE, self).__init__()
        self.encoder: nn.Module = self._get_encoder(
            input_size, layers, latent_size)
        self.decoder: nn.Module = self._get_decoder(
            latent_size, list(reversed(layers)), input_size)
        self.prior: torch.distributions.Distribution = (
            torch.distributions.Normal(0, 1))

        self.layer_mu: nn.Module = nn.Linear(latent_size, latent_size)
        self.layer_sigma: nn.Module = nn.Linear(latent_size, latent_size)
        self.decoder_input: nn.Module = nn.Linear(latent_size, latent_size)

    @classmethod
    def _get_encoder(cls, input_size, layers, output_size) -> nn.Module:
        nn_layers = build_layers(input_size, layers, output_size)
        encoder = build_network(nn_layers)
        return encoder

    @classmethod
    def _get_decoder(cls, input_size, layers, output_size) -> nn.Module:
        nn_layers = build_layers(input_size, layers, output_size)
        decoder = build_network(nn_layers)
        return decoder

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)

        mu = self.layer_mu(encoded)
        log_var = self.layer_sigma(encoded)

        return mu, log_var

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder_input(x)
        return self.decoder(x)

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, x, mu, log_var

    @classmethod
    def reparametrize(
            cls, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    @classmethod
    def loss_function(
        cls,
        model_outputs: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        (x_hat, x, mu, log_var) = model_outputs

        reconstruction_loss = F.mse_loss(x_hat, x)
        kl_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1),
            dim=0)

        total_loss = reconstruction_loss + kl_loss
        return total_loss

    def sample(self) -> np.ndarray:
        z = torch.randn(self._latent_size)
        smpl = self.decode(z)
        return smpl.detach().numpy()


class VariationalAutoEncoder(BaseDetector):

    def __init__(
        self,
        window_size: int,
        latent_size: int = 50,
        layers: Union[List[int], Tuple[int]] = (500, 200),
        l_samples: int = 10,
        use_gpu: bool = False,
    ) -> None:
        """
        Variational Auto-Encoder based anomaly detector. Detects anomalies
        based on reconstruction probability score.

        :param latent_size:
        :param l_samples:
        :param threshold:
        """
        super(VariationalAutoEncoder, self).__init__()
        self.model: Optional[nn.Module] = None
        self._input_size: Optional[int] = None
        self._window_size: int = window_size
        self._latent_size: int = latent_size
        self._layers: Union[List[int], Tuple[int]] = layers
        self._l_samples: int = l_samples
        self._device: torch.device = torch.device(
            'cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

    def train(
        self, train_data: pd.DataFrame, epochs: int = 30, batch_size: int = 32,
        learning_rate: float = 1e-4, validation_portion: float = 1e-2,
        verbose: bool = False,
    ) -> None:
        all_data_tensors = self._prepare_data(train_data)
        self._init_detector(all_data_tensors)

        train_data_loader, valid_data_loader = train_valid_split(
            all_data_tensors, validation_portion, batch_size)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            train_loss = self._train_model(train_data_loader, optimizer)
            valid_loss = self._validate_model(valid_data_loader)
            if verbose:
                print(f"Epoch {epoch} train_loss: {train_loss}, "
                      f"valid_loss: {valid_loss}")

    def _prepare_data(self, train_data: pd.DataFrame) -> List[torch.Tensor]:
        all_data = self._transform_data(train_data)
        all_data_tensors = self._data_to_tensors(all_data)
        return all_data_tensors

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        all_data_tensors = self._prepare_data(data)
        scores = [0.0] * (self._window_size - 1)    # To match input length.
        self.model.eval()
        with torch.no_grad():
            for x in all_data_tensors:
                mu, log_var = self.model.encode(x)
                score = 0
                for i in range(self._l_samples):
                    z = self.model.reparametrize(mu, log_var)
                    x_hat = self.model.decode(z)
                    score += F.mse_loss(x_hat, x).item()
                scores.append(score / self._l_samples)
        return np.array(scores)

    def _init_detector(self, all_data_tensors):
        if self._input_size is None:
            self._input_size = len(all_data_tensors[0])
        self._init_model_if_needed()

    def _train_model(
        self, train_data_loader: DataLoader, optimizer: torch.optim.Optimizer,
    ) -> float:
        total_loss = 0
        self.model.train()
        for batch in train_data_loader:
            optimizer.zero_grad()
            model_outputs = self.model.forward(batch)
            loss = self.model.loss_function(model_outputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_data_loader)

    def _validate_model(self, valid_data_loader: DataLoader) -> float:
        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch in valid_data_loader:
                model_outputs = self.model.forward(batch)
                loss = self.model.loss_function(model_outputs)
                total_loss += loss.item()

        return total_loss / len(valid_data_loader)

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
        self.model = _VAE(
            input_size=self._input_size,
            layers=self._layers,
            latent_size=self._latent_size,
        ).to(self._device)

    def sample(self) -> np.ndarray:
        return self.model.sample()
