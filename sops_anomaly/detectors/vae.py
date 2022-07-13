"""
Variational auto-encoder anomaly detector.

References:
    - An, Jinwon, and Sungzoon Cho. "Variational auto-encoder based anomaly
      detection using reconstruction probability."
"""
from typing import Union, List, Optional, Tuple

import numpy as np
import pandas as pd
from torch import nn
from torch.nn import functional as F
import torch.optim
import torch.distributions

from sops_anomaly.detectors.base_detector import BaseDetector
from sops_anomaly.utils import window_data


class _VAE(nn.Module):
    """Vanilla Variational Auto Encoder implementation with ELBO training.

     TODO: allow for batch-processing
    """
    def __init__(self, input_size: int, latent_size: int) -> None:
        super(_VAE, self).__init__()
        self._input_size: int = input_size
        self._latent_size: int = latent_size
        self.encoder: nn.Module = self._get_encoder()
        self.decoder: nn.Module = self._get_decoder()
        self.prior: torch.distributions.Distribution = (
            torch.distributions.Normal(0, 1))

        self.layer_mu: nn.Module = nn.Linear(100, latent_size)
        self.layer_sigma: nn.Module = nn.Linear(100, latent_size)
        self.decoder_input: nn.Module = nn.Linear(latent_size, 100)

    def _get_encoder(self) -> nn.Module:
        encoder = nn.Sequential(
            nn.Linear(self._input_size, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 100)
        )
        return encoder

    def _get_decoder(self) -> nn.Module:
        decoder = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 500),
            nn.ReLU(),
            nn.Linear(500, self._input_size)
        )
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

    def loss_function(
        self,
        model_outputs: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        (x_hat, x, mu, log_var) = model_outputs

        reconstruction_loss = F.mse_loss(x_hat, x)
        # TODO: batch processing loss definition
        # kld_loss = torch.mean(
        #     -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1),
        #     dim=0)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())

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
        latent_size: int,
        l_samples: int = 10,
        threshold: float = 0.9,
    ):
        """
        Variational Auto-Encoder based anomaly detector. Detects anomalies
        based on reconstruction probability score.

        :param latent_size:
        :param l_samples:
        :param threshold:
        """
        super(VariationalAutoEncoder, self).__init__()
        self._window_size: int = window_size
        self._latent_size: int = latent_size
        self._l_samples: int = l_samples
        self._threshold: float = threshold
        self._max_error: float = 0.0
        self.model: Optional[nn.Module] = None

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
                rec, _, _, _ = self.model.forward(sample)
                scores.append(F.mse_loss(rec, sample).item())
        scores = np.array(scores)
        return np.max(scores)

    def train(
        self,
        train_data: pd.DataFrame,
        epochs: int = 30,
        learning_rate: float = 1e-4,
    ) -> None:
        train_data = self._transform_data(train_data)
        train_data = self._data_to_tensors(train_data)

        self.model = _VAE(
            input_size=len(train_data[0]),
            latent_size=self._latent_size,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for sample in train_data:
                optimizer.zero_grad()
                results = self.model.forward(sample)
                loss = self.model.loss_function(results)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch} loss: {epoch_loss / len(train_data)}")

        self._max_error = self._compute_threshold(train_data)

    def sample(self) -> np.ndarray:
        return self.model.sample()

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        data = self._transform_data(data)
        data = self._data_to_tensors(data)
        scores = [0.0] * (self._window_size - 1)    # To match input length.
        self.model.eval()
        with torch.no_grad():
            for x in data:
                mu, log_var = self.model.encode(x)
                score = 0
                for i in range(self._l_samples):
                    z = self.model.reparametrize(mu, log_var)
                    x_hat = self.model.decode(z)
                    score += F.mse_loss(x_hat, x).item()
                scores.append(score / self._l_samples)
        return np.array(scores)

    def detect(self, data: pd.DataFrame) -> np.ndarray:
        scores = self.predict(data)
        return (scores >= self._threshold * self._max_error).astype(np.int32)
