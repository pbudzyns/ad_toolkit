"""
Variational auto-encoder anomaly detector.

References:

    [1] An, J., & Cho, S. (2015). Variational autoencoder based anomaly
        detection using reconstruction probability.

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
    build_linear_layers, build_network, dataframe_to_tensors,
    train_valid_data_loaders,
)
from ad_toolkit.utils import window_data


class VariationalAutoEncoder(BaseDetector):

    def __init__(
        self,
        window_size: int = 1,
        latent_size: int = 50,
        layers: Union[List[int], Tuple[int]] = (500, 200),
        l_samples: int = 10,
        use_gpu: bool = False,
    ) -> None:
        """Variational Auto-Encoder anomaly detector. Detects anomalies
        based on reconstruction probability score. The probability is computed
        as an averaged error made for `l_samples` reconstructions.

        Parameters
        ----------
        window_size
            Size of the window if multiple time steps should be used as
            an input. If `window_size` > 1 then samples from consecutive
            time steps will be concatenated together.
        latent_size
            Size of the latent space for variational auto-encoder model.
        layers
            Sizes of hidden layer of variational auto-encoder model.
        l_samples
            Number of outputs to be sampled to compute reconstruction
            probability.
        use_gpu
            Accelerated computation when GPU device is available.
        """
        super(VariationalAutoEncoder, self).__init__()
        self.model: Optional["_VAE"] = None
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
        """Trains variational auto-encoder model to fit given data.

        During the first execution a model is created. Multiple executions
        are possible but input data should always have the same dimension.

        Parameters
        ----------
        train_data
            ``pandas.DataFrame`` containing samples as rows. Features should
            correspond to columns.
        epochs
            Number of epochs to use during the training.
        batch_size
            Batch size to use during the training is batch processing
            is desired.
        learning_rate
            Learning rate for the optimizer.
        validation_portion
            Percent of training data to use as a validation during training.
        verbose
            Controls printing of progress messages.

        Returns
        -------
        None
        """
        all_data_tensors = self._prepare_data(train_data)
        self._init_detector(all_data_tensors)

        train_data_loader, valid_data_loader = train_valid_data_loaders(
            all_data_tensors, validation_portion, batch_size)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            train_loss = self._train_model(train_data_loader, optimizer)
            valid_loss = self._validate_model(valid_data_loader)
            if verbose:
                print(f"Epoch {epoch} train_loss: {train_loss}, "
                      f"valid_loss: {valid_loss}")

    def predict(
            self, data: pd.DataFrame, raw_errors: bool = False) -> np.ndarray:
        """Predict function returns reconstruction probability for each
        of the samples from `data`. If `raw_errors` is `True` errors are not
        averaged and the output scores have same shape as input data.

        Parameters
        ----------
        data
            ``pandas.DataFrame`` containing data samples.
        raw_errors
            Controls shape of the output.

        Returns
        -------
        np.ndarray
            Reconstruction error scores for the input data. Either averaged
            per sample or raw.
        """
        all_data_tensors = self._prepare_data(data)
        scores = []
        self.model.eval()
        with torch.no_grad():
            for x in all_data_tensors:
                x = x.to(self._device)
                mu, log_var = self.model.encode(x)
                if raw_errors:
                    score = self._raw_error(x, mu, log_var)
                else:
                    score = self._simple_error(x, mu, log_var)
                scores.append(score)
                del x, mu, log_var

        if raw_errors:
            results = np.zeros(data.shape)
        else:
            results = np.zeros((len(data),))
        results[self._window_size-1:] = scores
        return results

    def _prepare_data(self, train_data: pd.DataFrame) -> List[torch.Tensor]:
        all_data = self._transform_data(train_data)
        all_data_tensors = dataframe_to_tensors(all_data)
        return all_data_tensors

    def _simple_error(self, x, mu, log_var):
        """Computes reconstruction probability as described in ref. [1]."""
        score = 0
        for i in range(self._l_samples):
            z = self.model.reparametrize(mu, log_var)
            x_hat = self.model.decode(z)
            score += F.mse_loss(x_hat, x).item()
        return score / self._l_samples

    def _raw_error(self, x, mu, log_var):
        errors = np.zeros((self._input_size, ))
        for i in range(self._l_samples):
            z = self.model.reparametrize(mu, log_var)
            x_hat = self.model.decode(z)
            error = F.mse_loss(x_hat, x, reduction='none').cpu().numpy()
            error /= self._l_samples
            errors += error
        d = int(self._input_size / self._window_size)
        res = np.zeros((d, ))
        for i in range(self._window_size):
            res += errors[i*d:(i+1)*d]
        return res / self._window_size

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
            batch = batch.to(self._device)
            model_outputs = self.model(batch)
            loss = self.model.loss_function(model_outputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            del batch, model_outputs
        return total_loss / len(train_data_loader)

    def _validate_model(self, valid_data_loader: DataLoader) -> float:
        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch in valid_data_loader:
                batch = batch.to(self._device)
                model_outputs = self.model(batch)
                loss = self.model.loss_function(model_outputs)
                total_loss += loss.item()
                del batch, model_outputs
        return total_loss / len(valid_data_loader)

    def _transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return window_data(data, self._window_size)

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


class _VAE(nn.Module):
    def __init__(
        self,
        input_size: int,
        layers: Union[List[int], Tuple[int]],
        latent_size: int,
    ) -> None:
        """Variational Auto-Encoder implementation with ELBO.

        Parameters
        ----------
        input_size
            Size of the input layer.
        layers
            Sizes of the hidden layers.
        latent_size
            Size of the latent space.
        """
        super(_VAE, self).__init__()
        # Creates Encoder module from given sizes.
        self.encoder: nn.Module = self._get_module(
            input_size, layers, latent_size)
        # Creates Decoder module from given sizes.
        self.decoder: nn.Module = self._get_module(
            latent_size, list(reversed(layers)), input_size)

        self.layer_mu: nn.Module = nn.Linear(latent_size, latent_size)
        self.layer_sigma: nn.Module = nn.Linear(latent_size, latent_size)
        self.decoder_input: nn.Module = nn.Linear(latent_size, latent_size)
        self.latent_size: int = latent_size

    @classmethod
    def _get_module(cls, input_size, layers, output_size) -> nn.Module:
        nn_layers = build_linear_layers(input_size, layers, output_size)
        encoder = build_network(nn_layers)
        return encoder

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
        z = torch.randn(self.latent_size)
        smpl = self.decode(z)
        return smpl.detach().numpy()
