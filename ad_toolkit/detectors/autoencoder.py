"""
Auto-encoder anomaly detector.

References:

    [1] An, J., & Cho, S. (2015). Variational autoencoder based anomaly
        detection using reconstruction probability.

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
    build_linear_layers, build_network, dataframe_to_tensors, get_data_loader,
    train_valid_data_loaders,
)
from ad_toolkit.utils import window_data


class AutoEncoder(BaseDetector):

    def __init__(
        self,
        window_size: int = 1,
        latent_size: int = 100,
        layers: Union[List[int], Tuple[int]] = (500, 200),
        use_gpu: bool = False,
    ) -> None:
        """Auto-Encoder based anomaly detector. During the training
        an auto-encoder model is trained to fit the training data.
        The prediction scores are computed based on reconstruction error.

        Parameters
        ----------
        window_size
            Size of the window if multiple time steps should be used as
            an input. If `window_size` > 1 then samples from consecutive
            time steps will be concatenated together.
        latent_size
            Size of the latent space for auto-encoder model.
        layers
            Sizes of hidden layer of auto-encoder model.
        use_gpu
            Accelerated computation when GPU device is available.
        """
        self.model: Optional["_AEModel"] = None
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
        """Trains auto-encoder model to fit given data. The loss score is
        MSE for decoded reconstructions.

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

        all_data_tensors = self.prepare_data(train_data)
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
        self, data: pd.DataFrame, batch_size: int = 32,
        raw_errors: bool = False,
    ) -> np.ndarray:
        """Predict function returns mean reconstruction error for each
        of the samples from `data`. If `raw_errors` is `True` errors are not
        averaged and the output scores have same shape as input data.

        Parameters
        ----------
        data
            ``pandas.DataFrame`` containing data samples.
        batch_size
            Batch size to use during prediction.
        raw_errors
            Controls shape of the output.

        Returns
        -------
        np.ndarray
            Reconstruction error scores for the input data. Either averaged
            per sample or raw.
        """
        input_data = self.prepare_data(data)

        data_loader = get_data_loader(input_data, batch_size, prediction=True)
        scores = []
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self._device)
                rec = self.model(batch)
                errors = F.mse_loss(rec, batch, reduction='none')
                if not raw_errors:
                    errors = errors.mean(1)
                scores.append(errors.cpu().numpy())
                del batch, rec

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

    def _init_detector(self, all_data: List[torch.Tensor]) -> None:
        if self._input_size is None:
            self._input_size = all_data[0].shape[0]
        self._init_model_if_needed()

    def prepare_data(self, train_data: pd.DataFrame) -> List[torch.Tensor]:
        if self._window_size > 1:
            all_data = self._transform_data(train_data)
        else:
            all_data = train_data
        all_data_tensors = dataframe_to_tensors(all_data)
        return all_data_tensors

    def _train_model(
        self, train_data_loader: DataLoader, optimizer: torch.optim.Optimizer,
    ) -> float:
        """Run training loop."""
        epoch_loss = 0
        self.model.train()
        for batch in train_data_loader:
            optimizer.zero_grad()
            batch = batch.to(self._device)
            reconstructed = self.model(batch)
            loss = F.mse_loss(reconstructed, batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            del batch, reconstructed

        return epoch_loss / len(train_data_loader)

    def _validate_model(self, valid_data_loader: DataLoader) -> float:
        """Run validation loop."""
        epoch_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch in valid_data_loader:
                batch = batch.to(self._device)
                reconstructed = self.model(batch)
                loss = F.mse_loss(reconstructed, batch)
                epoch_loss += loss.item()
                del batch, reconstructed

        return epoch_loss / len(valid_data_loader)

    def _transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepares the data to use for the training."""
        return window_data(data, self._window_size)

    def _init_model_if_needed(self) -> None:
        if self.model is not None:
            return
        self.model = _AEModel(
            input_size=self._input_size,
            layers=self._layers,
            latent_size=self._latent_size,
        ).to(self._device)


class _AEModel(nn.Module):

    def __init__(
        self,
        input_size: int,
        layers: Union[List[int], Tuple[int]],
        latent_size: int,
    ) -> None:
        """Auto-Encoder torch module.

        Parameters
        ----------
        input_size
            Size of the input layer to initialize.
        layers
            List of hidden layer sizes to initialize.
        latent_size
            Size of the inner-most layer.

        """

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
        nn_layers = build_linear_layers(input_size, layers, output_size)
        encoder = build_network(nn_layers)
        return encoder

    @classmethod
    def _get_decoder(
        cls, input_size: int, layers: Union[List[int], Tuple[int]],
        output_size: int,
    ) -> nn.Module:
        nn_layers = build_linear_layers(input_size, layers, output_size)
        decoder = build_network(nn_layers)
        return decoder

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        return decoded
