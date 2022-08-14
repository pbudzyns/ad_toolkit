"""
Long-Short Term Memory based Encoder-Decoder anomaly detector.

References:

    [1] Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P.,
        & Shroff, G. (2016). LSTM-based encoder-decoder for multi-sensor
        anomaly detection.

    [2] DeepADoTS
        https://github.com/KDD-OpenSource/DeepADoTS/blob/master/src/algorithms/lstm_enc_dec_axl.py

"""
import functools
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.metrics
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ad_toolkit.detectors.base_detector import BaseDetector
from ad_toolkit.utils.torch_utils import (
    get_data_loader, train_valid_data_loaders)


class LSTM_ED(BaseDetector):

    def __init__(
        self,
        sequence_len: int = 20,
        stride: int = 1,
        hidden_size: int = 32,
        threshold: float = 0.5,
        use_gpu: bool = False,
    ) -> None:
        """LSTM encoder-decoder anomaly detector. Learns to reconstruct
        the sample of the data. At each time step the model generates
        tries to reconstruct a part of the time series of `sequence_len` size.
        The score for a point is computed as a probability of the overall error
        for this point coming from the distribution of errors learned
        during the training.

        Parameters
        ----------
        sequence_len
            Input size. Number of consecutive values to process at
            each time step.
        stride
            The distance between starts of consecutive sequences.
        hidden_size
            Hidden size of the LSTM model.
        threshold
            Anomaly detection threshold. If validation data is provided
            during the training it will be optimized to get
            the best predictions.
        use_gpu
            Accelerated computation when GPU device is available.
        """

        super(LSTM_ED, self).__init__()
        self.model: Optional["_LSTMEncoderDecoder"] = None
        self._error_dist: Optional[scipy.stats.multivariate_normal] = None
        self._n_dims: int = 0
        self._sequence_len: int = sequence_len
        self._stride: int = stride
        self._hidden_size: int = hidden_size
        self._threshold: float = threshold
        self._device: torch.device = torch.device(
            'cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

    def train(
        self,
        train_data: pd.DataFrame,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        epochs: int = 30,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> None:
        """Trains LSTM ED model to fit given data. The model is trained in
        an unsupervised manner with minimizing L1 loss of reconstruction
        as an objective.

        During the first execution a model is created. Multiple executions
        are possible but input data should always have the same dimension.

        Parameters
        ----------
        train_data
            ``pandas.DataFrame`` containing samples as rows. Features should
            correspond to columns.
        validation_data
            ``pandas.DataFrame`` with data to be used for threshold
            optimization.
        epochs
            Number of epochs to use during the training.
        learning_rate
            Learning rate for the optimizer.
        batch_size
            Batch size to use during the training.
        verbose
            Controls printing of progress messages.

        Returns
        -------
        None
        """
        sequences = self._data_to_sequences(train_data, self._stride)
        # Train eval data split.
        train_data_loader, eval_data_loader = train_valid_data_loaders(
            sequences, validation_portion=0.2, batch_size=batch_size)

        # Initialize the model.
        self._n_dims = sequences[0].shape[1]
        self._init_model_if_needed()

        self._fit_model(train_data_loader, epochs, learning_rate, verbose)
        self._fit_error_distribution(eval_data_loader)

        if validation_data is not None:
            self._optimize_prediction_threshold(validation_data)

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
            Batch size to use.
        raw_errors
            Controls shape of the output.

        Returns
        -------
        np.ndarray
            Reconstruction error scores for the input data. Either averaged
            per sample or raw.
        """
        sequences = self._data_to_sequences(data, 1)
        test_data_loader = get_data_loader(sequences, batch_size,
                                           prediction=True)

        scores = []
        self.model.eval()
        for inputs in test_data_loader:
            inputs = inputs.float().to(self._device)
            outputs = self.model(inputs)
            error = F.l1_loss(outputs, inputs, reduction='none')
            error = error.view(-1, data.shape[1]).cpu().detach().numpy()
            if raw_errors:
                error = error.reshape(
                    inputs.size(0), self._sequence_len * self._n_dims)
                scores.append(error)
            else:
                score = -self._dist(error)
                scores.append(score.reshape(inputs.size(0), self._sequence_len))
            del inputs, outputs

        scores = np.concatenate(scores)
        if raw_errors:
            return self._raw_errors(data, scores)

        lattice = np.zeros((self._sequence_len, data.shape[0]))
        for i, score in enumerate(scores):
            lattice[i % self._sequence_len, i:i + self._sequence_len] = score
        scores = np.nanmean(lattice, axis=0)
        return scores

    def detect(self, data: pd.DataFrame) -> np.ndarray:
        """Returns list of points classified as anomalies based on the threshold
        value.

        Parameters
        ----------
        data
            Data points to make prediction about.

        Returns
        -------
        np.ndarray
            Labels marking anomalous data point.
        """
        scores = self.predict(data)
        return (scores < self._threshold).astype(np.int32)

    def _raw_errors(self, data, errors):
        res = np.zeros(data.shape)
        averaged = np.zeros((errors.shape[0], self._n_dims))
        for i in range(self._sequence_len):
            averaged += errors[:, i*self._n_dims:(i+1)*self._n_dims]
        averaged /= self._sequence_len
        res[self._sequence_len-1:] = averaged
        return res

    def _optimize_prediction_threshold(
            self, validation_data: Tuple[pd.DataFrame, pd.Series]) -> None:
        data, labels = validation_data
        scores = self.predict(data)
        best_f1 = 0
        best_threshold = 0
        for threshold in np.linspace(np.min(scores), np.max(scores), 300):
            anomalies = (scores < threshold).astype(np.int32)
            f1_score = sklearn.metrics.f1_score(labels, anomalies)
            if f1_score > best_f1:
                best_f1 = f1_score
                best_threshold = threshold

        self._threshold = best_threshold

    def _fit_model(self, train_data_loader, epochs, learning_rate, verbose):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for inputs in train_data_loader:
                inputs = inputs.float().to(self._device)
                optimizer.zero_grad()
                output = self.model(inputs)
                loss = F.mse_loss(output, inputs, reduction='sum')
                epoch_loss += loss.item() / len(inputs)
                loss.backward()
                optimizer.step()
                del inputs, output
            if verbose:
                print(f"Epoch {epoch} loss: "
                      f"{epoch_loss / len(train_data_loader)}")

    def _fit_error_distribution(self, data_loader: DataLoader) -> None:
        """Fit multivariate gaussian distribution to a given sample using
        maximum likelihood estimation method.

        Source:
          https://stackoverflow.com/questions/27230824/fit-multivariate-gaussian-distribution-to-a-given-dataset
        """
        error_vectors = self._compute_errors(data_loader)

        means = np.mean(error_vectors, axis=0)
        cov = np.cov(error_vectors, rowvar=False)

        self._dist = functools.partial(
            scipy.stats.multivariate_normal.logpdf,
            mean=means,
            cov=cov,
            allow_singular=True,
        )

    def _compute_errors(self, data_loader: DataLoader) -> List[np.ndarray]:
        error_vectors = []
        self.model.eval()
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs.float().to(self._device)
                output = self.model(inputs)
                error = F.l1_loss(output, inputs, reduction='none')
                error_vectors += list(
                    error.view(-1, self._n_dims).cpu().detach().numpy())
                del inputs, output
        return error_vectors

    def _init_model_if_needed(self) -> None:
        if self.model is not None:
            return
        self.model = _LSTMEncoderDecoder(
            n_dims=self._n_dims,
            hidden_size=self._hidden_size,
            device=self._device,
        )

    def _data_to_sequences(
            self, data: pd.DataFrame, stride: int) -> List[np.ndarray]:
        values = data.values
        sequences = [
            values[i:i + self._sequence_len]
            for i
            in range(0, values.shape[0] - self._sequence_len + 1, stride)
        ]
        return sequences


class _LSTMEncoderDecoder(nn.Module):

    def __init__(
        self, n_dims: int, hidden_size: int, device: torch.device,
        num_layers: int = 1,
    ) -> None:
        """LSTM based encoder-decoder model.

        Parameters
        ----------
        n_dims
            Size of the input layer.
        hidden_size
            Size of the hidden layer.
        device
            Device to to use, ie. either CPU or GPU.
        num_layers
            Number of layers that encoder and decoder networks will have.
        """
        super(_LSTMEncoderDecoder, self).__init__()
        self.encoder = nn.LSTM(input_size=n_dims, hidden_size=hidden_size,
                               num_layers=num_layers)
        self.decoder = nn.LSTM(input_size=n_dims, hidden_size=hidden_size,
                               num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, n_dims)
        self._device = device
        self.to(self._device)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """LSTM based encoder-decoder architecture uses hidden state returned
        from the encoder after processing given sentence to reconstruct
        the sentence using a decoder.

        Parameters
        ----------
        input_tensor
            Data tensor containing model inputs.

        Returns
        -------
        torch.Tensor
            Model outputs.
        """
        dec_hidden = self._encode_sentence(input_tensor)
        outputs = self._decode_sequence(input_tensor, dec_hidden)
        return outputs

    def _encode_sentence(
        self, input_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run encoder to get a final hidden state after processing a sequence.

        Parameters
        ----------
        input_tensor
            Data tensor with model inputs.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Model outputs.
        """
        _, hidden = self.encoder(input_tensor)
        dec_hidden = (
            hidden[0][:, -1].unsqueeze(1),
            hidden[1][:, -1].unsqueeze(1),
        )
        return dec_hidden

    def _decode_sequence(
        self,
        input_tensor: torch.Tensor,
        dec_hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Use given hidden state to decode a sequence. While in train mode uses
        consecutive train data points as inputs. During evaluation uses sampled
        points.

        Parameters
        ----------
        input_tensor
            Data tensor with model inputs.
        dec_hidden
            Hidden state returned from the encoder.

        Returns
        -------
        torch.Tensor
            Model outputs.
        """
        outputs = torch.Tensor(input_tensor.shape).zero_().to(self._device)
        for i in reversed(range(outputs.shape[1])):
            # Projection hidden state to predicted data point.
            outputs[:, i, :] = self.linear(dec_hidden[0].squeeze(1))

            # During training use real data points, during eval use what was
            # returned in a previous time step from the decoder.
            inputs = input_tensor[:, i] if self.training else outputs[:, i]

            # Make a step using decoder to sample next data point
            # and get consecutive hidden state.
            _, dec_hidden = self.decoder(inputs.unsqueeze(1), dec_hidden)

        return outputs
