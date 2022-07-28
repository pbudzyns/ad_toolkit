"""
Long-Short Term Memory based Encoder-Decoder anomaly detector.

References:
    - Malhotra, Pankaj, et al. "LSTM-based encoder-decoder for multi-sensor
      anomaly detection."
    - Implementation from DeepADoTS
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
from torch.utils.data import DataLoader, SubsetRandomSampler

from sops_anomaly.detectors.base_detector import BaseDetector


class LSTM_ED(BaseDetector):

    def __init__(
        self,
        sequence_len: int = 20,
        stride: int = 1,
        hidden_size: int = 32,
        threshold: float = 0.5,
        use_gpu: bool = False,
    ) -> None:

        super(LSTM_ED, self).__init__()
        self.model: Optional[nn.Module] = None
        self._n_dims: int = 0
        self._sequence_len: int = sequence_len
        self._stride: int = stride
        self._hidden_size: int = hidden_size
        self._error_dist: Optional[scipy.stats.multivariate_normal] = None
        self._threshold: float = threshold
        self._device: torch.device = torch.device(
            'cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

    def train(
        self,
        train_data: pd.DataFrame,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        validation_steps: int = 10,
        epochs: int = 30,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> None:
        sequences = self._data_to_sequences(train_data, self._stride)
        # Train eval data split.
        split = int(0.8 * len(sequences))
        train_data_loader = self._get_train_data_loader(
            sequences[:split], batch_size)
        eval_data_loader = self._get_train_data_loader(
            sequences[split:], batch_size)

        # Initialize the model.
        self._n_dims = sequences[0].shape[1]
        self._init_model_if_needed()

        self._fit_model(train_data_loader, epochs, learning_rate, verbose)
        self._fit_error_distribution(eval_data_loader)

        if validation_data is not None:
            self._optimize_prediction_threshold(
                validation_data, validation_steps)

    def predict(self, data: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
        sequences = self._data_to_sequences(data, 1)
        test_data_loader = self._get_eval_data_loader(sequences, batch_size)

        scores = []
        self.model.eval()
        for inputs in test_data_loader:
            inputs = inputs.float().to(self._device)
            outputs = self.model.forward(inputs)
            error = F.l1_loss(outputs, inputs, reduction='none')
            error = error.view(-1, data.shape[1]).cpu().detach().numpy()
            score = -self._dist(error)
            scores.append(score.reshape(inputs.size(0), self._sequence_len))

        scores = np.concatenate(scores)
        lattice = np.zeros((self._sequence_len, data.shape[0]))
        for i, score in enumerate(scores):
            lattice[i % self._sequence_len, i:i + self._sequence_len] = score
        scores = np.nanmean(lattice, axis=0)

        return scores

    def detect(self, data: pd.DataFrame) -> np.ndarray:
        scores = self.predict(data)
        return (scores < self._threshold).astype(np.int32)

    def _optimize_prediction_threshold(
        self,
        validation_data: Tuple[pd.DataFrame, pd.Series],
        steps: int,
    ) -> None:
        data, labels = validation_data
        scores = self.predict(data)
        best_f1 = 0
        best_threshold = 0
        for threshold in np.linspace(np.min(scores), np.max(scores), steps):
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
                output = self.model.forward(inputs)
                loss = F.mse_loss(output, inputs, reduction='sum')
                epoch_loss += loss.item() / len(inputs)
                loss.backward()
                optimizer.step()
            if verbose:
                print(f"Epoch {epoch} loss: "
                      f"{epoch_loss / len(train_data_loader)}")

    def _fit_error_distribution(self, data_loader: DataLoader) -> None:
        """Fit multivariate gaussian distribution to a given sample using
        maximum likelihood estimation method.

        Source:
          https://stackoverflow.com/questions/27230824/fit-multivariate-gaussian-distribution-to-a-given-dataset


        :param sample:
        :return:
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
        self.model.eval()
        error_vectors = []
        for inputs in data_loader:
            inputs = inputs.float().to(self._device)
            output = self.model.forward(inputs)
            error = F.l1_loss(output, inputs, reduction='none')
            error_vectors += list(
                error.view(-1, self._n_dims).cpu().detach().numpy())
        return error_vectors

    def _init_model_if_needed(self) -> None:
        if self.model is not None:
            return
        self.model = _LSTMEncoderDecoder(
            n_dims=self._n_dims,
            hidden_size=self._hidden_size,
            device=self._device,
        )

    @classmethod
    def _get_train_data_loader(
            cls, sequences: List[np.ndarray], batch_size: int) -> DataLoader:
        indices = np.random.permutation(len(sequences))
        data_loader = DataLoader(
            dataset=sequences,
            batch_size=min(len(sequences), batch_size),
            drop_last=True,
            sampler=SubsetRandomSampler(indices),
        )
        return data_loader

    @classmethod
    def _get_eval_data_loader(
            cls, sequences: List[np.ndarray], batch_size: int) -> DataLoader:
        data_loader = DataLoader(
            dataset=sequences,
            batch_size=min(len(sequences), batch_size),
            drop_last=True,
        )
        return data_loader

    def _data_to_sequences(self, data: pd.DataFrame, stride: int) -> List[np.ndarray]:
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

        :param input_tensor:
        :return:
        """
        dec_hidden = self._encode_sentence(input_tensor)
        outputs = self._decode_sequence(input_tensor, dec_hidden)
        return outputs

    def _encode_sentence(
        self, input_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run encoder to get a final hidden state after processing a sequence.

        :param input_tensor:
        :return:
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

        :param input_tensor:
        :param dec_hidden:
        :return:
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


if __name__ == '__main__':
    data = pd.DataFrame(data=np.random.random((100, 10)).astype(np.float))
    module = LSTM_ED()
    module.train(data, epochs=3)
