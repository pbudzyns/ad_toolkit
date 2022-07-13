"""
Long-Short Term Memory based Encoder-Decoder anomaly detector.

References:
    - "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection"
      Malhotra et al.
      https://www.researchgate.net/publication/304758073_LSTM-based_Encoder-Decoder_for_Multi-sensor_Anomaly_Detection
    - Implementation from DeepADoTS
      https://github.com/KDD-OpenSource/DeepADoTS/blob/master/src/algorithms/lstm_enc_dec_axl.py
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

from sops_anomaly.detectors.base_detector import BaseDetector
from sops_anomaly.detectors.lstm_ad import ErrorDistribution


class _LSTMEncoderDecoder(nn.Module):

    def __init__(
            self, n_dims: int, hidden_size: int, num_layers: int = 1) -> None:
        super(_LSTMEncoderDecoder, self).__init__()
        self.encoder = nn.LSTM(
            input_size=n_dims, hidden_size=hidden_size, num_layers=num_layers,
        )
        self.decoder = nn.LSTM(
            input_size=n_dims, hidden_size=hidden_size, num_layers=num_layers,
        )
        self.linear = nn.Linear(hidden_size, n_dims)

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
        outputs = torch.Tensor(input_tensor.shape).zero_()
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


class LSTM_ED(BaseDetector):

    def __init__(
        self,
        hidden_size: int = 32,
        sequence_len: int = 20,
        batch_size: int = 32,
    ) -> None:

        super(LSTM_ED, self).__init__()
        self._sequence_len = sequence_len
        self._batch_size: int = batch_size
        self._hidden_size: int = hidden_size
        self._error_dist: Optional[ErrorDistribution] = None
        self.model: Optional[nn.Module] = None

    def train(
        self, train_data: pd.DataFrame, epochs: int = 30, verbose: bool = False,
    ) -> None:
        sequences = self._data_to_sequences(train_data)
        train_data_loader = self._get_data_loader(sequences)
        self._init_model(n_dims=sequences[0].shape[1])

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self._run_train_loop(epochs, optimizer, train_data_loader, verbose)

    def _run_train_loop(
        self, epochs: int, optimizer: torch.optim.Optimizer,
        train_data_loader: DataLoader, verbose: bool,
    ) -> None:
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for inputs in train_data_loader:
                inputs = inputs.float()
                optimizer.zero_grad()
                output = self.model.forward(inputs)
                loss = F.mse_loss(output, inputs)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            if verbose:
                print(f"Epoch {epoch} loss: "
                      f"{epoch_loss / len(train_data_loader)}")

    def _init_model(self, n_dims: int) -> None:
        self.model = _LSTMEncoderDecoder(
            n_dims=n_dims,
            hidden_size=self._hidden_size,
        )

    def _get_data_loader(self, sequences: List[np.ndarray]) -> DataLoader:
        indices = np.random.permutation(len(sequences))
        data_loader = DataLoader(
            dataset=sequences,
            batch_size=self._batch_size,
            drop_last=True,
            sampler=SubsetRandomSampler(indices),
            pin_memory=True,
        )
        return data_loader

    def _data_to_sequences(self, data: pd.DataFrame) -> List[np.ndarray]:
        values = data.values
        sequences = [values[i:i + self._sequence_len]
                     for i
                     in range(values.shape[0] - self._sequence_len + 1)]
        return sequences

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        pass

    def detect(self, data: pd.DataFrame) -> np.ndarray:
        pass


if __name__ == '__main__':
    data = pd.DataFrame(data=np.random.random((100, 10)).astype(np.float))
    module = LSTM_ED()
    module.train(data, epochs=3)
