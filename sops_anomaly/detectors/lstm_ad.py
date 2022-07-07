"""
LSTM Anomaly Detector based on reconstruction error.

"""
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from sops_anomaly.detectors.base_detector import BaseDetector


class LSTM(BaseDetector):

    def __init__(
        self,
        l_predictions: int = 10,
        hidden_size: int = 32,
    ) -> None:
        super(LSTM, self).__init__()
        self.model: Optional[nn.LSTM] = None
        self._hidden_size = hidden_size
        self._l_predictions = l_predictions

    def _get_lstm(
        self, input_size: int, n_layers: int = 2, dropout: float = 0.5,
    ) -> nn.LSTM:
        return nn.LSTM(
            input_size=input_size,
            hidden_size=self._hidden_size,
            proj_size=self._l_predictions,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=False,
        )

    def _transform_data(self, data: pd.DataFrame) -> List[torch.Tensor]:
        new_data = []
        for _, row in data.iterrows():
            new_data.append(torch.Tensor(np.array(row)))
        return new_data

    def train(self, train_data: pd.DataFrame, epochs: int = 50) -> None:
        train_data = self._transform_data(train_data)
        self.model = self._get_lstm(input_size=train_data[0].shape[0])
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        for epoch in range(epochs):
            outputs = self.model()

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        pass

    def detect(self, data: pd.DataFrame) -> np.ndarray:
        pass


if __name__ == '__main__':
    data = pd.DataFrame(data=[1,2,3,4,5,6,7,8,9,10])
    data2 = pd.DataFrame(data=[
        [1,2],
        [3,4],
        [5,6],
        [7,8],
        [9,10],
        [11,12],
    ])

    def transform_train_data_target(data: pd.DataFrame, l_preds: int):
        values = np.expand_dims(data, axis=0)
        train_data = values[:, :-l_preds, :]
        train_labels = []
        for l in range(l_preds-1):
            train_labels += [values[:, 1+l:-l_preds+l+1, :]]
        train_labels += [values[:, l_preds:, :]]
        train_labels = np.stack(train_labels, axis=3)
        return train_data, train_labels

    def transform_eval_data_target(data: pd.DataFrame, l_preds:int):
        values = np.expand_dims(data, axis=0)
        eval_data = values[:, :-l_preds, :]
        eval_target = values[:, l_preds:-l_preds+1, :]

        return eval_data, eval_target

    def get_errors(output: np.ndarray, target: np.ndarray, l_preds: int):
        errors = []
        print(output.shape)
        for l in range(l_preds-1):
            errors += [output[:, l:-l_preds+l+1, :, l_preds-1-l]]
        errors += [output[:, l_preds-1:, :, 0]]
        errors = np.stack(errors, axis=3)
        print(errors - target[..., np.newaxis])


    t_d, t_l = transform_train_data_target(data2, 3)
    # print(t_d, t_l)

    e_d, e_t = transform_eval_data_target(data2, 3)
    # print(e_d, e_t)

    # get_errors(t_l, e_t, 3)
