"""
LSTM Anomaly Detector based on reconstruction error density.

"""
import functools
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import scipy.stats

from sops_anomaly.detectors.base_detector import BaseDetector


class LSTM(BaseDetector):

    def __init__(
        self,
        l_predictions: int = 10,
        hidden_size: int = 400,
        threshold: float = 0.9,
    ) -> None:
        """

        :param l_predictions:
        :param hidden_size:
        """
        super(LSTM, self).__init__()
        self.model: Optional[nn.LSTM] = None
        self._threshold: float = threshold
        self._hidden_size: int = hidden_size
        # Model output dimensions (l, d)
        self._l_preds: int = l_predictions
        self._d_size: int = 0
        # Multivariate gaussian scipy.stats.multivariate_gaussian
        self._dist: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def _get_lstm(self, n_layers: int = 2, dropout: float = 0.5) -> nn.LSTM:
        return nn.LSTM(
            input_size=self._d_size,
            hidden_size=self._hidden_size,
            proj_size=self._l_preds * self._d_size,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=False,
        )

    @classmethod
    def _to_tensor(cls, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array.astype(np.float32))

    def _transform_train_data_target(
            self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transforms given data into train and target sets. The sequential
        nature of the model requires targets to include windowed slices of
        data from consequent time steps.

        Ex. if
              data = [1,2,3,4,5,6,7], self._l_preds = 3
            then
              train = [1,2,3,4]
              targets = [[2,3,4],[3,4,5],[4,5,6],[5,6,7]]

        :param data:
        :return:
        """
        values = np.expand_dims(data, axis=0)
        train_data = values[:, :-self._l_preds, :]
        train_targets = []
        for i in range(self._l_preds-1):
            train_targets += [values[:, 1+i:-self._l_preds+i+1, :]]
        train_targets += [values[:, self._l_preds:, :]]
        train_targets = np.stack(train_targets, axis=3)

        train_data, train_targets = (
            self._to_tensor(train_data),
            self._to_tensor(train_targets),
        )
        return train_data, train_targets

    def _transform_eval_data_target(
            self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms given data into evaluation data input and targets.
        The evaluation is realised as computing reconstruction errors
        for a given point x(t) over all its reconstructions. Because of that
        evaluation can be only performed for points x(t), where t comes from:
        self._l_preds < t <= len(data)-self._l_preds

        :param data:
        :return:
        """
        values = np.expand_dims(data, axis=0)
        eval_data = values[:, :-self._l_preds, :]
        eval_target = values[:, self._l_preds:-self._l_preds+1, :]

        return eval_data, eval_target

    def get_errors(
            self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Computes reconstruction error of a point x(t) over consecutive
        reconstructions. All self._l_preds reconstructions are needed to
        construct error vector hence only a range of outputs contribute
        to the result.

        Ex. outputs = [
            [x2_1, x3_1, x4_1],
            [x3_2, x4_2, x5_2],
            [x4_3, x5_3, x6_3],
        ]
        Then error can be only computed for x4 using [x4_1, x4_2, x4_3].

        :param output:
        :param target:
        :return:
        """
        errors = []
        for i in range(self._l_preds-1):
            errors += [output[:, i:-self._l_preds+i+1, :, self._l_preds-1-i]]
        errors += [output[:, self._l_preds-1:, :, 0]]
        errors = np.stack(errors, axis=3)
        errors = errors - target[..., np.newaxis]
        return errors.squeeze(axis=0)

    def _reshape_output(self, output: torch.Tensor) -> torch.Tensor:
        """Model returns self._l_preds predicted values for each of self._d_size
        dimensions.

        :param output:
        :return:
        """
        d1, d2, _ = output.shape
        return output.reshape(d1, d2, self._d_size, self._l_preds)

    def train(
        self,
        train_data: pd.DataFrame,
        epochs: int = 50,
        learning_rate: float = 1e-4,
    ) -> None:
        """

        :param train_data:
        :param epochs:
        :param learning_rate:
        :return:
        """
        # Shape: (time_steps, d)
        data = train_data
        # Shape: (batch_size, time_steps-l, d), (batch_size, time_steps-l, d, l)
        train_data, train_targets = self._transform_train_data_target(data)
        self._d_size = train_data[0].shape[-1]
        self.model = self._get_lstm()
        # TODO: do test-eval split to estimate distribution on eval set
        self._train_model(train_data, train_targets, epochs, learning_rate)
        self._fit_error_distribution(data)

    def _fit_error_distribution(self, data: pd.DataFrame) -> None:
        # Shape: (time_steps-l, d), (time_steps-2*l, d)
        eval_data, eval_targets = self._transform_eval_data_target(data)
        self.model.eval()
        with torch.no_grad():
            # Shape: (batch_size, time_steps, d*l)
            outputs, _ = self.model(self._to_tensor(eval_data))
        # Shape: (batch_size, time_steps, d, l)
        outputs = self._reshape_output(outputs)
        # Shape: (time_steps, d, l)
        errors = self.get_errors(outputs.detach().numpy(), eval_targets)
        # Shape: (time_steps, d*l)
        errors = errors.reshape((errors.shape[0], self._l_preds * self._d_size))
        means, cov = self._fit_multivariate_gauss(errors)
        self._dist = functools.partial(
            scipy.stats.multivariate_normal.pdf,
            mean=means,
            cov=cov,
            allow_singular=True,
        )

    @classmethod
    def _fit_multivariate_gauss(
            cls, sample: np.ndarray,) -> Tuple[np.ndarray, np.ndarray]:
        """Fit multivariate gaussian distribution to a given sample using
        maximum likelihood estimation method.

        Source:
          https://stackoverflow.com/questions/27230824/fit-multivariate-gaussian-distribution-to-a-given-dataset

        :param sample:
        :return:
        """
        mean = np.mean(sample, axis=0)
        cov = np.cov(sample, rowvar=0)
        return mean, cov

    def _train_model(
        self, train_data: torch.Tensor, train_targets: torch.Tensor,
        epochs: int, learning_rate: float,
    ) -> None:

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs, _ = self.model(train_data)
            outputs = self._reshape_output(outputs)
            loss = F.mse_loss(outputs, train_targets)
            print(f"Epoch {epoch} loss: {loss.item()}")
            loss.backward()
            optimizer.step()

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self._dist is None:
            raise RuntimeError("Model not trained.")

        self.model.eval()
        inputs, targets = self._transform_eval_data_target(data)
        with torch.no_grad():
            outputs, _ = self.model(self._to_tensor(inputs))
        outputs = self._reshape_output(outputs)

        errors = self.get_errors(outputs.detach().numpy(), targets)
        errors = errors.reshape((errors.shape[0], self._l_preds * self._d_size))
        p = -self._dist(errors)
        scores = np.zeros((len(data),))
        scores[self._l_preds:-self._l_preds+1] = p

        return scores

    def detect(self, data: pd.DataFrame) -> np.ndarray:
        scores = self.predict(data)
        return (scores < self._threshold).astype(np.int32)


if __name__ == '__main__':
    data = pd.DataFrame(data=[1,2,3,4,5,6,7,8,9,10])
    data2 = pd.DataFrame(data=[
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
        [11, 12],
        [11, 12],
        [11, 12],
    ])

    from sops_anomaly.datasets import NabDataset

    data = NabDataset().get_train_samples(n_samples=1)
    # print(data)
    lstm = LSTM(l_predictions=3, threshold=0.05)
    lstm.train(data)
    a = lstm.predict(data)
    print(a[a<0])
    print(len(a), np.sum(a))

    # def transform_train_data_target(data: pd.DataFrame, l_preds: int):
    #     values = np.expand_dims(data, axis=0)
    #     train_data = values[:, :-l_preds, :]
    #     train_labels = []
    #     for l in range(l_preds-1):
    #         train_labels += [values[:, 1+l:-l_preds+l+1, :]]
    #     train_labels += [values[:, l_preds:, :]]
    #     train_labels = np.stack(train_labels, axis=3)
    #     return train_data, train_labels
    #
    # def transform_eval_data_target(data: pd.DataFrame, l_preds:int):
    #     values = np.expand_dims(data, axis=0)
    #     eval_data = values[:, :-l_preds, :]
    #     eval_target = values[:, l_preds:-l_preds+1, :]
    #
    #     return eval_data, eval_target
    #
    # def get_errors(output: np.ndarray, target: np.ndarray, l_preds: int):
    #     errors = []
    #     print(output.shape)
    #     for l in range(l_preds-1):
    #         errors += [output[:, l:-l_preds+l+1, :, l_preds-1-l]]
    #     errors += [output[:, l_preds-1:, :, 0]]
    #     errors = np.stack(errors, axis=3)
    #     print(errors - target[..., np.newaxis])
    #
    #
    # t_d, t_l = transform_train_data_target(data2, 3)
    # print(t_d.shape, t_l.shape)
    #
    # # e_d, e_t = transform_eval_data_target(data2, 3)
    # # print(e_d, e_t)
    # #
    # # get_errors(t_l, e_t, 3)
