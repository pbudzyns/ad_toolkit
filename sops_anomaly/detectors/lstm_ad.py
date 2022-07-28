"""
LSTM Anomaly Detector based on reconstruction error density.

References:
    - Malhotra, Pankaj, et al. "Long short term memory networks for anomaly
      detection in time series."
    - Implementation from DeepADoTS
      https://github.com/KDD-OpenSource/DeepADoTS/blob/master/src/algorithms/lstm_ad.py

"""
from typing import Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.metrics
import torch
from torch import nn
import torch.nn.functional as F

from sops_anomaly.detectors.base_detector import BaseDetector


class LSTM_AD(BaseDetector):

    def __init__(
        self, window_size: int = 10, hidden_size: int = 32,
        threshold: float = 0.9, use_gpu: bool = False,
    ) -> None:
        """

        :param window_size:
        :param hidden_size:
        """
        super(LSTM_AD, self).__init__()
        self.model: Optional[_LSTM] = None
        self._threshold: float = threshold
        self._hidden_size: int = hidden_size
        # Model output dimensions (l, d)
        self._l_preds: int = window_size
        self._d_size: int = 0
        # Multivariate gaussian scipy.stats.multivariate_gaussian
        self._error_dist = None
        self._device: torch.device = torch.device(
            'cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

    def train(
        self,
        train_data: pd.DataFrame,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        validation_steps: int = 10,
        epochs: int = 50,
        learning_rate: float = 1e-4,
        verbose: bool = False,
    ) -> None:
        """

        :param train_data:
        :param validation_data:
        :param validation_steps:
        :param epochs:
        :param learning_rate:
        :param verbose:
        :return:
        """
        if len(train_data) > 1e4:
            warnings.warn(
                "Input very long, consider using train_with_slices.")
        # Train eval data split.
        # Shape: (time_steps, d)
        split = int(0.7 * len(train_data))
        train_df = train_data.iloc[:split, :]
        eval_df = train_data.iloc[split:, :]

        # Initialize model.
        self._d_size = train_data.shape[-1]
        self._initialize_model_if_needed()
        # Fit model to train data.
        self._fit_model(train_df, epochs, learning_rate, verbose)
        # Compute error distribution using eval data.
        self._fit_error_distribution(eval_df)

        if validation_data is not None:
            # Use validation data to compute optimal anomaly threshold.
            self._optimize_prediction_threshold(
                validation_data, validation_steps)

    def train_with_slices(
        self,
        train_data: pd.DataFrame,
        slice_len: int = 1000,
        # validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        # validation_steps: int = 10,
        epochs: int = 3,
        learning_rate: float = 1e-4,
        verbose: bool = False,
    ) -> None:
        """Workaround for very long time series."""
        errors = None
        # Initialize model.
        self._d_size = train_data.shape[-1]
        self._initialize_model_if_needed()
        for epoch in range(epochs):
            for i in range(len(train_data) // slice_len):
                data_slice = train_data[i*slice_len:(i+1)*slice_len]
                split = int(0.7 * len(data_slice))
                train_slice = data_slice[:split]
                eval_slice = data_slice[split:]

                # Fit model to train data.
                self._fit_model(train_slice, 1, learning_rate, verbose=verbose)

                eval_data, eval_targets = self._transform_eval_data_target(
                    eval_slice)
                self.model.eval()
                # Shape: (batch_size, time_steps, d, l)
                outputs = self._get_model_outputs(self._to_tensor(eval_data))
                error_slice = self._error_dist.get_errors(
                    outputs.cpu().detach().numpy(),
                    eval_targets,
                )
                if errors is None:
                    errors = error_slice
                else:
                    errors = np.concatenate((errors, error_slice))

        self._error_dist.fit_multivariate_gauss(errors)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores for data points.

        :param data:
        :return:
        """
        self.model.eval()
        inputs, targets = self._transform_eval_data_target(data)
        outputs = self._get_model_outputs(self._to_tensor(inputs))
        errors = self._get_errors(outputs, targets)
        scores = self._get_scores(data, errors)
        return scores

    def detect(self, data: pd.DataFrame) -> np.ndarray:
        """Detect anomalous data points in the provided data.

        :param data:
        :return:
        """
        scores = self.predict(data)
        return (scores < self._threshold).astype(np.int32)

    def _initialize_model_if_needed(self) -> None:
        if self.model is not None:
            return
        self.model = _LSTM(
            input_size=self._d_size,
            output_size=self._l_preds,
            hidden_size=self._hidden_size,
            device=self._device,
        )
        self.model.double()
        self._error_dist = _ErrorDistribution(self._d_size, self._l_preds)

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array.astype(np.float64)).to(self._device)

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
        self._l_preds < t <= len(data)

        :param data:
        :return:
        """
        values = np.expand_dims(data, axis=0)
        eval_data = values
        eval_target = values[:, self._l_preds:, :]
        return eval_data, eval_target

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

    def _fit_model(
        self, train_df: pd.DataFrame, epochs: int, learning_rate: float,
        verbose: bool,
    ) -> None:
        # Shape: (batch_size, time_steps-l, d), (batch_size, time_steps-l, d, l)
        train_data, train_targets = self._transform_train_data_target(train_df)
        self._train_model(train_data, train_targets, epochs, learning_rate,
                          verbose)

    def _fit_error_distribution(self, data: pd.DataFrame):
        # Shape: (time_steps-l, d), (time_steps-2*l, d)
        eval_data, eval_targets = self._transform_eval_data_target(data)
        self.model.eval()
        # Shape: (batch_size, time_steps, d, l)
        outputs = self._get_model_outputs(self._to_tensor(eval_data))
        self._error_dist.fit(outputs.cpu().detach().numpy(), eval_targets)

    def _train_model(
        self, train_data: torch.Tensor, train_targets: torch.Tensor,
        epochs: int, learning_rate: float, verbose: bool,
    ) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self._get_model_outputs(train_data)
            loss = F.mse_loss(outputs, train_targets)
            loss.backward()
            optimizer.step()
            if verbose:
                print(f"Epoch {epoch} loss: {loss.item()}")

    def _get_scores(self, data: pd.DataFrame, errors: np.ndarray) -> np.ndarray:
        p = self._error_dist(errors)
        scores = np.zeros((len(data),))
        scores[self._l_preds:] = p
        return scores

    def _get_errors(
            self, outputs: torch.Tensor, targets: np.ndarray) -> np.ndarray:
        errors = self._error_dist.get_errors(
            outputs.cpu().detach().numpy(), targets)
        errors = errors.reshape((errors.shape[0], self._l_preds * self._d_size))
        return errors

    def _get_model_outputs(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.model.training:
            outputs = self.model(inputs)
        else:
            with torch.no_grad():
                outputs = self.model(inputs)
        return outputs


class _LSTM(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, hidden_size: int,
        device: torch.device, batch_size: int = 1,
    ) -> None:
        super().__init__()
        self._device = device
        self._d_size = input_size
        self._l_preds = output_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lstm_layer_1 = nn.LSTMCell(
            input_size, self.hidden_size)
        self.lstm_layer_2 = nn.LSTMCell(
            self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(
            self.hidden_size, input_size * output_size)
        self.to(self._device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = []
        h_t = torch.zeros(
            self.batch_size, self.hidden_size).double().to(self._device)
        c_t = torch.zeros(
            self.batch_size, self.hidden_size).double().to(self._device)
        h_t2 = torch.zeros(
            self.batch_size, self.hidden_size).double().to(self._device)
        c_t2 = torch.zeros(
            self.batch_size, self.hidden_size).double().to(self._device)

        for input_t in inputs.chunk(inputs.size(1), dim=1):
            h_t, c_t = self.lstm_layer_1(input_t.squeeze(dim=1), (h_t, c_t))
            h_t2, c_t2 = self.lstm_layer_2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze()

        return outputs.view(
            inputs.size(0), inputs.size(1), self._d_size, self._l_preds)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs)


class _ErrorDistribution:

    def __init__(self, n_dims: int, l_preds: int) -> None:
        self._d_size: int = n_dims
        self._l_preds: int = l_preds
        self._dist: scipy.stats.multivariate_normal = None
        self.means = None
        self.cov = None

    def __call__(self, errors: np.ndarray) -> np.ndarray:
        return -scipy.stats.multivariate_normal.logpdf(
            errors,
            mean=self.means,
            cov=self.cov,
            allow_singular=True,
        )

    def get_errors(
            self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        # """Computes reconstruction error of a point x(t) over consecutive
        # reconstructions. All self._l_preds reconstructions are needed to
        # construct error vector hence only a range of outputs contribute
        # to the result.
        #
        # Ex. outputs = [
        #     [x2_1, x3_1, x4_1],
        #     [x3_2, x4_2, x5_2],
        #     [x4_3, x5_3, x6_3],
        # ]
        # Then error can be only computed for x4 using [x4_1, x4_2, x4_3].
        #
        # :param output:
        # :param target:
        # :return:
        # """
        errors = [output[:, self._l_preds - 1:-1, :, 0]]
        for i in range(1, self._l_preds):
            errors += [output[:, self._l_preds - 1 - i:-1 - i, :, i]]
        errors = np.stack(errors, axis=3)
        errors = target[..., np.newaxis] - errors
        # Shape: (batch_size, time_steps, d, l)
        errors = errors.squeeze(axis=0)
        # Shape: (time_steps, d*l)
        errors = errors.reshape((errors.shape[0], self._l_preds * self._d_size))
        return errors

    def fit(self, outputs: np.ndarray, targets: np.ndarray) -> None:
        self._fit_error_distribution(outputs, targets)

    def _fit_error_distribution(
            self, outputs: np.ndarray, eval_targets: np.ndarray) -> None:
        errors = self.get_errors(outputs, eval_targets)
        self.fit_multivariate_gauss(errors)

    def fit_multivariate_gauss(self, sample: np.ndarray) -> None:
        """Fit multivariate gaussian distribution to a given sample using
        maximum likelihood estimation method.

        Source:
          https://stackoverflow.com/questions/27230824/fit-multivariate-gaussian-distribution-to-a-given-dataset

        :param sample:
        :return:
        """
        mean = np.mean(sample, axis=0)
        cov = np.cov(sample, rowvar=False)
        self.means, self.cov = mean, cov
