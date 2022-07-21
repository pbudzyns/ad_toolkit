"""
LSTM Anomaly Detector based on reconstruction error density.

References:
    - Malhotra, Pankaj, et al. "Long short term memory networks for anomaly
      detection in time series."
    - Implementation from DeepADoTS
      https://github.com/KDD-OpenSource/DeepADoTS/blob/master/src/algorithms/lstm_ad.py

"""
import functools
from typing import Optional, Tuple

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
        self,
        l_predictions: int = 10,
        hidden_size: int = 400,
        threshold: float = 0.9,
    ) -> None:
        """

        :param l_predictions:
        :param hidden_size:
        """
        super(LSTM_AD, self).__init__()
        self.model: Optional[nn.LSTM] = None
        self.linear: Optional[nn.Module] = None
        self._threshold: float = threshold
        self._hidden_size: int = hidden_size
        # Model output dimensions (l, d)
        self._l_preds: int = l_predictions
        self._d_size: int = 0
        # Multivariate gaussian scipy.stats.multivariate_gaussian
        self._error_dist = None

    def _initialize_model(
            self, n_layers: int = 2, dropout: float = 0.5) -> None:
        self.model = nn.LSTM(
            input_size=self._d_size,
            hidden_size=self._hidden_size,
            proj_size=self._l_preds * self._d_size,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=False,
        )
        self._error_dist = _ErrorDistribution(self._d_size, self._l_preds)

    def _reshape_outputs(self, output: torch.Tensor) -> torch.Tensor:
        """Model returns self._l_preds predicted values for each of self._d_size
        dimensions.

        :param output:
        :return:
        """
        d1, d2, _ = output.shape
        return output.reshape(d1, d2, self._d_size, self._l_preds)

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

    def train(
        self,
        train_data: pd.DataFrame,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        validation_steps: int = 10,
        epochs: int = 50,
        batch_size: int = 32,
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
        # Train eval data split.
        # Shape: (time_steps, d)
        train_df = train_data.sample(frac=0.8)
        eval_df = train_data.drop(index=train_df.index)

        # Initialize model.
        self._d_size = train_data.shape[-1]
        self._initialize_model()
        # Fit model to train data.
        self._fit_model(train_df, epochs, learning_rate, verbose)
        # Compute error distribution using eval data.
        self._fit_error_distribution(eval_df)

        if validation_data is not None:
            # Use validation data to compute optimal anomaly threshold.
            self._optimize_prediction_threshold(
                validation_data, validation_steps)

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
        self._error_dist.fit(outputs.detach().numpy(), eval_targets)

    def _train_model(
        self, train_data: torch.Tensor, train_targets: torch.Tensor,
        epochs: int, learning_rate: float, verbose: bool,
    ) -> None:

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self._run_train_loop(
            epochs, optimizer, train_data, train_targets, verbose)

    def _run_train_loop(
        self, epochs: int, optimizer: torch.optim.Optimizer,
        train_data: torch.Tensor, train_targets: torch.Tensor, verbose: bool,
    ) -> None:
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self._get_model_outputs(train_data)
            loss = F.mse_loss(outputs, train_targets)
            loss.backward()
            optimizer.step()
            if verbose:
                print(f"Epoch {epoch} loss: {loss.item()}")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        inputs, targets = self._transform_eval_data_target(data)
        outputs = self._get_model_outputs(self._to_tensor(inputs))
        errors = self._get_errors(outputs, targets)
        scores = self._get_scores(data, errors)

        return scores

    def _get_scores(self, data: pd.DataFrame, errors: np.ndarray) -> np.ndarray:
        p = self._error_dist(errors)
        scores = np.zeros((len(data),))
        scores[self._l_preds:-self._l_preds + 1] = p
        return scores

    def _get_errors(
            self, outputs: torch.Tensor, targets: np.ndarray) -> np.ndarray:
        errors = -self._error_dist.get_errors(outputs.detach().numpy(), targets)
        errors = errors.reshape((errors.shape[0], self._l_preds * self._d_size))
        return errors

    def _get_model_outputs(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.model.training:
            outputs, _ = self.model(inputs)
        else:
            with torch.no_grad():
                outputs, _ = self.model(inputs)
        return self._reshape_outputs(outputs)

    def detect(self, data: pd.DataFrame) -> np.ndarray:
        scores = self.predict(data)
        return (scores < self._threshold).astype(np.int32)


class _ErrorDistribution:

    def __init__(self, n_dims: int, l_preds: int) -> None:
        self._d_size: int = n_dims
        self._l_preds: int = l_preds
        self._dist: scipy.stats.multivariate_normal = None

    def __call__(self, errors: np.ndarray) -> np.ndarray:
        return self._dist(errors)

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

    def fit(self, outputs: np.ndarray, targets: np.ndarray) -> None:
        self._fit_error_distribution(outputs, targets)

    def _fit_error_distribution(
            self, outputs: np.ndarray, eval_targets: np.ndarray) -> None:
        # Shape: (batch_size, time_steps, d, l)
        # outputs = self._reshape_output(outputs)
        # Shape: (time_steps, d, l)
        errors = self.get_errors(outputs, eval_targets)
        # Shape: (time_steps, d*l)
        errors = errors.reshape((errors.shape[0], self._l_preds * self._d_size))
        means, cov = self._fit_multivariate_gauss(errors)
        self._dist = functools.partial(
            scipy.stats.multivariate_normal.logpdf,
            mean=means,
            cov=cov,
            allow_singular=True,
        )

    @classmethod
    def _fit_multivariate_gauss(
            cls, sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
