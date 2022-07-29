"""
Variational auto-encoder with MCMC.

References:
    - Xu, Haowen, et al. "Unsupervised anomaly detection via variational
      auto-encoder for seasonal kpis in web applications."
    - Implementation from DeepADoTS
     https://github.com/KDD-OpenSource/DeepADoTS/blob/master/src/algorithms/donut.py

"""
from typing import Any, List, Tuple, Union

from donut import (
    Donut as _Donut,
    DonutTrainer as _DonutTrainer,
    DonutPredictor as _DonutPredictor,
)
from donut.preprocessing import complete_timestamp, standardize_kpi
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tfsnippet.modules import Sequential

from ad_toolkit.detectors.base_detector import BaseDetector


class Donut(BaseDetector):

    def __init__(
        self, window_size: int = 120, latent_size: int = 10,
        layers: Union[List[int], Tuple[int]] = (100, 100),
    ) -> None:
        self._x_dim: int = window_size
        self._latent_size: int = latent_size
        self._layers: Union[List[int], Tuple[int]] = layers
        self._means: List[float] = []
        self._stds: List[float] = []
        self._models: List[_Donut] = []
        self._sessions: List[tf.Session] = []
        self._predictors: List[_DonutPredictor] = []
        self._model_counter = 0

    def __del__(self):
        for sess in self._sessions:
            sess.close()

    def train(self, train_data: pd.DataFrame, epochs: int = 30):
        # train_data = window_data(train_data, self._window_size)
        timestamp = np.array(train_data.index)
        for _, column in train_data.items():
            values = np.array(column)
            labels = np.zeros_like(values, dtype=np.int32)
            timestamp, missing, (values, labels) = complete_timestamp(
                timestamp, (values, labels))
            train_values, mean, std = standardize_kpi(
                values, excludes=np.logical_or(labels, missing))

            session = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True))
            model, model_vs = self._build_model()
            trainer = _DonutTrainer(
                model=model,
                model_vs=model_vs,
                max_epoch=epochs,
                missing_data_injection_rate=0.0,
            )
            with session.as_default():
                trainer.fit(train_values, labels, missing, mean, std,
                            valid_portion=0.25)

            self._models.append(model)
            self._means.append(mean)
            self._stds.append(std)
            self._sessions.append(session)

    def _build_model(self) -> Tuple[_Donut, Any]:
        with tf.variable_scope('model') as model_vs:
            encoder_layers = [
                layers.Dense(
                    layer_size,
                    kernel_regularizer=keras.regularizers.l2(0.001),
                    activation=tf.nn.relu,
                )
                for layer_size
                in self._layers
            ]
            decoder_layers = [
                layers.Dense(
                    layer_size,
                    kernel_regularizer=keras.regularizers.l2(0.001),
                    activation=tf.nn.relu,
                )
                for layer_size
                in self._layers[::-1]
            ]
            model = _Donut(
                h_for_p_x=Sequential(encoder_layers),
                h_for_q_z=Sequential(decoder_layers),
                x_dims=self._x_dim,
                z_dims=self._latent_size,
            )
        return model, model_vs

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        prediction_scores = np.ones((len(data),)) * (-1)
        # results = []
        for i, (_, column) in enumerate(data.items()):
            values = np.array(column)
            mean, std = self._means[i], self._stds[i]
            values, _, _ = standardize_kpi(values, mean=mean, std=std)
            missing = np.zeros_like(values, dtype=np.int32)
            session = self._sessions[i]
            predictor = _DonutPredictor(model=self._models[i])
            with session.as_default():
                scores = predictor.get_score(values, missing)
            scores = np.exp(scores)
            # results.append(scores)

        prediction_scores[-len(scores):] = scores
        return prediction_scores

    def detect(self, data: pd.DataFrame) -> np.ndarray:
        # TODO: implement detection, check in the paper how is it done
        pass
