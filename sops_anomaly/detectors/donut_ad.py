"""
Variational auto-encoder with MCMC.

"""
from typing import Any, List, Tuple

from donut import (
    Donut as _Donut,
    DonutTrainer as _DonutTrainer,
    DonutPredictor as _DonutPredictor)
from donut.preprocessing import complete_timestamp, standardize_kpi
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tfsnippet.modules import Sequential

from sops_anomaly.detectors.base_detector import BaseDetector
# from sops_anomaly.utils import window_data


class Donut(BaseDetector):

    def __init__(self, window_size: int, latent_size: int = 10) -> None:
        self._window_size: int = window_size
        self._latent_size: int = latent_size
        self._models: List[_Donut] = []
        self._sessions: List[tf.Session] = []
        self._predictors: List[_DonutPredictor] = []
        self._model_counter = 0

    def train(self, train_data: pd.DataFrame, epochs: int = 30):
        # train_data = window_data(train_data, self._window_size)
        timestamp = np.array(train_data.index)
        for _, column in train_data.items():
            values = np.array(column)
            labels = np.zeros_like(values, dtype=np.int32)
            timestamp, missing, (values, labels) = complete_timestamp(timestamp,
                                                                      (values,
                                                                       labels))
            train_values, mean, std = standardize_kpi(
                values, excludes=np.logical_or(labels, missing))

            model, model_vs = self._build_model()
            trainer = _DonutTrainer(
                model=model, model_vs=model_vs, max_epoch=epochs)
            predictor = _DonutPredictor(model)
            session = tf.Session()

            with session.as_default():
                trainer.fit(train_values, labels, missing, mean, std)

            self._models.append(model)
            self._predictors.append(predictor)
            self._sessions.append(session)

    def _build_model(self) -> Tuple[_Donut, Any]:
        with tf.variable_scope(f'model{self._model_counter}') as model_vs:
            model = _Donut(
                h_for_p_x=Sequential([
                    layers.Dense(100,
                                 kernel_regularizer=keras.regularizers.l2(0.001),
                                 activation=tf.nn.relu),
                    layers.Dense(100,
                                 kernel_regularizer=keras.regularizers.l2(0.001),
                                 activation=tf.nn.relu),
                ]),
                h_for_q_z=Sequential([
                    layers.Dense(100,
                                 kernel_regularizer=keras.regularizers.l2(0.001),
                                 activation=tf.nn.relu),
                    layers.Dense(100,
                                 kernel_regularizer=keras.regularizers.l2(0.001),
                                 activation=tf.nn.relu),
                ]),
                x_dims=self._window_size,
                z_dims=self._latent_size,
            )
        self._model_counter += 1
        return model, model_vs

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        # data = window_data(data, self._window_size)
        timestamp = np.array(data.index)
        results = []
        for i, (_, column) in enumerate(data.items()):
            values = np.array(column)
            labels = np.zeros_like(values, dtype=np.int32)
            timestamp, missing, (values, labels) = complete_timestamp(timestamp,
                                                                      (values,
                                                                       labels))
            session = self._sessions[i]
            with session.as_default():
                scores = self._predictors[i].get_score(values, missing)
            results.append(scores)

        return np.mean(results, axis=0)

    def detect(self, data: pd.DataFrame) -> np.ndarray:
        # TODO: implement detection, check in the paper how is it done
        pass


if __name__ == '__main__':
    from sops_anomaly.datasets.nab_samples import NabDataset

    dataset = NabDataset().data
    dataset['value2'] = dataset['value']
    model = Donut()
    model.train(dataset)

    p = model.predict(dataset)
    print(p)
