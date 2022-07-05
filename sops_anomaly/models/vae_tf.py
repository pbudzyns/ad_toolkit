"""
Variational auto-encoder implementation.

Reference:
    - https://keras.io/examples/generative/vae/
    - "Variational autoencoder based anomaly detection using reconstruction
    probability" J.An, S.Cho.

"""
from typing import Tuple, Union, List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sops_anomaly.models.base_model import BaseDetector


class _SamplingLayer(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __call__(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        # dim2 = tf.shape(z_mean)
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim, 2))
        # print(z_mean.shape, z_log_var.shape, epsilon.shape)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class _VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(_VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (
                        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class VariationalAutoEncoder(BaseDetector):

    def __init__(self, input_size: Tuple[int, int], latent_dim: int = 2):
        self._input_size = input_size
        self._latent_dim = latent_dim
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()
        self.model = _VAE(self.encoder, self.decoder)

    def _get_encoder(self) -> keras.Model:
        encoder_inputs = keras.Input(shape=self._input_size)
        # x = layers.Conv1D(32, 3, activation="relu", strides=2, padding="same")(
        #     encoder_inputs)
        # x = layers.Conv1D(64, 3, activation="relu", strides=2, padding="same")(
        #     x)
        # x = layers.Flatten()(x)
        x = layers.Dense(500, activation="relu")(encoder_inputs)
        x = layers.Dense(200, activation="relu")(x)
        # x = layers.Dense(16, activation="relu")(x)

        z_mean = layers.Dense(self._latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self._latent_dim, name="z_log_var")(x)
        z = _SamplingLayer()([z_mean, z_log_var])

        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z],
                              name="encoder")

        return encoder

    def _get_decoder(self) -> keras.Model:
        latent_inputs = keras.Input(shape=(self._latent_dim,))
        # x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        # x = layers.Reshape((7, 7, 64))(x)
        # x = layers.Conv1DTranspose(64, 3, activation="relu", strides=2,
        #                            padding="same")(x)
        # x = layers.Conv1DTranspose(32, 3, activation="relu", strides=2,
        #                            padding="same")(x)
        # decoder_outputs = layers.Conv1DTranspose(1, 3, activation="sigmoid",
        #                                          padding="same")(x)
        x = layers.Dense(200, activation="relu")(latent_inputs)
        x = layers.Dense(500, activation="relu")(x)
        decoder_outputs = layers.Dense(self._input_size[0], activation="sigmoid")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        return decoder

    def train(self, train_data: np.ndarray, epochs: int = 50):
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
        history = self.model.fit(train_data, epochs=epochs, batch_size=128)
        return history

    def predict(self, data: np.ndarray) -> np.ndarray:
        pass

    def detect(self, data: np.ndarray) -> Union[List[int], np.ndarray]:
        pass


if __name__ == '__main__':
    from sops_anomaly.datasets import MNIST
    mnist = MNIST()
    x = mnist.get_train_samples(n_samples=100)

    vae = VariationalAutoEncoder(input_size=(MNIST.sample_size(), 1))
    vae.train(x)

    encoder_output = vae.encoder.predict(x[:2])
    print(encoder_output)
