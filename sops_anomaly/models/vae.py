from typing import Union, List

import numpy as np
from torch import nn
from torch.nn import functional as F
import torch.distributions
import torch.optim

from sops_anomaly.models.model import BaseDetector


class _VAE(nn.Module):
    """Vanilla Variational Auto Encoder implementation with ELBO training.

     TODO: allow for batch-processing
    """
    def __init__(self, input_size: int, latent_size: int):
        super(_VAE, self).__init__()
        self._input_size = input_size
        self._latent_size = latent_size
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()
        self.prior = torch.distributions.Normal(0, 1)

        self.layer_mu = nn.Linear(100, latent_size)
        self.layer_sigma = nn.Linear(100, latent_size)

        self.decoder_input = nn.Linear(latent_size, 100)

    def _get_encoder(self):
        encoder = nn.Sequential(
            nn.Linear(self._input_size, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 100)
        )
        return encoder

    def _get_decoder(self):
        decoder = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 500),
            nn.ReLU(),
            nn.Linear(500, self._input_size)
        )
        return decoder

    def encode(self, x):
        encoded = self.encoder(x)

        mu = self.layer_mu(encoded)
        log_var = self.layer_sigma(encoded)

        return mu, log_var

    def decode(self, x):
        x = self.decoder_input(x)
        return self.decoder(x)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, x, mu, log_var

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss_function(self, model_outputs):
        (x_hat, x, mu, log_var) = model_outputs

        reconstruction_loss = F.mse_loss(x_hat, x)
        # TODO: batch processing loss definition
        # kld_loss = torch.mean(
        #     -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1),
        #     dim=0)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())

        total_loss = reconstruction_loss + kl_loss
        return total_loss

    def sample(self):
        z = torch.randn(self._latent_size)
        smpl = self.decode(z)
        return smpl


class VariationalAutoEncoder(BaseDetector):

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        l_samples: int = 10,
        threshold: float = 0.9,
    ):
        """
        Variational Auto-Encoder based anomaly detector. Detects anomalies
        based on reconstruction probability score.

        :param input_size:
        :param latent_size:
        :param l_samples:
        :param threshold:
        """
        super(VariationalAutoEncoder, self).__init__()
        self._l_samples = l_samples
        self._threshold = threshold
        self.model = _VAE(input_size, latent_size)

    def train(
        self,
        train_data: List[torch.Tensor],
        epochs: int = 30,
        # batch_size: int = 64,
        learning_rate: float = 1e-4,
    ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            epoch_loss = 0
            for sample in train_data:
                results = self.model.forward(sample)
                loss = self.model.loss_function(results)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(epoch_loss / len(train_data))

    def sample(self):
        return self.model.sample()

    def predict(self, x: torch.Tensor) -> float:
        mu, log_var = self.model.encode(x)
        score = 0
        for i in range(self._l_samples):
            z = self.model.reparametrize(mu, log_var)
            x_hat = self.model.decode(z)
            score += F.mse_loss(x_hat, x).item()
        return score / self._l_samples

    def detect(self, data: np.ndarray) -> Union[List[int], np.ndarray]:
        scores = []
        for sample in data:
            score = self.predict(sample)
            scores.append(score)
        mean_score = np.mean(scores)
        return np.where(np.array(scores) > mean_score * self._threshold)


if __name__ == '__main__':
    from sops_anomaly.datasets import MNIST
    mnist = MNIST(anomaly_class=7)
    x, y = mnist.get_test_samples(n_samples=1000)

    # model = VariationalAutoEncoder(input_size=MNIST.sample_size(), latent_size=10)
    # model.train(x)
    # batch_size = 10
    # print(x[0])
    # print(torch.Tensor(x[0]))
    x = [torch.Tensor(s) for s in x[-100:]]
    # x = torch.Tensor(x)

    vae = VariationalAutoEncoder(input_size=MNIST.sample_size(), latent_size=10, l_samples=10)
    vae.train(x)
    sample = vae.sample()

    import matplotlib.pyplot as plt
    # plt.imshow(sample.detach().numpy().reshape((28,28)), cmap='gray_r')
    # plt.show()
    score = 0
    score_anomaly = 0
    for xx in x[:10]:
        score += vae.predict(xx)
    an, _ = MNIST(anomaly_class=3).get_test_samples(n_samples=100)
    for xx in an[-10:]:
        score_anomaly += vae.predict(torch.Tensor(xx))

    print("Normal:", score / 10)
    print("Anomalous: ", score_anomaly / 10)
    plt.subplot(1,2,1)
    plt.imshow(x[3].detach().numpy().reshape((28,28)), cmap='gray_r')
    plt.subplot(1,2,2)
    plt.imshow(an[-1].reshape((28, 28)), cmap='gray_r')
    plt.show()
