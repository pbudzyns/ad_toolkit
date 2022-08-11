![CI status](https://github.com/pbudzyns/ad_toolkit/actions/workflows/python-package.yml/badge.svg)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/pbudzyns/46a4b56814f345bdc80420ad8bda204f/raw/coverage.json)
![Licence](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/pbudzyns/9315c025fb9e5bb6afe874a219d74fc2/raw/licence.json)
![Version](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/pbudzyns/6f89659da7879f763fa9f44310496ba8/raw/version.json)
# Deep Learning Algorithms for Anomaly Detection

A collection of anomaly detection algorithms and relevant datasets with a common
interface that allows for fast and easy prototyping and benchmarking.



## Installation
The simplest way to install the package is to use `pip`. By cloning the
repository:
```commandline
$ git colne https://github.com/pbudzyns/ad_toolkit.git
$ cd ad_toolkit
$ pip install .
```
or directly using `pip`:
```commandline
$ pip install git+https://github.com/pbudzyns/ad_toolkit.git
```

#### Extras
`donut` model requires extra (deprecated) dependencies and works only with older
versions of `python` that supports `tensorflow <= 1.15`. To install extra
dependencies use
```commandline
$ pip install .[donut]
```
or
```commandline
$ pip install "ad_toolkit[donut] @ git+https://github.com/pbudzyns/ad_toolkit.git"
```

## Example

#### Loading dataset
```python
from ad_toolkit.datasets import NabDataset

nab = NabDataset()
nab.plot()
```
![Time series](docs/nab_sample.png?raw=true "Time series")

#### Training a model
```python
from ad_toolkit.detectors import AutoEncoder

x_train, _ = nab.get_train_samples()

model = AutoEncoder(window_size=100, layers=(64,32,16), latent_size=8)
model.train(x_train, epochs=20, learning_rate=1e-4)

x, y = nab.get_test_samples()
scores = model.predict(x)
```

#### Evaluation
```python
import numpy as np

from ad_toolkit.evaluation import Result

labels = (scores > np.mean(scores)*2.2)

print(Result(labels, y))
nab.plot(anomalies={'ae': labels})

# Sample output:
# ... Result(accuracy=0.93,
# ...	(tp, fp, tn, fn)=(135, 0, 3629, 268),
# ...	precision=1.0,
# ...	recall=0.33,
# ...	f1=0.5,
# ...	roc_auc=0.67,
# ...	y_pred%=0.033482142857142856,
# ...	y_label%=0.09995039682539683,
# ... )
```
![Time series with anomalies](docs/nab_anomalies.png?raw=true
"Time series with anomalies")

## Package content

### Detectors

#### AutoEncoder with Time-Series Segmentation (``ad_toolkit.detectors.AutoEncoderTSS``)
Change-point detector searching for points of significant change of behaviour.

__References:__
1. _Lee, W. H., Ortiz, J., Ko, B., & Lee, R. (2018). Time series segmentation
   through automatic feature learning._

2. _Boumghar, R., Venkataswaran, A., Brown, H., & Crespo, X. Behaviour-based
   anomaly detection in spacecraft using deep learning._

#### AutoEncoder (``ad_toolkit.detectors.AutoEncoder``)
Detector based on auto-encoder model. Returns prediction score based on
reconstruction error. This model is capable of working with multivariate time
series.

__References:__
1. _An, J., & Cho, S. (2015). Variational autoencoder based anomaly detection
   using reconstruction probability._

#### Variational AutoEncoder (``ad_toolkit.detectors.VariationalAutoEncoder``)
Detector based on variational auto-encoder model trained with ELBO.
On prediction phase model generates `n` reconstructions of a data point and
returns resulting reconstruction probability.

__References:__
1. _An, J., & Cho, S. (2015). Variational autoencoder based anomaly detection
   using reconstruction probability._

#### LSTM_AD (``ad_toolkit.detectors.LSTM_AD``)
LSTM Anomaly Detection model. Trained on a task of predicting future values of
the time series given previous values. The prediction score is computed as a
probability of the prediction error coming from the multivariate error
distribution estimated with validation data.

__References:__
1. _Malhotra, P., Vig, L., Shroff, G., & Agarwal, P. (2015, April). Long
   short term memory networks for anomaly detection in time series._

#### LSTM_ED (``ad_toolkit.detectors.LSTM_ED``)
LSTM Encoder Decoder model. Trained on a task of reconstructing a window of
values of the time series. The prediction score is computed as a
probability of the reconstruction error coming from the multivariate error
distribution estimated with validation data.

__References:__
1. _Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P.,
   & Shroff, G. (2016). LSTM-based encoder-decoder for multi-sensor
   anomaly detection._

#### Donut (``ad_toolkit.detectors.Donut``)

__References:__
1. _Xu, H., Chen, W., Zhao, N., Li, Z., Bu, J., Li, Z., ... & Qiao, H.
   (2018, April). Unsupervised anomaly detection via variational
   auto-encoder for seasonal kpis in web applications._

### Datasets

#### Supervised Dataset (``ad_toolkit.datasets.SupervisedDataset``)
A wrapper class for semi-supervised learning. Capable of returning training sets
with anomalous rows filtered out or limited to a requested percentage.

#### MNIST (``ad_toolkit.datasets.MNIST``)
Collection of handwritten digits. http://yann.lecun.com/exdb/mnist/

#### KDD Cup Dataset (``ad_toolkit.datasets.KddCup``)
A dataset from competition task to build a network intrusion detector. https://archive.ics.uci.edu/ml/datasets/kdd+cup+1999+data

#### NAB Dataset (``ad_toolkit.datasets.NabDataset``)
A set of datasets coming from NAB Benchmark for anomaly detection. https://github.com/numenta/NAB

## Testing
```commandline
$ pip install .[test]
$ coverage run -m pytest .
```

## Lint
```commandline
$ pip install .[lint]
$ flake8
```
