import datetime

import numpy as np
import pandas as pd
import pytest
from torch import nn

from ad_toolkit.utils import window_data
from ad_toolkit.utils.torch_utils import build_linear_layers, build_network


def time_stamps(n_steps):
    now = datetime.datetime.now()
    stamps = [now]
    for i in range(1, n_steps):
        stamps += [now + datetime.timedelta(seconds=5*i)]
    return stamps


@pytest.mark.parametrize("data,window_size,expected_output", (
    (
        pd.DataFrame(),
        10,
        pd.DataFrame(),
    ),
    (
        pd.DataFrame(data=[1, 2, 3]),
        5,
        pd.DataFrame(),
    ),
    (
        pd.DataFrame(data=[[1, 2, 3, 4, 5]]),
        1,
        pd.DataFrame(data=[[1, 2, 3, 4, 5]]),
    ),
    (
        pd.DataFrame(data=[1, 2, 3, 4]),
        2,
        pd.DataFrame(data=[[1, 2], [2, 3], [3, 4]], index=[1, 2, 3]),
    ),
    (
        pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8]]),
        3,
        pd.DataFrame(data=[[1, 2, 3, 4, 5, 6], [3, 4, 5, 6, 7, 8]],
                     index=[2, 3]),
    ),
))
def test_window_data_expected_output(data, window_size, expected_output):
    windowed_data = window_data(data, window_size)
    assert windowed_data.compare(expected_output).empty


@pytest.mark.parametrize("data", (
    pd.DataFrame(data=np.random.random((10,))),
    pd.DataFrame(data=np.random.random((10, 1))),
    pd.DataFrame(data=np.random.random((10, 10)),
                 index=time_stamps(10)),
    pd.DataFrame(data=np.random.random((200, 7)),
                 index=time_stamps(200)),
    pd.DataFrame(data=np.random.random((500, 765))),
))
@pytest.mark.parametrize("window_size", (1, 3, 10))
def test_window_data_output_shape(data, window_size):
    windowed_data = window_data(data, window_size)
    length = len(data) - window_size + 1
    width = data.shape[1] * window_size
    assert windowed_data.shape == (length, width)
    assert np.all(windowed_data.index == data.index[window_size-1:])


@pytest.mark.parametrize("inputs,inner,outputs", (
    (1, (), 2),
    (10, (20, ), 10),
    (10, (20, 30, 40), 10),
))
def test_build_layers(inputs, inner, outputs):
    layers = build_linear_layers(inputs, inner, outputs)
    expected_sizes = [inputs] + list(inner) + [outputs]
    for i, layer in enumerate(layers):
        assert layer.in_features == expected_sizes[i]
        assert layer.out_features == expected_sizes[i+1]


@pytest.mark.parametrize("layers", (
    (nn.Linear(100, 100), ),
    (nn.Linear(100, 200), nn.Linear(200, 100)),
    (
        nn.Linear(100, 200), nn.Linear(200, 300), nn.Linear(300, 200),
        nn.Linear(200, 100),
    ),
))
def test_build_network(layers):
    network = build_network(layers)
    linear_layers = [network[i] for i in range(0, len(network), 2)]
    activations = [network[i] for i in range(1, len(network), 2)]
    assert all(isinstance(act, nn.ReLU) for act in activations)
    assert all(
        l.in_features == ll.in_features and l.out_features == ll.out_features
        for (l, ll)
        in zip(linear_layers, layers)
    )
