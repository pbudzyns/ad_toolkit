import math
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch import nn


def build_linear_layers(
    input_size: int, layer_sizes: Union[Tuple[int], List[int]],
    output_size: int,
) -> List[nn.Module]:
    """Translates given sizes into a list of ``nn.Linear`` objects.

    Parameters
    ----------
    input_size
        Size of the input layer.
    layer_sizes
        Sizes of the inner layers.
    output_size
        Size of the output layer.

    Returns
    -------
    List[nn.Module]
        List of linear layers.
    """
    if len(layer_sizes) > 0:
        input_layer = nn.Linear(input_size, layer_sizes[0])
        output_layer = nn.Linear(layer_sizes[-1], output_size)
    else:
        return [nn.Linear(input_size, output_size)]

    inner_layers = []
    if len(layer_sizes) > 1:
        inner_layers = [
            nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            for i
            in range(1, len(layer_sizes))
        ]
    all_layers = [input_layer] + inner_layers + [output_layer]
    return all_layers


def build_network(layers: List[nn.Module]) -> nn.Sequential:
    """Translates a list of ``nn.Module`` objects into a
    ``nn.Sequential`` model with ``nn.ReLU`` as an activation function
     between them.

    Parameters
    ----------
    layers
        List of ``nn.Module`` objects.

    Returns
    -------
    nn.Sequential
        Sequential model consisting of provided layers.
    """
    network = []
    for layer in layers[:-1]:
        network.extend((
            layer,
            nn.ReLU(),
        ))
    network.append(layers[-1])
    return nn.Sequential(*network)


def dataframe_to_tensors(data: pd.DataFrame) -> List[torch.Tensor]:
    """Transforms data rows from the ``pd.DataFrame`` into ``torch.Tensor``
    objects.

    Parameters
    ----------
    data
        ``pd.DataFrame`` with data rows.

    Returns
    -------
    List[torch.Tensor]
        List of data tensors.
    """
    tensors = [torch.Tensor(row) for _, row in data.iterrows()]
    return tensors


def get_data_loader(
    data: Union[List[torch.Tensor], List[np.ndarray]], batch_size: int,
    prediction: bool = False,
) -> torch.utils.data.DataLoader:
    """Creates ``torch.utils.data.DataLoader`` object as a data provider
    with desired parameters. Data loader for training purposes would return
    batches of equal sizes in random order. While `prediction` is `True`
    all data is being returned in fixed order.

    Parameters
    ----------
    data
        Data to be included in the data loader.
    batch_size
        Batch size to be used.
    prediction
        Whether data loader is used for prediction or training.

    Returns
    -------
    torch.utils.data.DataLoader
    """
    if prediction:
        sampler = None
    else:
        indices = np.random.permutation(len(data))
        sampler = torch.utils.data.SubsetRandomSampler(indices)

    return torch.utils.data.DataLoader(
        dataset=data,
        batch_size=min(len(data), batch_size),
        drop_last=not prediction,
        sampler=sampler,
    )


def train_valid_data_loaders(
    data: Union[List[torch.Tensor], List[np.ndarray]],
    validation_portion: float, batch_size: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Splits data into training and validation parts and returns
    corresponding data loaders.

    Parameters
    ----------
    data
        Data tensors or arrays to be used.
    validation_portion
        Percent of data to be used for validation purposes.
    batch_size
        Batch size to be used.

    Returns
    -------
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        Tuple with train data loader and validation data loader.
    """
    split = math.ceil(validation_portion * len(data))
    train_data = data[:split]
    valid_data = data[split:]

    train_data_loader = get_data_loader(train_data, batch_size)
    valid_data_loader = get_data_loader(valid_data, batch_size)

    return train_data_loader, valid_data_loader
