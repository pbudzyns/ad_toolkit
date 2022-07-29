import math
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler


def build_layers(input_size, layers, output_size):
    if len(layers) > 0:
        input_layer = nn.Linear(input_size, layers[0])
        output_layer = nn.Linear(layers[-1], output_size)
    else:
        return [nn.Linear(input_size, output_size)]

    inner_layers = []
    if len(layers) > 1:
        inner_layers = [
            nn.Linear(layers[i - 1], layers[i])
            for i
            in range(1, len(layers))
        ]
    all_layers = [input_layer] + inner_layers + [output_layer]
    return all_layers


def build_network(layers: List[nn.Module]) -> nn.Sequential:
    network = []
    for layer in layers[:-1]:
        network.extend((
            layer,
            nn.ReLU(),
        ))
    network.append(layers[-1])
    return nn.Sequential(*network)


def get_data_loader(
    data: Union[List[torch.Tensor], List[np.ndarray]], batch_size: int,
    test: bool = False,
) -> DataLoader:
    if test:
        sampler = None
    else:
        indices = np.random.permutation(len(data))
        sampler = SubsetRandomSampler(indices)

    return DataLoader(
        dataset=data,
        batch_size=min(len(data), batch_size),
        drop_last=not test,
        sampler=sampler,
    )


def train_valid_split(
    data: Union[List[torch.Tensor], List[np.ndarray]],
    validation_portion: float, batch_size: int,
) -> Tuple[DataLoader, DataLoader]:

    split = math.ceil(validation_portion * len(data))
    train_data = data[:split]
    valid_data = data[split:]

    train_data_loader = get_data_loader(train_data, batch_size)
    valid_data_loader = get_data_loader(valid_data, batch_size)

    return train_data_loader, valid_data_loader
