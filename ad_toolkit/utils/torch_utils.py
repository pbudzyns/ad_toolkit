from typing import List

from torch import nn


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
