import torch
from torch import nn


def count_parameters(layer: nn.Module):
    return sum(p.numel() for p in layer.parameters())