import torch
from torch import float32


def function02(tensor: torch.Tensor):
    weights = torch.randn(tensor.shape[1]).to(float32)
    weights.requires_grad = True
    return weights
