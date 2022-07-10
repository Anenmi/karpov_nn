import torch
from torch import nn


def function04(x: torch.Tensor, y: torch.Tensor):
    model = torch.nn.Linear(x.size()[1], 1, bias=False)
    learning_rate = 1e-2
    for i in range(10000):
        y_pred = model(x).flatten()
        loss = torch.mean((y_pred - y) ** 2)
        loss.backward()
        with torch.no_grad():
            model.weight -= model.weight.grad * learning_rate
        model.weight.grad.zero_()
        if loss < 0.3:
            break
    return model
