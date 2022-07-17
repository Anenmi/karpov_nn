import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Optimizer
from torch import nn


def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn):
    history = []
    model.eval()
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.view(x_batch.shape[0], -1)
        with torch.set_grad_enabled(False):
            y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        # print(round(loss.item(),5))
        history.append(loss.item())
    return sum(history) / len(history)
