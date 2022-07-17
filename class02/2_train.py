import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Optimizer
from torch import nn


def train(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn):
    model.train()
    history = []
    for x_batch, y_batch in data_loader:
        optimizer.zero_grad()
        x_batch = x_batch.view(x_batch.shape[0], -1)
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        # print(round(loss.item(),5))
        history.append(loss.item())
        loss.backward()
        optimizer.step()
    return sum(history) / len(history)
