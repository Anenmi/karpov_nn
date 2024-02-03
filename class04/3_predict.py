import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_pred = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        y_batch_pred = torch.argmax(output, dim=1)
        y_pred.append(y_batch_pred.cpu())
    return torch.cat(y_pred)
