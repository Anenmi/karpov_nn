import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def predict_tta(
    model: nn.Module, loader: DataLoader, device: torch.device, iterations: int = 2
):
    outputs = []
    model.eval()
    for i in range(iterations):
        iter_output = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            batch_output = model(x)
            iter_output.append(batch_output.cpu())
        iter_output = torch.cat(iter_output)
        outputs.append(iter_output)
    outputs = torch.stack(outputs)
    outputs = torch.mean(outputs, 0)
    y_pred = torch.argmax(outputs, dim=1)
    return y_pred
