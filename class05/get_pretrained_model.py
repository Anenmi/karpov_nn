from torchvision.models import alexnet
from torchvision.models import vgg11
from torchvision.models import googlenet
from torchvision.models import resnet18
import torch.nn as nn


def get_pretrained_model(model_name: str, num_classes: int, pretrained: bool = True):
    if model_name == "alexnet":
        model = alexnet(pretrained=pretrained)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

    if model_name == "resnet18":
        model = resnet18(pretrained=pretrained)
        model.fc = nn.Linear(in_features=512, out_features=num_classes)

    if model_name == "vgg11":
        model = vgg11(pretrained=pretrained)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

    if model_name == "googlenet":
        model = googlenet(pretrained=pretrained)
        model.fc = nn.Linear(in_features=1024, out_features=num_classes)
        model.aux1.fc2 = nn.Linear(in_features=1024, out_features=num_classes)
        model.aux2.fc2 = nn.Linear(in_features=1024, out_features=num_classes)

    return model
