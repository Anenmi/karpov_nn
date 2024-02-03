import torch
import torchvision.transforms as T


def get_augmentations(train: bool = True) -> T.Compose:

    if train:
        transforms = T.Compose(
            [
                T.Resize(size=(224, 224)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAdjustSharpness(sharpness_factor=2),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    else:
        transforms = T.Compose(
            [
                T.Resize(size=(224, 224)),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    return transforms
