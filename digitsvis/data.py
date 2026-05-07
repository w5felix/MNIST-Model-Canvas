from typing import Callable, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Dataset registry
_DATASETS: Dict[str, Callable[..., Tuple[DataLoader, DataLoader]]] = {}


def register_dataset(name: str):
    """Decorator to register a dataset builder under a name.

    A dataset builder has signature: (root: str, batch_size: int, num_workers: int) -> (train_loader, test_loader)
    """

    def _decorator(fn: Callable[..., Tuple[DataLoader, DataLoader]]):
        _DATASETS[name.lower()] = fn
        return fn

    return _decorator


def get_dataloaders(dataset: str = "mnist", root: str = "data", batch_size: int = 64, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    """Return train/test dataloaders using a registered dataset builder.

    To add your own dataset, create a function matching the builder signature and decorate it with @register_dataset("name").
    Then call get_dataloaders(dataset="name", ...).
    """
    name = dataset.lower()
    if name not in _DATASETS:
        raise KeyError(f"Unknown dataset '{dataset}'. Available: {sorted(_DATASETS.keys())}")
    return _DATASETS[name](root=root, batch_size=batch_size, num_workers=num_workers)


@register_dataset("mnist")
def _build_mnist(root: str = "data", batch_size: int = 64, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
