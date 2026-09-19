"""Data loading for MNIST, held-out splits, and Permuted-MNIST drift."""

from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import torchvision
import torchvision.transforms as transforms


def get_default_data_dir() -> Path:
    """Return default data directory path."""
    return Path(__file__).resolve().parent.parent / "data"


def load_mnist_datasets(
    data_dir: Union[str, Path, None] = None,
    b_train_size: int = 50000,
    held_out_size: int = 10000,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset, int]:
    """Load MNIST and create:
    1. b_train_dataset (50,000 samples for Model B & Observer training)
    2. held_out_dataset (10,000 samples B never saw, for probe/observer-head training)
    3. test_dataset (10,000 test samples for final evaluation)

    Returns:
        (b_train_dataset, held_out_dataset, test_dataset, input_dim)
    """
    target_dir = Path(data_dir) if data_dir else get_default_data_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ])
        full_train = torchvision.datasets.MNIST(
            root=str(target_dir), train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=str(target_dir), train=False, download=True, transform=transform
        )

        total_train = len(full_train)
        indices = np.random.RandomState(seed).permutation(total_train)

        b_train_indices = indices[:b_train_size]
        held_out_indices = indices[b_train_size : b_train_size + held_out_size]

        b_train_dataset = Subset(full_train, b_train_indices)
        held_out_dataset = Subset(full_train, held_out_indices)

        return b_train_dataset, held_out_dataset, test_dataset, 784

    except Exception as e:
        print(f"Torchvision MNIST fallback to sklearn digits: {e}")
        digits = load_digits()
        x_norm = digits.data / 16.0
        y_targets = digits.target
        x_tr, x_te, y_tr, y_te = train_test_split(
            x_norm, y_targets, test_size=0.2, random_state=seed, stratify=y_targets
        )
        x_b_tr, x_held, y_b_tr, y_held = train_test_split(
            x_tr, y_tr, test_size=0.3, random_state=seed, stratify=y_tr
        )

        b_train_dataset = TensorDataset(torch.from_numpy(x_b_tr).float(), torch.from_numpy(y_b_tr).long())
        held_out_dataset = TensorDataset(torch.from_numpy(x_held).float(), torch.from_numpy(y_held).long())
        test_dataset = TensorDataset(torch.from_numpy(x_te).float(), torch.from_numpy(y_te).long())
        return b_train_dataset, held_out_dataset, test_dataset, 64


class PermutedDataset(Dataset):
    """Dataset with fixed pixel permutation applied to inputs."""

    def __init__(self, base_dataset: Dataset, perm_indices: torch.Tensor):
        self.base_dataset = base_dataset
        self.perm_indices = perm_indices

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        x, y = self.base_dataset[idx]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        flat_x = torch.flatten(x)
        return flat_x[self.perm_indices], y


def get_permuted_dataset(
    base_dataset: Dataset,
    permutation_seed: int,
    input_dim: int = 784,
) -> Dataset:
    """Create a dataset with a deterministic pixel permutation."""
    rng = np.random.RandomState(permutation_seed)
    perm_indices = torch.from_numpy(rng.permutation(input_dim)).long()
    return PermutedDataset(base_dataset, perm_indices)
