from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from .pcam import PCAMDataset


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to create Train and Validation DataLoaders
    using pre-split H5 files.
    """
    data_cfg = config["data"]
    base_path = Path(data_cfg["data_path"])
    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 4)
    # train_transform = ...
    # val_transform = ...
    train_transform = transforms.Compose([])
    val_transform = transforms.Compose([])
    # TODO: Define Paths for X and Y (train and val)
    x_train_name = data_cfg.get("x_train", "camelyonpatch_level_2_split_train_x.h5")
    y_train_name = data_cfg.get("y_train", "camelyonpatch_level_2_split_train_y.h5")
    x_val_name   = data_cfg.get("x_val",   "camelyonpatch_level_2_split_valid_x.h5")
    y_val_name   = data_cfg.get("y_val",   "camelyonpatch_level_2_split_valid_y.h5")    
    x_train_path = base_path / x_train_name
    y_train_path = base_path / y_train_name
    x_val_path   = base_path / x_val_name
    y_val_path   = base_path / y_val_name
    # TODO: Instantiate PCAMDataset for train and val
    train_ds = PCAMDataset(
        x_path=str(x_train_path),
        y_path=str(y_train_path),
        transform=train_transform,
    )
    val_ds = PCAMDataset(
        x_path=str(x_val_path),
        y_path=str(y_val_path),
        transform=val_transform,
    )

    labels = []
    for _, y in train_ds:
        labels.append(y.item())
    labels = torch.tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    # TODO: Create DataLoaders
    # train_loader = ...
    # val_loader = ...
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
