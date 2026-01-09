from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import transforms

from .pcam import PCAMDataset


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to create Train and Validation DataLoaders
    using pre-split H5 files.
    """
    data_cfg = config["data"]
    base_path = Path(data_cfg["data_path"])

    # train_transform = ...
    # val_transform = ...
    train_transform = transforms.Compose([])
    val_transform = transforms.Compose([])
    # TODO: Define Paths for X and Y (train and val)
    x_train_path = base_path / data_cfg["x_train"]
    y_train_path = base_path / data_cfg["y_train"]
    x_val_path = base_path / data_cfg["x_val"]
    y_val_path = base_path / data_cfg["y_val"]
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
