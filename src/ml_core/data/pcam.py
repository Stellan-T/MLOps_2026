from pathlib import Path
from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PCAMDataset(Dataset):
    """
    PatchCamelyon (PCAM) Dataset reader for H5 format.
    """

    def __init__(self, x_path: str, y_path: str, transform: Optional[Callable] = None, filter_data=False):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.transform = transform
        self.filter_data = filter_data

        # 1. Check if files exist
        if not self.x_path.exists():
            raise FileNotFoundError(f"x_path not found: {self.x_path}")
        if not self.y_path.exists():
            raise FileNotFoundError(f"y_path not found: {self.y_path}")
        # 2. Open h5 files in read mode
        self.x_file = h5py.File(self.x_path, "r")
        self.y_file = h5py.File(self.y_path, "r")
        self.x_data = self.x_file['x']
        self.y_data = self.y_file['y']

        if self.filter_data:
            means = self.x_data[...].mean(axis=(1, 2, 3))
            lo, hi = np.percentile(means, [1, 99])
            self.indices = np.where((means >= lo) & (mean <= hi))[0]
        else:
            self.indixes = np.arange(len(self.x_data))
    def __len__(self) -> int:
        # The dataloader will know hence how many batches to create
        return len(self.x_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Read data at idx
        image = self.x_data[idx]
        label = self.y_data[idx]
        # 2. Convert to uint8 (for PIL compatibility if using transforms)
        image = np.clip(image, 0, 255).astype(np.uint8)
        # 3. Apply transforms if they exist
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).float()
            image = image.permute(2, 0, 1)
            image = image / 255.0
        # 4. Return tensor image and label (as long)
        label = torch.tensor(label, dtype=torch.long).squeeze()
        return image, label

