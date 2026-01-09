from pathlib import Path
import matplotlib.pyplot as plt
import torch

from ml_core.data.pcam import PCAMDataset

data_dir = Path("/scratch-shared/scur2385/surfdrive")
x_train = data_dir / "camelyonpatch_level_2_split_train_x.h5"
y_train = data_dir / "camelyonpatch_level_2_split_train_y.h5"

ds = PCAMDataset(str(x_train), str(y_train), filter_data=False)

import numpy as np

labels = []
for i in range(len(ds)):
    _, y = ds[i]
    labels.append(int(y))

labels = np.array(labels)

plt.figure()
unique, counts = np.unique(labels, return_counts=True)
plt.bar(unique, counts)
plt.xticks([0, 1], ["class 0", "class 1"])
plt.ylabel("count")
plt.title("PCAM train label distribution")
plt.tight_layout()
plt.savefig("eda_label_distribution.png")

import numpy as np
import random

n_samples = min(3000, len(ds))
indices = random.sample(range(len(ds)), n_samples)

means = []
for idx in indices:
    x, _ = ds[idx]
    m = x.mean().item()
    means.append(m)

means = np.array(means)

plt.figure()
plt.hist(means, bins=50)
plt.xlabel("mean pixel intensity (0–1)")
plt.ylabel("count")
plt.title("Mean patch intensity (subset of train)")
plt.tight_layout()
plt.savefig("eda_intensity_hist.png")

def show_patch(idx, title):
    x, y = ds[idx]          # (C,H,W)
    img = x.numpy().transpose(1, 2, 0)  # terug naar (H,W,C)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{title} (label={int(y)})")

plt.figure(figsize=(8, 3))

plt.subplot(1, 3, 1)
show_patch(indices[0], "normal?")

plt.subplot(1, 3, 2)
show_patch(indices[1], "maybe dark")

plt.subplot(1, 3, 3)
show_patch(indices[2], "maybe bright")

plt.tight_layout()
plt.savefig("eda_example_patches.png")
