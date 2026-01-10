from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: List[int],
        hidden_units: List[int],
        num_classes: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        
        # TODO: Build the MLP architecture
        # If you are up to the task, explore other architectures or model types
        # Hint: Flatten -> [Linear -> ReLU -> Dropout] * N_layers -> Linear
        c, h, w = input_shape
        in_features = c * h * w
        layers = []
        for u in hidden_units:
            layers.append(nn.Linear(in_features, u))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = u
        layers.append(nn.Linear(in_features, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        x = torch.flatten(x, start_dim=1)
        out = self.model(x)
        return out
