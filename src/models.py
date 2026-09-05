"""
Model definitions for earthquake ground-motion prediction.
"""

import torch
import torch.nn as nn


class PGA_MLP(nn.Module):
    """Multilayer perceptron for log-PGA prediction."""

    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.BatchNorm1d(64),
            nn.Dropout(0.15),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.BatchNorm1d(32),
            nn.Dropout(0.10),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)
