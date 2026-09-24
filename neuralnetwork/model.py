import torch
from torch import nn


import torch
from torch import nn


class ModalNet(nn.Module):
    def __init__(self, n_modes=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.geometry_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
        )

        self.frequency_head = nn.Linear(
            64,
            n_modes,
        )

        self.point_head = nn.Sequential(
            nn.Linear(64 + 2, 64),
            nn.ReLU(),
            nn.Linear(64, n_modes),
            nn.Tanh(),
        )

    def forward(self, shape, point):
        features = self.encoder(shape)

        z = self.geometry_head(features)

        log_factors = self.frequency_head(z)

        point_input = torch.cat(
            [z, point],
            dim=1,
        )

        gains = self.point_head(point_input)

        return log_factors, gains