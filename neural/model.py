import math
import torch
import torch.nn.functional as F
from torch import nn


class PlateNet(nn.Module):
    def __init__(self, n_modes=32, latent_dim=256):
        super().__init__()

        self.n_modes = n_modes

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
        )

        self.geometry_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.SiLU(),
            nn.Linear(512, latent_dim),
            nn.SiLU(),
        )

        self.frequency_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, n_modes),
        )

        self.point_head = nn.Sequential(
            nn.Linear(latent_dim + 2, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, n_modes),
        )

        self._init_frequency_head()


    def _init_frequency_head(self):
        output = self.frequency_head[-1]

        nn.init.normal_(
            output.weight,
            mean=0.0,
            std=1e-3,
        )

        with torch.no_grad():
            output.bias.zero_()
            output.bias[0] = math.log(20.0)
            output.bias[1:] = -2.1


    def encode(self, shape):
        return self.geometry_head(
            self.encoder(shape)
        )


    def predict_log_factors(self, z):
        raw = self.frequency_head(z)

        first = raw[:, :1]

        spacing = (
            F.softplus(raw[:, 1:])
            + 1e-5
        )

        return torch.cat(
            (
                first,
                first + torch.cumsum(
                    spacing,
                    dim=1,
                ),
            ),
            dim=1,
        )


    def predict_gains(self, z, points):
        single_point = points.ndim == 2

        if single_point:
            points = points[:, None, :]

        batch_size, n_points, _ = points.shape

        z = z[:, None, :].expand(
            -1,
            n_points,
            -1,
        )

        x = torch.cat(
            (z, points),
            dim=-1,
        )

        gains = self.point_head(
            x.reshape(
                batch_size * n_points,
                -1,
            )
        )

        gains = gains.reshape(
            batch_size,
            n_points,
            self.n_modes,
        )

        if single_point:
            gains = gains[:, 0]

        return gains


    def forward(self, shape, points):
        z = self.encode(shape)

        return (
            self.predict_log_factors(z),
            self.predict_gains(z, points),
        )