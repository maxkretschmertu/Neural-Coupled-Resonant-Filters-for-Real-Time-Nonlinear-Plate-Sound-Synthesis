import math

import torch
import torch.nn.functional as F
from torch import nn


class PlateNet(nn.Module):
    def __init__(
        self,
        n_modes=32,
        latent_dim=128,
        fourier_bands=8,
    ):
        super().__init__()

        self.n_modes = n_modes
        self.fourier_bands = fourier_bands

        # morph + aspect -> geometry latent
        self.geometry_net = nn.Sequential(
            nn.Linear(2, 128),
            nn.SiLU(),
            nn.Linear(128, latent_dim),
            nn.SiLU(),
        )

        # geometry latent -> modal factors
        self.frequency_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, n_modes),
        )

        # x, y plus Fourier features
        point_dim = 2 + 4 * fourier_bands

        self.point_head = nn.Sequential(
            nn.Linear(latent_dim + point_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, n_modes),
        )

        self._init_frequencies()


    def _init_frequencies(self):
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


    def encode_geometry(self, geometry):
        """
        geometry[:, 0] = morph  [0, 1]
        geometry[:, 1] = aspect [0.5, 2]
        """

        morph = 2.0 * geometry[:, 0] - 1.0
        aspect = (
            geometry[:, 1] - 1.25
        ) / 0.75

        x = torch.stack(
            (morph, aspect),
            dim=-1,
        )

        return self.geometry_net(x)


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
                first
                + torch.cumsum(
                    spacing,
                    dim=1,
                ),
            ),
            dim=1,
        )


    def point_features(self, points):
        features = [points]

        x = points[..., 0:1]
        y = points[..., 1:2]

        for k in range(1, self.fourier_bands + 1):
            frequency = k * math.pi

            features.extend(
                (
                    torch.sin(frequency * x),
                    torch.cos(frequency * x),
                    torch.sin(frequency * y),
                    torch.cos(frequency * y),
                )
            )

        return torch.cat(
            features,
            dim=-1,
        )


    def predict_gains(self, z, points):
        """
        z:      (B, latent)
        points: (B, P, 2)

        returns:
            (B, P, modes)
        """

        features = self.point_features(
            points
        )

        z = z[:, None, :].expand(
            -1,
            points.shape[1],
            -1,
        )

        x = torch.cat(
            (z, features),
            dim=-1,
        )

        return self.point_head(x)


    def forward(self, geometry, points):
        z = self.encode_geometry(
            geometry
        )

        factors = torch.exp(
            self.predict_log_factors(z)
        )

        gains = self.predict_gains(
            z,
            points,
        )

        return factors, gains