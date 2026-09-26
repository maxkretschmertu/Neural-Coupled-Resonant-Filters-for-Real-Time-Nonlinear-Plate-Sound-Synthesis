import math

import torch
import torch.nn.functional as F
from torch import nn


class FrequencyNetV2(nn.Module):
    """
    Geometry-only network for ordered modal factors.

    Input:
        shape: (B, 1, 64, 64)

    Output:
        log_factors: (B, n_modes)

    Parameterization:
        log(mu_1) = a_1

        log(mu_k) =
            log(mu_1)
            + sum_{j=2..k} softplus(d_j)

    Thus:
        mu_1 < mu_2 < ... < mu_N
    """

    def __init__(
        self,
        n_modes=32,
        latent_dim=256,
    ):
        super().__init__()

        self.n_modes = n_modes
        self.latent_dim = latent_dim

        # =================================================
        # Geometry encoder
        #
        # 64 -> 32 -> 16 -> 8 -> 4
        # =================================================

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.GroupNorm(
                4,
                16,
            ),
            nn.SiLU(),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                32,
            ),
            nn.SiLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
            nn.SiLU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(
                16,
                128,
            ),
            nn.SiLU(),
        )

        # =================================================
        # Geometry latent
        # =================================================

        self.geometry_head = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 4 * 4,
                512,
            ),
            nn.SiLU(),

            nn.Linear(
                512,
                latent_dim,
            ),
            nn.SiLU(),
        )

        # =================================================
        # Frequency head
        #
        # raw[:, 0]  -> log(mu_1)
        # raw[:, 1:] -> positive log-frequency increments
        # =================================================

        self.frequency_head = nn.Sequential(

            nn.Linear(
                latent_dim,
                256,
            ),
            nn.SiLU(),

            nn.Linear(
                256,
                128,
            ),
            nn.SiLU(),

            nn.Linear(
                128,
                n_modes,
            ),
        )

        self._initialize_frequency_output()

    # =====================================================
    # Initialize output around plausible modal factors
    # =====================================================

    def _initialize_frequency_output(
        self,
    ):

        output_layer = self.frequency_head[-1]

        nn.init.normal_(
            output_layer.weight,
            mean=0.0,
            std=1e-3,
        )

        with torch.no_grad():

            output_layer.bias.zero_()

            # Initial first modal factor ~20
            output_layer.bias[0] = math.log(
                20.0
            )

            # softplus(-2.1) ~ 0.115
            #
            # Reasonable initial spacing in log(mu).
            output_layer.bias[
                1:
            ] = -2.1

    # =====================================================
    # Encode shape
    # =====================================================

    def encode_geometry(
        self,
        shape,
    ):

        features = self.encoder(
            shape
        )

        z = self.geometry_head(
            features
        )

        return z

    # =====================================================
    # Predict ordered log modal factors
    # =====================================================

    def predict_log_factors(
        self,
        z,
    ):

        raw = self.frequency_head(
            z
        )

        # -------------------------------------------------
        # First modal factor
        # -------------------------------------------------

        log_mu_1 = raw[
            :,
            0:1,
        ]

        if self.n_modes == 1:

            return log_mu_1

        # -------------------------------------------------
        # Positive log-frequency spacings
        # -------------------------------------------------

        log_spacing = (
            F.softplus(
                raw[
                    :,
                    1:
                ]
            )
            + 1e-5
        )

        # -------------------------------------------------
        # Cumulative sequence
        #
        # log(mu_2)
        # log(mu_3)
        # ...
        # -------------------------------------------------

        remaining_log_factors = (
            log_mu_1
            + torch.cumsum(
                log_spacing,
                dim=1,
            )
        )

        # -------------------------------------------------
        # Complete ordered vector
        # -------------------------------------------------

        log_factors = torch.cat(
            [
                log_mu_1,
                remaining_log_factors,
            ],
            dim=1,
        )

        return log_factors

    # =====================================================
    # Forward
    # =====================================================

    def forward(
        self,
        shape,
    ):

        z = self.encode_geometry(
            shape
        )

        log_factors = self.predict_log_factors(
            z
        )

        return log_factors