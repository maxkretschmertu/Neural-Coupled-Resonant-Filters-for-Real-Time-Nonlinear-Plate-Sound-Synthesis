import torch
from torch import nn


class ModalNet(nn.Module):
    def __init__(
        self,
        n_modes=32,
        latent_dim=64,
    ):
        super().__init__()

        self.n_modes = n_modes


        # -------------------------------------------------
        # Geometry encoder
        #
        # Input:
        #     (B, 1, 64, 64)
        #
        # Output:
        #     (B, latent_dim)
        # -------------------------------------------------

        self.encoder = nn.Sequential(
            nn.Conv2d(
                1,
                8,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                8,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (4, 4)
            ),

            nn.Flatten(),

            nn.Linear(
                32 * 4 * 4,
                latent_dim,
            ),

            nn.ReLU(),
        )


        # -------------------------------------------------
        # Modal-frequency head
        #
        # Predict log(mu) because modal factors are
        # strictly positive and span a fairly large range.
        #
        # Output:
        #     (B, 32)
        # -------------------------------------------------

        self.frequency_head = nn.Sequential(
            nn.Linear(
                latent_dim,
                128,
            ),
            nn.ReLU(),

            nn.Linear(
                128,
                n_modes,
            ),
        )


        # -------------------------------------------------
        # Strike -> pickup modal-residue head
        #
        # Input:
        #
        #     geometry latent
        #     strike_x
        #     strike_y
        #     pickup_x
        #     pickup_y
        #
        # Output:
        #     32 modal residues
        #
        # No tanh:
        # mass-normalized residues are not restricted
        # to [-1, 1].
        # -------------------------------------------------

        self.pair_head = nn.Sequential(
            nn.Linear(
                latent_dim + 4,
                128,
            ),
            nn.ReLU(),

            nn.Linear(
                128,
                128,
            ),
            nn.ReLU(),

            nn.Linear(
                128,
                n_modes,
            ),
        )


    # -----------------------------------------------------
    # Encode geometry only once
    # -----------------------------------------------------

    def encode_geometry(
        self,
        shape,
    ):
        return self.encoder(
            shape
        )


    # -----------------------------------------------------
    # Frequency prediction
    # -----------------------------------------------------

    def predict_frequencies(
        self,
        z,
    ):
        log_factors = (
            self.frequency_head(
                z
            )
        )

        return log_factors


    # -----------------------------------------------------
    # Pair prediction
    #
    # Supports:
    #
    # z:
    #     (B, latent)
    #
    # strike_points:
    #     (B, P, 2)
    #
    # pickup_points:
    #     (B, P, 2)
    #
    # Result:
    #     (B, P, N_MODES)
    # -----------------------------------------------------

    def predict_residues(
        self,
        z,
        strike_points,
        pickup_points,
    ):
        batch_size = (
            z.shape[0]
        )

        n_pairs = (
            strike_points.shape[1]
        )

        z_expanded = (
            z[:, None, :]
            .expand(
                -1,
                n_pairs,
                -1,
            )
        )

        pair_input = torch.cat(
            (
                z_expanded,
                strike_points,
                pickup_points,
            ),
            dim=-1,
        )

        pair_input = (
            pair_input.reshape(
                batch_size
                * n_pairs,
                -1,
            )
        )

        residues = (
            self.pair_head(
                pair_input
            )
        )

        residues = (
            residues.reshape(
                batch_size,
                n_pairs,
                self.n_modes,
            )
        )

        return residues


    # -----------------------------------------------------
    # Complete forward pass
    # -----------------------------------------------------

    def forward(
        self,
        shape,
        strike_points,
        pickup_points,
    ):
        z = self.encode_geometry(
            shape
        )

        log_factors = (
            self.predict_frequencies(
                z
            )
        )

        residues = (
            self.predict_residues(
                z,
                strike_points,
                pickup_points,
            )
        )

        return (
            log_factors,
            residues,
        )


# ---------------------------------------------------------
# Small shape test
# ---------------------------------------------------------

if __name__ == "__main__":

    model = ModalNet(
        n_modes=32
    )

    B = 4
    P = 16

    masks = torch.randn(
        B,
        1,
        64,
        64,
    )

    strike = torch.randn(
        B,
        P,
        2,
    )

    pickup = torch.randn(
        B,
        P,
        2,
    )

    (
        log_factors,
        residues,
    ) = model(
        masks,
        strike,
        pickup,
    )

    print(
        "log_factors:",
        log_factors.shape,
    )

    print(
        "residues:",
        residues.shape,
    )