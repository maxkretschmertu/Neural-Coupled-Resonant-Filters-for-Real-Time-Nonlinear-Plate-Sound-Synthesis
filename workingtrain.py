from pathlib import Path
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import PlateNet


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

N_MODES = 32

BATCH_SIZE = 16
EPOCHS = 500
PATIENCE = 80

LEARNING_RATE = 5e-4
FREQUENCY_WEIGHT = 10.0

PAIRS_PER_GEOMETRY = 32
N_FREQUENCIES = 64
DAMPING = 0.03

ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "plate_dataset.npz"

MODEL_PATH = (
    ROOT.parent
    / "models"
    / "plate_nn.pt"
)

MODEL_PATH.parent.mkdir(
    exist_ok=True
)


# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

torch.manual_seed(0)
np.random.seed(0)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

data = np.load(
    DATASET_PATH
)

geometry = np.column_stack(
    (
        data["morphs"],
        data["aspects"],
    )
).astype(np.float32)


def make_loader(indices, shuffle):
    dataset = TensorDataset(
        torch.tensor(
            geometry[indices],
            dtype=torch.float32,
        ),

        torch.tensor(
            data["factors"][indices],
            dtype=torch.float32,
        ),

        torch.tensor(
            data["points"][indices],
            dtype=torch.float32,
        ),

        torch.tensor(
            data["point_gains"][indices],
            dtype=torch.float32,
        ),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


train_loader = make_loader(
    data["train_indices"],
    True,
)

val_loader = make_loader(
    data["val_indices"],
    False,
)


print(
    "Train geometries:",
    len(data["train_indices"]),
)

print(
    "Validation geometries:",
    len(data["val_indices"]),
)


# ---------------------------------------------------------
# Frequency grid for differentiable transfer function
# ---------------------------------------------------------

train_factors = data["factors"][
    data["train_indices"]
]

mu_min = float(
    train_factors.min()
)

mu_max = float(
    train_factors.max()
)

omega = torch.logspace(
    np.log10(mu_min * 0.9),
    np.log10(mu_max * 1.05),
    N_FREQUENCIES,
    device=device,
)


# ---------------------------------------------------------
# Model
#
# Fresh model: no old checkpoints are loaded.
# ---------------------------------------------------------

model = PlateNet(
    n_modes=N_MODES,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5,
)


# ---------------------------------------------------------
# Transfer function
# ---------------------------------------------------------

def modal_response(
    factors,
    residues,
):
    """
    factors:
        (B, modes)

    residues:
        (B, pairs, modes)

    returns:
        complex response (B, pairs, frequencies)
    """

    mu = factors[
        :, None, :, None
    ]

    w = omega[
        None, None, None, :
    ]

    denominator = (
        mu.square()
        - w.square()
        + 1j
        * 2.0
        * DAMPING
        * mu
        * w
    )

    return (
        residues[..., None]
        / denominator
    ).sum(dim=2)


# ---------------------------------------------------------
# Random strike/pickup pairs
# ---------------------------------------------------------

def sample_pairs(
    points,
    gains,
    n_pairs,
):
    batch_size = points.shape[0]
    n_points = points.shape[1]

    i = torch.randint(
        n_points,
        (
            batch_size,
            n_pairs,
        ),
        device=device,
    )

    j = torch.randint(
        n_points,
        (
            batch_size,
            n_pairs,
        ),
        device=device,
    )

    # Some self-pairs help determine gain magnitudes.
    n_self = n_pairs // 4

    j[:, :n_self] = i[:, :n_self]

    batch = torch.arange(
        batch_size,
        device=device,
    )[:, None]

    strike_points = points[
        batch,
        i,
    ]

    pickup_points = points[
        batch,
        j,
    ]

    target_residues = (
        gains[batch, i]
        * gains[batch, j]
    )

    return (
        strike_points,
        pickup_points,
        target_residues,
    )


# ---------------------------------------------------------
# Loss
# ---------------------------------------------------------

def transfer_loss(
    predicted,
    target,
):
    return (
        (predicted - target)
        .abs()
        .square()
        .sum()
        /
        target
        .abs()
        .square()
        .sum()
        .clamp_min(1e-8)
    )


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

@torch.no_grad()
def evaluate():
    model.eval()

    frequency_errors = []

    response_error = 0.0
    response_energy = 0.0

    # Same validation pairs every epoch.
    generator = torch.Generator()
    generator.manual_seed(1234)

    for (
        geometry_batch,
        factors,
        points,
        gains,
    ) in val_loader:

        geometry_batch = (
            geometry_batch.to(device)
        )

        factors = factors.to(device)
        points = points.to(device)
        gains = gains.to(device)

        z = model.encode_geometry(
            geometry_batch
        )

        predicted_factors = torch.exp(
            model.predict_log_factors(z)
        )

        frequency_errors.append(
            (
                torch.abs(
                    predicted_factors
                    - factors
                )
                / factors
            ).cpu()
        )


        batch_size = points.shape[0]
        n_points = points.shape[1]

        i = torch.randint(
            n_points,
            (
                batch_size,
                PAIRS_PER_GEOMETRY,
            ),
            generator=generator,
        ).to(device)

        j = torch.randint(
            n_points,
            (
                batch_size,
                PAIRS_PER_GEOMETRY,
            ),
            generator=generator,
        ).to(device)

        batch = torch.arange(
            batch_size,
            device=device,
        )[:, None]


        strike_points = points[
            batch,
            i,
        ]

        pickup_points = points[
            batch,
            j,
        ]


        true_residues = (
            gains[batch, i]
            * gains[batch, j]
        )


        predicted_strike = (
            model.predict_gains(
                z,
                strike_points,
            )
        )

        predicted_pickup = (
            model.predict_gains(
                z,
                pickup_points,
            )
        )

        predicted_residues = (
            predicted_strike
            * predicted_pickup
        )


        true_response = modal_response(
            factors,
            true_residues,
        )

        predicted_response = modal_response(
            predicted_factors,
            predicted_residues,
        )


        response_error += (
            (
                predicted_response
                - true_response
            )
            .abs()
            .square()
            .sum()
            .item()
        )

        response_energy += (
            true_response
            .abs()
            .square()
            .sum()
            .item()
        )


    frequency_errors = torch.cat(
        frequency_errors
    )


    modal_error = (
        frequency_errors.mean().item()
        * 100.0
    )


    transfer_rms = (
        np.sqrt(
            response_error
            / response_energy
        )
        * 100.0
    )


    return (
        modal_error,
        transfer_rms,
    )


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

best_transfer = float("inf")
best_state = None
no_improvement = 0


for epoch in range(
    1,
    EPOCHS + 1,
):

    model.train()

    total_loss = 0.0


    for (
        geometry_batch,
        factors,
        points,
        gains,
    ) in train_loader:

        geometry_batch = (
            geometry_batch.to(device)
        )

        factors = factors.to(device)
        points = points.to(device)
        gains = gains.to(device)


        # Geometry representation
        z = model.encode_geometry(
            geometry_batch
        )


        # Modal frequencies
        predicted_log_factors = (
            model.predict_log_factors(z)
        )

        predicted_factors = torch.exp(
            predicted_log_factors
        )


        # Random strike/pickup pairs
        (
            strike_points,
            pickup_points,
            true_residues,
        ) = sample_pairs(
            points,
            gains,
            PAIRS_PER_GEOMETRY,
        )


        predicted_strike = (
            model.predict_gains(
                z,
                strike_points,
            )
        )

        predicted_pickup = (
            model.predict_gains(
                z,
                pickup_points,
            )
        )

        predicted_residues = (
            predicted_strike
            * predicted_pickup
        )


        # Physical responses
        true_response = modal_response(
            factors,
            true_residues,
        )

        predicted_response = modal_response(
            predicted_factors,
            predicted_residues,
        )


        # Frequency accuracy
        loss_frequency = (
            predicted_log_factors
            - torch.log(factors)
        ).square().mean()


        # Sign- and basis-robust physical target
        loss_transfer = transfer_loss(
            predicted_response,
            true_response,
        )


        # Slowly introduce transfer loss while frequencies
        # settle during the first epochs.
        transfer_weight = min(
            1.0,
            epoch / 50.0,
        )


        loss = (
            FREQUENCY_WEIGHT
            * loss_frequency
            + transfer_weight
            * loss_transfer
        )


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    modal_error, transfer_rms = (
        evaluate()
    )


    if transfer_rms < best_transfer:

        best_transfer = transfer_rms

        best_state = copy.deepcopy(
            model.state_dict()
        )

        no_improvement = 0

    else:
        no_improvement += 1


    if (
        epoch == 1
        or epoch % 10 == 0
    ):

        print(
            f"epoch {epoch:3d}"
            f" | loss={total_loss / len(train_loader):.4f}"
            f" | modal={modal_error:.2f}%"
            f" | transfer={transfer_rms:.2f}%"
        )


    if no_improvement >= PATIENCE:
        print("Early stopping.")
        break


# ---------------------------------------------------------
# Save best model
# ---------------------------------------------------------

model.load_state_dict(
    best_state
)

modal_error, transfer_rms = (
    evaluate()
)


torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "n_modes":
            N_MODES,

        "modal_error":
            modal_error,

        "transfer_rms":
            transfer_rms,
    },
    MODEL_PATH,
)


print()
print(
    f"Validation modal mean error: "
    f"{modal_error:.3f}%"
)

print(
    f"Validation transfer RMS: "
    f"{transfer_rms:.3f}%"
)

print(
    "Saved:",
    MODEL_PATH,
)