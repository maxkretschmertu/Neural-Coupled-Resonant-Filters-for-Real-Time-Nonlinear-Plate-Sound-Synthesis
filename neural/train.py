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
EPOCHS = 300
PATIENCE = 60

POINT_LR = 5e-4
ENCODER_LR = 2e-5
FREQUENCY_WEIGHT = 1000.0

ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "plate_dataset.npz"
MODEL_PATH = ROOT.parent / "models" / "plate_nn.pt"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(0)
np.random.seed(0)

print("Device:", device)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

data = np.load(DATASET_PATH)


def make_loader(indices, shuffle):
    dataset = TensorDataset(
        torch.tensor(
            data["masks"][indices],
            dtype=torch.float32,
        ).unsqueeze(1),

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
# Model
# ---------------------------------------------------------

model = PlateNet(
    n_modes=N_MODES,
).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# Frequency head stays fixed.
for parameter in model.parameters():
    parameter.requires_grad = True

for parameter in model.frequency_head.parameters():
    parameter.requires_grad = False


optimizer = torch.optim.AdamW(
    [
        {
            "params": model.point_head.parameters(),
            "lr": POINT_LR,
        },
        {
            "params": (
                list(model.encoder.parameters())
                + list(model.geometry_head.parameters())
            ),
            "lr": ENCODER_LR,
        },
    ]
)


# ---------------------------------------------------------
# Losses
# ---------------------------------------------------------

mode_weights = torch.ones(
    N_MODES,
    device=device,
)

mode_weights[:4] = 4.0
mode_weights[4:8] = 2.0
mode_weights /= mode_weights.mean()


def frequency_loss(predicted, target):
    return (
        (predicted - target).square()
        * mode_weights
    ).mean()


def residue_loss(predicted_gains, target_gains):
    predicted = (
        predicted_gains[:, :, None, :]
        * predicted_gains[:, None, :, :]
    )

    target = (
        target_gains[:, :, None, :]
        * target_gains[:, None, :, :]
    )

    return (
        (predicted - target).square().sum()
        / target.square().sum().clamp_min(1e-8)
    )

# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

@torch.no_grad()
def evaluate():
    model.eval()

    residue_error = 0.0
    residue_energy = 0.0

    factor_error = []

    for masks, factors, points, gains in val_loader:
        masks = masks.to(device)
        factors = factors.to(device)
        points = points.to(device)
        gains = gains.to(device)

        z = model.encode(masks)

        predicted_log_factors = (
            model.predict_log_factors(z)
        )

        predicted_factors = torch.exp(
            predicted_log_factors
        )

        predicted_gains = model.predict_gains(
            z,
            points,
        )

        predicted_residues = (
            predicted_gains[:, :, None, :]
            * predicted_gains[:, None, :, :]
        )

        target_residues = (
            gains[:, :, None, :]
            * gains[:, None, :, :]
        )

        residue_error += (
            (predicted_residues - target_residues)
            .square()
            .sum()
            .item()
        )

        residue_energy += (
            target_residues
            .square()
            .sum()
            .item()
        )

        factor_error.append(
            (
                torch.abs(
                    predicted_factors - factors
                )
                / factors
            ).cpu()
        )

    residue_rms = (
        np.sqrt(
            residue_error / residue_energy
        )
        * 100.0
    )

    factor_error = torch.cat(
        factor_error
    )

    modal_mean = (
        factor_error.mean().item()
        * 100.0
    )

    return residue_rms, modal_mean


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

best_residue = float("inf")
best_state = None
no_improvement = 0


for epoch in range(1, EPOCHS + 1):
    model.train()

    total_loss = 0.0

    for masks, factors, points, gains in train_loader:
        masks = masks.to(device)
        factors = factors.to(device)
        points = points.to(device)
        gains = gains.to(device)

        z = model.encode(masks)

        predicted_log_factors = (
            model.predict_log_factors(z)
        )

        predicted_gains = model.predict_gains(
            z,
            points,
        )

        loss_points = residue_loss(
            predicted_gains,
            gains,
        )

        loss_frequency = frequency_loss(
            predicted_log_factors,
            torch.log(factors),
        )

        loss = (
            loss_points
            + FREQUENCY_WEIGHT
            * loss_frequency
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    val_residue, val_frequency = evaluate()


    if val_residue < best_residue:
        best_residue = val_residue

        best_state = copy.deepcopy(
            model.state_dict()
        )

        no_improvement = 0

    else:
        no_improvement += 1


    if epoch == 1 or epoch % 10 == 0:
        print(
            f"epoch {epoch:3d}"
            f" | loss={total_loss / len(train_loader):.4f}"
            f" | residue={val_residue:.2f}%"
            f" | modal={val_frequency:.2f}%"
        )


    if no_improvement >= PATIENCE:
        print("Early stopping.")
        break


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

model.load_state_dict(
    best_state
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "n_modes": N_MODES,
        "validation_residue_rms": best_residue,
    },
    MODEL_PATH,
)

final_residue, final_frequency = evaluate()

print()
print(
    f"Best validation residue RMS: "
    f"{final_residue:.3f}%"
)

print(
    f"Validation modal mean error: "
    f"{final_frequency:.3f}%"
)

print(
    "Saved:",
    MODEL_PATH,
)