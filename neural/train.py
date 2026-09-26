from pathlib import Path

import numpy as np
import torch

from model import PlateNet


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

N_MODES = 32
STEPS = 20000
BATCH = 16
PAIRS = 32
N_FREQS = 64

LR = 5e-4
FREQ_WEIGHT = 10.0
DAMPING = 0.03

ROOT = Path(__file__).parent
DATA = np.load(ROOT / "plate_dataset.npz")
MODEL_PATH = ROOT.parent / "models" / "plate_nn.pt"
MODEL_PATH.parent.mkdir(exist_ok=True)

torch.manual_seed(0)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------

geometry = torch.tensor(
    np.column_stack((DATA["morphs"], DATA["aspects"])),
    dtype=torch.float32,
    device=device,
)

factors = torch.tensor(
    DATA["factors"],
    dtype=torch.float32,
    device=device,
)

points = torch.tensor(
    DATA["points"],
    dtype=torch.float32,
    device=device,
)

gains = torch.tensor(
    DATA["point_gains"],
    dtype=torch.float32,
    device=device,
)

train = torch.tensor(
    DATA["train_indices"],
    device=device,
)

val = torch.tensor(
    DATA["val_indices"],
    device=device,
)

print("Train geometries:", len(train))
print("Validation geometries:", len(val))


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = PlateNet(N_MODES).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-5,
)


# ---------------------------------------------------------
# Modal transfer function
# ---------------------------------------------------------

omega = torch.logspace(
    np.log10(float(factors[train].min()) * 0.9),
    np.log10(float(factors[train].max()) * 1.05),
    N_FREQS,
    device=device,
)


def response(mu, residues):
    mu = mu[:, None, :, None]
    w = omega[None, None, None, :]

    return (
        residues[..., None]
        / (
            mu**2
            - w**2
            + 1j * 2.0 * DAMPING * mu * w
        )
    ).sum(dim=2)


def transfer_loss(predicted, target):
    return (
        (predicted - target).abs().square().sum()
        / target.abs().square().sum().clamp_min(1e-8)
    )


# ---------------------------------------------------------
# Fixed validation pairs
# ---------------------------------------------------------

torch.manual_seed(1234)

val_i = torch.randint(
    points.shape[1],
    (len(val), PAIRS),
    device=device,
)

val_j = torch.randint(
    points.shape[1],
    (len(val), PAIRS),
    device=device,
)

torch.manual_seed(0)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

@torch.no_grad()
def validate():
    z = model.encode_geometry(
        geometry[val]
    )

    mu_pred = torch.exp(
        model.predict_log_factors(z)
    )

    batch = torch.arange(
        len(val),
        device=device,
    )[:, None]

    p = points[val]
    phi = gains[val]

    strike = p[batch, val_i]
    pickup = p[batch, val_j]

    true_residues = (
        phi[batch, val_i]
        * phi[batch, val_j]
    )

    pred_residues = (
        model.predict_gains(z, strike)
        * model.predict_gains(z, pickup)
    )

    modal_error = (
        (
            (mu_pred - factors[val]).abs()
            / factors[val]
        ).mean()
        * 100.0
    )

    transfer_error = (
        torch.sqrt(
            transfer_loss(
                response(
                    mu_pred,
                    pred_residues,
                ),
                response(
                    factors[val],
                    true_residues,
                ),
            )
        )
        * 100.0
    )

    return (
        modal_error.item(),
        transfer_error.item(),
    )


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

best_transfer = float("inf")

for step in range(1, STEPS + 1):

    idx = train[
        torch.randint(
            len(train),
            (BATCH,),
            device=device,
        )
    ]

    p = points[idx]
    phi = gains[idx]

    i = torch.randint(
        p.shape[1],
        (BATCH, PAIRS),
        device=device,
    )

    j = torch.randint(
        p.shape[1],
        (BATCH, PAIRS),
        device=device,
    )

    # 25 % self-pairs stabilize modal gain magnitudes.
    j[:, :PAIRS // 4] = i[:, :PAIRS // 4]

    batch = torch.arange(
        BATCH,
        device=device,
    )[:, None]

    strike = p[batch, i]
    pickup = p[batch, j]

    true_residues = (
        phi[batch, i]
        * phi[batch, j]
    )

    z = model.encode_geometry(
        geometry[idx]
    )

    log_mu = model.predict_log_factors(z)
    mu_pred = torch.exp(log_mu)

    pred_residues = (
        model.predict_gains(z, strike)
        * model.predict_gains(z, pickup)
    )

    loss_frequency = (
        log_mu
        - torch.log(factors[idx])
    ).square().mean()

    loss_transfer = transfer_loss(
        response(
            mu_pred,
            pred_residues,
        ),
        response(
            factors[idx],
            true_residues,
        ),
    )

    # Same gradual transfer-loss introduction
    # as in the successful training run.
    transfer_weight = min(
        1.0,
        step / 1250.0,
    )

    loss = (
        FREQ_WEIGHT * loss_frequency
        + transfer_weight * loss_transfer
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if step == 1 or step % 250 == 0:

        model.eval()

        modal_error, transfer_error = (
            validate()
        )

        model.train()

        print(
            f"{step:5d}/{STEPS}"
            f" | loss={loss.item():.4f}"
            f" | modal={modal_error:.2f}%"
            f" | transfer={transfer_error:.2f}%"
        )

        if transfer_error < best_transfer:
            best_transfer = transfer_error

            torch.save(
                model.state_dict(),
                MODEL_PATH,
            )


# ---------------------------------------------------------
# Result
# ---------------------------------------------------------

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
)

model.eval()

modal_error, transfer_error = validate()

print()
print(f"Modal mean error: {modal_error:.3f}%")
print(f"Transfer RMS:     {transfer_error:.3f}%")
print("Saved:", MODEL_PATH)