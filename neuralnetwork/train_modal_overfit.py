from pathlib import Path
import copy

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from model import ModalNet


# =========================================================
# Settings
# =========================================================

N_MODES = 32
N_GEOMETRIES = 4

N_FREQUENCY_POINTS = 128
DAMPING_RATIO = 0.02

RANDOM_SEED = 0


# ---------------------------------------------------------
# Stage 1
#
# Geometry -> modal frequencies
# ---------------------------------------------------------

STAGE1_EPOCHS = 4000
STAGE1_LEARNING_RATE = 1e-3


# ---------------------------------------------------------
# Stage 2
#
# Multitask:
#
# geometry -> frequencies
# geometry + points -> residues
#
# Transfer still uses exact FEM frequencies.
# ---------------------------------------------------------

STAGE2_EPOCHS = 2000

STAGE2_ENCODER_LR = 2e-4
STAGE2_FREQUENCY_LR = 2e-4
STAGE2_PAIR_LR = 1e-3

STAGE2_FREQUENCY_WEIGHT = 25.0
STAGE2_RESIDUE_WEIGHT = 1.0
STAGE2_TRANSFER_WEIGHT = 1.0


# ---------------------------------------------------------
# Stage 3
#
# Full end-to-end.
#
# Both modal factors and residues are predicted by NN.
# ---------------------------------------------------------

STAGE3_EPOCHS = 2000

STAGE3_ENCODER_LR = 2e-5
STAGE3_FREQUENCY_LR = 2e-5
STAGE3_PAIR_LR = 1e-4

STAGE3_FREQUENCY_WEIGHT = 25.0
STAGE3_RESIDUE_WEIGHT = 0.25
STAGE3_TRANSFER_WEIGHT = 1.0

# Slowly introduce end-to-end transfer loss.
STAGE3_TRANSFER_RAMP_EPOCHS = 500


DATASET_PATH = (
    Path(__file__).parent
    / "plate_dataset_test.npz"
)


# =========================================================
# Reproducibility
# =========================================================

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)


# =========================================================
# Device
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device,
)


# =========================================================
# Load dataset
# =========================================================

data = np.load(
    DATASET_PATH
)


masks = data[
    "masks"
][
    :N_GEOMETRIES
]

factors = data[
    "factors"
][
    :N_GEOMETRIES
]

strike_points = data[
    "strike_points"
][
    :N_GEOMETRIES
]

pickup_points = data[
    "pickup_points"
][
    :N_GEOMETRIES
]

modal_weights = data[
    "modal_weights"
][
    :N_GEOMETRIES
]


print()

print(
    "Loaded:"
)

print(
    "masks:",
    masks.shape,
)

print(
    "factors:",
    factors.shape,
)

print(
    "strike:",
    strike_points.shape,
)

print(
    "pickup:",
    pickup_points.shape,
)

print(
    "weights:",
    modal_weights.shape,
)


# =========================================================
# Torch tensors
# =========================================================

shape_x = torch.from_numpy(
    masks
).unsqueeze(
    1
).to(
    device
)


factor_y = torch.from_numpy(
    factors
).to(
    device
)


strike_x = torch.from_numpy(
    strike_points
).to(
    device
)


pickup_x = torch.from_numpy(
    pickup_points
).to(
    device
)


weight_y = torch.from_numpy(
    modal_weights
).to(
    device
)


log_factor_y = torch.log(
    factor_y
)


# =========================================================
# Frequency axis
# =========================================================

omega_min = (
    0.5
    * torch.min(
        factor_y[:, 0]
    )
)


omega_max = (
    0.90
    * torch.min(
        factor_y[
            :,
            N_MODES - 1,
        ]
    )
)


omega = torch.linspace(
    omega_min.item(),
    omega_max.item(),
    N_FREQUENCY_POINTS,
    device=device,
)


print()

print(
    "Omega range:",
    f"{omega_min.item():.3f}",
    "...",
    f"{omega_max.item():.3f}",
)


# =========================================================
# Modal response
#
#
# H(Omega) =
#
# sum_k
#
#             r_k
# -----------------------------
# mu_k² - Omega²
# + j 2 zeta mu_k Omega
#
#
# log_factors:
#     (B, M)
#
# residues:
#     (B, P, M)
#
# return:
#     (B, P, F)
# =========================================================

def modal_response(
    log_factors,
    residues,
    omega,
):

    modal_factors = torch.exp(
        log_factors
    )


    # (B, 1, 1, M)
    mu = modal_factors[
        :,
        None,
        None,
        :
    ]


    # (B, P, 1, M)
    r = residues[
        :,
        :,
        None,
        :
    ]


    # (1, 1, F, 1)
    w = omega[
        None,
        None,
        :,
        None
    ]


    denominator_real = (
        mu**2
        - w**2
    )


    denominator_imag = (
        2.0
        * DAMPING_RATIO
        * mu
        * w
    )


    denominator = torch.complex(
        denominator_real,
        denominator_imag,
    )


    return torch.sum(
        r
        / denominator,
        dim=-1,
    )


# =========================================================
# Global energy-weighted transfer loss
#
#
#          sum |H_pred - H_true|²
# loss = --------------------------
#             sum |H_true|²
#
#
# sqrt(loss) = relative RMS
# =========================================================

def transfer_loss(
    predicted,
    target,
):

    error_energy = torch.sum(
        torch.abs(
            predicted
            - target
        ) ** 2
    )


    target_energy = torch.sum(
        torch.abs(
            target
        ) ** 2
    )


    return (
        error_energy
        /
        torch.clamp(
            target_energy,
            min=1e-12,
        )
    )


# =========================================================
# Helper:
# modal-factor diagnostics
# =========================================================

def factor_errors(
    predicted_log_factors,
):

    predicted_factors = torch.exp(
        predicted_log_factors
    )


    relative_error = (
        torch.abs(
            predicted_factors
            - factor_y
        )
        / factor_y
    )


    mean_error = (
        relative_error.mean()
        * 100.0
    )


    max_error = (
        relative_error.max()
        * 100.0
    )


    return (
        predicted_factors,
        mean_error,
        max_error,
    )


# =========================================================
# Exact FEM transfer target
# =========================================================

with torch.no_grad():

    target_response = modal_response(
        log_factors=log_factor_y,
        residues=weight_y,
        omega=omega,
    )


print()

print(
    "Target response:",
    target_response.shape,
)


# =========================================================
# Model
# =========================================================

model = ModalNet(
    n_modes=N_MODES
).to(
    device
)


# =========================================================
#
# STAGE 1
#
# Geometry -> modal factors
#
# =========================================================

print()

print(
    "=" * 70
)

print(
    "STAGE 1: modal frequencies"
)

print(
    "=" * 70
)

print()


optimizer = torch.optim.Adam(
    list(
        model.encoder.parameters()
    )
    + list(
        model.frequency_head.parameters()
    ),
    lr=STAGE1_LEARNING_RATE,
)


scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=200,
        min_lr=1e-6,
    )
)


for epoch in range(
    STAGE1_EPOCHS + 1
):

    model.train()


    z = model.encode_geometry(
        shape_x
    )


    predicted_log_factors = (
        model.predict_frequencies(
            z
        )
    )


    frequency_loss = F.mse_loss(
        predicted_log_factors,
        log_factor_y,
    )


    optimizer.zero_grad()

    frequency_loss.backward()

    optimizer.step()


    scheduler.step(
        frequency_loss.detach().item()
    )


    if (
        epoch % 50 == 0
        or epoch == STAGE1_EPOCHS
    ):

        model.eval()

        with torch.no_grad():

            z_eval = model.encode_geometry(
                shape_x
            )


            log_factors_eval = (
                model.predict_frequencies(
                    z_eval
                )
            )


            frequency_loss_eval = F.mse_loss(
                log_factors_eval,
                log_factor_y,
            )


            (
                factors_eval,
                mean_error,
                max_error,
            ) = factor_errors(
                log_factors_eval
            )


        current_lr = (
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )


        print(
            f"epoch "
            f"{epoch:4d}"
            f" | "
            f"freq="
            f"{frequency_loss_eval.item():.8f}"
            f" | "
            f"mu mean="
            f"{mean_error.item():6.3f}%"
            f" | "
            f"mu max="
            f"{max_error.item():6.3f}%"
            f" | "
            f"lr="
            f"{current_lr:.2e}"
        )


# =========================================================
# Save Stage-1 state
# =========================================================

stage1_state = copy.deepcopy(
    model.state_dict()
)


model.eval()

with torch.no_grad():

    z_stage1 = model.encode_geometry(
        shape_x
    )


    stage1_log_factors = (
        model.predict_frequencies(
            z_stage1
        )
    )


    (
        stage1_predicted_factors,
        stage1_mean_error,
        stage1_max_error,
    ) = factor_errors(
        stage1_log_factors
    )


print()

print(
    "=" * 70
)

print(
    "FINAL STAGE-1 RESULT"
)

print(
    "=" * 70
)

print(
    f"Mean modal-factor error: "
    f"{stage1_mean_error.item():.4f}%"
)

print(
    f"Max modal-factor error: "
    f"{stage1_max_error.item():.4f}%"
)


# =========================================================
#
# STAGE 2
#
# MULTITASK TRAINING
#
#
# encoder
#   ├── frequency_head
#   └── pair_head
#
#
# Frequency supervision remains active so the shared
# encoder cannot destroy its Stage-1 representation.
#
# Transfer still uses exact FEM frequencies.
#
# =========================================================

print()

print(
    "=" * 70
)

print(
    "STAGE 2: multitask residues + transfer"
)

print(
    "=" * 70
)

print()


for parameter in (
    model.encoder.parameters()
):
    parameter.requires_grad = True


for parameter in (
    model.frequency_head.parameters()
):
    parameter.requires_grad = True


for parameter in (
    model.pair_head.parameters()
):
    parameter.requires_grad = True


optimizer = torch.optim.Adam(
    [
        {
            "params":
                model.encoder.parameters(),
            "lr":
                STAGE2_ENCODER_LR,
        },
        {
            "params":
                model.frequency_head.parameters(),
            "lr":
                STAGE2_FREQUENCY_LR,
        },
        {
            "params":
                model.pair_head.parameters(),
            "lr":
                STAGE2_PAIR_LR,
        },
    ]
)


for epoch in range(
    STAGE2_EPOCHS + 1
):

    model.train()


    # -----------------------------------------------------
    # Shared geometry latent
    # -----------------------------------------------------

    z = model.encode_geometry(
        shape_x
    )


    # -----------------------------------------------------
    # Frequency prediction
    # -----------------------------------------------------

    predicted_log_factors = (
        model.predict_frequencies(
            z
        )
    )


    # -----------------------------------------------------
    # Residue prediction
    # -----------------------------------------------------

    predicted_residues = (
        model.predict_residues(
            z,
            strike_x,
            pickup_x,
        )
    )


    # -----------------------------------------------------
    # Frequency supervision
    # -----------------------------------------------------

    frequency_loss = F.mse_loss(
        predicted_log_factors,
        log_factor_y,
    )


    # -----------------------------------------------------
    # Residue supervision
    # -----------------------------------------------------

    residue_loss = F.mse_loss(
        predicted_residues,
        weight_y,
    )


    # -----------------------------------------------------
    # Transfer uses EXACT FEM frequencies in Stage 2
    # -----------------------------------------------------

    predicted_response = modal_response(
        log_factors=log_factor_y,
        residues=predicted_residues,
        omega=omega,
    )


    response_loss = transfer_loss(
        predicted_response,
        target_response,
    )


    # -----------------------------------------------------
    # Stage-2 multitask objective
    # -----------------------------------------------------

    total_loss = (
        STAGE2_FREQUENCY_WEIGHT
        * frequency_loss

        + STAGE2_RESIDUE_WEIGHT
        * residue_loss

        + STAGE2_TRANSFER_WEIGHT
        * response_loss
    )


    optimizer.zero_grad()

    total_loss.backward()


    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=5.0,
    )


    optimizer.step()


    # -----------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------

    if (
        epoch % 50 == 0
        or epoch == STAGE2_EPOCHS
    ):

        model.eval()

        with torch.no_grad():

            z_eval = model.encode_geometry(
                shape_x
            )


            log_factors_eval = (
                model.predict_frequencies(
                    z_eval
                )
            )


            residues_eval = (
                model.predict_residues(
                    z_eval,
                    strike_x,
                    pickup_x,
                )
            )


            frequency_loss_eval = F.mse_loss(
                log_factors_eval,
                log_factor_y,
            )


            (
                factors_eval,
                mean_error,
                max_error,
            ) = factor_errors(
                log_factors_eval
            )


            residue_loss_eval = F.mse_loss(
                residues_eval,
                weight_y,
            )


            residue_mae_eval = torch.mean(
                torch.abs(
                    residues_eval
                    - weight_y
                )
            )


            response_eval = modal_response(
                log_factors=log_factor_y,
                residues=residues_eval,
                omega=omega,
            )


            response_loss_eval = transfer_loss(
                response_eval,
                target_response,
            )


            response_rms_eval = (
                torch.sqrt(
                    response_loss_eval
                )
                * 100.0
            )


        print(
            f"epoch "
            f"{epoch:4d}"
            f" | "
            f"freq="
            f"{frequency_loss_eval.item():.6f}"
            f" | "
            f"mu mean="
            f"{mean_error.item():6.2f}%"
            f" | "
            f"mu max="
            f"{max_error.item():6.2f}%"
            f" | "
            f"residue="
            f"{residue_loss_eval.item():.6f}"
            f" | "
            f"MAE="
            f"{residue_mae_eval.item():.4f}"
            f" | "
            f"H RMS="
            f"{response_rms_eval.item():6.2f}%"
        )


# =========================================================
# Save Stage-2 state
# =========================================================

stage2_state = copy.deepcopy(
    model.state_dict()
)


model.eval()

with torch.no_grad():

    z_stage2 = model.encode_geometry(
        shape_x
    )


    stage2_log_factors = (
        model.predict_frequencies(
            z_stage2
        )
    )


    (
        stage2_predicted_factors,
        stage2_mean_error,
        stage2_max_error,
    ) = factor_errors(
        stage2_log_factors
    )


    stage2_residues = (
        model.predict_residues(
            z_stage2,
            strike_x,
            pickup_x,
        )
    )


    stage2_response = modal_response(
        log_factors=log_factor_y,
        residues=stage2_residues,
        omega=omega,
    )


    stage2_response_loss = transfer_loss(
        stage2_response,
        target_response,
    )


    stage2_response_rms = (
        torch.sqrt(
            stage2_response_loss
        )
        * 100.0
    )


print()

print(
    "=" * 70
)

print(
    "FINAL STAGE-2 RESULT"
)

print(
    "=" * 70
)

print(
    f"Modal-factor mean error: "
    f"{stage2_mean_error.item():.3f}%"
)

print(
    f"Modal-factor max error: "
    f"{stage2_max_error.item():.3f}%"
)

print(
    f"Transfer H RMS "
    f"(exact FEM frequencies): "
    f"{stage2_response_rms.item():.3f}%"
)


# =========================================================
#
# STAGE 3
#
# FULL END-TO-END FINE TUNING
#
#
# mask
#   ↓
# encoder
#   ├── predicted frequencies
#   └── predicted residues
#            ↓
#       modal response
#
#
# Predicted response contains NO FEM modal parameters.
#
# =========================================================

print()

print(
    "=" * 70
)

print(
    "STAGE 3: full end-to-end"
)

print(
    "=" * 70
)

print()


for parameter in model.parameters():
    parameter.requires_grad = True


optimizer = torch.optim.Adam(
    [
        {
            "params":
                model.encoder.parameters(),
            "lr":
                STAGE3_ENCODER_LR,
        },
        {
            "params":
                model.frequency_head.parameters(),
            "lr":
                STAGE3_FREQUENCY_LR,
        },
        {
            "params":
                model.pair_head.parameters(),
            "lr":
                STAGE3_PAIR_LR,
        },
    ]
)


# ---------------------------------------------------------
# Best-model selection
#
# IMPORTANT:
#
# Training uses a ramped transfer weight.
#
# Checkpoint selection ALWAYS uses the final/full
# Stage-3 objective with transfer weight = 1.
#
# This makes all epochs directly comparable.
# ---------------------------------------------------------

best_stage3_loss = float(
    "inf"
)

best_stage3_state = None
best_stage3_epoch = 0


for epoch in range(
    STAGE3_EPOCHS + 1
):

    model.train()


    (
        predicted_log_factors,
        predicted_residues,
    ) = model(
        shape_x,
        strike_x,
        pickup_x,
    )


    # -----------------------------------------------------
    # Frequency target
    # -----------------------------------------------------

    frequency_loss = F.mse_loss(
        predicted_log_factors,
        log_factor_y,
    )


    # -----------------------------------------------------
    # Residue target
    # -----------------------------------------------------

    residue_loss = F.mse_loss(
        predicted_residues,
        weight_y,
    )


    # -----------------------------------------------------
    # FULL predicted transfer
    #
    # Both frequencies and residues come from the NN.
    # -----------------------------------------------------

    predicted_response = modal_response(
        log_factors=predicted_log_factors,
        residues=predicted_residues,
        omega=omega,
    )


    response_loss = transfer_loss(
        predicted_response,
        target_response,
    )


    # -----------------------------------------------------
    # Ramp transfer loss in gradually
    # -----------------------------------------------------

    transfer_scale = min(
        1.0,
        epoch
        / STAGE3_TRANSFER_RAMP_EPOCHS,
    )


    effective_transfer_weight = (
        STAGE3_TRANSFER_WEIGHT
        * transfer_scale
    )


    # -----------------------------------------------------
    # TRAINING objective
    #
    # Uses ramped transfer weight.
    # -----------------------------------------------------

    total_loss = (
        STAGE3_FREQUENCY_WEIGHT
        * frequency_loss

        + STAGE3_RESIDUE_WEIGHT
        * residue_loss

        + effective_transfer_weight
        * response_loss
    )


    optimizer.zero_grad()

    total_loss.backward()


    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=5.0,
    )


    optimizer.step()


    # -----------------------------------------------------
    # Diagnostics / best model
    # -----------------------------------------------------

    if (
        epoch % 50 == 0
        or epoch == STAGE3_EPOCHS
    ):

        model.eval()

        with torch.no_grad():

            (
                log_factors_eval,
                residues_eval,
            ) = model(
                shape_x,
                strike_x,
                pickup_x,
            )


            # ---------------------------------------------
            # Frequency metrics
            # ---------------------------------------------

            frequency_loss_eval = F.mse_loss(
                log_factors_eval,
                log_factor_y,
            )


            (
                factors_eval,
                mean_error,
                max_error,
            ) = factor_errors(
                log_factors_eval
            )


            # ---------------------------------------------
            # Residue metrics
            # ---------------------------------------------

            residue_loss_eval = F.mse_loss(
                residues_eval,
                weight_y,
            )


            # ---------------------------------------------
            # Full end-to-end response
            # ---------------------------------------------

            response_eval = modal_response(
                log_factors=log_factors_eval,
                residues=residues_eval,
                omega=omega,
            )


            response_loss_eval = transfer_loss(
                response_eval,
                target_response,
            )


            response_rms_eval = (
                torch.sqrt(
                    response_loss_eval
                )
                * 100.0
            )


            # ---------------------------------------------
            # Training-objective value
            #
            # Uses current ramped transfer weight.
            # ---------------------------------------------

            total_loss_eval = (
                STAGE3_FREQUENCY_WEIGHT
                * frequency_loss_eval

                + STAGE3_RESIDUE_WEIGHT
                * residue_loss_eval

                + effective_transfer_weight
                * response_loss_eval
            )


            # ---------------------------------------------
            # FIX:
            #
            # Checkpoint-selection objective.
            #
            # ALWAYS uses full final weights.
            #
            # Therefore epoch 0 cannot "win" merely
            # because its transfer weight is still zero.
            # ---------------------------------------------

            selection_loss = (
                STAGE3_FREQUENCY_WEIGHT
                * frequency_loss_eval

                + STAGE3_RESIDUE_WEIGHT
                * residue_loss_eval

                + STAGE3_TRANSFER_WEIGHT
                * response_loss_eval
            )


            # ---------------------------------------------
            # Save best checkpoint
            # ---------------------------------------------

            if (
                selection_loss.item()
                < best_stage3_loss
            ):

                best_stage3_loss = (
                    selection_loss.item()
                )

                best_stage3_epoch = (
                    epoch
                )

                best_stage3_state = (
                    copy.deepcopy(
                        model.state_dict()
                    )
                )


        print(
            f"epoch "
            f"{epoch:4d}"
            f" | "
            f"train="
            f"{total_loss_eval.item():.6f}"
            f" | "
            f"select="
            f"{selection_loss.item():.6f}"
            f" | "
            f"freq="
            f"{frequency_loss_eval.item():.6f}"
            f" | "
            f"mu mean="
            f"{mean_error.item():6.2f}%"
            f" | "
            f"mu max="
            f"{max_error.item():6.2f}%"
            f" | "
            f"residue="
            f"{residue_loss_eval.item():.6f}"
            f" | "
            f"H RMS="
            f"{response_rms_eval.item():6.2f}%"
            f" | "
            f"H weight="
            f"{effective_transfer_weight:.2f}"
        )


# =========================================================
# Restore best Stage-3 model
# =========================================================

if best_stage3_state is None:

    raise RuntimeError(
        "No Stage-3 checkpoint was saved."
    )


model.load_state_dict(
    best_stage3_state
)


print()

print(
    f"Best Stage-3 epoch: "
    f"{best_stage3_epoch}"
)

print(
    f"Best Stage-3 selection loss: "
    f"{best_stage3_loss:.8f}"
)


# =========================================================
#
# FINAL END-TO-END EVALUATION
#
# =========================================================

model.eval()


with torch.no_grad():

    final_z = model.encode_geometry(
        shape_x
    )


    # -----------------------------------------------------
    # Predicted frequencies
    # -----------------------------------------------------

    predicted_log_factors = (
        model.predict_frequencies(
            final_z
        )
    )


    (
        predicted_factors,
        final_factor_mean_error,
        final_factor_max_error,
    ) = factor_errors(
        predicted_log_factors
    )


    # -----------------------------------------------------
    # Predicted residues
    # -----------------------------------------------------

    predicted_residues = (
        model.predict_residues(
            final_z,
            strike_x,
            pickup_x,
        )
    )


    # -----------------------------------------------------
    # Complete NN transfer
    # -----------------------------------------------------

    predicted_response = modal_response(
        log_factors=predicted_log_factors,
        residues=predicted_residues,
        omega=omega,
    )


    # -----------------------------------------------------
    # Residue errors
    # -----------------------------------------------------

    final_residue_mse = F.mse_loss(
        predicted_residues,
        weight_y,
    )


    final_residue_rmse = torch.sqrt(
        final_residue_mse
    )


    final_residue_mae = torch.mean(
        torch.abs(
            predicted_residues
            - weight_y
        )
    )


    # -----------------------------------------------------
    # End-to-end transfer error
    # -----------------------------------------------------

    final_response_loss = transfer_loss(
        predicted_response,
        target_response,
    )


    final_response_rms = (
        torch.sqrt(
            final_response_loss
        )
        * 100.0
    )


    # -----------------------------------------------------
    # Final fixed selection metric
    # -----------------------------------------------------

    final_frequency_loss = F.mse_loss(
        predicted_log_factors,
        log_factor_y,
    )


    final_selection_loss = (
        STAGE3_FREQUENCY_WEIGHT
        * final_frequency_loss

        + STAGE3_RESIDUE_WEIGHT
        * final_residue_mse

        + STAGE3_TRANSFER_WEIGHT
        * final_response_loss
    )


# =========================================================
# Final result
# =========================================================

print()

print(
    "=" * 70
)

print(
    "FINAL END-TO-END RESULT"
)

print(
    "=" * 70
)


print(
    f"Best Stage-3 epoch: "
    f"{best_stage3_epoch}"
)


print(
    f"Selection loss: "
    f"{final_selection_loss.item():.8f}"
)


print(
    f"Modal-factor mean error: "
    f"{final_factor_mean_error.item():.3f}%"
)


print(
    f"Modal-factor max error: "
    f"{final_factor_max_error.item():.3f}%"
)


print(
    f"Residue MSE: "
    f"{final_residue_mse.item():.8f}"
)


print(
    f"Residue RMSE: "
    f"{final_residue_rmse.item():.6f}"
)


print(
    f"Residue MAE: "
    f"{final_residue_mae.item():.6f}"
)


print(
    f"Global transfer H RMS: "
    f"{final_response_rms.item():.3f}%"
)


# =========================================================
# Per-pair transfer diagnostics
# =========================================================

with torch.no_grad():

    error_power_per_pair = torch.mean(
        torch.abs(
            predicted_response
            - target_response
        ) ** 2,
        dim=-1,
    )


    target_power_per_pair = torch.mean(
        torch.abs(
            target_response
        ) ** 2,
        dim=-1,
    )


    relative_rms_per_pair = torch.sqrt(
        error_power_per_pair
        /
        torch.clamp(
            target_power_per_pair,
            min=1e-12,
        )
    )


    global_relative_rms = torch.sqrt(
        torch.sum(
            error_power_per_pair
        )
        /
        torch.sum(
            target_power_per_pair
        )
    )


    flat_relative_rms = (
        relative_rms_per_pair
        .flatten()
    )


    flat_target_power = (
        target_power_per_pair
        .flatten()
    )


    sorted_rms, _ = torch.sort(
        flat_relative_rms
    )


    n = sorted_rms.numel()


    median_rms = sorted_rms[
        n // 2
    ]


    p90_rms = sorted_rms[
        int(
            0.90
            * (n - 1)
        )
    ]


    worst_index = torch.argmax(
        flat_relative_rms
    )


    weakest_index = torch.argmin(
        flat_target_power
    )


print()

print(
    "=" * 70
)

print(
    "END-TO-END TRANSFER DIAGNOSTICS"
)

print(
    "=" * 70
)


print(
    f"Global energy-weighted RMS: "
    f"{global_relative_rms.item() * 100.0:.2f}%"
)


print(
    f"Median pair RMS: "
    f"{median_rms.item() * 100.0:.2f}%"
)


print(
    f"90th percentile pair RMS: "
    f"{p90_rms.item() * 100.0:.2f}%"
)


print(
    f"Worst pair RMS: "
    f"{flat_relative_rms[worst_index].item() * 100.0:.2f}%"
)


print(
    f"Smallest target power: "
    f"{flat_target_power[weakest_index].item():.8e}"
)


print(
    f"Largest target power: "
    f"{flat_target_power.max().item():.8e}"
)


# =========================================================
# Plots
#
# Geometry 0
# Point pair 0
# =========================================================

mode_numbers = np.arange(
    1,
    N_MODES + 1,
)


# ---------------------------------------------------------
# Modal factors
# ---------------------------------------------------------

true_factors_plot = (
    factor_y[0]
    .detach()
    .cpu()
    .numpy()
)


stage1_factors_plot = (
    stage1_predicted_factors[0]
    .detach()
    .cpu()
    .numpy()
)


stage2_factors_plot = (
    stage2_predicted_factors[0]
    .detach()
    .cpu()
    .numpy()
)


final_factors_plot = (
    predicted_factors[0]
    .detach()
    .cpu()
    .numpy()
)


# ---------------------------------------------------------
# Residues
# ---------------------------------------------------------

true_weights_plot = (
    weight_y[
        0,
        0,
    ]
    .detach()
    .cpu()
    .numpy()
)


predicted_weights_plot = (
    predicted_residues[
        0,
        0,
    ]
    .detach()
    .cpu()
    .numpy()
)


# ---------------------------------------------------------
# Transfer
# ---------------------------------------------------------

true_response_plot = (
    target_response[
        0,
        0,
    ]
    .detach()
    .cpu()
    .numpy()
)


predicted_response_plot = (
    predicted_response[
        0,
        0,
    ]
    .detach()
    .cpu()
    .numpy()
)


omega_plot = (
    omega
    .detach()
    .cpu()
    .numpy()
)


# =========================================================
# Plot 1:
# Modal factors through stages
# =========================================================

plt.figure()


plt.plot(
    mode_numbers,
    true_factors_plot,
    "o-",
    label="FEM",
)


plt.plot(
    mode_numbers,
    stage1_factors_plot,
    "x--",
    label="Stage 1",
)


plt.plot(
    mode_numbers,
    stage2_factors_plot,
    ".-",
    label="Stage 2",
)


plt.plot(
    mode_numbers,
    final_factors_plot,
    "+--",
    label="Stage 3",
)


plt.xlabel(
    "Mode"
)


plt.ylabel(
    "Modal factor"
)


plt.title(
    "Modal-factor preservation"
)


plt.grid(
    True
)


plt.legend()


# =========================================================
# Plot 2:
# Final residues
# =========================================================

plt.figure()


plt.plot(
    mode_numbers,
    true_weights_plot,
    "o-",
    label="FEM residue",
)


plt.plot(
    mode_numbers,
    predicted_weights_plot,
    "x--",
    label="NN residue",
)


plt.xlabel(
    "Mode"
)


plt.ylabel(
    "Modal residue"
)


plt.title(
    "Stage 3 end-to-end: modal residues"
)


plt.grid(
    True
)


plt.legend()


# =========================================================
# Plot 3:
# Final transfer
# =========================================================

true_magnitude = np.abs(
    true_response_plot
)


predicted_magnitude = np.abs(
    predicted_response_plot
)


reference = np.max(
    true_magnitude
)


true_db = (
    20.0
    * np.log10(
        true_magnitude
        / reference
        + 1e-12
    )
)


predicted_db = (
    20.0
    * np.log10(
        predicted_magnitude
        / reference
        + 1e-12
    )
)


plt.figure()


plt.plot(
    omega_plot,
    true_db,
    label="FEM target",
)


plt.plot(
    omega_plot,
    predicted_db,
    label="NN end-to-end",
)


plt.xlabel(
    "Normalized frequency Ω"
)


plt.ylabel(
    "Magnitude [dB rel.]"
)


plt.title(
    "Stage 3 end-to-end: strike → pickup"
)


plt.ylim(
    -60.0,
    5.0,
)


plt.grid(
    True
)


plt.legend()


plt.show()