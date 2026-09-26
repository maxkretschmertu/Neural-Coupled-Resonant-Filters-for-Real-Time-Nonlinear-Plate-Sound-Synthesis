from pathlib import Path
import copy

import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from model import ModalNet


# =========================================================
# Settings
# =========================================================

N_MODES = 32
N_FREQUENCY_POINTS = 128

DAMPING_RATIO = 0.02

BATCH_SIZE = 32

RANDOM_SEED = 0


# ---------------------------------------------------------
# Stage 1
#
# Geometry -> modal frequencies
# ---------------------------------------------------------

STAGE1_EPOCHS = 400
STAGE1_LR = 1e-3
STAGE1_PATIENCE = 60


# ---------------------------------------------------------
# Stage 2
#
# Multitask training.
#
# Transfer still uses exact FEM frequencies.
# ---------------------------------------------------------

STAGE2_EPOCHS = 500

STAGE2_ENCODER_LR = 1e-4
STAGE2_FREQUENCY_LR = 1e-4
STAGE2_PAIR_LR = 5e-4

STAGE2_FREQUENCY_WEIGHT = 25.0
STAGE2_RESIDUE_WEIGHT = 1.0
STAGE2_TRANSFER_WEIGHT = 1.0

STAGE2_PATIENCE = 80


# ---------------------------------------------------------
# Stage 3
#
# Full end-to-end training.
#
# Both modal frequencies and residues come from NN.
# ---------------------------------------------------------

STAGE3_EPOCHS = 600

STAGE3_ENCODER_LR = 2e-5
STAGE3_FREQUENCY_LR = 2e-5
STAGE3_PAIR_LR = 1e-4

STAGE3_FREQUENCY_WEIGHT = 25.0
STAGE3_RESIDUE_WEIGHT = 0.25
STAGE3_TRANSFER_WEIGHT = 1.0

STAGE3_TRANSFER_RAMP_EPOCHS = 100
STAGE3_PATIENCE = 100


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

PRINT_EVERY = 10


# ---------------------------------------------------------
# Files
# ---------------------------------------------------------

ROOT = Path(__file__).parent

DATASET_PATH = (
    ROOT
    / "plate_dataset_pilot_500.npz"
)

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


STAGE1_CHECKPOINT = (
    CHECKPOINT_DIR
    / "modalnet_pilot_stage1.pt"
)

STAGE2_CHECKPOINT = (
    CHECKPOINT_DIR
    / "modalnet_pilot_stage2.pt"
)

STAGE3_CHECKPOINT = (
    CHECKPOINT_DIR
    / "modalnet_pilot_stage3.pt"
)

FINAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "modalnet_pilot_final.pt"
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

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
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
# Dataset
# =========================================================

class PlateDataset(
    Dataset
):

    def __init__(
        self,
        data,
        indices,
    ):

        indices = np.asarray(
            indices,
            dtype=np.int64,
        )


        self.masks = torch.from_numpy(
            np.asarray(
                data["masks"][
                    indices
                ],
                dtype=np.float32,
            )
        )


        self.factors = torch.from_numpy(
            np.asarray(
                data["factors"][
                    indices
                ],
                dtype=np.float32,
            )
        )


        self.strike_points = torch.from_numpy(
            np.asarray(
                data["strike_points"][
                    indices
                ],
                dtype=np.float32,
            )
        )


        self.pickup_points = torch.from_numpy(
            np.asarray(
                data["pickup_points"][
                    indices
                ],
                dtype=np.float32,
            )
        )


        self.modal_weights = torch.from_numpy(
            np.asarray(
                data["modal_weights"][
                    indices
                ],
                dtype=np.float32,
            )
        )


        self.indices = torch.from_numpy(
            indices
        )


    def __len__(
        self,
    ):

        return len(
            self.indices
        )


    def __getitem__(
        self,
        index,
    ):

        return (
            self.masks[
                index
            ],
            self.factors[
                index
            ],
            self.strike_points[
                index
            ],
            self.pickup_points[
                index
            ],
            self.modal_weights[
                index
            ],
            self.indices[
                index
            ],
        )


# =========================================================
# Load NPZ
# =========================================================

data = np.load(
    DATASET_PATH
)


train_indices = np.asarray(
    data[
        "train_indices"
    ],
    dtype=np.int64,
)


val_indices = np.asarray(
    data[
        "val_indices"
    ],
    dtype=np.int64,
)


train_dataset = PlateDataset(
    data=data,
    indices=train_indices,
)


val_dataset = PlateDataset(
    data=data,
    indices=val_indices,
)


print()

print(
    "Train geometries:",
    len(
        train_dataset
    ),
)

print(
    "Validation geometries:",
    len(
        val_dataset
    ),
)


# =========================================================
# DataLoaders
#
# num_workers=0 is deliberate:
# easiest / safest on Windows.
# =========================================================

generator = torch.Generator()

generator.manual_seed(
    RANDOM_SEED
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(
        device.type
        == "cuda"
    ),
    generator=generator,
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(
        device.type
        == "cuda"
    ),
)


# =========================================================
# Response frequency axis
#
# IMPORTANT:
#
# Build this only from TRAINING data.
# Validation targets do not define the training domain.
# =========================================================

train_factors = np.asarray(
    data["factors"][
        train_indices
    ],
    dtype=np.float32,
)


omega_min = (
    0.5
    * np.min(
        train_factors[
            :,
            0,
        ]
    )
)


omega_max = (
    0.90
    * np.min(
        train_factors[
            :,
            N_MODES - 1,
        ]
    )
)


omega = torch.linspace(
    float(
        omega_min
    ),
    float(
        omega_max
    ),
    N_FREQUENCY_POINTS,
    dtype=torch.float32,
    device=device,
)


print()

print(
    "Omega range:",
    f"{omega_min:.3f}",
    "...",
    f"{omega_max:.3f}",
)


# =========================================================
# Move batch to device
# =========================================================

def move_batch(
    batch,
):

    (
        masks,
        factors,
        strike,
        pickup,
        weights,
        indices,
    ) = batch


    masks = (
        masks
        .unsqueeze(
            1
        )
        .to(
            device,
            non_blocking=True,
        )
    )


    factors = factors.to(
        device,
        non_blocking=True,
    )


    strike = strike.to(
        device,
        non_blocking=True,
    )


    pickup = pickup.to(
        device,
        non_blocking=True,
    )


    weights = weights.to(
        device,
        non_blocking=True,
    )


    return (
        masks,
        factors,
        strike,
        pickup,
        weights,
        indices,
    )


# =========================================================
# Physical modal response
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
        None,
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


    response = torch.sum(
        r
        / denominator,
        dim=-1,
    )


    return response


# =========================================================
# Global energy-weighted transfer loss
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
# Validation
#
# response_mode:
#
#     "exact"
#         target FEM frequencies +
#         predicted residues
#
#     "predicted"
#         predicted frequencies +
#         predicted residues
#
# =========================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    response_mode,
):

    model.eval()


    frequency_squared_error = 0.0
    frequency_count = 0


    residue_squared_error = 0.0
    residue_absolute_error = 0.0
    residue_count = 0


    factor_relative_sum = 0.0
    factor_relative_count = 0

    factor_relative_max = 0.0


    transfer_error_energy = 0.0
    transfer_target_energy = 0.0


    pair_relative_rms = []


    for batch in loader:

        (
            masks,
            factors,
            strike,
            pickup,
            weights,
            indices,
        ) = move_batch(
            batch
        )


        log_factor_target = torch.log(
            factors
        )


        (
            predicted_log_factors,
            predicted_residues,
        ) = model(
            masks,
            strike,
            pickup,
        )


        # -------------------------------------------------
        # Frequency MSE
        # -------------------------------------------------

        frequency_squared_error += (
            torch.sum(
                (
                    predicted_log_factors
                    - log_factor_target
                ) ** 2
            )
            .item()
        )


        frequency_count += (
            predicted_log_factors.numel()
        )


        # -------------------------------------------------
        # Physical modal-factor errors
        # -------------------------------------------------

        predicted_factors = torch.exp(
            predicted_log_factors
        )


        relative_factor_error = (
            torch.abs(
                predicted_factors
                - factors
            )
            / factors
        )


        factor_relative_sum += (
            relative_factor_error
            .sum()
            .item()
        )


        factor_relative_count += (
            relative_factor_error
            .numel()
        )


        factor_relative_max = max(
            factor_relative_max,
            relative_factor_error
            .max()
            .item(),
        )


        # -------------------------------------------------
        # Residue metrics
        # -------------------------------------------------

        difference = (
            predicted_residues
            - weights
        )


        residue_squared_error += (
            torch.sum(
                difference**2
            )
            .item()
        )


        residue_absolute_error += (
            torch.sum(
                torch.abs(
                    difference
                )
            )
            .item()
        )


        residue_count += (
            difference.numel()
        )


        # -------------------------------------------------
        # Exact target response
        # -------------------------------------------------

        target_response = modal_response(
            log_factors=log_factor_target,
            residues=weights,
            omega=omega,
        )


        # -------------------------------------------------
        # Predicted response
        # -------------------------------------------------

        if response_mode == "exact":

            response_log_factors = (
                log_factor_target
            )

        elif response_mode == "predicted":

            response_log_factors = (
                predicted_log_factors
            )

        else:

            raise ValueError(
                f"Unknown response mode: "
                f"{response_mode}"
            )


        predicted_response = modal_response(
            log_factors=(
                response_log_factors
            ),
            residues=(
                predicted_residues
            ),
            omega=omega,
        )


        # -------------------------------------------------
        # Global energy
        # -------------------------------------------------

        transfer_error_energy += (
            torch.sum(
                torch.abs(
                    predicted_response
                    - target_response
                ) ** 2
            )
            .item()
        )


        transfer_target_energy += (
            torch.sum(
                torch.abs(
                    target_response
                ) ** 2
            )
            .item()
        )


        # -------------------------------------------------
        # Per-pair diagnostic
        # -------------------------------------------------

        pair_error_power = torch.mean(
            torch.abs(
                predicted_response
                - target_response
            ) ** 2,
            dim=-1,
        )


        pair_target_power = torch.mean(
            torch.abs(
                target_response
            ) ** 2,
            dim=-1,
        )


        pair_rms = torch.sqrt(
            pair_error_power
            /
            torch.clamp(
                pair_target_power,
                min=1e-12,
            )
        )


        pair_relative_rms.append(
            pair_rms
            .detach()
            .cpu()
            .reshape(
                -1
            )
        )


    # =====================================================
    # Aggregate
    # =====================================================

    frequency_mse = (
        frequency_squared_error
        / frequency_count
    )


    residue_mse = (
        residue_squared_error
        / residue_count
    )


    residue_rmse = np.sqrt(
        residue_mse
    )


    residue_mae = (
        residue_absolute_error
        / residue_count
    )


    factor_mean_error = (
        factor_relative_sum
        / factor_relative_count
        * 100.0
    )


    factor_max_error = (
        factor_relative_max
        * 100.0
    )


    transfer_loss_value = (
        transfer_error_energy
        /
        max(
            transfer_target_energy,
            1e-12,
        )
    )


    transfer_rms = (
        np.sqrt(
            transfer_loss_value
        )
        * 100.0
    )


    pair_relative_rms = torch.cat(
        pair_relative_rms
    ).numpy()


    median_pair_rms = (
        np.median(
            pair_relative_rms
        )
        * 100.0
    )


    p90_pair_rms = (
        np.percentile(
            pair_relative_rms,
            90.0,
        )
        * 100.0
    )


    return {

        "frequency_mse":
            frequency_mse,

        "factor_mean_error":
            factor_mean_error,

        "factor_max_error":
            factor_max_error,

        "residue_mse":
            residue_mse,

        "residue_rmse":
            residue_rmse,

        "residue_mae":
            residue_mae,

        "transfer_loss":
            transfer_loss_value,

        "transfer_rms":
            transfer_rms,

        "median_pair_rms":
            median_pair_rms,

        "p90_pair_rms":
            p90_pair_rms,
    }


# =========================================================
# Checkpoint helper
# =========================================================

def save_checkpoint(
    path,
    model,
    stage,
    epoch,
    validation,
):

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "stage":
                stage,

            "epoch":
                epoch,

            "validation":
                validation,

            "n_modes":
                N_MODES,

            "damping_ratio":
                DAMPING_RATIO,

            "omega_min":
                float(
                    omega_min
                ),

            "omega_max":
                float(
                    omega_max
                ),

            "n_frequency_points":
                N_FREQUENCY_POINTS,

            "dataset":
                DATASET_PATH.name,
        },
        path,
    )


# =========================================================
# Model
# =========================================================

model = ModalNet(
    n_modes=N_MODES
).to(
    device
)


# #########################################################
#
# STAGE 1
#
# Geometry -> modal frequencies
#
# #########################################################

print()

print(
    "=" * 78
)

print(
    "STAGE 1: FREQUENCY PRETRAINING"
)

print(
    "=" * 78
)


optimizer = torch.optim.Adam(
    list(
        model.encoder.parameters()
    )
    + list(
        model.frequency_head.parameters()
    ),
    lr=STAGE1_LR,
)


scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=20,
        min_lr=1e-6,
    )
)


best_score = float(
    "inf"
)

best_state = None

epochs_without_improvement = 0


for epoch in range(
    1,
    STAGE1_EPOCHS + 1,
):

    model.train()

    training_loss_sum = 0.0

    n_batches = 0


    for batch in train_loader:

        (
            masks,
            factors,
            strike,
            pickup,
            weights,
            indices,
        ) = move_batch(
            batch
        )


        log_factor_target = torch.log(
            factors
        )


        z = model.encode_geometry(
            masks
        )


        predicted_log_factors = (
            model.predict_frequencies(
                z
            )
        )


        loss = F.mse_loss(
            predicted_log_factors,
            log_factor_target,
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        training_loss_sum += (
            loss.item()
        )

        n_batches += 1


    # -----------------------------------------------------
    # Validation every epoch
    # -----------------------------------------------------

    validation = evaluate(
        model=model,
        loader=val_loader,
        response_mode="predicted",
    )


    score = validation[
        "frequency_mse"
    ]


    scheduler.step(
        score
    )


    # -----------------------------------------------------
    # Best Stage-1 checkpoint
    # -----------------------------------------------------

    if score < best_score:

        best_score = score

        best_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0


        save_checkpoint(
            path=STAGE1_CHECKPOINT,
            model=model,
            stage=1,
            epoch=epoch,
            validation=validation,
        )


    else:

        epochs_without_improvement += 1


    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    if (
        epoch == 1
        or epoch % PRINT_EVERY == 0
    ):

        current_lr = (
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )


        print(
            f"epoch {epoch:4d}"
            f" | train={training_loss_sum / n_batches:.6f}"
            f" | val freq={validation['frequency_mse']:.6f}"
            f" | mu mean={validation['factor_mean_error']:6.2f}%"
            f" | mu max={validation['factor_max_error']:6.2f}%"
            f" | lr={current_lr:.2e}"
        )


    # -----------------------------------------------------
    # Early stopping
    # -----------------------------------------------------

    if (
        epochs_without_improvement
        >= STAGE1_PATIENCE
    ):

        print()

        print(
            "Stage 1 early stopping."
        )

        break


# ---------------------------------------------------------
# Restore best Stage 1
# ---------------------------------------------------------

model.load_state_dict(
    best_state
)


stage1_validation = evaluate(
    model=model,
    loader=val_loader,
    response_mode="predicted",
)


print()

print(
    "FINAL STAGE-1 VALIDATION"
)

print(
    f"mu mean: "
    f"{stage1_validation['factor_mean_error']:.3f}%"
)

print(
    f"mu max:  "
    f"{stage1_validation['factor_max_error']:.3f}%"
)


# #########################################################
#
# STAGE 2
#
# Multitask:
#
# frequency + residue + physical transfer
#
# Transfer uses exact FEM frequencies.
#
# #########################################################

print()

print(
    "=" * 78
)

print(
    "STAGE 2: MULTITASK TRAINING"
)

print(
    "=" * 78
)


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


scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=25,
        min_lr=1e-6,
    )
)


best_score = float(
    "inf"
)

best_state = None

epochs_without_improvement = 0


for epoch in range(
    1,
    STAGE2_EPOCHS + 1,
):

    model.train()

    training_loss_sum = 0.0

    n_batches = 0


    for batch in train_loader:

        (
            masks,
            factors,
            strike,
            pickup,
            weights,
            indices,
        ) = move_batch(
            batch
        )


        log_factor_target = torch.log(
            factors
        )


        # -------------------------------------------------
        # Shared encoder
        # -------------------------------------------------

        z = model.encode_geometry(
            masks
        )


        # -------------------------------------------------
        # Frequency prediction
        # -------------------------------------------------

        predicted_log_factors = (
            model.predict_frequencies(
                z
            )
        )


        # -------------------------------------------------
        # Residue prediction
        # -------------------------------------------------

        predicted_residues = (
            model.predict_residues(
                z,
                strike,
                pickup,
            )
        )


        # -------------------------------------------------
        # Frequency loss
        # -------------------------------------------------

        frequency_loss = F.mse_loss(
            predicted_log_factors,
            log_factor_target,
        )


        # -------------------------------------------------
        # Residue loss
        # -------------------------------------------------

        residue_loss = F.mse_loss(
            predicted_residues,
            weights,
        )


        # -------------------------------------------------
        # Exact FEM target response
        # -------------------------------------------------

        target_response = modal_response(
            log_factors=log_factor_target,
            residues=weights,
            omega=omega,
        )


        # -------------------------------------------------
        # Predicted residue response
        #
        # Exact FEM frequencies intentionally used.
        # -------------------------------------------------

        predicted_response = modal_response(
            log_factors=log_factor_target,
            residues=predicted_residues,
            omega=omega,
        )


        response_loss = transfer_loss(
            predicted_response,
            target_response,
        )


        # -------------------------------------------------
        # Multitask loss
        # -------------------------------------------------

        loss = (
            STAGE2_FREQUENCY_WEIGHT
            * frequency_loss

            + STAGE2_RESIDUE_WEIGHT
            * residue_loss

            + STAGE2_TRANSFER_WEIGHT
            * response_loss
        )


        optimizer.zero_grad()

        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )


        optimizer.step()


        training_loss_sum += (
            loss.item()
        )

        n_batches += 1


    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    validation = evaluate(
        model=model,
        loader=val_loader,
        response_mode="exact",
    )


    # Fixed validation objective
    score = (
        STAGE2_FREQUENCY_WEIGHT
        * validation[
            "frequency_mse"
        ]

        + STAGE2_RESIDUE_WEIGHT
        * validation[
            "residue_mse"
        ]

        + STAGE2_TRANSFER_WEIGHT
        * validation[
            "transfer_loss"
        ]
    )


    scheduler.step(
        score
    )


    # -----------------------------------------------------
    # Checkpoint
    # -----------------------------------------------------

    if score < best_score:

        best_score = score

        best_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0


        save_checkpoint(
            path=STAGE2_CHECKPOINT,
            model=model,
            stage=2,
            epoch=epoch,
            validation=validation,
        )


    else:

        epochs_without_improvement += 1


    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    if (
        epoch == 1
        or epoch % PRINT_EVERY == 0
    ):

        print(
            f"epoch {epoch:4d}"
            f" | train={training_loss_sum / n_batches:.5f}"
            f" | mu={validation['factor_mean_error']:5.2f}%"
            f" | residue MAE={validation['residue_mae']:.4f}"
            f" | H RMS={validation['transfer_rms']:6.2f}%"
            f" | H med={validation['median_pair_rms']:6.2f}%"
        )


    # -----------------------------------------------------
    # Early stopping
    # -----------------------------------------------------

    if (
        epochs_without_improvement
        >= STAGE2_PATIENCE
    ):

        print()

        print(
            "Stage 2 early stopping."
        )

        break


# ---------------------------------------------------------
# Restore best Stage 2
# ---------------------------------------------------------

model.load_state_dict(
    best_state
)


stage2_validation = evaluate(
    model=model,
    loader=val_loader,
    response_mode="exact",
)


print()

print(
    "FINAL STAGE-2 VALIDATION"
)

print(
    f"mu mean:     "
    f"{stage2_validation['factor_mean_error']:.3f}%"
)

print(
    f"mu max:      "
    f"{stage2_validation['factor_max_error']:.3f}%"
)

print(
    f"residue MAE: "
    f"{stage2_validation['residue_mae']:.4f}"
)

print(
    f"H RMS:       "
    f"{stage2_validation['transfer_rms']:.3f}%"
)

print(
    f"H median:    "
    f"{stage2_validation['median_pair_rms']:.3f}%"
)


# #########################################################
#
# STAGE 3
#
# Full end-to-end training
#
# predicted mu + predicted residues
#
# #########################################################

print()

print(
    "=" * 78
)

print(
    "STAGE 3: FULL END-TO-END"
)

print(
    "=" * 78
)


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


scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=30,
        min_lr=1e-6,
    )
)


best_score = float(
    "inf"
)

best_state = None

best_epoch = 0

epochs_without_improvement = 0


for epoch in range(
    1,
    STAGE3_EPOCHS + 1,
):

    model.train()

    training_loss_sum = 0.0

    n_batches = 0


    # -----------------------------------------------------
    # Transfer ramp
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


    for batch in train_loader:

        (
            masks,
            factors,
            strike,
            pickup,
            weights,
            indices,
        ) = move_batch(
            batch
        )


        log_factor_target = torch.log(
            factors
        )


        # -------------------------------------------------
        # Complete NN forward
        # -------------------------------------------------

        (
            predicted_log_factors,
            predicted_residues,
        ) = model(
            masks,
            strike,
            pickup,
        )


        # -------------------------------------------------
        # Frequency loss
        # -------------------------------------------------

        frequency_loss = F.mse_loss(
            predicted_log_factors,
            log_factor_target,
        )


        # -------------------------------------------------
        # Residue auxiliary loss
        # -------------------------------------------------

        residue_loss = F.mse_loss(
            predicted_residues,
            weights,
        )


        # -------------------------------------------------
        # FEM target response
        # -------------------------------------------------

        target_response = modal_response(
            log_factors=log_factor_target,
            residues=weights,
            omega=omega,
        )


        # -------------------------------------------------
        # FULL NN response
        #
        # No FEM modal parameters in this branch.
        # -------------------------------------------------

        predicted_response = modal_response(
            log_factors=predicted_log_factors,
            residues=predicted_residues,
            omega=omega,
        )


        response_loss = transfer_loss(
            predicted_response,
            target_response,
        )


        # -------------------------------------------------
        # Training loss
        # -------------------------------------------------

        loss = (
            STAGE3_FREQUENCY_WEIGHT
            * frequency_loss

            + STAGE3_RESIDUE_WEIGHT
            * residue_loss

            + effective_transfer_weight
            * response_loss
        )


        optimizer.zero_grad()

        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )


        optimizer.step()


        training_loss_sum += (
            loss.item()
        )

        n_batches += 1


    # -----------------------------------------------------
    # Validation:
    #
    # FULL predicted response.
    # -----------------------------------------------------

    validation = evaluate(
        model=model,
        loader=val_loader,
        response_mode="predicted",
    )


    # -----------------------------------------------------
    # IMPORTANT:
    #
    # Checkpoint selection always uses FULL final transfer
    # weight, even while training is still ramping it in.
    #
    # Therefore every epoch is comparable.
    # -----------------------------------------------------

    selection_score = (
        STAGE3_FREQUENCY_WEIGHT
        * validation[
            "frequency_mse"
        ]

        + STAGE3_RESIDUE_WEIGHT
        * validation[
            "residue_mse"
        ]

        + STAGE3_TRANSFER_WEIGHT
        * validation[
            "transfer_loss"
        ]
    )


    scheduler.step(
        selection_score
    )


    # -----------------------------------------------------
    # Best validation model
    # -----------------------------------------------------

    if selection_score < best_score:

        best_score = selection_score

        best_epoch = epoch

        best_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0


        save_checkpoint(
            path=STAGE3_CHECKPOINT,
            model=model,
            stage=3,
            epoch=epoch,
            validation=validation,
        )


    else:

        # Do not early-stop before transfer ramp is done.
        if (
            epoch
            >= STAGE3_TRANSFER_RAMP_EPOCHS
        ):

            epochs_without_improvement += 1


    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    if (
        epoch == 1
        or epoch % PRINT_EVERY == 0
    ):

        print(
            f"epoch {epoch:4d}"
            f" | train={training_loss_sum / n_batches:.5f}"
            f" | select={selection_score:.5f}"
            f" | mu={validation['factor_mean_error']:5.2f}%"
            f" | mu max={validation['factor_max_error']:6.2f}%"
            f" | residue MAE={validation['residue_mae']:.4f}"
            f" | H RMS={validation['transfer_rms']:6.2f}%"
            f" | H med={validation['median_pair_rms']:6.2f}%"
            f" | H90={validation['p90_pair_rms']:6.2f}%"
            f" | H weight={effective_transfer_weight:.2f}"
        )


    # -----------------------------------------------------
    # Early stopping
    # -----------------------------------------------------

    if (
        epoch
        >= STAGE3_TRANSFER_RAMP_EPOCHS
        and epochs_without_improvement
        >= STAGE3_PATIENCE
    ):

        print()

        print(
            "Stage 3 early stopping."
        )

        break


# =========================================================
# Restore best end-to-end validation checkpoint
# =========================================================

if best_state is None:

    raise RuntimeError(
        "No Stage-3 checkpoint was created."
    )


model.load_state_dict(
    best_state
)


# =========================================================
# Final TRAIN metrics
# =========================================================

train_final = evaluate(
    model=model,
    loader=train_loader,
    response_mode="predicted",
)


# =========================================================
# Final VALIDATION metrics
# =========================================================

val_final = evaluate(
    model=model,
    loader=val_loader,
    response_mode="predicted",
)


# =========================================================
# Final report
# =========================================================

print()

print(
    "=" * 78
)

print(
    "FINAL PILOT RESULT"
)

print(
    "=" * 78
)


print(
    f"Best Stage-3 epoch: "
    f"{best_epoch}"
)


print()

print(
    "TRAIN"
)

print(
    f"  mu mean:      "
    f"{train_final['factor_mean_error']:.3f}%"
)

print(
    f"  mu max:       "
    f"{train_final['factor_max_error']:.3f}%"
)

print(
    f"  residue MAE:  "
    f"{train_final['residue_mae']:.5f}"
)

print(
    f"  H RMS:        "
    f"{train_final['transfer_rms']:.3f}%"
)

print(
    f"  H median:     "
    f"{train_final['median_pair_rms']:.3f}%"
)

print(
    f"  H 90th pct.:  "
    f"{train_final['p90_pair_rms']:.3f}%"
)


print()

print(
    "VALIDATION"
)

print(
    f"  mu mean:      "
    f"{val_final['factor_mean_error']:.3f}%"
)

print(
    f"  mu max:       "
    f"{val_final['factor_max_error']:.3f}%"
)

print(
    f"  residue MAE:  "
    f"{val_final['residue_mae']:.5f}"
)

print(
    f"  H RMS:        "
    f"{val_final['transfer_rms']:.3f}%"
)

print(
    f"  H median:     "
    f"{val_final['median_pair_rms']:.3f}%"
)

print(
    f"  H 90th pct.:  "
    f"{val_final['p90_pair_rms']:.3f}%"
)


# =========================================================
# Save final deployable checkpoint
# =========================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "n_modes":
            N_MODES,

        "damping_ratio":
            DAMPING_RATIO,

        "omega_min":
            float(
                omega_min
            ),

        "omega_max":
            float(
                omega_max
            ),

        "n_frequency_points":
            N_FREQUENCY_POINTS,

        "dataset":
            DATASET_PATH.name,

        "best_stage3_epoch":
            best_epoch,

        "train_metrics":
            train_final,

        "validation_metrics":
            val_final,
    },
    FINAL_CHECKPOINT,
)


print()

print(
    "Saved final checkpoint:"
)

print(
    FINAL_CHECKPOINT
)