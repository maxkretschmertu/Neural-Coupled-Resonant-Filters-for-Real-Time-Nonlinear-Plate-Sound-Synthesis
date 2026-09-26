from pathlib import Path
import copy
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
)

from model_frequency_v2 import (
    FrequencyNetV2,
)


# =========================================================
# Settings
# =========================================================

N_MODES = 32

BATCH_SIZE = 32

MAX_EPOCHS = 1000

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

EARLY_STOPPING_PATIENCE = 150

SCHEDULER_PATIENCE = 30

PRINT_EVERY = 10

RANDOM_SEED = 0


# ---------------------------------------------------------
# Physical response diagnostic
# ---------------------------------------------------------

DAMPING_RATIO = 0.02

N_FREQUENCY_POINTS = 128


# =========================================================
# Files
# =========================================================

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


CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "frequency_v2_best.pt"
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

class FrequencyDataset(
    Dataset
):

    def __init__(
        self,
        data,
        indices,
    ):

        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )


        self.masks = torch.from_numpy(
            np.asarray(
                data[
                    "masks"
                ][
                    self.indices
                ],
                dtype=np.float32,
            )
        )


        self.factors = torch.from_numpy(
            np.asarray(
                data[
                    "factors"
                ][
                    self.indices
                ],
                dtype=np.float32,
            )
        )


        # Exact FEM residues are NOT training targets here.
        #
        # They are loaded only for the physical frequency
        # diagnostic:
        #
        # NN mu + exact r -> H
        self.modal_weights = torch.from_numpy(
            np.asarray(
                data[
                    "modal_weights"
                ][
                    self.indices
                ],
                dtype=np.float32,
            )
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

            self.modal_weights[
                index
            ],
        )


# =========================================================
# Load dataset
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


train_dataset = FrequencyDataset(
    data=data,
    indices=train_indices,
)


val_dataset = FrequencyDataset(
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
# Frequency-loss weights
#
# Modes 1-4:
#     weight 4
#
# Modes 5-8:
#     weight 2
#
# Modes 9-32:
#     weight 1
#
# Normalize to mean = 1 so the overall loss magnitude
# stays comparable to ordinary MSE.
# =========================================================

mode_weights = torch.ones(
    N_MODES,
    dtype=torch.float32,
    device=device,
)


mode_weights[
    0:4
] = 4.0


mode_weights[
    4:8
] = 2.0


mode_weights = (
    mode_weights
    / torch.mean(
        mode_weights
    )
)


print()

print(
    "Mode weights:"
)

print(
    mode_weights
    .detach()
    .cpu()
    .numpy()
)


# =========================================================
# Physical frequency range
#
# Use TRAIN data only.
# =========================================================

train_factors_np = np.asarray(
    data[
        "factors"
    ][
        train_indices
    ],
    dtype=np.float32,
)


omega_min = (
    0.5
    * np.min(
        train_factors_np[
            :,
            0,
        ]
    )
)


omega_max = (
    0.90
    * np.min(
        train_factors_np[
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
# Weighted frequency loss
# =========================================================

def weighted_frequency_loss(
    predicted_log_factors,
    target_log_factors,
):

    squared_error = (
        predicted_log_factors
        - target_log_factors
    ) ** 2


    weighted_squared_error = (
        squared_error
        * mode_weights[
            None,
            :
        ]
    )


    return torch.mean(
        weighted_squared_error
    )


# =========================================================
# Physical modal response
#
# Used only as diagnostic.
#
# Residues are exact FEM residues.
# =========================================================

def modal_response(
    log_factors,
    residues,
):

    mu = torch.exp(
        log_factors
    )[
        :,
        None,
        None,
        :
    ]


    r = residues[
        :,
        :,
        None,
        :
    ]


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


    return torch.sum(
        r
        / denominator,
        dim=-1,
    )


# =========================================================
# Move batch
# =========================================================

def move_batch(
    batch,
):

    (
        masks,
        factors,
        residues,
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


    residues = residues.to(
        device,
        non_blocking=True,
    )


    return (
        masks,
        factors,
        residues,
    )


# =========================================================
# Evaluation
# =========================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
):

    model.eval()


    weighted_loss_sum = 0.0

    n_geometries = 0


    all_relative_errors = []


    transfer_error_energy = 0.0

    transfer_target_energy = 0.0


    for batch in loader:

        (
            masks,
            factors,
            residues,
        ) = move_batch(
            batch
        )


        batch_size = (
            factors.shape[
                0
            ]
        )


        target_log_factors = torch.log(
            factors
        )


        predicted_log_factors = model(
            masks
        )


        # -------------------------------------------------
        # Weighted frequency loss
        # -------------------------------------------------

        batch_loss = weighted_frequency_loss(
            predicted_log_factors,
            target_log_factors,
        )


        weighted_loss_sum += (
            batch_loss.item()
            * batch_size
        )


        n_geometries += (
            batch_size
        )


        # -------------------------------------------------
        # Physical modal-factor error
        # -------------------------------------------------

        predicted_factors = torch.exp(
            predicted_log_factors
        )


        relative_error = (
            torch.abs(
                predicted_factors
                - factors
            )
            / factors
        )


        all_relative_errors.append(
            relative_error
            .cpu()
            .numpy()
        )


        # -------------------------------------------------
        # Physical response:
        #
        # exact mu + exact residues
        # -------------------------------------------------

        target_response = modal_response(
            log_factors=(
                target_log_factors
            ),
            residues=residues,
        )


        # -------------------------------------------------
        # Frequency-only error:
        #
        # NN mu + EXACT residues
        # -------------------------------------------------

        predicted_response = modal_response(
            log_factors=(
                predicted_log_factors
            ),
            residues=residues,
        )


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


    # =====================================================
    # Modal-factor metrics
    # =====================================================

    relative_errors = np.concatenate(
        all_relative_errors,
        axis=0,
    )


    mean_error = (
        np.mean(
            relative_errors
        )
        * 100.0
    )


    max_error = (
        np.max(
            relative_errors
        )
        * 100.0
    )


    low4_mean = (
        np.mean(
            relative_errors[
                :,
                0:4,
            ]
        )
        * 100.0
    )


    low8_mean = (
        np.mean(
            relative_errors[
                :,
                0:8,
            ]
        )
        * 100.0
    )


    # =====================================================
    # Transfer error
    # =====================================================

    transfer_rms = (
        np.sqrt(
            transfer_error_energy
            / max(
                transfer_target_energy,
                1e-12,
            )
        )
        * 100.0
    )


    return {

        "loss":
            weighted_loss_sum
            / n_geometries,

        "mean_error":
            mean_error,

        "max_error":
            max_error,

        "low4_mean":
            low4_mean,

        "low8_mean":
            low8_mean,

        "transfer_rms":
            transfer_rms,

        "relative_errors":
            relative_errors,
    }


# =========================================================
# Model
# =========================================================

model = FrequencyNetV2(
    n_modes=N_MODES,
    latent_dim=256,
).to(
    device
)


parameter_count = sum(
    parameter.numel()
    for parameter
    in model.parameters()
)


print()

print(
    "Parameters:",
    f"{parameter_count:,}",
)


# =========================================================
# Optimizer
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=SCHEDULER_PATIENCE,
        min_lr=1e-6,
    )
)


# =========================================================
# Training
# =========================================================

best_validation_loss = float(
    "inf"
)


best_state = None

best_epoch = 0


epochs_without_improvement = 0


start_time = time.perf_counter()


print()

print(
    "=" * 82
)

print(
    "FREQUENCY V2 TRAINING"
)

print(
    "=" * 82
)


for epoch in range(
    1,
    MAX_EPOCHS + 1,
):

    model.train()


    train_loss_sum = 0.0

    train_count = 0


    for batch in train_loader:

        (
            masks,
            factors,
            residues,
        ) = move_batch(
            batch
        )


        target_log_factors = torch.log(
            factors
        )


        predicted_log_factors = model(
            masks
        )


        loss = weighted_frequency_loss(
            predicted_log_factors,
            target_log_factors,
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )


        optimizer.step()


        batch_size = (
            factors.shape[
                0
            ]
        )


        train_loss_sum += (
            loss.item()
            * batch_size
        )


        train_count += (
            batch_size
        )


    train_loss = (
        train_loss_sum
        / train_count
    )


    # =====================================================
    # Validation
    # =====================================================

    validation = evaluate(
        model=model,
        loader=val_loader,
    )


    validation_loss = (
        validation[
            "loss"
        ]
    )


    scheduler.step(
        validation_loss
    )


    # =====================================================
    # Checkpoint
    # =====================================================

    if (
        validation_loss
        < best_validation_loss
    ):

        best_validation_loss = (
            validation_loss
        )


        best_epoch = (
            epoch
        )


        best_state = copy.deepcopy(
            model.state_dict()
        )


        epochs_without_improvement = 0


        torch.save(
            {
                "model_state_dict":
                    best_state,

                "epoch":
                    best_epoch,

                "n_modes":
                    N_MODES,

                "latent_dim":
                    256,

                "validation":
                    {
                        key:
                            value
                        for key, value
                        in validation.items()
                        if key
                        != "relative_errors"
                    },

                "omega_min":
                    float(
                        omega_min
                    ),

                "omega_max":
                    float(
                        omega_max
                    ),

                "damping_ratio":
                    DAMPING_RATIO,

                "dataset":
                    DATASET_PATH.name,
            },
            CHECKPOINT_PATH,
        )


    else:

        epochs_without_improvement += 1


    # =====================================================
    # Logging
    # =====================================================

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
            f" | "
            f"train={train_loss:.6f}"
            f" | "
            f"val={validation_loss:.6f}"
            f" | "
            f"mu={validation['mean_error']:5.2f}%"
            f" | "
            f"low4={validation['low4_mean']:5.2f}%"
            f" | "
            f"low8={validation['low8_mean']:5.2f}%"
            f" | "
            f"max={validation['max_error']:6.2f}%"
            f" | "
            f"Hfreq={validation['transfer_rms']:6.2f}%"
            f" | "
            f"lr={current_lr:.2e}"
        )


    # =====================================================
    # Early stopping
    # =====================================================

    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):

        print()

        print(
            "Early stopping."
        )

        break


# =========================================================
# Restore best checkpoint
# =========================================================

if best_state is None:

    raise RuntimeError(
        "No checkpoint was created."
    )


model.load_state_dict(
    best_state
)


# =========================================================
# Final train / validation metrics
# =========================================================

train_result = evaluate(
    model=model,
    loader=train_loader,
)


val_result = evaluate(
    model=model,
    loader=val_loader,
)


elapsed = (
    time.perf_counter()
    - start_time
)


# =========================================================
# Final report
# =========================================================

print()

print(
    "=" * 82
)

print(
    "FINAL FREQUENCY V2 RESULT"
)

print(
    "=" * 82
)


print(
    f"Best epoch: "
    f"{best_epoch}"
)


print(
    f"Training time: "
    f"{elapsed:.1f} s"
)


print()

print(
    "TRAIN"
)

print(
    f"  modal-factor mean: "
    f"{train_result['mean_error']:.3f}%"
)

print(
    f"  modes 1-4 mean:    "
    f"{train_result['low4_mean']:.3f}%"
)

print(
    f"  modes 1-8 mean:    "
    f"{train_result['low8_mean']:.3f}%"
)

print(
    f"  modal-factor max:  "
    f"{train_result['max_error']:.3f}%"
)

print(
    f"  frequency-only "
    f"H RMS: "
    f"{train_result['transfer_rms']:.3f}%"
)


print()

print(
    "VALIDATION"
)

print(
    f"  modal-factor mean: "
    f"{val_result['mean_error']:.3f}%"
)

print(
    f"  modes 1-4 mean:    "
    f"{val_result['low4_mean']:.3f}%"
)

print(
    f"  modes 1-8 mean:    "
    f"{val_result['low8_mean']:.3f}%"
)

print(
    f"  modal-factor max:  "
    f"{val_result['max_error']:.3f}%"
)

print(
    f"  frequency-only "
    f"H RMS: "
    f"{val_result['transfer_rms']:.3f}%"
)


# =========================================================
# Per-mode diagnostics
# =========================================================

relative_errors = (
    val_result[
        "relative_errors"
    ]
)


mean_per_mode = (
    np.mean(
        relative_errors,
        axis=0,
    )
    * 100.0
)


median_per_mode = (
    np.median(
        relative_errors,
        axis=0,
    )
    * 100.0
)


p90_per_mode = (
    np.percentile(
        relative_errors,
        90.0,
        axis=0,
    )
    * 100.0
)


max_per_mode = (
    np.max(
        relative_errors,
        axis=0,
    )
    * 100.0
)


print()

print(
    "=" * 82
)

print(
    "VALIDATION ERROR BY MODE"
)

print(
    "=" * 82
)


print(
    "mode | mean [%] | median [%] | p90 [%] | max [%]"
)

print(
    "-" * 58
)


for mode in range(
    N_MODES
):

    print(
        f"{mode + 1:4d}"
        f" | "
        f"{mean_per_mode[mode]:8.3f}"
        f" | "
        f"{median_per_mode[mode]:10.3f}"
        f" | "
        f"{p90_per_mode[mode]:7.3f}"
        f" | "
        f"{max_per_mode[mode]:7.3f}"
    )


# =========================================================
# Verify ordering
# =========================================================

with torch.no_grad():

    all_ordered = True


    for batch in val_loader:

        (
            masks,
            factors,
            residues,
        ) = move_batch(
            batch
        )


        log_factors = model(
            masks
        )


        predicted_factors = torch.exp(
            log_factors
        )


        differences = (
            predicted_factors[
                :,
                1:
            ]
            - predicted_factors[
                :,
                :-1
            ]
        )


        if torch.any(
            differences <= 0.0
        ):

            all_ordered = False

            break


print()

print(
    "All predicted modal factors ordered:",
    all_ordered,
)


# =========================================================
# Plot
# =========================================================

mode_numbers = np.arange(
    1,
    N_MODES + 1,
)


plt.figure(
    figsize=(
        10,
        5,
    )
)


plt.plot(
    mode_numbers,
    mean_per_mode,
    "o-",
    label="Mean",
)


plt.plot(
    mode_numbers,
    p90_per_mode,
    "x--",
    label="90th percentile",
)


plt.plot(
    mode_numbers,
    max_per_mode,
    ".-",
    label="Maximum",
)


plt.xlabel(
    "Mode"
)


plt.ylabel(
    "Modal-factor error [%]"
)


plt.title(
    "Frequency V2 validation error by mode"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


# =========================================================
# One validation example
# =========================================================

example_index = 0


with torch.no_grad():

    example_mask = torch.from_numpy(
        np.asarray(
            data[
                "masks"
            ][
                val_indices[
                    example_index
                ]
            ],
            dtype=np.float32,
        )
    )[
        None,
        None,
        :,
        :
    ].to(
        device
    )


    example_true = np.asarray(
        data[
            "factors"
        ][
            val_indices[
                example_index
            ]
        ],
        dtype=np.float32,
    )


    example_predicted = (
        torch.exp(
            model(
                example_mask
            )
        )[
            0
        ]
        .cpu()
        .numpy()
    )


plt.figure(
    figsize=(
        10,
        5,
    )
)


plt.plot(
    mode_numbers,
    example_true,
    "o-",
    label="FEM",
)


plt.plot(
    mode_numbers,
    example_predicted,
    "x--",
    label="FrequencyNetV2",
)


plt.xlabel(
    "Mode"
)


plt.ylabel(
    "Modal factor"
)


plt.title(
    "Frequency V2: validation example"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()