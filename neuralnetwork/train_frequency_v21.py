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

MAX_EPOCHS = 500

LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-5

RANDOM_SEED = 0


# ---------------------------------------------------------
# Physical response
# ---------------------------------------------------------

DAMPING_RATIO = 0.02

N_FREQUENCY_POINTS = 128


# ---------------------------------------------------------
# Fine-tuning objective
#
# At the V2 checkpoint:
#
# frequency loss ~ 1.2e-4
# transfer loss  ~ 0.04
#
# Therefore lambda_H = 0.005 gives both terms
# roughly comparable importance.
# ---------------------------------------------------------

TRANSFER_WEIGHT = 0.005

TRANSFER_RAMP_EPOCHS = 50


# ---------------------------------------------------------
# Optimization control
# ---------------------------------------------------------

SCHEDULER_PATIENCE = 30

EARLY_STOPPING_PATIENCE = 120

PRINT_EVERY = 10


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


SOURCE_CHECKPOINT = (
    CHECKPOINT_DIR
    / "frequency_v2_best.pt"
)


BEST_HFREQ_CHECKPOINT = (
    CHECKPOINT_DIR
    / "frequency_v21_best_hfreq.pt"
)


BEST_BALANCED_CHECKPOINT = (
    CHECKPOINT_DIR
    / "frequency_v21_best_balanced.pt"
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


        # Exact FEM modal residues.
        #
        # These are NOT predicted by this network.
        # They are used only to construct the physical
        # transfer response for frequency fine-tuning.
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
# Mode weights
#
# Same weighting as Frequency V2.
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


# =========================================================
# Frequency grid
#
# Same range as previous experiment.
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
# Weighted modal-factor loss
# =========================================================

def weighted_frequency_loss(
    predicted_log_factors,
    target_log_factors,
):

    squared_error = (
        predicted_log_factors
        - target_log_factors
    ) ** 2


    weighted_error = (
        squared_error
        * mode_weights[
            None,
            :
        ]
    )


    return torch.mean(
        weighted_error
    )


# =========================================================
# Modal transfer function
#
# H(Omega) =
#
# sum_k
#
#             r_k
# --------------------------------
# mu_k^2 - Omega^2
# + j 2 zeta mu_k Omega
#
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


    frequency_loss_sum = 0.0
    geometry_count = 0


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
        # Frequency loss
        # -------------------------------------------------

        frequency_loss = (
            weighted_frequency_loss(
                predicted_log_factors,
                target_log_factors,
            )
        )


        frequency_loss_sum += (
            frequency_loss.item()
            * batch_size
        )


        geometry_count += (
            batch_size
        )


        # -------------------------------------------------
        # Relative mu error
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
        # Exact physical response
        # -------------------------------------------------

        target_response = modal_response(
            log_factors=target_log_factors,
            residues=residues,
        )


        # -------------------------------------------------
        # NN mu + exact FEM residues
        # -------------------------------------------------

        predicted_response = modal_response(
            log_factors=predicted_log_factors,
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
    # Aggregate
    # =====================================================

    frequency_loss_value = (
        frequency_loss_sum
        / geometry_count
    )


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


    low4_mean = (
        np.mean(
            relative_errors[
                :,
                :4,
            ]
        )
        * 100.0
    )


    low8_mean = (
        np.mean(
            relative_errors[
                :,
                :8,
            ]
        )
        * 100.0
    )


    max_error = (
        np.max(
            relative_errors
        )
        * 100.0
    )


    transfer_loss_value = (
        transfer_error_energy
        / max(
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


    balanced_score = (
        frequency_loss_value
        + TRANSFER_WEIGHT
        * transfer_loss_value
    )


    return {

        "frequency_loss":
            frequency_loss_value,

        "mean_error":
            mean_error,

        "low4_mean":
            low4_mean,

        "low8_mean":
            low8_mean,

        "max_error":
            max_error,

        "transfer_loss":
            transfer_loss_value,

        "transfer_rms":
            transfer_rms,

        "balanced_score":
            balanced_score,

        "relative_errors":
            relative_errors,
    }


# =========================================================
# Load Frequency V2 checkpoint
# =========================================================

try:

    source_checkpoint = torch.load(
        SOURCE_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

except TypeError:

    source_checkpoint = torch.load(
        SOURCE_CHECKPOINT,
        map_location=device,
    )


latent_dim = int(
    source_checkpoint.get(
        "latent_dim",
        256,
    )
)


model = FrequencyNetV2(
    n_modes=N_MODES,
    latent_dim=latent_dim,
).to(
    device
)


model.load_state_dict(
    source_checkpoint[
        "model_state_dict"
    ]
)


print()

print(
    "Loaded:"
)

print(
    SOURCE_CHECKPOINT
)


print(
    "Source epoch:",
    source_checkpoint.get(
        "epoch",
        "unknown",
    ),
)


# =========================================================
# Baseline evaluation
# =========================================================

baseline_train = evaluate(
    model=model,
    loader=train_loader,
)


baseline_val = evaluate(
    model=model,
    loader=val_loader,
)


print()

print(
    "=" * 82
)

print(
    "V2 BASELINE"
)

print(
    "=" * 82
)


print()

print(
    "TRAIN"
)


print(
    f"  mu mean:   "
    f"{baseline_train['mean_error']:.3f}%"
)


print(
    f"  low4:      "
    f"{baseline_train['low4_mean']:.3f}%"
)


print(
    f"  low8:      "
    f"{baseline_train['low8_mean']:.3f}%"
)


print(
    f"  Hfreq RMS: "
    f"{baseline_train['transfer_rms']:.3f}%"
)


print()

print(
    "VALIDATION"
)


print(
    f"  mu mean:   "
    f"{baseline_val['mean_error']:.3f}%"
)


print(
    f"  low4:      "
    f"{baseline_val['low4_mean']:.3f}%"
)


print(
    f"  low8:      "
    f"{baseline_val['low8_mean']:.3f}%"
)


print(
    f"  Hfreq RMS: "
    f"{baseline_val['transfer_rms']:.3f}%"
)


# =========================================================
# Save initial state
#
# Useful if fine-tuning becomes unstable.
# =========================================================

initial_state = copy.deepcopy(
    model.state_dict()
)


# =========================================================
# Optimizer
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ---------------------------------------------------------
# Scheduler follows validation physical transfer.
# ---------------------------------------------------------

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
# Checkpoint state
# =========================================================

best_hfreq = (
    baseline_val[
        "transfer_rms"
    ]
)


best_hfreq_epoch = 0


best_hfreq_state = copy.deepcopy(
    model.state_dict()
)


best_balanced = (
    baseline_val[
        "balanced_score"
    ]
)


best_balanced_epoch = 0


best_balanced_state = copy.deepcopy(
    model.state_dict()
)


# =========================================================
# History
# =========================================================

history_epoch = []

history_mu = []

history_low4 = []

history_hfreq = []

history_balanced = []


# =========================================================
# Fine tuning
# =========================================================

print()

print(
    "=" * 82
)

print(
    "FREQUENCY V2.1 PHYSICAL FINE-TUNING"
)

print(
    "=" * 82
)


start_time = time.perf_counter()


epochs_without_physical_improvement = 0


for epoch in range(
    1,
    MAX_EPOCHS + 1,
):

    model.train()


    # -----------------------------------------------------
    # Slowly introduce physical loss.
    # -----------------------------------------------------

    transfer_scale = min(
        1.0,
        epoch
        / TRANSFER_RAMP_EPOCHS,
    )


    effective_transfer_weight = (
        TRANSFER_WEIGHT
        * transfer_scale
    )


    training_frequency_sum = 0.0

    training_transfer_sum = 0.0

    training_total_sum = 0.0

    training_count = 0


    for batch in train_loader:

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


        # -------------------------------------------------
        # Predict modal factors
        # -------------------------------------------------

        predicted_log_factors = model(
            masks
        )


        # -------------------------------------------------
        # Frequency supervision
        # -------------------------------------------------

        frequency_loss = (
            weighted_frequency_loss(
                predicted_log_factors,
                target_log_factors,
            )
        )


        # -------------------------------------------------
        # Exact target response
        #
        # exact mu + exact FEM residues
        # -------------------------------------------------

        with torch.no_grad():

            target_response = modal_response(
                log_factors=target_log_factors,
                residues=residues,
            )


        # -------------------------------------------------
        # Frequency-only NN response
        #
        # NN mu + exact FEM residues
        # -------------------------------------------------

        predicted_response = modal_response(
            log_factors=predicted_log_factors,
            residues=residues,
        )


        physical_loss = transfer_loss(
            predicted_response,
            target_response,
        )


        # -------------------------------------------------
        # Combined objective
        # -------------------------------------------------

        total_loss = (
            frequency_loss

            + effective_transfer_weight
            * physical_loss
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        total_loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )


        optimizer.step()


        training_frequency_sum += (
            frequency_loss.item()
            * batch_size
        )


        training_transfer_sum += (
            physical_loss.item()
            * batch_size
        )


        training_total_sum += (
            total_loss.item()
            * batch_size
        )


        training_count += (
            batch_size
        )


    # =====================================================
    # Validation
    # =====================================================

    validation = evaluate(
        model=model,
        loader=val_loader,
    )


    scheduler.step(
        validation[
            "transfer_rms"
        ]
    )


    # =====================================================
    # Save best PHYSICAL model
    #
    # Criterion:
    #     lowest Hfreq RMS
    # =====================================================

    physical_improved = (
        validation[
            "transfer_rms"
        ]
        < best_hfreq
    )


    if physical_improved:

        best_hfreq = (
            validation[
                "transfer_rms"
            ]
        )


        best_hfreq_epoch = epoch


        best_hfreq_state = copy.deepcopy(
            model.state_dict()
        )


        epochs_without_physical_improvement = 0


        torch.save(
            {
                "model_state_dict":
                    best_hfreq_state,

                "epoch":
                    epoch,

                "selection":
                    "minimum_validation_hfreq",

                "validation":
                    {
                        key:
                            value
                        for key, value
                        in validation.items()
                        if key
                        != "relative_errors"
                    },

                "n_modes":
                    N_MODES,

                "latent_dim":
                    latent_dim,

                "transfer_weight":
                    TRANSFER_WEIGHT,

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

                "dataset":
                    DATASET_PATH.name,
            },
            BEST_HFREQ_CHECKPOINT,
        )


    else:

        epochs_without_physical_improvement += 1


    # =====================================================
    # Save best BALANCED model
    #
    # Criterion:
    #
    # frequency_loss
    # +
    # fixed final lambda_H * transfer_loss
    #
    # Note:
    # no ramp here, so epochs are directly comparable.
    # =====================================================

    if (
        validation[
            "balanced_score"
        ]
        < best_balanced
    ):

        best_balanced = (
            validation[
                "balanced_score"
            ]
        )


        best_balanced_epoch = epoch


        best_balanced_state = copy.deepcopy(
            model.state_dict()
        )


        torch.save(
            {
                "model_state_dict":
                    best_balanced_state,

                "epoch":
                    epoch,

                "selection":
                    "minimum_balanced_score",

                "validation":
                    {
                        key:
                            value
                        for key, value
                        in validation.items()
                        if key
                        != "relative_errors"
                    },

                "n_modes":
                    N_MODES,

                "latent_dim":
                    latent_dim,

                "transfer_weight":
                    TRANSFER_WEIGHT,

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

                "dataset":
                    DATASET_PATH.name,
            },
            BEST_BALANCED_CHECKPOINT,
        )


    # =====================================================
    # History
    # =====================================================

    history_epoch.append(
        epoch
    )


    history_mu.append(
        validation[
            "mean_error"
        ]
    )


    history_low4.append(
        validation[
            "low4_mean"
        ]
    )


    history_hfreq.append(
        validation[
            "transfer_rms"
        ]
    )


    history_balanced.append(
        validation[
            "balanced_score"
        ]
    )


    # =====================================================
    # Logging
    # =====================================================

    if (
        epoch == 1
        or epoch % PRINT_EVERY == 0
    ):

        train_frequency = (
            training_frequency_sum
            / training_count
        )


        train_transfer_rms = (
            np.sqrt(
                training_transfer_sum
                / training_count
            )
            * 100.0
        )


        train_total = (
            training_total_sum
            / training_count
        )


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
            f"train={train_total:.6f}"
            f" | "
            f"freq={train_frequency:.6f}"
            f" | "
            f"train H={train_transfer_rms:6.2f}%"
            f" | "
            f"val mu={validation['mean_error']:5.2f}%"
            f" | "
            f"low4={validation['low4_mean']:5.2f}%"
            f" | "
            f"low8={validation['low8_mean']:5.2f}%"
            f" | "
            f"max={validation['max_error']:5.2f}%"
            f" | "
            f"Hfreq={validation['transfer_rms']:6.2f}%"
            f" | "
            f"lambda={effective_transfer_weight:.4f}"
            f" | "
            f"lr={current_lr:.2e}"
        )


    # =====================================================
    # Early stopping
    #
    # We care primarily about physical H improvement.
    # =====================================================

    if (
        epoch
        >= TRANSFER_RAMP_EPOCHS
        and epochs_without_physical_improvement
        >= EARLY_STOPPING_PATIENCE
    ):

        print()

        print(
            "Early stopping."
        )

        break


# =========================================================
# Training time
# =========================================================

elapsed = (
    time.perf_counter()
    - start_time
)


# =========================================================
# Evaluate original V2
# =========================================================

model.load_state_dict(
    initial_state
)


original_train = evaluate(
    model=model,
    loader=train_loader,
)


original_val = evaluate(
    model=model,
    loader=val_loader,
)


# =========================================================
# Evaluate best physical checkpoint
# =========================================================

model.load_state_dict(
    best_hfreq_state
)


physical_train = evaluate(
    model=model,
    loader=train_loader,
)


physical_val = evaluate(
    model=model,
    loader=val_loader,
)


physical_relative_errors = (
    physical_val[
        "relative_errors"
    ]
)


# =========================================================
# Evaluate best balanced checkpoint
# =========================================================

model.load_state_dict(
    best_balanced_state
)


balanced_train = evaluate(
    model=model,
    loader=train_loader,
)


balanced_val = evaluate(
    model=model,
    loader=val_loader,
)


# =========================================================
# Final report
# =========================================================

print()

print(
    "=" * 82
)

print(
    "FINAL FREQUENCY V2.1 RESULT"
)

print(
    "=" * 82
)


print(
    f"Training time: "
    f"{elapsed:.1f} s"
)


print()

print(
    "ORIGINAL V2"
)


print(
    f"  validation mu mean: "
    f"{original_val['mean_error']:.3f}%"
)


print(
    f"  validation low4:    "
    f"{original_val['low4_mean']:.3f}%"
)


print(
    f"  validation low8:    "
    f"{original_val['low8_mean']:.3f}%"
)


print(
    f"  validation Hfreq:   "
    f"{original_val['transfer_rms']:.3f}%"
)


print()

print(
    "BEST PHYSICAL CHECKPOINT"
)


print(
    f"  epoch:              "
    f"{best_hfreq_epoch}"
)


print(
    f"  train mu mean:      "
    f"{physical_train['mean_error']:.3f}%"
)


print(
    f"  train Hfreq:        "
    f"{physical_train['transfer_rms']:.3f}%"
)


print(
    f"  validation mu mean: "
    f"{physical_val['mean_error']:.3f}%"
)


print(
    f"  validation low4:    "
    f"{physical_val['low4_mean']:.3f}%"
)


print(
    f"  validation low8:    "
    f"{physical_val['low8_mean']:.3f}%"
)


print(
    f"  validation max:     "
    f"{physical_val['max_error']:.3f}%"
)


print(
    f"  validation Hfreq:   "
    f"{physical_val['transfer_rms']:.3f}%"
)


print()

print(
    "BEST BALANCED CHECKPOINT"
)


print(
    f"  epoch:              "
    f"{best_balanced_epoch}"
)


print(
    f"  train mu mean:      "
    f"{balanced_train['mean_error']:.3f}%"
)


print(
    f"  train Hfreq:        "
    f"{balanced_train['transfer_rms']:.3f}%"
)


print(
    f"  validation mu mean: "
    f"{balanced_val['mean_error']:.3f}%"
)


print(
    f"  validation low4:    "
    f"{balanced_val['low4_mean']:.3f}%"
)


print(
    f"  validation low8:    "
    f"{balanced_val['low8_mean']:.3f}%"
)


print(
    f"  validation max:     "
    f"{balanced_val['max_error']:.3f}%"
)


print(
    f"  validation Hfreq:   "
    f"{balanced_val['transfer_rms']:.3f}%"
)


# =========================================================
# Per-mode errors for best physical checkpoint
# =========================================================

mean_per_mode = (
    np.mean(
        physical_relative_errors,
        axis=0,
    )
    * 100.0
)


p90_per_mode = (
    np.percentile(
        physical_relative_errors,
        90.0,
        axis=0,
    )
    * 100.0
)


max_per_mode = (
    np.max(
        physical_relative_errors,
        axis=0,
    )
    * 100.0
)


print()

print(
    "=" * 82
)

print(
    "BEST PHYSICAL CHECKPOINT: ERROR BY MODE"
)

print(
    "=" * 82
)


print(
    "mode | mean [%] | p90 [%] | max [%]"
)


print(
    "-" * 45
)


for mode in range(
    N_MODES
):

    print(
        f"{mode + 1:4d}"
        f" | "
        f"{mean_per_mode[mode]:8.3f}"
        f" | "
        f"{p90_per_mode[mode]:7.3f}"
        f" | "
        f"{max_per_mode[mode]:7.3f}"
    )


# =========================================================
# Plot 1:
# Physical validation error during fine tuning
# =========================================================

plt.figure(
    figsize=(
        10,
        5,
    )
)


plt.plot(
    history_epoch,
    history_hfreq,
)


plt.axhline(
    original_val[
        "transfer_rms"
    ],
    linestyle="--",
    label="Original V2",
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Validation Hfreq RMS [%]"
)


plt.title(
    "Frequency V2.1 physical fine-tuning"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


# =========================================================
# Plot 2:
# Mu accuracy during fine tuning
# =========================================================

plt.figure(
    figsize=(
        10,
        5,
    )
)


plt.plot(
    history_epoch,
    history_mu,
    label="All modes",
)


plt.plot(
    history_epoch,
    history_low4,
    label="Modes 1-4",
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Validation modal-factor error [%]"
)


plt.title(
    "Frequency accuracy during physical fine-tuning"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


# =========================================================
# Plot 3:
# Per-mode error for best physical checkpoint
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
    "Frequency V2.1: best physical checkpoint"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


# =========================================================
# Plot 4:
# Validation example
#
# FEM vs original V2 vs V2.1 physical
# =========================================================

example_index = 0


dataset_index = (
    val_indices[
        example_index
    ]
)


example_mask = torch.from_numpy(
    np.asarray(
        data[
            "masks"
        ][
            dataset_index
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
        dataset_index
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------
# Original V2
# ---------------------------------------------------------

model.load_state_dict(
    initial_state
)


model.eval()


with torch.no_grad():

    example_original = (
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


# ---------------------------------------------------------
# V2.1 physical
# ---------------------------------------------------------

model.load_state_dict(
    best_hfreq_state
)


model.eval()


with torch.no_grad():

    example_physical = (
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
    example_original,
    "x--",
    label="Frequency V2",
)


plt.plot(
    mode_numbers,
    example_physical,
    ".-",
    label="Frequency V2.1 physical",
)


plt.xlabel(
    "Mode"
)


plt.ylabel(
    "Modal factor"
)


plt.title(
    "Frequency V2.1 validation example"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()