from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch

from model import ModalNet


# =========================================================
# Settings
# =========================================================

BATCH_SIZE = 16

ROOT = Path(__file__).parent

DATASET_PATH = (
    ROOT
    / "plate_dataset_pilot_500.npz"
)

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "modalnet_pilot_final.pt"
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
# Load checkpoint
# =========================================================

try:

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

except TypeError:

    # Compatibility with older PyTorch versions
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )


N_MODES = int(
    checkpoint[
        "n_modes"
    ]
)

DAMPING_RATIO = float(
    checkpoint[
        "damping_ratio"
    ]
)

OMEGA_MIN = float(
    checkpoint[
        "omega_min"
    ]
)

OMEGA_MAX = float(
    checkpoint[
        "omega_max"
    ]
)

N_FREQUENCY_POINTS = int(
    checkpoint[
        "n_frequency_points"
    ]
)


print()

print(
    "Checkpoint:"
)

print(
    "  modes:",
    N_MODES,
)

print(
    "  damping ratio:",
    DAMPING_RATIO,
)

print(
    "  omega:",
    f"{OMEGA_MIN:.3f}",
    "...",
    f"{OMEGA_MAX:.3f}",
)

print(
    "  frequency points:",
    N_FREQUENCY_POINTS,
)

print(
    "  best Stage-3 epoch:",
    checkpoint.get(
        "best_stage3_epoch",
        "unknown",
    ),
)


# =========================================================
# Model
# =========================================================

model = ModalNet(
    n_modes=N_MODES
).to(
    device
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model.eval()


# =========================================================
# Load dataset
# =========================================================

data = np.load(
    DATASET_PATH
)


val_indices = np.asarray(
    data[
        "val_indices"
    ],
    dtype=np.int64,
)


masks = np.asarray(
    data[
        "masks"
    ][
        val_indices
    ],
    dtype=np.float32,
)


factors = np.asarray(
    data[
        "factors"
    ][
        val_indices
    ],
    dtype=np.float32,
)


strike_points = np.asarray(
    data[
        "strike_points"
    ][
        val_indices
    ],
    dtype=np.float32,
)


pickup_points = np.asarray(
    data[
        "pickup_points"
    ][
        val_indices
    ],
    dtype=np.float32,
)


modal_weights = np.asarray(
    data[
        "modal_weights"
    ][
        val_indices
    ],
    dtype=np.float32,
)


morphs = np.asarray(
    data[
        "morphs"
    ][
        val_indices
    ],
    dtype=np.float32,
)


aspects = np.asarray(
    data[
        "aspects"
    ][
        val_indices
    ],
    dtype=np.float32,
)


N_GEOMETRIES = len(
    val_indices
)


print()

print(
    "Validation geometries:",
    N_GEOMETRIES,
)

print(
    "Point pairs:",
    strike_points.shape[1],
)


# =========================================================
# Frequency axis
# =========================================================

omega = torch.linspace(
    OMEGA_MIN,
    OMEGA_MAX,
    N_FREQUENCY_POINTS,
    dtype=torch.float32,
    device=device,
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


    return torch.sum(
        r
        / denominator,
        dim=-1,
    )


# =========================================================
# Global relative RMS helper
# =========================================================

def global_relative_rms(
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


    ratio = (
        error_energy
        /
        torch.clamp(
            target_energy,
            min=1e-12,
        )
    )


    return torch.sqrt(
        ratio
    )


# =========================================================
# Accumulators
#
# Four response cases:
#
# A:
# exact mu + exact residues
#
# B:
# exact mu + NN residues
#
# C:
# NN mu + exact residues
#
# D:
# NN mu + NN residues
# =========================================================

case_names = [
    "Exact μ + exact r",
    "Exact μ + NN r",
    "NN μ + exact r",
    "NN μ + NN r",
]


error_energy = np.zeros(
    4,
    dtype=np.float64,
)


target_energy = 0.0


# ---------------------------------------------------------
# Frequency-error storage
# ---------------------------------------------------------

factor_relative_errors = []


# ---------------------------------------------------------
# Residue-error storage
# ---------------------------------------------------------

residue_absolute_errors = []


# ---------------------------------------------------------
# Per-geometry end-to-end metrics
# ---------------------------------------------------------

geometry_end_to_end_rms = []


# =========================================================
# Inference
# =========================================================

with torch.no_grad():

    for start in range(
        0,
        N_GEOMETRIES,
        BATCH_SIZE,
    ):

        stop = min(
            start
            + BATCH_SIZE,
            N_GEOMETRIES,
        )


        # -------------------------------------------------
        # Batch
        # -------------------------------------------------

        mask_batch = torch.from_numpy(
            masks[
                start:stop
            ]
        ).unsqueeze(
            1
        ).to(
            device
        )


        factor_batch = torch.from_numpy(
            factors[
                start:stop
            ]
        ).to(
            device
        )


        strike_batch = torch.from_numpy(
            strike_points[
                start:stop
            ]
        ).to(
            device
        )


        pickup_batch = torch.from_numpy(
            pickup_points[
                start:stop
            ]
        ).to(
            device
        )


        weight_batch = torch.from_numpy(
            modal_weights[
                start:stop
            ]
        ).to(
            device
        )


        exact_log_factors = torch.log(
            factor_batch
        )


        # -------------------------------------------------
        # NN inference
        # -------------------------------------------------

        (
            predicted_log_factors,
            predicted_residues,
        ) = model(
            mask_batch,
            strike_batch,
            pickup_batch,
        )


        predicted_factors = torch.exp(
            predicted_log_factors
        )


        # =================================================
        # Parameter errors
        # =================================================

        factor_relative_error = (
            torch.abs(
                predicted_factors
                - factor_batch
            )
            / factor_batch
        )


        factor_relative_errors.append(
            factor_relative_error
            .cpu()
            .numpy()
        )


        residue_absolute_error = (
            torch.abs(
                predicted_residues
                - weight_batch
            )
        )


        residue_absolute_errors.append(
            residue_absolute_error
            .cpu()
            .numpy()
        )


        # =================================================
        # Reference
        #
        # exact mu + exact residues
        # =================================================

        H_exact = modal_response(
            log_factors=exact_log_factors,
            residues=weight_batch,
            omega=omega,
        )


        # =================================================
        # CASE B
        #
        # exact mu + predicted residues
        # =================================================

        H_exact_mu_nn_r = modal_response(
            log_factors=exact_log_factors,
            residues=predicted_residues,
            omega=omega,
        )


        # =================================================
        # CASE C
        #
        # predicted mu + exact residues
        # =================================================

        H_nn_mu_exact_r = modal_response(
            log_factors=predicted_log_factors,
            residues=weight_batch,
            omega=omega,
        )


        # =================================================
        # CASE D
        #
        # predicted mu + predicted residues
        # =================================================

        H_nn = modal_response(
            log_factors=predicted_log_factors,
            residues=predicted_residues,
            omega=omega,
        )


        responses = [
            H_exact,
            H_exact_mu_nn_r,
            H_nn_mu_exact_r,
            H_nn,
        ]


        # -------------------------------------------------
        # Reference energy
        # -------------------------------------------------

        batch_target_energy = torch.sum(
            torch.abs(
                H_exact
            ) ** 2
        )


        target_energy += (
            batch_target_energy.item()
        )


        # -------------------------------------------------
        # Error energy for all four cases
        # -------------------------------------------------

        for case_index, response in enumerate(
            responses
        ):

            batch_error_energy = torch.sum(
                torch.abs(
                    response
                    - H_exact
                ) ** 2
            )


            error_energy[
                case_index
            ] += (
                batch_error_energy.item()
            )


        # =================================================
        # Per-geometry END-TO-END RMS
        # =================================================

        geometry_error_energy = torch.sum(
            torch.abs(
                H_nn
                - H_exact
            ) ** 2,
            dim=(
                1,
                2,
            ),
        )


        geometry_target_energy = torch.sum(
            torch.abs(
                H_exact
            ) ** 2,
            dim=(
                1,
                2,
            ),
        )


        geometry_rms = torch.sqrt(
            geometry_error_energy
            /
            torch.clamp(
                geometry_target_energy,
                min=1e-12,
            )
        )


        geometry_end_to_end_rms.append(
            geometry_rms
            .cpu()
            .numpy()
        )


# =========================================================
# Aggregate parameter errors
# =========================================================

factor_relative_errors = np.concatenate(
    factor_relative_errors,
    axis=0,
)


residue_absolute_errors = np.concatenate(
    residue_absolute_errors,
    axis=0,
)


geometry_end_to_end_rms = np.concatenate(
    geometry_end_to_end_rms,
    axis=0,
)


# =========================================================
# Four-case response decomposition
# =========================================================

case_rms = (
    np.sqrt(
        error_energy
        /
        max(
            target_energy,
            1e-12,
        )
    )
    * 100.0
)


print()

print(
    "=" * 78
)

print(
    "TRANSFER ERROR DECOMPOSITION"
)

print(
    "=" * 78
)

print()

for name, rms in zip(
    case_names,
    case_rms,
):

    print(
        f"{name:24s}"
        f" : "
        f"{rms:8.3f}%"
    )


# =========================================================
# Frequency diagnostics per mode
# =========================================================

factor_mean_per_mode = (
    np.mean(
        factor_relative_errors,
        axis=0,
    )
    * 100.0
)


factor_median_per_mode = (
    np.median(
        factor_relative_errors,
        axis=0,
    )
    * 100.0
)


factor_p90_per_mode = (
    np.percentile(
        factor_relative_errors,
        90.0,
        axis=0,
    )
    * 100.0
)


factor_max_per_mode = (
    np.max(
        factor_relative_errors,
        axis=0,
    )
    * 100.0
)


print()

print(
    "=" * 78
)

print(
    "MODAL-FACTOR ERROR BY MODE"
)

print(
    "=" * 78
)

print()

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
        f"{factor_mean_per_mode[mode]:8.3f}"
        f" | "
        f"{factor_median_per_mode[mode]:10.3f}"
        f" | "
        f"{factor_p90_per_mode[mode]:7.3f}"
        f" | "
        f"{factor_max_per_mode[mode]:7.3f}"
    )


# =========================================================
# Residue error by mode
# =========================================================

# shape:
# geometry, point-pair, mode

residue_mae_per_mode = np.mean(
    residue_absolute_errors,
    axis=(
        0,
        1,
    ),
)


print()

print(
    "=" * 78
)

print(
    "RESIDUE MAE BY MODE"
)

print(
    "=" * 78
)

print()


for mode in range(
    N_MODES
):

    print(
        f"mode "
        f"{mode + 1:2d}"
        f" : "
        f"{residue_mae_per_mode[mode]:.5f}"
    )


# =========================================================
# Identify worst frequency modes
# =========================================================

worst_mode_order = np.argsort(
    factor_mean_per_mode
)[
    ::-1
]


print()

print(
    "=" * 78
)

print(
    "WORST MODAL-FACTOR MODES"
)

print(
    "=" * 78
)

print()


for rank in range(
    min(
        10,
        N_MODES,
    )
):

    mode = worst_mode_order[
        rank
    ]


    print(
        f"{rank + 1:2d}. "
        f"mode "
        f"{mode + 1:2d}"
        f" | "
        f"mean="
        f"{factor_mean_per_mode[mode]:6.2f}%"
        f" | "
        f"p90="
        f"{factor_p90_per_mode[mode]:6.2f}%"
        f" | "
        f"max="
        f"{factor_max_per_mode[mode]:6.2f}%"
    )


# =========================================================
# Geometry diagnostics
# =========================================================

geometry_rms_percent = (
    geometry_end_to_end_rms
    * 100.0
)


print()

print(
    "=" * 78
)

print(
    "END-TO-END ERROR BY GEOMETRY"
)

print(
    "=" * 78
)

print()


print(
    f"mean:   "
    f"{np.mean(geometry_rms_percent):.2f}%"
)

print(
    f"median: "
    f"{np.median(geometry_rms_percent):.2f}%"
)

print(
    f"p90:    "
    f"{np.percentile(geometry_rms_percent, 90):.2f}%"
)

print(
    f"max:    "
    f"{np.max(geometry_rms_percent):.2f}%"
)


# ---------------------------------------------------------
# Worst validation geometries
# ---------------------------------------------------------

worst_geometry_order = np.argsort(
    geometry_rms_percent
)[
    ::-1
]


print()

print(
    "Worst validation geometries:"
)

print()

print(
    "rank | dataset idx | morph | aspect | H RMS [%]"
)

print(
    "-" * 55
)


for rank in range(
    min(
        10,
        N_GEOMETRIES,
    )
):

    local_index = (
        worst_geometry_order[
            rank
        ]
    )


    dataset_index = (
        val_indices[
            local_index
        ]
    )


    print(
        f"{rank + 1:4d}"
        f" | "
        f"{dataset_index:11d}"
        f" | "
        f"{morphs[local_index]:5.3f}"
        f" | "
        f"{aspects[local_index]:6.3f}"
        f" | "
        f"{geometry_rms_percent[local_index]:9.2f}"
    )


# =========================================================
# Plot 1:
# Four-way transfer decomposition
# =========================================================

plt.figure(
    figsize=(
        9,
        5,
    )
)


plt.bar(
    np.arange(
        4
    ),
    case_rms,
)


plt.xticks(
    np.arange(
        4
    ),
    [
        "exact μ\nexact r",
        "exact μ\nNN r",
        "NN μ\nexact r",
        "NN μ\nNN r",
    ],
)


plt.ylabel(
    "Global transfer RMS [%]"
)


plt.title(
    "Transfer error decomposition"
)


plt.grid(
    True,
    axis="y",
)


plt.tight_layout()


# =========================================================
# Plot 2:
# Frequency error by mode
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
    factor_mean_per_mode,
    "o-",
    label="Mean",
)


plt.plot(
    mode_numbers,
    factor_p90_per_mode,
    "x--",
    label="90th percentile",
)


plt.plot(
    mode_numbers,
    factor_max_per_mode,
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
    "Validation modal-factor error by mode"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


# =========================================================
# Plot 3:
# Residue error by mode
# =========================================================

plt.figure(
    figsize=(
        10,
        5,
    )
)


plt.plot(
    mode_numbers,
    residue_mae_per_mode,
    "o-",
)


plt.xlabel(
    "Mode"
)


plt.ylabel(
    "Residue MAE"
)


plt.title(
    "Validation residue error by mode"
)


plt.grid(
    True
)


plt.tight_layout()


# =========================================================
# Plot 4:
# End-to-end error versus morph
# =========================================================

plt.figure(
    figsize=(
        8,
        5,
    )
)


plt.scatter(
    morphs,
    geometry_rms_percent,
)


plt.xlabel(
    "Morph"
)


plt.ylabel(
    "End-to-end H RMS [%]"
)


plt.title(
    "Validation error vs morph"
)


plt.grid(
    True
)


plt.tight_layout()


# =========================================================
# Plot 5:
# End-to-end error versus aspect
# =========================================================

plt.figure(
    figsize=(
        8,
        5,
    )
)


plt.scatter(
    aspects,
    geometry_rms_percent,
)


plt.xlabel(
    "Aspect"
)


plt.ylabel(
    "End-to-end H RMS [%]"
)


plt.title(
    "Validation error vs aspect"
)


plt.grid(
    True
)


plt.tight_layout()


plt.show()