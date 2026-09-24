import torch
from torch import nn

from model import ModalNet

import numpy as np

from shapes import make_rectangle_mask
from reference import (
    rectangle_modal_factors,
    rectangle_modal_gains,
)


N_MODES = 32
N_SAMPLES = 5000

ASPECT_MIN = 0.5
ASPECT_MAX = 2.0


np.random.seed(0)
torch.manual_seed(0)


def make_dataset(n_samples=N_SAMPLES):
    masks = []
    points = []
    factor_targets = []
    gain_targets = []

    for _ in range(n_samples):
        aspect = np.random.uniform(
            ASPECT_MIN,
            ASPECT_MAX,
        )

        x = np.random.uniform(0.0, 1.0)
        y = np.random.uniform(0.0, 1.0)

        mask = make_rectangle_mask(aspect)

        modes, modal_factors = rectangle_modal_factors(
            aspect
        )

        modal_gains = rectangle_modal_gains(
            modes,
            x,
            y,
        )

        masks.append(mask)
        points.append([x, y])
        factor_targets.append(modal_factors)
        gain_targets.append(modal_gains)

    return (
        np.asarray(masks, dtype=np.float32),
        np.asarray(points, dtype=np.float32),
        np.asarray(factor_targets, dtype=np.float32),
        np.asarray(gain_targets, dtype=np.float32),
    )


def train():
    masks, points, targets, gain_targets = make_dataset()

    x = torch.from_numpy(masks).unsqueeze(1)

    point_x = torch.from_numpy(points)

    factor_y = torch.from_numpy(targets)

    gain_y = torch.from_numpy(gain_targets)

    print("Shapes:", x.shape)
    print("Points:", point_x.shape)
    print("Factor targets:", factor_y.shape)
    print("Gain targets:", gain_y.shape)

    model = ModalNet(N_MODES)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    loss_fn = nn.MSELoss()

    for epoch in range(500):
        predicted_factors, predicted_gains = model(
            x,
            point_x,
        )

        factor_loss = loss_fn(
            predicted_factors,
            torch.log(factor_y),
        )

        gain_loss = loss_fn(
            predicted_gains,
            gain_y,
        )

        loss = factor_loss + gain_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Total {loss.item():.6f} | "
                f"Factors {factor_loss.item():.6f} | "
                f"Gains {gain_loss.item():.6f}"
            )

    test_aspects = [
        0.55,
        0.73,
        0.91,
        1.00,
        1.17,
        1.37,
        1.63,
        1.91,
    ]

    model.eval()

    for test_aspect in test_aspects:
        test_mask = make_rectangle_mask(
            test_aspect
        )

        test_x = torch.from_numpy(
            test_mask
        ).unsqueeze(0).unsqueeze(0)

        test_point = torch.tensor(
            [[0.3, 0.7]],
            dtype=torch.float32,
        )

        with torch.no_grad():
            (
                predicted_log_factors,
                predicted_gains,
            ) = model(
                test_x,
                test_point,
            )

        predicted_factors = torch.exp(
            predicted_log_factors
        )[0].numpy()

        predicted_gains = (
            predicted_gains[0].numpy()
        )

        modes, true_factors = (
            rectangle_modal_factors(
                test_aspect
            )
        )

        true_gains = rectangle_modal_gains(
            modes,
            x=0.3,
            y=0.7,
        )

        relative_error = (
            np.abs(
                predicted_factors
                - true_factors
            )
            / true_factors
        )

        gain_mae = np.mean(
            np.abs(
                predicted_gains
                - true_gains
            )
        )

        print(
            f"\naspect={test_aspect:.2f}"
        )

        print(
            f"Factor mean error="
            f"{relative_error.mean() * 100:.2f}%"
        )

        print(
            f"Factor max error="
            f"{relative_error.max() * 100:.2f}%"
        )

        print(
            f"Gain MAE={gain_mae:.4f}"
        )

        print("\nPredicted gains:")
        print(predicted_gains)

        print("\nTrue gains:")
        print(true_gains)


if __name__ == "__main__":
    train()