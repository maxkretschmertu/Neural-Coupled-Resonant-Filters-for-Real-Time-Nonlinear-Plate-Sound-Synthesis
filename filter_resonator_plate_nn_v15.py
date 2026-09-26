from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from numba import njit
from skimage.measure import points_in_poly
from tkinter import *
from tkinter import ttk

from neural.model import PlateNet
from neural.shapes import make_morph_contour


fs = 48000.0
n_Modes = 32
MAX_STATE_MAG = 10.0

params = {
    "alpha_g": 0.3322, "alpha_r": 4e-5,
    "nonlinearity": 0.5,
    "modes": 32,
    "size": 1.0, "aspect": 1.0, "morph": 0.0,
    "D": 18300.0, "rho": 7800.0, "H": 0.01,
    "excitation": 0,

    "x_e": -0.3, "y_e": 0.3,
    "x_l": -0.3, "y_l": -0.3,
    "x_r": 0.3, "y_r": -0.3,

    "N_ex": 100, "A": 0.5,
    "impact_start": None, "changed": False,
}

plate_net = PlateNet(n_modes=n_Modes)
plate_net.load_state_dict(torch.load(
    Path(__file__).parent / "models" / "plate_nn.pt",
    map_location="cpu", weights_only=True,
))
plate_net.eval()


@torch.no_grad()
def plate_model():
    geometry = torch.tensor(
        [[params["morph"], params["aspect"]]],
        dtype=torch.float32,
    )

    points = torch.tensor([[
        [params["x_e"], params["y_e"]],
        [params["x_l"], params["y_l"]],
        [params["x_r"], params["y_r"]],
    ]], dtype=torch.float32)

    mu, gains = plate_net(geometry, points)

    mu = mu[0].numpy().astype(np.float64)
    gains = gains[0].numpy().astype(np.float64)

    freqs = (
        mu
        * np.sqrt(params["D"] / (params["rho"] * params["H"]))
        / (2 * np.pi * params["size"] ** 2)
    )

    return freqs, gains[0], gains[1], gains[2]


def nonlinear_params():
    n = params["nonlinearity"]

    tau = 2.0 * (1.0 - n)
    eta = 0.02 * n
    lamb = 0.02 * n

    return tau, eta, lamb


def distribution_matrix(freqs):
    _, eta, lamb = nonlinear_params()

    diff = np.abs(freqs[:, None] - freqs[None, :])
    a = 1.0 - diff / np.mean(freqs)

    np.fill_diagonal(a, 0.0)

    denom = a.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0

    return np.ascontiguousarray(
        eta * lamb * a / denom
        - lamb * np.eye(len(freqs)),
        dtype=np.float64,
    )


def build_runtime():
    freqs, strike, left, right = plate_model()

    n = params["modes"]

    freqs = freqs[:n]
    strike = strike[:n]
    left = left[:n]
    right = right[:n]

    alphas = np.exp(
        np.minimum(
            params["alpha_g"]
            + params["alpha_r"] * freqs,
            700.0,
        )
    )

    Z = (
        np.exp(-alphas / fs)
        * np.exp(1j * 2 * np.pi * freqs / fs)
    )

    return (
        freqs,
        np.ascontiguousarray(Z, dtype=np.complex128),
        distribution_matrix(freqs),
        np.ascontiguousarray(strike, dtype=np.float64),
        np.ascontiguousarray(left, dtype=np.float64),
        np.ascontiguousarray(right, dtype=np.float64),
    )


def excitation_signal(pos, gains):
    if params["excitation"] == 0:
        u = np.zeros_like(pos, dtype=np.float64)

        inside = pos < params["N_ex"]

        u[inside] = (
            params["A"]
            * 2.0 / params["N_ex"]
            * np.sin(
                np.pi
                * pos[inside]
                / params["N_ex"]
            ) ** 2
        )

    else:
        u = (
            pos % int(fs) == 0
        ).astype(np.float64)

    return np.ascontiguousarray(
        gains[:, None] * u[None, :]
    )


@njit(cache=True, fastmath=True)
def process_block(
    states,
    Z,
    M,
    u,
    left,
    right,
    tau,
    max_state_mag,
    out,
):
    n = len(states)

    rectified = np.empty(n)
    T = np.empty(n)

    for i in range(u.shape[1]):

        for k in range(n):
            zr = states[k].real
            zi = states[k].imag

            r = (
                0.5 * (zr * zr + zi * zi)
                - tau
            )

            rectified[k] = (
                r if r > 0.0 else 0.0
            )

        for k in range(n):
            acc = 0.0

            for j in range(n):
                acc += (
                    M[k, j]
                    * rectified[j]
                )

            T[k] = (
                acc if acc > 0.0 else 0.0
            )

        for k in range(n):
            zr = states[k].real
            zi = states[k].imag

            mag2 = (
                zr * zr
                + zi * zi
            )

            if mag2 < 1e-20:
                new_z = (
                    np.sqrt(2.0 * T[k])
                    * Z[k]
                    + u[k, i]
                )

            else:
                new_z = (
                    np.sqrt(
                        1.0
                        + 2.0 * T[k] / mag2
                    )
                    * Z[k]
                    * states[k]
                    + u[k, i]
                )

            mag = np.sqrt(
                new_z.real ** 2
                + new_z.imag ** 2
            )

            if mag > max_state_mag:
                new_z = (
                    max_state_mag
                    * np.tanh(
                        mag / max_state_mag
                    )
                    / mag
                )

            states[k] = new_z

        out_l = 0.0
        out_r = 0.0

        for k in range(n):
            value = states[k].imag

            out_l += left[k] * value
            out_r += right[k] * value

        out[i, 0] = out_l
        out[i, 1] = out_r


states = np.zeros(
    n_Modes,
    dtype=np.complex128,
)

runtime = build_runtime()


# Numba warmup
_, Z0, M0, strike0, left0, right0 = runtime

tau0, _, _ = nonlinear_params()

process_block(
    states[:len(Z0)].copy(),
    Z0,
    M0,
    np.zeros((len(strike0), 4)),
    left0,
    right0,
    tau0,
    MAX_STATE_MAG,
    np.zeros((4, 2)),
)

print("Numba-Kernel kompiliert.")


def callback(outdata, frames, time, status):
    global runtime

    if status:
        print(status)

    if params["changed"]:
        runtime = build_runtime()
        params["changed"] = False

    _, Z, M, strike, left, right = runtime

    n = len(Z)

    pos = (
        callback.pos
        + np.arange(frames)
    )

    u = (
        np.zeros(
            (n, frames),
            dtype=np.float64,
        )
        if params["impact_start"] is None

        else excitation_signal(
            pos - params["impact_start"],
            strike,
        )
    )

    out = np.empty(
        (frames, 2),
        dtype=np.float64,
    )

    tau, _, _ = nonlinear_params()

    process_block(
        states[:n],
        Z,
        M,
        u,
        left,
        right,
        tau,
        MAX_STATE_MAG,
        out,
    )

    callback.pos += frames

    out = np.nan_to_num(
        out / np.sqrt(n),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    outdata[:] = np.clip(
        out,
        -1.0,
        1.0,
    )


callback.pos = 0


root = Tk()
root.title("Resonator Filter GUI (Numba)")

canvas_size = 300

ttk.Label(
    root,
    text=(
        "Strike: left drag | "
        "Pickup L: right drag | "
        "Pickup R: middle drag"
    ),
).pack()

canvas = Canvas(
    root,
    width=canvas_size,
    height=canvas_size,
    bg="white",
)

canvas.pack(
    pady=6
)

shape_id = canvas.create_polygon(
    0,
    0,
    fill="",
    outline="black",
    width=2,
)


def contour():
    return make_morph_contour(
        params["morph"],
        params["aspect"],
    )


def point_inside(x, y):
    return points_in_poly(
        np.array([[x, y]]),
        contour(),
    )[0]


def draw():
    c = contour()

    xy = np.column_stack((
        (c[:, 0] + 1)
        * 0.5
        * canvas_size,

        (1 - c[:, 1])
        * 0.5
        * canvas_size,
    ))

    canvas.coords(
        shape_id,
        *xy.ravel(),
    )

    for tag, x, y, color in (
        (
            "strike",
            params["x_e"],
            params["y_e"],
            "red",
        ),
        (
            "left",
            params["x_l"],
            params["y_l"],
            "green",
        ),
        (
            "right",
            params["x_r"],
            params["y_r"],
            "blue",
        ),
    ):
        canvas.delete(tag)

        px = (
            (x + 1)
            * 0.5
            * (canvas_size - 1)
        )

        py = (
            (1 - y)
            * 0.5
            * (canvas_size - 1)
        )

        canvas.create_oval(
            px - 6,
            py - 6,
            px + 6,
            py + 6,
            fill=color,
            tags=tag,
        )


def set_param(name, value):
    if name == "modes":
        params[name] = int(float(value))

        # avoid old stored states returning
        # when increasing mode count again
        states[:] = 0.0
    else:
        params[name] = float(value)

    if name in ("morph", "aspect"):
        for x, y in (
            ("x_e", "y_e"),
            ("x_l", "y_l"),
            ("x_r", "y_r"),
        ):
            if not point_inside(
                params[x],
                params[y],
            ):
                params[x] = 0.0
                params[y] = 0.0

        draw()

    params["changed"] = True


def set_point(
    event,
    x_name,
    y_name,
):
    x = (
        2
        * np.clip(
            event.x,
            0,
            canvas_size - 1,
        )
        / (canvas_size - 1)
        - 1
    )

    y = (
        1
        - 2
        * np.clip(
            event.y,
            0,
            canvas_size - 1,
        )
        / (canvas_size - 1)
    )

    if point_inside(x, y):
        params[x_name] = x
        params[y_name] = y
        params["changed"] = True

        draw()


def add_slider(
    label,
    name,
    lo,
    hi,
    model_param=True,
):
    ttk.Label(
        root,
        text=label,
    ).pack()

    command = (
        lambda v:
            set_param(name, v)
        if model_param

        else lambda v:
            params.__setitem__(
                name,
                float(v),
            )
    )

    s = ttk.Scale(
        root,
        from_=lo,
        to=hi,
        orient="horizontal",
        command=command,
    )

    s.set(
        params[name]
    )

    s.pack(
        fill="x"
    )


ttk.Button(
    root,
    text="Strike",
    command=lambda:
        params.__setitem__(
            "impact_start",
            callback.pos,
        ),
).pack(
    pady=6
)


ttk.Button(
    root,
    text="Toggle Excitation",
    command=lambda:
        params.__setitem__(
            "excitation",
            1 - params["excitation"],
        ),
).pack(
    pady=6
)


add_slider(
    "Size",
    "size",
    0.1,
    3.0,
)

add_slider(
    "Morph",
    "morph",
    0.0,
    1.0,
)

add_slider(
    "Shape",
    "aspect",
    0.5,
    2.0,
)

add_slider(
    "Damping",
    "alpha_g",
    0.001,
    5.0,
)

add_slider(
    "Damping Tilt",
    "alpha_r",
    0.0,
    0.05,
)

add_slider(
    "Nonlinearity",
    "nonlinearity",
    0.0,
    100.0,
)

add_slider(
    "Excitation Length",
    "N_ex",
    2,
    192,
    False,
)

add_slider(
    "Gain",
    "A",
    0.0,
    2.0,
    False,
)


# Modes should be integer-valued
ttk.Label(
    root,
    text="Modes",
).pack()

mode_slider = Scale(
    root,
    from_=1,
    to=n_Modes,
    orient=HORIZONTAL,
    resolution=1,
    command=lambda v:
        set_param(
            "modes",
            v,
        ),
)

mode_slider.set(
    params["modes"]
)

mode_slider.pack(
    fill="x"
)


# Canvas controls
canvas.bind(
    "<Button-1>",
    lambda e:
        set_point(
            e,
            "x_e",
            "y_e",
        ),
)

canvas.bind(
    "<B1-Motion>",
    lambda e:
        set_point(
            e,
            "x_e",
            "y_e",
        ),
)

canvas.bind(
    "<Button-3>",
    lambda e:
        set_point(
            e,
            "x_l",
            "y_l",
        ),
)

canvas.bind(
    "<B3-Motion>",
    lambda e:
        set_point(
            e,
            "x_l",
            "y_l",
        ),
)

canvas.bind(
    "<Button-2>",
    lambda e:
        set_point(
            e,
            "x_r",
            "y_r",
        ),
)

canvas.bind(
    "<B2-Motion>",
    lambda e:
        set_point(
            e,
            "x_r",
            "y_r",
        ),
)

# alternative for mice without middle-button dragging
canvas.bind(
    "<Shift-Button-3>",
    lambda e:
        set_point(
            e,
            "x_r",
            "y_r",
        ),
)

canvas.bind(
    "<Shift-B3-Motion>",
    lambda e:
        set_point(
            e,
            "x_r",
            "y_r",
        ),
)


draw()

print(
    "Initial frequencies:",
    np.round(
        runtime[0],
        2,
    ),
)


with sd.OutputStream(
    channels=2,
    samplerate=int(fs),
    callback=callback,
):
    root.mainloop()