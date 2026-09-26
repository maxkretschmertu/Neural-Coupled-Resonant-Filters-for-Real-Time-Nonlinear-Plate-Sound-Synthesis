import numpy as np
import sounddevice as sd
import torch
from pathlib import Path
from tkinter import *
from tkinter import ttk

from numba import njit
from skimage.measure import points_in_poly

from neural.model import PlateNet
from neural.shapes import make_morph_contour


fs = 48000.0
n_Modes = 32
root = Tk()


plate_net = PlateNet(n_modes=n_Modes)
plate_net.load_state_dict(torch.load(
    Path(__file__).parent / "models" / "plate_nn.pt",
    map_location="cpu",
    weights_only=True,
))
plate_net.eval()


canvas_size = 300
ttk.Label(root, text="Strike: left drag | Pickup: right drag").pack()
canvas_id = Canvas(root, width=canvas_size, height=canvas_size, bg="white")
canvas_id.pack(pady=6)
shape_id = canvas_id.create_polygon(0, 0, fill="", outline="black", width=2)


params = {
    "alpha_g": 0.3322,
    "alpha_r": 4e-5,

    "tau": 1,
    "eta": 0.01,
    "lamb": 0.01,

    "size": 1.0,
    "aspect": 1.0,
    "morph": 0.0,

    "D": 18300,
    "rho": 7800.0,
    "H": 0.01,

    "excitation": 0,

    "x_e": -0.3,
    "y_e": 0.3,

    "x_p": 0.3,
    "y_p": 0.3,

    "N_ex": 192,
    "changed": False,
    "A": 0.5,
}


def modal_factors_to_freqs(modal_factors, size, D, rho, H):
    omega = modal_factors * np.sqrt(D / (rho * H)) / size**2
    return omega / (2 * np.pi)


@torch.no_grad()
def neural_plate_model(
    size, aspect, morph,
    D, rho, H,
    x_e, y_e,
    x_p, y_p,
):
    geometry = torch.tensor(
        [[morph, aspect]],
        dtype=torch.float32,
    )

    points = torch.tensor(
        [[
            [x_e, y_e],
            [x_p, y_p],
        ]],
        dtype=torch.float32,
    )

    modal_factors, gains = plate_net(
        geometry,
        points,
    )

    modal_factors = modal_factors[0].numpy().astype(np.float64)
    gains = gains[0].numpy().astype(np.float64)

    freqs = modal_factors_to_freqs(
        modal_factors,
        size,
        D,
        rho,
        H,
    )

    return modal_factors, freqs, gains[0], gains[1]


def distribution_matrix(freqs_arr, eta=0.01, lamb=1.0):
    freqs_arr = np.asarray(freqs_arr, dtype=np.float64)
    n = len(freqs_arr)

    if n <= 1:
        return np.zeros((n, n), dtype=np.float64)

    diff = np.abs(freqs_arr[:, None] - freqs_arr[None, :])
    a = 1.0 - diff / np.mean(freqs_arr)
    np.fill_diagonal(a, 0.0)

    denom = np.sum(a, axis=1, keepdims=True)
    denom[denom == 0] = 1.0

    M = eta * lamb * a / denom - lamb * np.eye(n)
    return np.ascontiguousarray(M, dtype=np.float64)


def excitation_signal(pos, mode_gains, N_ex=2, A=0.5):
    pos = np.asarray(pos, dtype=np.float64)
    n = len(mode_gains)
    frames = pos.shape[0]

    if params.get("excitation") == 0:
        u = np.zeros_like(pos)
        within = pos < N_ex
        scale = 2.0 / N_ex if N_ex > 0 else 1.0
        u[within] = A * scale * np.sin(np.pi * pos[within] / N_ex) ** 2
    else:
        u = (pos % int(fs) == 0).astype(np.float64)

    u_ex = np.empty((n, frames), dtype=np.float64)

    for k in range(n):
        u_ex[k, :] = mode_gains[k] * u

    return np.ascontiguousarray(u_ex, dtype=np.float64)


@njit(cache=True, fastmath=True)
def process_block(
    states,
    Z,
    M,
    u,
    pickup_gains,
    tau,
    max_state_mag,
    s_out,
    T_last,
):
    n = states.shape[0]
    frames = u.shape[1]

    rectified = np.empty(n, dtype=np.float64)
    T = np.empty(n, dtype=np.float64)

    for i in range(frames):

        for k in range(n):
            zr = states[k].real
            zi = states[k].imag
            p = 0.5 * (zr * zr + zi * zi)
            r = p - tau
            rectified[k] = r if r > 0.0 else 0.0

        for k in range(n):
            acc = 0.0

            for j in range(n):
                acc += M[k, j] * rectified[j]

            T[k] = acc if acc > 0.0 else 0.0

        for k in range(n):
            zr = states[k].real
            zi = states[k].imag
            mag2 = zr * zr + zi * zi

            if mag2 < 1e-20:
                amp = np.sqrt(2.0 * T[k])
                new_z = amp * Z[k] + u[k, i]
            else:
                factor = np.sqrt(1.0 + (2.0 * T[k]) / mag2)
                new_z = factor * Z[k] * states[k] + u[k, i]

            nr = new_z.real
            ni = new_z.imag
            mag = np.sqrt(nr * nr + ni * ni)

            if mag > max_state_mag:
                comp = max_state_mag * np.tanh(mag / max_state_mag)
                new_z = new_z * (comp / mag)

            states[k] = new_z

        acc = 0.0

        for k in range(n):
            acc += pickup_gains[k] * states[k].imag

        s_out[i] = acc

        if i == frames - 1:
            for k in range(n):
                T_last[k] = T[k]


def _strike():
    params["impact_start"] = callback.pos
    print(f"Strike scheduled at sample {callback.pos}")


def current_contour():
    return make_morph_contour(
        params["morph"],
        params["aspect"],
    )


def point_inside(x, y):
    return points_in_poly(
        np.array([[x, y]]),
        current_contour(),
    )[0]


def _draw_points():
    canvas_id.delete("strike")
    canvas_id.delete("pickup")

    for tag, x, y, color in (
        ("strike", params["x_e"], params["y_e"], "red"),
        ("pickup", params["x_p"], params["y_p"], "green"),
    ):
        px = int((x + 1.0) * 0.5 * (canvas_size - 1))
        py = int((1.0 - y) * 0.5 * (canvas_size - 1))
        r = 6

        canvas_id.create_oval(
            px-r,
            py-r,
            px+r,
            py+r,
            fill=color,
            tags=(tag,),
        )


def update_canvas():
    contour = current_contour()

    xy = np.column_stack((
        (contour[:, 0] + 1.0) * 0.5 * canvas_size,
        (1.0 - contour[:, 1]) * 0.5 * canvas_size,
    ))

    canvas_id.coords(
        shape_id,
        *xy.ravel(),
    )

    _draw_points()


modal_factors, filter_freqs, strike_gains, pickup_gains = neural_plate_model(
    params["size"],
    params["aspect"],
    params["morph"],
    params["D"],
    params["rho"],
    params["H"],
    params["x_e"],
    params["y_e"],
    params["x_p"],
    params["y_p"],
)


_freqs_np_init = np.asarray(
    filter_freqs,
    dtype=np.float64,
)

_alphas_init = np.exp(np.minimum(
    params["alpha_g"]
    + params["alpha_r"] * _freqs_np_init,
    700.0,
))

Z = (
    np.exp(-_alphas_init / fs)
    * np.exp(
        1j
        * 2
        * np.pi
        * _freqs_np_init
        / fs
    )
)

Z = np.ascontiguousarray(
    Z,
    dtype=np.complex128,
)

states = np.zeros(
    len(filter_freqs),
    dtype=np.complex128,
)

last_params = params.copy()

M = distribution_matrix(
    filter_freqs,
    params["eta"],
    params["lamb"],
)

last_T = np.zeros(
    len(filter_freqs),
    dtype=np.float64,
)

MAX_STATE_MAG = 10.0


_dummy_states = states.copy()
_dummy_u = np.zeros(
    (len(filter_freqs), 4),
    dtype=np.float64,
)
_dummy_out = np.zeros(
    4,
    dtype=np.float64,
)
_dummy_T = np.zeros(
    len(filter_freqs),
    dtype=np.float64,
)

process_block(
    _dummy_states,
    Z,
    M,
    _dummy_u,
    pickup_gains,
    float(params["tau"]),
    MAX_STATE_MAG,
    _dummy_out,
    _dummy_T,
)

print("Numba-Kernel kompiliert.")


def callback(outdata, frames, time, status):
    if status:
        print(status)

    global modal_factors, filter_freqs, strike_gains, pickup_gains
    global Z, states, last_params, M, last_T

    if params["changed"]:
        last_params = params.copy()

        modal_factors, filter_freqs, strike_gains, pickup_gains = neural_plate_model(
            params["size"],
            params["aspect"],
            params["morph"],
            params["D"],
            params["rho"],
            params["H"],
            params["x_e"],
            params["y_e"],
            params["x_p"],
            params["y_p"],
        )

        freqs_np = np.asarray(
            filter_freqs,
            dtype=np.float64,
        )

        alphas = np.exp(np.minimum(
            params["alpha_g"]
            + params["alpha_r"] * freqs_np,
            700.0,
        ))

        Z = (
            np.exp(-alphas / fs)
            * np.exp(
                1j
                * 2
                * np.pi
                * freqs_np
                / fs
            )
        )

        Z = np.ascontiguousarray(
            Z,
            dtype=np.complex128,
        )

        M = distribution_matrix(
            filter_freqs,
            params["eta"],
            params["lamb"],
        )

        last_T = np.zeros(
            len(filter_freqs),
            dtype=np.float64,
        )

        params["changed"] = False

    n = len(states)

    pos = callback.pos + np.arange(frames)

    if params.get("impact_start") is None:
        u = np.zeros(
            (n, frames),
            dtype=np.float64,
        )
    else:
        u = excitation_signal(
            pos - params["impact_start"],
            strike_gains,
            params["N_ex"],
            params["A"],
        )

    u = np.ascontiguousarray(
        u,
        dtype=np.float64,
    )

    s_out = np.empty(
        frames,
        dtype=np.float64,
    )

    process_block(
        states,
        Z,
        M,
        u,
        pickup_gains,
        float(params["tau"]),
        MAX_STATE_MAG,
        s_out,
        last_T,
    )

    callback.pos += frames

    out = s_out / max(np.sqrt(n), 1)
    out = np.nan_to_num(
        out,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    out = np.clip(
        out,
        -1.0,
        1.0,
    )

    outdata[:] = out.reshape(-1, 1)


callback.pos = 0
params["impact_start"] = None


stream = sd.OutputStream(
    channels=1,
    samplerate=int(fs),
    callback=callback,
)

sd.sleep(1)
stream.start()


for f in filter_freqs:
    print(f"Filter frequency: {f:.2f} Hz")


def set_model_param(name, value):
    params[name] = float(value)

    if name in ("morph", "aspect"):
        for x_name, y_name in (
            ("x_e", "y_e"),
            ("x_p", "y_p"),
        ):
            if not point_inside(
                params[x_name],
                params[y_name],
            ):
                params[x_name] = 0.0
                params[y_name] = 0.0

        update_canvas()

    params["changed"] = True


root.title("Resonator Filter GUI (Numba)")
frm = ttk.Frame(root, padding=10)


gain_slider = ttk.Scale(
    root,
    from_=0.0,
    to=2,
    orient="horizontal",
    command=lambda v:
        params.__setitem__(
            "A",
            float(v),
        ),
)

ttk.Label(
    root,
    text="Excitation Amplitude",
).pack()

gain_slider.pack(
    fill="x"
)

gain_slider.set(
    params["A"]
)


strike_btn = ttk.Button(
    root,
    text="Strike",
    command=_strike,
)

strike_btn.pack(
    pady=6
)


excitation_btn = ttk.Button(
    root,
    text="Toggle Excitation",
    command=lambda:
        params.__setitem__(
            "excitation",
            1 - params["excitation"],
        ),
)

excitation_btn.pack(
    pady=6
)


morph_slider = ttk.Scale(
    root,
    from_=0.0,
    to=1.0,
    orient="horizontal",
    command=lambda v:
        set_model_param(
            "morph",
            v,
        ),
)

ttk.Label(
    root,
    text="Morph",
).pack()

morph_slider.pack(
    fill="x"
)

morph_slider.set(
    params["morph"]
)


size_slider = ttk.Scale(
    root,
    from_=0.01,
    to=5.0,
    orient="horizontal",
    command=lambda v:
        set_model_param(
            "size",
            v,
        ),
)

ttk.Label(
    root,
    text="Size [m]",
).pack()

size_slider.pack(
    fill="x"
)

size_slider.set(
    params["size"]
)


aspect_slider = ttk.Scale(
    root,
    from_=0.5,
    to=2.0,
    orient="horizontal",
    command=lambda v:
        set_model_param(
            "aspect",
            v,
        ),
)

ttk.Label(
    root,
    text="Aspect",
).pack()

aspect_slider.pack(
    fill="x"
)

aspect_slider.set(
    params["aspect"]
)


spacing_slider = ttk.Scale(
    root,
    from_=0.001,
    to=5,
    orient="horizontal",
    command=lambda v:
        set_model_param(
            "alpha_g",
            v,
        ),
)

ttk.Label(
    root,
    text="Alpha_g",
).pack()

spacing_slider.pack(
    fill="x"
)

spacing_slider.set(
    params["alpha_g"]
)


excitation_slider = Scale(
    root,
    from_=2,
    to=192,
    orient="horizontal",
    resolution=1,
    command=lambda v:
        params.__setitem__(
            "N_ex",
            int(float(v)),
        ),
)

ttk.Label(
    root,
    text="Excitation Length",
).pack()

excitation_slider.pack(
    fill="x"
)

excitation_slider.set(
    params["N_ex"]
)


ttk.Label(
    frm,
    text="Filter GUI",
).grid(
    column=0,
    row=0,
)


def _set_point(event, x_name, y_name):
    x = (
        2.0
        * min(
            max(event.x, 0),
            canvas_size - 1,
        )
        / (canvas_size - 1)
        - 1.0
    )

    y = (
        1.0
        - 2.0
        * min(
            max(event.y, 0),
            canvas_size - 1,
        )
        / (canvas_size - 1)
    )

    if point_inside(x, y):
        params[x_name] = x
        params[y_name] = y
        params["changed"] = True
        _draw_points()


_drag = {
    "on": False
}


def _on_press(event):
    _drag["on"] = True
    _on_move(event)


def _on_move(event):
    if _drag["on"]:
        _set_point(
            event,
            "x_e",
            "y_e",
        )


def _on_release(event):
    _drag["on"] = False


canvas_id.bind(
    "<Button-1>",
    _on_press,
)

canvas_id.bind(
    "<B1-Motion>",
    _on_move,
)

canvas_id.bind(
    "<ButtonRelease-1>",
    _on_release,
)

canvas_id.bind(
    "<Button-3>",
    lambda e:
        _set_point(
            e,
            "x_p",
            "y_p",
        ),
)

canvas_id.bind(
    "<B3-Motion>",
    lambda e:
        _set_point(
            e,
            "x_p",
            "y_p",
        ),
)


update_canvas()
root.mainloop()