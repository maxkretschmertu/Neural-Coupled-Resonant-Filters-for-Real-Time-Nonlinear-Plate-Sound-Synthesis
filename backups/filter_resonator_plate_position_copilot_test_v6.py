import numpy as np
import sounddevice as sd   # optional, for saving output
import os
from tkinter import *
from tkinter import ttk
import threading
import time


fs = 48000.0
f0 = 200
alpha_r = 4e-5 
alpha_g = 0.3322
Lx = 1.0  # plate length in x-direction (meters)
D = 18300  # flexural rigidity (N*m)
rho = 7800.0  # density (kg/m^3)
H = 0.01  # plate thickness (meters)
Ly = 1.0  # plate length in y-direction (meters)
omega = (np.pi**2/(Lx*Ly)) * np.sqrt(D/(rho*H))
f_m = omega/(2*np.pi)  # fundamental frequency of the plate


params = {
    "alpha_g": 0.3322,
    "alpha_r": 4e-5,
    "base_freq": f_m,
    "spacing": 1.4,
    "N": 5,
    "tau": 1,
    "eta": 1.0,
    "lamb": 0.2,
    "Lx": 1.0,
    "Ly": 1.0,
    "D": 18300,
    "rho": 7800.0,
    "H": 0.01,
    "x": 2,            # number of modes in x-direction
    "x_e": 0.5,        # excitation position in x (normalized 0..1)
    "y_e": 0.5,        # excitation position in y (normalized 0..1)
    "impact_start": None,
    "A": 1.0,          # excitation amplitude
    "N_ex": 48,        # excitation length in samples
}





def freqs(Lx, Ly, D, rho, H, x):
    v = Lx/Ly
    modes = [(m, n) for m in range(1, x+1) for n in range(1, x)]  # first x modes
    for l, m in modes:
        omega = (np.pi**2/(Lx*Ly)) * np.sqrt(D/(rho*H)) * (l**2 + v * m**2)
        f_m = omega/(2*np.pi)
        yield f_m

def resonator_filter(signal, fs, f0, alpha):
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim > 1:
        x = x[:, 0]
    y = np.empty_like(x, dtype=np.float64)
    Z = np.exp(-alpha / fs) * np.exp(1j * 2 * np.pi * f0 / fs)
    z_n = 0j
    for i, u in enumerate(x):
        z_n = Z * z_n + u
        y[i] = np.real(z_n)
    return y



def power_function(states):
    """states can be a scalar or array of complex resonator states."""
    P = 0.5 * (np.real(states) ** 2 + np.imag(states) ** 2)
    return P


def distribution_matrix(freqs_arr, eta=0.01, lamb=1.0):
    """
    Vectorized version of the original nested-loop matrix build.
    freqs_arr : (N,) array of filter center frequencies
    power_arr : (N,) array of per-filter power values (currently unused
                in the matrix build itself -- kept for signature parity)
    eta       : coupling strength between filters (off-diagonal scale)
    lamb      : self-inhibition strength (diagonal = -lamb)

    Returns M, the (N, N) coupling/distribution matrix.

    Each row's off-diagonal coupling terms sum to eta*lamb (normalized
    by that row's own total similarity, not the whole matrix), so the
    coupling strength stays predictable as N changes. With eta and
    lamb of comparable magnitude, the coupling term (off-diagonal) and
    self-inhibition term (diagonal, -lamb) are then on comparable
    scales, so T_i(n) = M @ [p(n)-tau]_+ isn't always clipped to 0 by
    max(T, 0) -- which is what made the matrix inaudible before this
    fix (it was normalized by the sum over the *entire* matrix, which
    grows with N^2 and drove the coupling term toward zero).
    """
    freqs_arr = np.asarray(freqs_arr, dtype=np.float64)
    n = len(freqs_arr)

    if n <= 1:
        return np.zeros((n, n))

    # pairwise |f_i - f_j| via broadcasting, no Python loop
    diff = np.abs(freqs_arr[:, None] - freqs_arr[None, :])
    a = 1.0 - diff / np.mean(freqs_arr)
    np.fill_diagonal(a, 0.0)  # i != j condition from the original

    denom = np.sum(a, axis=0, keepdims=True)  # per-row sum, NOT whole-matrix sum
    denom[denom == 0] = 1.0  # avoid div-by-zero for isolated/degenerate rows

    M = eta * lamb * a / denom - lamb * np.eye(n)
    return M


def compute_t(M, power_arr, tau):
    """
    t(n) = M [p(n) - tau]_+

    M        : (N, N) distribution/coupling matrix from distribution_matrix()
    power_arr: (N,) array of per-filter power values, p(n)
    tau      : scalar or (N,) array of per-filter thresholds

    [x]_+ denotes the rectified (positive-part) operator: max(x, 0),
    applied elementwise before the matrix multiply.
    """
    power_arr = np.asarray(power_arr, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)

    rectified = np.maximum(power_arr - tau, 0.0)  # [p(n) - tau]_+
    t = M @ rectified                              # M [p(n) - tau]_+
    return t


def excitation_signal(A, l, m, x, y, N_ex, fs, pos=None, start=None):
    """Return spatially-weighted raised-sinusoid excitation.

    u(n) = A * sin^2(pi * n / N_ex) for 0 <= n <= N_ex, else 0.

    If `pos` is None, returns the envelope for n=0..N_ex (length N_ex+1).
    If `pos` is an array of absolute sample indices, returns the
    corresponding values at those indices (same shape as `pos`). This
    lets the audio callback request the block at arbitrary absolute
    positions via `pos = callback.pos + np.arange(frames)`.
    """
    spatial = np.sin(l * np.pi * x) * np.sin(m * np.pi * y)

    if pos is None:
        n = np.arange(N_ex + 1)
        u = np.zeros_like(n, dtype=float)
        mask = (n >= 0) & (n <= N_ex)
        u[mask] = A * np.sin(np.pi * n[mask] / N_ex) ** 2
        return spatial * u

    pos = np.asarray(pos, dtype=int)
    u = np.zeros_like(pos, dtype=float)
    if start is None:
        # treat pos as absolute sample indices with impact starting at sample 0
        mask = (pos >= 0) & (pos <= N_ex)
        if np.any(mask):
            n_rel = pos[mask]
            u[mask] = A * np.sin(np.pi * n_rel / N_ex) ** 2
    else:
        # compute relative sample index from provided absolute start
        n_rel_all = pos - int(start)
        mask = (n_rel_all >= 0) & (n_rel_all <= N_ex)
        if np.any(mask):
            n_rel = n_rel_all[mask]
            u[mask] = A * np.sin(np.pi * n_rel / N_ex) ** 2
    return spatial * u

# initial frequencies
filter_freqs = list(freqs(params["Lx"], params["Ly"], params["D"], params["rho"], params["H"], params["x"]))

# resonator coefficients and states will be produced by a background
# worker thread so continuous GUI interactions (slider drags) don't
# trigger heavy work directly in the main thread and cause audio dropouts.
_freqs_np_init = np.asarray(filter_freqs, dtype=np.float64)
_alphas_init = np.exp(params["alpha_g"] + params["alpha_r"] * _freqs_np_init)
_Z_init = np.exp(-_alphas_init / fs) * np.exp(1j * 2 * np.pi * _freqs_np_init / fs)

states = np.zeros(len(filter_freqs), dtype=np.complex128)

# shared computed container and lock
computed_lock = threading.Lock()
computed = {
    'filter_freqs': list(filter_freqs),
    'Z': _Z_init,
    'M': distribution_matrix(filter_freqs, params['eta'], params['lamb']),
    'alphas': _alphas_init,
    'version': 0,
}
last_params = params.copy()
last_T = None

# Hard ceiling on |z_i(n)|: this nonlinear recurrence can enter runaway
# positive feedback (see callback for why) and diverge to inf/NaN within
# a few thousand samples at higher N/eta. This clamp keeps it bounded
# and audible instead of blowing up.
MAX_STATE_MAG = 10.0


def callback(outdata, frames, time, status):
    if status:
        print(status)

    global states, last_params, last_T
    # Read precomputed arrays from the worker thread (fast, under lock)
    with computed_lock:
        cur_version = computed['version']
        cur_Z = computed['Z']
        cur_M = computed['M']
        cur_filter_freqs = list(computed['filter_freqs'])

    # If the number of filters changed, resize state accordingly.
    if len(states) != len(cur_filter_freqs):
        states = np.zeros(len(cur_filter_freqs), dtype=np.complex128)

    n = len(states)

    # Build the excitation vector for this block using the precomputed
    # parameters (fast). `excitation_signal` is lightweight.
    pos = callback.pos + np.arange(frames)
    u = excitation_signal(A=params.get("A", 1.0), l=1, m=1,
                         x=params["x_e"], y=params["y_e"],
                         N_ex=params.get("N_ex", 48), fs=fs, pos=pos,
                         start=params.get("impact_start", None))

    # Nonlinear resonator bank update:
    #
    #   z_i(n+1) = sqrt(2*T_i(n)) * Z_i + u_i(n)                          if z_i(n) == 0
    #            = sqrt(1 + 2*T_i(n)/|z_i(n)|^2) * Z_i * z_i(n) + u_i(n)  else
    #
    # T_i(n) = t(n) = M [p(n) - tau]_+ depends on the *current* power at
    # each sample, so it's recomputed every sample (M itself is not --
    # it only changes when filter_freqs change, see above).
    tau = params["tau"]
    s_out = np.empty(frames, dtype=np.float64)

    for i in range(frames):
        z = states

        p = power_function(z)                     # p(n), per filter
        rectified = np.maximum(p - tau, 0.0)       # [p(n) - tau]_+
        T = cur_M @ rectified                      # T_i(n)
        T = np.maximum(T, 0.0)                     # guard: sqrt() needs T >= 0

        new_z = np.empty_like(z)
        # Use a small-magnitude threshold rather than exact `== 0`: in
        # floating point, z(n) essentially never hits exact zero, but it
        # can get extremely close on its way down, and the "else" branch
        # divides by |z(n)|^2 -- so a near-zero (not exactly zero) state
        # causes a divide blow-up (inf/NaN) that the exact-equality check
        # would miss entirely. This guard catches that case too.
        mag2_all = np.abs(z) ** 2
        zero_mask = mag2_all < 1e-20
        nz_mask = ~zero_mask

        if np.any(zero_mask):
            new_z[zero_mask] = np.sqrt(2.0 * T[zero_mask]) * cur_Z[zero_mask] + u[i]

        if np.any(nz_mask):
            mag2 = mag2_all[nz_mask]
            factor = np.sqrt(1.0 + (2.0 * T[nz_mask]) / mag2)
            new_z[nz_mask] = factor * cur_Z[nz_mask] * z[nz_mask] + u[i]

        states[:] = new_z

        # Safety clamp: this recurrence is a positive-feedback loop
        # (T > 0 grows a filter, which raises power, which raises T for
        # coupled neighbors...). At higher N / eta it can diverge to
        # inf/NaN within a few thousand samples, which must never reach
        # the sound device. Soft-clamp any state exceeding MAX_STATE_MAG
        # back down instead of letting it blow up.
        mag = np.abs(states)
        over = mag > MAX_STATE_MAG
        if np.any(over):
            states[over] *= (MAX_STATE_MAG / mag[over])

        s_out[i] = np.imag(states).sum()

    last_T = T  # last sample's T_i(n), kept around for inspection/GUI

    callback.pos += frames

    out = s_out / max(n, 1)
    # Last line of defense: never let NaN/Inf or an extreme transient
    # reach the sound device, no matter what the state clamp above missed.
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, -1.0, 1.0)

    outdata[:] = out.reshape(-1, 1)


callback.pos = 0

stream = sd.OutputStream(channels=1, samplerate=int(fs), callback=callback)
sd.sleep(1)  # Give the stream time to start
stream.start()


# Background worker: recompute filter_freqs, alphas, Z, M at a low rate
# when GUI params change. This lets sliders update continuously without
# doing heavy work on the main thread.
_worker_stop = False
def _worker():
    global computed
    last_snapshot = None
    keys = ('Lx','Ly','D','rho','H','x','alpha_g','alpha_r','eta','lamb')
    while not _worker_stop:
        snapshot = tuple(params[k] for k in keys)
        if snapshot != last_snapshot:
            last_snapshot = snapshot
            # recompute
            try:
                freqs_new = list(freqs(params['Lx'], params['Ly'], params['D'], params['rho'], params['H'], params['x']))
                freqs_np = np.asarray(freqs_new, dtype=np.float64)
                alphas = np.exp(params['alpha_g'] + params['alpha_r'] * freqs_np)
                Z_new = np.exp(-alphas / fs) * np.exp(1j * 2 * np.pi * freqs_np / fs)
                M_new = distribution_matrix(freqs_new, params['eta'], params['lamb'])
                with computed_lock:
                    computed['filter_freqs'] = freqs_new
                    computed['Z'] = Z_new
                    computed['M'] = M_new
                    computed['alphas'] = alphas
                    computed['version'] += 1
                print(f"[worker] recomputed version={computed['version']}; alpha_g={params['alpha_g']:.6g}")
            except Exception:
                pass
        time.sleep(0.02)  # ~50 Hz update

worker_thread = threading.Thread(target=_worker, daemon=True)
worker_thread.start()


for f in filter_freqs:
    print(f"Filter frequency: {f:.2f} Hz")

root = Tk()
root.title("Resonator Filter GUI")
frm = ttk.Frame(root, padding=10)


# Lx slider (update on release to avoid continuous GUI callbacks)
base_slider = ttk.Scale(root, from_=0.001, to=2, orient="horizontal")
ttk.Label(root, text="Lx").pack()
base_slider.pack(fill="x")
base_slider.set(params["Lx"])
base_slider.bind('<ButtonRelease-1>', lambda e, s=base_slider: params.__setitem__("Lx", float(s.get())))
base_slider.bind('<B1-Motion>', lambda e, s=base_slider: params.__setitem__("Lx", float(s.get())))

# Ly slider (update on release)
base_slider = ttk.Scale(root, from_=0.001, to=2, orient="horizontal")
ttk.Label(root, text="Ly").pack()
base_slider.pack(fill="x")
base_slider.set(params["Ly"])
base_slider.bind('<ButtonRelease-1>', lambda e, s=base_slider: params.__setitem__("Ly", float(s.get())))
base_slider.bind('<B1-Motion>', lambda e, s=base_slider: params.__setitem__("Ly", float(s.get())))

# Alpha_g slider (update on release)
spacing_slider = ttk.Scale(root, from_=0.001, to=5, orient="horizontal")
ttk.Label(root, text="Alpha_g").pack()
spacing_slider.pack(fill="x")
spacing_slider.set(params["alpha_g"])
spacing_slider.bind('<ButtonRelease-1>', lambda e, s=spacing_slider: params.__setitem__("alpha_g", float(s.get())))
spacing_slider.bind('<B1-Motion>', lambda e, s=spacing_slider: params.__setitem__("alpha_g", float(s.get())))

# Number of Modes slider (update on release)
number_slider = Scale(root, from_=1, to=15, orient="horizontal", resolution=1)
ttk.Label(root, text="Number of Modes").pack()
number_slider.pack(fill="x")
number_slider.set(params["x"])
number_slider.bind('<ButtonRelease-1>', lambda e, s=number_slider: params.__setitem__("x", int(float(s.get()))))
number_slider.bind('<B1-Motion>', lambda e, s=number_slider: params.__setitem__("x", int(float(s.get()))))


ttk.Label(frm, text="Filter GUI").grid(column=0, row=0)

# Canvas to pick excitation position (normalized coords 0..1)
canvas_size = 300
ttk.Label(root, text="Excitation Position (drag the red dot)").pack()
canvas = Canvas(root, width=canvas_size, height=canvas_size, bg='white')
canvas.pack(pady=6)
canvas.create_rectangle(2, 2, canvas_size-2, canvas_size-2, outline='black')

def _draw_point():
    canvas.delete('point')
    px = int(params['x_e'] * (canvas_size - 1))
    py = int((1.0 - params['y_e']) * (canvas_size - 1))
    r = 6
    canvas.create_oval(px-r, py-r, px+r, py+r, fill='red', tags=('point',))

_draw_point()

_drag = {'on': False}
def _on_press(event):
    _drag['on'] = True
    _on_move(event)
def _on_move(event):
    if not _drag['on']:
        return
    px = min(max(event.x, 0), canvas_size - 1)
    py = min(max(event.y, 0), canvas_size - 1)
    params['x_e'] = px / (canvas_size - 1)
    params['y_e'] = 1.0 - (py / (canvas_size - 1))
    _draw_point()
def _on_release(event):
    _drag['on'] = False

canvas.bind('<Button-1>', _on_press)
canvas.bind('<B1-Motion>', _on_move)
canvas.bind('<ButtonRelease-1>', _on_release)

# Amplitude slider for excitation (update on release)
ttk.Label(root, text="Excitation Amplitude (A)").pack()
amp_slider = ttk.Scale(root, from_=0.0, to=5.0, orient="horizontal")
amp_slider.pack(fill="x")
amp_slider.set(params.get('A', 1.0))
amp_slider.bind('<ButtonRelease-1>', lambda e, s=amp_slider: params.__setitem__("A", float(s.get())))
amp_slider.bind('<B1-Motion>', lambda e, s=amp_slider: params.__setitem__("A", float(s.get())))
def _strike():
    # schedule an immediate impact at the current absolute sample position
    params["impact_start"] = callback.pos
    print(f"Strike scheduled at sample {callback.pos}")

strike_btn = ttk.Button(root, text="Strike", command=_strike)
strike_btn.pack(pady=6)
root.mainloop()