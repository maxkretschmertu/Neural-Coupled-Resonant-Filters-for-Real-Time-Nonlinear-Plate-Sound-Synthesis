import numpy as np
import sounddevice as sd   # optional, for saving output
import os
from tkinter import *
from tkinter import ttk


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
    "alpha_g": 0.3322,  # damping / decay rate
    "alpha_r": 4e-5,  # damping / decay rate
    "base_freq": f_m,
    "spacing": 1.4,
    "N": 5,       # >1 filters needed for the distribution matrix to do
                  # anything at all -- with N=1, M is a 1x1 zero matrix
                  # and T_i(n) is always 0 regardless of tau/eta/lamb


    "tau": 1,   # user-set threshold for [p(n) - tau]_+
    "eta": 1,   # coupling strength between filters (off-diagonal scale)
    "lamb": 0.2,   # self-inhibition strength (diagonal = -lamb) -- lowered
                  # from 1.0 so eta's coupling term can actually compete
                  # with it instead of always being clipped to 0

    "Lx": 1.0,  # plate length in x-direction (meters)
    "Ly": 1.0,  # plate length in y-direction (meters)
    "D": 18300,   # flexural rigidity (N*m)
    "rho": 7800.0,   # density (kg/m^3)
    "H": 0.01,   # plate thickness (meters)
    "x": 2,
    "impact_start": None,  # sample index for the next scheduled impact
    "feedback_gain": 0.20,  # scales the nonlinear feedback term
    "state_decay": 0.999,   # extra per-sample damping so the sound decays
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
    t = M @ rectified                             # M[p(n) - tau]_+
    return t

def excitation_signal(pos, x=0.3, y=0.7, l=2, m=2, N_ex=192, A=1.0):
    """
    Kurzer Anregungsimpuls (sin^2-Hüllkurve), Dauer N_ex Samples.
    pos    : Sample-Index (int oder Array), NICHT durch fs geteilt
    N_ex   : Pulsdauer in Samples (Default 192 = 4 ms bei fs=48000)
    """
    pos = np.asarray(pos, dtype=np.float64)
    u = np.zeros_like(pos)

    within = pos < N_ex
    u[within] = A * np.sin(np.pi * (pos[within]) / N_ex) ** 2

    u_ex = np.sin(l * np.pi * x) * np.sin(m * np.pi * y) * u
    return u_ex

def _strike():
    # schedule an immediate impact at the current absolute sample position
    params["impact_start"] = callback.pos
    print(f"Strike scheduled at sample {callback.pos}")


# initial frequencies
filter_freqs = list(freqs(params["Lx"], params["Ly"], params["D"], params["rho"], params["H"], params["x"]))

# resonator coefficients and states
# (frequency-scaled alpha, matching the formula used inside callback() --
# see the comment there for why this matters)
_freqs_np_init = np.asarray(filter_freqs, dtype=np.float64)
_alphas_init = np.exp(params["alpha_g"] + params["alpha_r"] * _freqs_np_init)
Z = np.exp(-_alphas_init / fs) * np.exp(1j * 2 * np.pi * _freqs_np_init / fs)

states = np.zeros(len(filter_freqs), dtype=np.complex128)

last_params = params.copy()

# M only depends on filter_freqs (not on power/tau), so it's computed
# once here and recomputed only when N/base_freq/spacing change --
# no need to rebuild it every sample. T_i(n), by contrast, DOES change
# every sample (it depends on the live power p(n)), so it's computed
# inside the callback's per-sample loop.
M = distribution_matrix(filter_freqs, params["eta"], params["lamb"])
last_T = None  # last computed T_i(n) vector, kept for inspection

# Hard ceiling on |z_i(n)|: this nonlinear recurrence can enter runaway
# positive feedback (see callback for why) and diverge to inf/NaN within
# a few thousand samples at higher N/eta. This clamp keeps it bounded
# and audible instead of blowing up.
MAX_STATE_MAG = 10.0


def callback(outdata, frames, time, status):
    if status:
        print(status)

    global filter_freqs, Z, states, last_params, M, last_T

    # Check if ANY parameter changed
    if params != last_params:
        last_params = params.copy()

        # Recompute frequencies
        filter_freqs = list(freqs(params["Lx"], params["Ly"], params["D"], params["rho"], params["H"], params["x"]))
        freqs_np = np.asarray(filter_freqs, dtype=np.float64)

        # Recompute Z coefficients (vectorized).
        #
        # IMPORTANT: alpha is scaled per filter by frequency here, not
        # shared as one flat value. With one shared alpha, every filter
        # has identical |Z| (decay magnitude) and receives the same
        # impulse train, so every filter's power stays IDENTICAL to
        # every other filter's at every sample -- there's no asymmetry
        # for the distribution matrix to redistribute, so M's coupling
        # terms collapse to a uniform per-row constant (which is why
        # tau/eta/lamb had no differentiated audible effect). Scaling
        # decay by frequency (higher partials decay faster, which is
        # also physically realistic) breaks that symmetry so the
        # coupling term actually varies per filter.
        alphas = np.exp(params["alpha_g"] + params["alpha_r"] * freqs_np)
        Z = np.exp(-alphas / fs) * np.exp(1j * 2 * np.pi * freqs_np / fs)

        # M depends only on filter_freqs, so it only needs rebuilding
        # here -- not every sample or every block.
        M = distribution_matrix(filter_freqs,
                                 params["eta"], params["lamb"])

        # Reset states
        states = np.zeros(len(filter_freqs), dtype=np.complex128)

    n = len(states)

    # Vectorized impulse train: u[i] = 1.0 at sample positions that are
    # multiples of fs, else 0.0. Replaces the per-sample Python `if`.
    pos = callback.pos + np.arange(frames)
    u = excitation_signal(pos)

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
        T = M @ rectified                          # T_i(n)
        T = np.maximum(T, 0.0)                     # guard: sqrt() needs T >= 0
        T *= float(params.get("feedback_gain", 0.20))

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
            new_z[zero_mask] = np.sqrt(2.0 * T[zero_mask]) * Z[zero_mask] + u[i]

        if np.any(nz_mask):
            mag2 = mag2_all[nz_mask]
            factor = np.sqrt(1.0 + (2.0 * T[nz_mask]) / mag2)
            new_z[nz_mask] = factor * Z[nz_mask] * z[nz_mask] + u[i]

        states[:] = new_z
        state_decay = float(params.get("state_decay", 0.999))
        if state_decay < 1.0:
            states *= state_decay

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


for f in filter_freqs:
    print(f"Filter frequency: {f:.2f} Hz")

root = Tk()
root.title("Resonator Filter GUI")
frm = ttk.Frame(root, padding=10)


# Lx slider
base_slider = ttk.Scale(root, from_=0.001, to=2, orient="horizontal",
                         command=lambda v: params.__setitem__("Lx", float(v)))

ttk.Label(root, text="Lx").pack()
base_slider.pack(fill="x")
base_slider.set(params["Lx"])

# Ly slider
base_slider = ttk.Scale(root, from_=0.001, to=2, orient="horizontal",
                         command=lambda v: params.__setitem__("Ly", float(v)))

ttk.Label(root, text="Ly").pack()
base_slider.pack(fill="x")
base_slider.set(params["Ly"])

# Alpha_g slider
spacing_slider = ttk.Scale(root, from_=0.001, to=5, orient="horizontal",
                            command=lambda v: params.__setitem__("alpha_g", float(v)))

ttk.Label(root, text="Alpha_g").pack()
spacing_slider.pack(fill="x")
spacing_slider.set(params["alpha_g"])

# Number of Modes slider
number_slider = Scale(root, from_=1, to=15, orient="horizontal", resolution=1,
                       command=lambda v: params.__setitem__("x", int(float(v))))

ttk.Label(root, text="Number of Modes").pack()
number_slider.pack(fill="x")
number_slider.set(params["x"])


ttk.Label(frm, text="Filter GUI").grid(column=0, row=0)


strike_btn = ttk.Button(root, text="Strike", command=_strike)
strike_btn.pack(pady=6)

root.mainloop()