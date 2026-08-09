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
    "alpha": np.exp(alpha_g + omega * alpha_r),  # damping / decay rate
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
    "D": 100.0,   # flexural rigidity (N*m)
    "rho": 7800.0,   # density (kg/m^3)
    "H": 0.01   # plate thickness (meters)
}

def plate_fundamental_frequency(Lx, Ly, D, rho, H):
    omega = (np.pi**2/(Lx*Ly)) * np.sqrt(D/(rho*H))
    f_m = omega/(2*np.pi)
    return f_m, omega

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


def freqs(N, base_freq, spacing):
    f = base_freq
    for _ in range(N):
        yield f
        f *= spacing


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


# initial frequencies
filter_freqs = list(freqs(params["N"], params["base_freq"], params["spacing"]))

# resonator coefficients and states
# (frequency-scaled alpha, matching the formula used inside callback() --
# see the comment there for why this matters)
_freqs_np_init = np.asarray(filter_freqs, dtype=np.float64)
_alphas_init = params["alpha"] * np.sqrt(_freqs_np_init / params["base_freq"])
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
        filter_freqs = list(freqs(params["N"], params["base_freq"], params["spacing"]))
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
        alphas = params["alpha"] * np.sqrt(freqs_np / params["base_freq"])
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
    u = (pos % int(fs) == 0).astype(np.float64)

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

# Alpha slider
#alpha_slider = ttk.Scale(root, from_=0.1, to=50, orient="horizontal",
 #                         command=lambda v: params.__setitem__("alpha", float(v)))

#ttk.Label(root, text="Alpha").pack()
#alpha_slider.pack(fill="x")
#alpha_slider.set(params["alpha"])

# Base frequency slider
base_slider = ttk.Scale(root, from_=20, to=2000, orient="horizontal",
                         command=lambda v: params.__setitem__("base_freq", float(v)))

ttk.Label(root, text="Base Frequency").pack()
base_slider.pack(fill="x")
base_slider.set(params["base_freq"])

# Spacing slider
spacing_slider = ttk.Scale(root, from_=1.01, to=2.0, orient="horizontal",
                            command=lambda v: params.__setitem__("spacing", float(v)))

ttk.Label(root, text="Spacing").pack()
spacing_slider.pack(fill="x")
spacing_slider.set(params["spacing"])

# Number of filters slider
number_slider = Scale(root, from_=1, to=15, orient="horizontal", resolution=1,
                       command=lambda v: params.__setitem__("N", int(float(v))))

ttk.Label(root, text="Number of Filters").pack()
number_slider.pack(fill="x")
number_slider.set(params["N"])

# Threshold (tau) slider
tau_slider = ttk.Scale(root, from_=0.0, to=5.0, orient="horizontal",
                        command=lambda v: params.__setitem__("tau", float(v)))

ttk.Label(root, text="Threshold (tau)").pack()
tau_slider.pack(fill="x")
tau_slider.set(params["tau"])

# Coupling strength (eta) slider
eta_slider = ttk.Scale(root, from_=0.0, to=2.0, orient="horizontal",
                        command=lambda v: params.__setitem__("eta", float(v)))

ttk.Label(root, text="Coupling Strength (eta)").pack()
eta_slider.pack(fill="x")
eta_slider.set(params["eta"])

# Self-inhibition strength (lambda) slider
lamb_slider = ttk.Scale(root, from_=0.01, to=2.0, orient="horizontal",
                         command=lambda v: params.__setitem__("lamb", float(v)))

ttk.Label(root, text="Self-Inhibition (lambda)").pack()
lamb_slider.pack(fill="x")
lamb_slider.set(params["lamb"])


ttk.Label(frm, text="Filter GUI").grid(column=0, row=0)
root.mainloop()