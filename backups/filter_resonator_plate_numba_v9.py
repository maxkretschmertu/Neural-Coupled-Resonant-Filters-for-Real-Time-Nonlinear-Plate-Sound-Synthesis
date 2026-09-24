import numpy as np
import sounddevice as sd   # optional, for saving output
import os
from tkinter import *
from tkinter import ttk

from numba import njit


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

    "tau": 1,   # user-set threshold for [p(n) - tau]_+
    "eta": 1,   # coupling strength between filters (off-diagonal scale)
    "lamb": 0.01,   # self-inhibition strength (diagonal = -lamb)

    "Lx": 1.0,  # plate length in x-direction (meters)
    "Ly": 1.0,  # plate length in y-direction (meters)
    "D": 18300,   # flexural rigidity (N*m)
    "rho": 7800.0,   # density (kg/m^3)
    "H": 0.01,   # plate thickness (meters)
    "x": 2,
    "excitation": 1
}


def freqs(Lx, Ly, D, rho, H, x):
    v = Lx/Ly
    modes = [(m, n) for m in range(1, x+1) for n in range(1, x)]  # first x modes
    for l, m in modes:
        omega = (np.pi**2/(Lx*Ly)) * np.sqrt(D/(rho*H)) * (l**2 + v * m**2)
        f_m = omega/(2*np.pi)
        yield f_m


def distribution_matrix(freqs_arr, eta=0.01, lamb=1.0):
    """
    Same math as the original: row-normalized similarity coupling on the
    off-diagonal, -lamb self-inhibition on the diagonal. Only rebuilt when
    filter_freqs/eta/lamb change (i.e. not per-sample), so it stays plain
    NumPy -- no need to JIT this one.
    """
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


def excitation_signal(pos, x=0.3, y=0.7, l=2, m=2, N_ex=192, A=1.0):
    """
    Kurzer Anregungsimpuls (sin^2-Huellkurve), Dauer N_ex Samples.
    pos    : Sample-Index (int oder Array), NICHT durch fs geteilt
    N_ex   : Pulsdauer in Samples (Default 192 = 4 ms bei fs=48000)
    """
    if params.get("excitation") == 0:
        pos = np.asarray(pos, dtype=np.float64)
        u = np.zeros_like(pos)
        within = pos < N_ex
        scale = 3.0 / N_ex if N_ex > 0 else 1.0
        u[within] = A * scale * np.sin(np.pi * (pos[within]) / N_ex) ** 2

        u_ex = np.sin(l * np.pi * x) * np.sin(m * np.pi * y) * u
        return u_ex
    else:
        u_ex = (pos % int(fs) == 0).astype(np.float64)
        return u_ex


# ---------------------------------------------------------------------------
# cache=True: kompiliertes Maschinencode wird auf Platte zwischengespeichert,
# damit der (einmalige, ca. 0.5-2s dauernde) Kompilier-Schritt nicht bei
# jedem Programmstart erneut passiert.
# fastmath=True: erlaubt dem Compiler etwas laxere Fliesskomma-Reihenfolge
# fuer zusaetzliche Vektorisierung -- bei einer Audio-Rekursion mit
# harten Clamp/NaN-Guards unkritisch.
@njit(cache=True, fastmath=True)
def process_block(states, Z, M, u, tau, max_state_mag, s_out, T_last):
    """
    Verarbeitet einen kompletten Audio-Block (frames Samples) fuer n Filter.

    states   : (n,) complex128, in-place aktualisiert
    Z        : (n,) complex128, Resonator-Koeffizienten
    M        : (n, n) float64, Kopplungsmatrix
    u        : (frames,) float64, Anregungssignal fuer diesen Block
    tau      : float64, Schwellwert
    max_state_mag : float64, Clamp-Grenze
    s_out    : (frames,) float64, wird befuellt (Output-Signal)
    T_last   : (n,) float64, wird mit T_i(n) des letzten Samples befuellt
               (fuer GUI-Inspektion, entspricht `last_T` im Original)
    """
    n = states.shape[0]
    frames = u.shape[0]

    rectified = np.empty(n, dtype=np.float64)
    T = np.empty(n, dtype=np.float64)

    for i in range(frames):
        # p(n) und [p(n) - tau]_+ pro Filter
        for k in range(n):
            zr = states[k].real
            zi = states[k].imag
            p = 0.5 * (zr * zr + zi * zi)
            r = p - tau
            rectified[k] = r if r > 0.0 else 0.0

        # T_i(n) = M @ [p(n)-tau]_+ , anschliessend max(T, 0)
        for k in range(n):
            acc = 0.0
            for j in range(n):
                acc += M[k, j] * rectified[j]
            T[k] = acc if acc > 0.0 else 0.0

        # Zustandsupdate (Fallunterscheidung z(n)==0 vs. sonst, mit
        # Epsilon-Schwelle statt exakter Gleichheit -- wie im Original)
        for k in range(n):
            zr = states[k].real
            zi = states[k].imag
            mag2 = zr * zr + zi * zi

            if mag2 < 1e-20:
                amp = np.sqrt(2.0 * T[k])
                new_z = amp * Z[k] + u[i]
            else:
                factor = np.sqrt(1.0 + (2.0 * T[k]) / mag2)
                new_z = factor * Z[k] * states[k] + u[i]

            # Clamp gegen die positive Feedback-Divergenz
            nr = new_z.real
            ni = new_z.imag
            mag = np.sqrt(nr * nr + ni * ni)
            if mag > max_state_mag:
                new_z = new_z * (max_state_mag / mag)

            states[k] = new_z

        # Summe von Im(z_k) ueber alle Filter -> Ausgangssignal
        acc = 0.0
        for k in range(n):
            acc += states[k].imag
        s_out[i] = acc

        # letztes T fuer GUI-Inspektion merken
        if i == frames - 1:
            for k in range(n):
                T_last[k] = T[k]


def _strike():
    params["impact_start"] = callback.pos
    print(f"Strike scheduled at sample {callback.pos}")


# initial frequencies
filter_freqs = list(freqs(params["Lx"], params["Ly"], params["D"], params["rho"], params["H"], params["x"]))

_freqs_np_init = np.asarray(filter_freqs, dtype=np.float64)
_alphas_init = np.exp(params["alpha_g"] + params["alpha_r"] * _freqs_np_init)
Z = np.exp(-_alphas_init / fs) * np.exp(1j * 2 * np.pi * _freqs_np_init / fs)
Z = np.ascontiguousarray(Z, dtype=np.complex128)

states = np.zeros(len(filter_freqs), dtype=np.complex128)

last_params = params.copy()

M = distribution_matrix(filter_freqs, params["eta"], params["lamb"])
last_T = np.zeros(len(filter_freqs), dtype=np.float64)

MAX_STATE_MAG = 10.0

# Einmaliger Warmup-Aufruf: kompiliert process_block() JIT, BEVOR der
# Audio-Callback zum ersten Mal laeuft (sonst wuerde der allererste Audio-
# Block durch die Kompilierzeit von ~0.5-2s blockieren -> garantierter
# Dropout beim Start).
_dummy_states = states.copy()
_dummy_u = np.zeros(4, dtype=np.float64)
_dummy_out = np.zeros(4, dtype=np.float64)
_dummy_T = np.zeros(len(filter_freqs), dtype=np.float64)
process_block(_dummy_states, Z, M, _dummy_u, float(params["tau"]), MAX_STATE_MAG, _dummy_out, _dummy_T)
print("Numba-Kernel kompiliert.")


def callback(outdata, frames, time, status):
    if status:
        print(status)

    global filter_freqs, Z, states, last_params, M, last_T

    if params != last_params:
        last_params = params.copy()

        filter_freqs = list(freqs(params["Lx"], params["Ly"], params["D"], params["rho"], params["H"], params["x"]))
        freqs_np = np.asarray(filter_freqs, dtype=np.float64)

        alphas = np.exp(params["alpha_g"] + params["alpha_r"] * freqs_np)
        Z = np.exp(-alphas / fs) * np.exp(1j * 2 * np.pi * freqs_np / fs)
        Z = np.ascontiguousarray(Z, dtype=np.complex128)

        M = distribution_matrix(filter_freqs, params["eta"], params["lamb"])

        states = np.zeros(len(filter_freqs), dtype=np.complex128)
        last_T = np.zeros(len(filter_freqs), dtype=np.float64)

    n = len(states)

    pos = callback.pos + np.arange(frames)

    if params.get("impact_start") is None:
        u = excitation_signal(pos)
    else:
        u = excitation_signal(pos - params["impact_start"])
    u = np.ascontiguousarray(u, dtype=np.float64)

    s_out = np.empty(frames, dtype=np.float64)

    process_block(states, Z, M, u, float(params["tau"]), MAX_STATE_MAG, s_out, last_T)

    callback.pos += frames

    out = s_out / max(np.sqrt(n), 1)
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
root.title("Resonator Filter GUI (Numba)")
frm = ttk.Frame(root, padding=10)


base_slider = ttk.Scale(root, from_=0.001, to=2, orient="horizontal",
                         command=lambda v: params.__setitem__("Lx", float(v)))
ttk.Label(root, text="Lx").pack()
base_slider.pack(fill="x")
base_slider.set(params["Lx"])

base_slider = ttk.Scale(root, from_=0.001, to=2, orient="horizontal",
                         command=lambda v: params.__setitem__("Ly", float(v)))
ttk.Label(root, text="Ly").pack()
base_slider.pack(fill="x")
base_slider.set(params["Ly"])

spacing_slider = ttk.Scale(root, from_=0.001, to=5, orient="horizontal",
                            command=lambda v: params.__setitem__("alpha_g", float(v)))
ttk.Label(root, text="Alpha_g").pack()
spacing_slider.pack(fill="x")
spacing_slider.set(params["alpha_g"])

number_slider = Scale(root, from_=1, to=15, orient="horizontal", resolution=1,
                       command=lambda v: params.__setitem__("x", int(float(v))))
ttk.Label(root, text="Number of Modes").pack()
number_slider.pack(fill="x")
number_slider.set(params["x"])


ttk.Label(frm, text="Filter GUI").grid(column=0, row=0)

strike_btn = ttk.Button(root, text="Strike", command=_strike)
strike_btn.pack(pady=6)

excitation_btn = ttk.Button(root, text="Toggle Excitation", command=lambda: params.__setitem__("excitation", 1 - params["excitation"]))
excitation_btn.pack(pady=6)


root.mainloop()