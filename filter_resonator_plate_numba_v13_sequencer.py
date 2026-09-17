import numpy as np
import sounddevice as sd   # optional, for saving output
import os
import math
from tkinter import *
from tkinter import ttk

from numba import njit


fs = 48000.0                                            #sample rate
#f0 = 200
alpha_r = 4e-5                                          #damping coeffincient inducing linear plate vibration charactersitics
alpha_g = 0.3322                                        #damping coeffincient inducing linear plate vibration charactersitics
Lx = 1.0                                                # plate length in x-direction (meters)
D = 18300                                               # flexural rigidity (N*m)
rho = 7800.0                                            # density (kg/m^3)
H = 0.01                                               # plate thickness (meters)
Ly = 1.0                                                # plate length in y-direction (meters)
omega = (np.pi**2/(Lx*Ly)) * np.sqrt(D/(rho*H))
f_m = omega/(2*np.pi)                                   # fundamental frequency of the plate
root = Tk() 

# Canvas to pick excitation position (normalized coords 0..1)
canvas_size = 300
ttk.Label(root, text="Excitation Position (drag the red dot)").pack()
canvas_id = Canvas(root, width=canvas_size, height=canvas_size, bg='white')
canvas_id.pack(pady=6)
moving_rect = canvas_id.create_rectangle(2, 2, canvas_size-2, canvas_size-2, outline='black')
moving_rect_2 = canvas_id.create_rectangle(20, 20, 40, 40, fill="blue")

#real-time adjustable Parameters
params = {
    "alpha_g": 0.3322,
    "alpha_r": 4e-5,
    #"base_freq": f_m,
    #"spacing": 1.4,

    "tau": 1,                                           # user-set threshold for [p(n) - tau]_+
    "eta": 0.01,                                           # coupling strength between filters (off-diagonal scale)
    "lamb": 0.01,                                          # self-inhibition strength (diagonal = -lamb)

    "Lx": 1.0,                                          # plate length in x-direction (meters)
    "Ly": 1.0,                                          # plate length in y-direction (meters)
    "D": 10000,                                         # flexural rigidity (N*m)
    "rho": 7800.0,                                      # density (kg/m^3)
    "H": 0.02,                                          # plate thickness (meters)
    "x": 10,                                             # number of modes in directions of Lx and Ly
    "excitation": 0,
    "x_e": 0.3,                                         # excitation position x (normalized 0..1)
    "y_e": 0.3,                                         # excitation position y (normalized 0..1)
    "N_ex": 192,                                        # excitation duration in samples
    "changed": False

}


# -----------------------------------------------------------------------------
# 8-step sequencer
# X = strike, O = rest. Each step stores its own Lx, Ly and alpha_g.
# One step is a 16th note: samples_per_step = fs * 60 / BPM / 4.
# -----------------------------------------------------------------------------
N_STEPS = 8
sequencer_steps = [
    {
        "active": (i % 2 == 0),
        "Lx": float(params["Lx"]),
        "Ly": float(params["Ly"]),
        "alpha_g": float(params["alpha_g"]),
        "cache": None,
    }
    for i in range(N_STEPS)
]

sequencer = {
    "running": False,
    "bpm": 120.0,
    "step_index": -1,
    "current_step": -1,
    "next_step_sample": 0,
}


def get_modes(x):
    """(l, m) mode index pairs, same enumeration order used everywhere else."""
    return [(m, n) for m in range(1, x + 1) for n in range(1, x)]  # first x modes


def modes_to_freqs(modes, Lx, Ly, D, rho, H):
    """Modal frequencies for a given list of (l, m) mode indices."""
    v = Lx / Ly
    out = []
    for l, m in modes:
        om = (np.pi**2 / (Lx * Ly)) * np.sqrt(D / (rho * H)) * (l**2 + v * m**2)
        out.append(om / (2 * np.pi))
    return out


#Calculation of the distribution matrix M
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


def build_model_cache(Lx_value, Ly_value, alpha_g_value):
    """Build all modal data needed by the audio callback for one parameter set."""
    modes = get_modes(int(params["x"]))
    freqs = modes_to_freqs(
        modes, Lx_value, Ly_value, params["D"], params["rho"], params["H"]
    )
    freqs_np = np.asarray(freqs, dtype=np.float64)
    alphas = np.exp(alpha_g_value + params["alpha_r"] * freqs_np)
    z_coeff = np.exp(-alphas / fs) * np.exp(1j * 2 * np.pi * freqs_np / fs)
    z_coeff = np.ascontiguousarray(z_coeff, dtype=np.complex128)
    coupling = distribution_matrix(freqs, params["eta"], params["lamb"])
    return {
        "modes": modes,
        "freqs": freqs,
        "Z": z_coeff,
        "M": coupling,
        "mode_setting": int(params["x"]),
    }


def refresh_step_cache(step_index):
    """Rebuild one step outside the real-time callback (normally from the GUI thread)."""
    step = sequencer_steps[step_index]
    step["cache"] = build_model_cache(step["Lx"], step["Ly"], step["alpha_g"])


def refresh_all_step_caches():
    for i in range(N_STEPS):
        refresh_step_cache(i)


#Defining the excitation signal for the case of simulating impact
def excitation_signal(pos, modes, x=0.3, y=0.7, N_ex=2, A=1.0):
    """
    Kurzer Anregungsimpuls (sin^2-Huellkurve), Dauer N_ex Samples.
    pos    : Sample-Index array, NICHT durch fs geteilt
    modes  : list of (l, m) mode indices, one per filter -- used to weight
             each filter's excitation by its own mode shape at (x, y)
    N_ex   : Pulsdauer in Samples (Default 192 = 4 ms bei fs=48000)

    Returns a (n_modes, frames) array: one excitation trace per mode,
    scaled by that mode's spatial sensitivity to a strike at (x, y).
    """
    pos = np.asarray(pos, dtype=np.float64)
    n = len(modes)
    frames = pos.shape[0]

    if params.get("excitation") == 0:
        u = np.zeros_like(pos)
        within = pos < N_ex
        scale = 2.0 / N_ex if N_ex > 0 else 1.0
        u[within] = A * scale * np.sin(np.pi * (pos[within]) / N_ex) ** 2
    else:
        u = (pos % int(fs) == 0).astype(np.float64)

    u_ex = np.empty((n, frames), dtype=np.float64)
    for k, (l, m) in enumerate(modes):
        weight = np.sin(l * np.pi * x) * np.sin(m * np.pi * y)
        u_ex[k, :] = weight * u

    return np.ascontiguousarray(u_ex, dtype=np.float64)



@njit(cache=True, fastmath=True)
def process_block(states, Z, M, u, tau, max_state_mag, s_out, T_last):
    """
    Verarbeitet einen kompletten Audio-Block (frames Samples) fuer n Filter.

    states   : (n,) complex128, in-place aktualisiert
    Z        : (n,) complex128, Resonator-Koeffizienten
    M        : (n, n) float64, Kopplungsmatrix
    u        : (n, frames) float64, PER-MODE Anregungssignal fuer diesen Block
    tau      : float64, Schwellwert
    max_state_mag : float64, Clamp-Grenze
    s_out    : (frames,) float64, wird befuellt (Output-Signal)
    T_last   : (n,) float64, wird mit T_i(n) des letzten Samples befuellt
               (fuer GUI-Inspektion, entspricht `last_T` im Original)
    """
    n = states.shape[0]
    frames = u.shape[1]

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

        for k in range(n):
            acc = 0.0
            for j in range(n):
                acc += M[k, j] * rectified[j]
            T[k] = acc if acc > 0.0 else 0.0

        # Zustandsupdate (Fallunterscheidung z(n)==0 vs. sonst, mit
        # Epsilon-Schwelle statt exakter Gleichheit -- wie im Original).
        # Jeder Modus bekommt jetzt SEIN EIGENES Anregungssignal u[k, i],
        # gewichtet nach seiner Modenform an der Anschlagposition.
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

            # Clamp gegen die positive Feedback-Divergenz
            nr = new_z.real
            ni = new_z.imag
            mag = np.sqrt(nr * nr + ni * ni)
            if mag > max_state_mag:
                comp = max_state_mag * np.tanh(mag / max_state_mag)
                new_z = new_z * (comp / mag)

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
    params["changed"]= True

def update_canvas():
    # Canvas center
    cx = canvas_size / 2
    cy = canvas_size / 2

    # Rectangle size based on Lx and Ly
    width  = 100 * params['Lx']
    height = 100 * params['Ly']

    # Compute top-left and bottom-right so the center stays fixed
    x1 = cx - width  / 2
    y1 = cy - height / 2
    x2 = cx + width  / 2
    y2 = cy + height / 2

    # Update rectangle
    canvas_id.coords(moving_rect_2, x1, y1, x2, y2)

    



# initial modes / frequencies
_manual_cache = build_model_cache(params["Lx"], params["Ly"], params["alpha_g"])
filter_modes = _manual_cache["modes"]
filter_freqs = _manual_cache["freqs"]
Z = _manual_cache["Z"]
M = _manual_cache["M"]

states = np.zeros(len(filter_freqs), dtype=np.complex128)
last_params = params.copy()
last_T = np.zeros(len(filter_freqs), dtype=np.float64)

# Precompute all step models so a sequencer boundary only swaps array references.
refresh_all_step_caches()

MAX_STATE_MAG = 10.0

# Einmaliger Warmup-Aufruf: kompiliert process_block() JIT, BEVOR der
# Audio-Callback zum ersten Mal laeuft (sonst wuerde der allererste Audio-
# Block durch die Kompilierzeit von ~0.5-2s blockieren -> garantierter
# Dropout beim Start).
_dummy_states = states.copy()
_dummy_u = np.zeros((len(filter_freqs), 4), dtype=np.float64)
_dummy_out = np.zeros(4, dtype=np.float64)
_dummy_T = np.zeros(len(filter_freqs), dtype=np.float64)
process_block(_dummy_states, Z, M, _dummy_u, float(params["tau"]), MAX_STATE_MAG, _dummy_out, _dummy_T)
print("Numba-Kernel kompiliert.")


def _apply_model_cache(cache):
    """Swap the active modal model. State is preserved when the mode count is unchanged."""
    global filter_modes, filter_freqs, Z, M, states, last_T

    filter_modes = cache["modes"]
    filter_freqs = cache["freqs"]
    Z = cache["Z"]
    M = cache["M"]

    n_new = len(filter_freqs)
    if len(states) != n_new:
        # A change of the global mode-count changes the state-vector dimension.
        states = np.zeros(n_new, dtype=np.complex128)
    last_T = np.zeros(n_new, dtype=np.float64)


def _rebuild_manual_model():
    cache = build_model_cache(params["Lx"], params["Ly"], params["alpha_g"])
    _apply_model_cache(cache)


def _samples_per_step():
    bpm = max(float(sequencer["bpm"]), 1.0)
    return max(1, int(round(fs * 60.0 / bpm / 4.0)))  # 16th-note steps


def _apply_sequencer_step(step_index, sample_index):
    """Called only at an exact step boundary in the audio callback."""
    step = sequencer_steps[step_index]
    cache = step.get("cache")

    # Safety fallback if the cache is stale (e.g. mode count changed just now).
    if cache is None or cache.get("mode_setting") != int(params["x"]):
        cache = build_model_cache(step["Lx"], step["Ly"], step["alpha_g"])
        step["cache"] = cache

    _apply_model_cache(cache)

    # Keep the shared parameter dictionary in sync so the plate drawing and
    # existing controls can inspect the currently sounding step.
    params["Lx"] = float(step["Lx"])
    params["Ly"] = float(step["Ly"])
    params["alpha_g"] = float(step["alpha_g"])
    params["changed"] = False

    if step["active"]:
        params["impact_start"] = int(sample_index)
        print(f"Seq step {step_index + 1}: X @ sample {sample_index}")
    else:
        # No new strike. Existing modal energy is allowed to ring out.
        params["impact_start"] = None

    sequencer["current_step"] = step_index


def _advance_sequencer(sample_index):
    next_index = (int(sequencer["step_index"]) + 1) % N_STEPS
    sequencer["step_index"] = next_index
    _apply_sequencer_step(next_index, sample_index)
    sequencer["next_step_sample"] = int(sample_index) + _samples_per_step()


def _render_segment(abs_start, length):
    """Render a contiguous segment for the currently active modal model."""
    if length <= 0:
        return np.zeros(0, dtype=np.float64)

    n = len(states)
    pos = int(abs_start) + np.arange(length)

    if params.get("impact_start") is None:
        u = np.zeros((n, length), dtype=np.float64)
    else:
        u = excitation_signal(
            pos - params["impact_start"],
            filter_modes,
            params["x_e"],
            params["y_e"],
            params["N_ex"],
        )
    u = np.ascontiguousarray(u, dtype=np.float64)

    s_segment = np.empty(length, dtype=np.float64)
    process_block(
        states,
        Z,
        M,
        u,
        float(params["tau"]),
        MAX_STATE_MAG,
        s_segment,
        last_T,
    )
    return s_segment / max(np.sqrt(n), 1.0)


def callback(outdata, frames, time, status):
    if status:
        print(status)

    global last_params

    block_start = int(callback.pos)
    block_end = block_start + int(frames)

    # Manual parameter edits are rebuilt at the start of the next audio block.
    # During sequencer playback, Lx/Ly/alpha_g come from the active step instead.
    if params["changed"] and not sequencer["running"]:
        last_params = params.copy()
        _rebuild_manual_model()
        params["changed"] = False
    elif params["changed"] and sequencer["running"]:
        params["changed"] = False

    out = np.zeros(frames, dtype=np.float64)
    cursor = block_start
    out_offset = 0

    if sequencer["running"] and sequencer["step_index"] < 0:
        # Start step 1 exactly at the first sample of this callback. This also
        # avoids missing the strike if Play was pressed while another callback
        # was already in progress.
        _advance_sequencer(cursor)

    while cursor < block_end:
        if sequencer["running"]:
            next_boundary = int(sequencer["next_step_sample"])

            # Catch up safely after an underrun or a tempo/control change.
            if next_boundary <= cursor:
                _advance_sequencer(cursor)
                continue

            seg_end = min(block_end, next_boundary)
        else:
            seg_end = block_end

        seg_len = int(seg_end - cursor)
        out[out_offset:out_offset + seg_len] = _render_segment(cursor, seg_len)
        cursor = seg_end
        out_offset += seg_len

        if sequencer["running"] and cursor == int(sequencer["next_step_sample"]):
            _advance_sequencer(cursor)

    callback.pos = block_end

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, -1.0, 1.0)
    outdata[:] = out.reshape(-1, 1)


callback.pos = 0
params["impact_start"] = None

stream = sd.OutputStream(channels=1, samplerate=int(fs), callback=callback)
sd.sleep(1)  # Give the stream time to start
stream.start()


for f in filter_freqs:
    print(f"Filter frequency: {f:.2f} Hz")

root.title("Resonator Plate + 8-Step Sequencer (Numba)")


# -----------------------------------------------------------------------------
# GUI helpers
# -----------------------------------------------------------------------------
def _set_manual_param(name, value):
    if name == "x":
        params[name] = int(float(value))
        # Step caches depend on the global mode count. Rebuild them outside the
        # audio callback. The active state vector is resized at the next model swap.
        refresh_all_step_caches()
    else:
        params[name] = float(value)
    params["changed"] = True


def _set_bpm(value):
    sequencer["bpm"] = float(value)
    bpm_value_label.config(text=f"{sequencer['bpm']:.1f} BPM")


def _sequencer_play():
    sequencer["step_index"] = -1
    sequencer["current_step"] = -1
    sequencer["next_step_sample"] = int(callback.pos)
    sequencer["running"] = True


def _sequencer_stop():
    sequencer["running"] = False
    sequencer["step_index"] = -1
    sequencer["current_step"] = -1


def _toggle_step(index):
    sequencer_steps[index]["active"] = not sequencer_steps[index]["active"]
    _update_step_buttons()


def _update_step_buttons():
    current = int(sequencer["current_step"])
    for i, btn in enumerate(step_buttons):
        symbol = "X" if sequencer_steps[i]["active"] else "O"
        if sequencer["running"] and i == current:
            btn.config(text=f"▶ {symbol}")
        else:
            btn.config(text=symbol)


class Knob(Canvas):
    """Small Tkinter rotary control. Drag vertically or use the mouse wheel."""

    def __init__(self, master, label, minimum, maximum, value,
                 on_change, on_commit=None, resolution=0.001, size=58):
        super().__init__(master, width=size, height=size + 28,
                         highlightthickness=0, bd=0)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.value = float(value)
        self.on_change = on_change
        self.on_commit = on_commit
        self.resolution = float(resolution)
        self.size = size
        self.drag_y = None
        self.drag_value = None
        self.label = label

        self.bind("<Button-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<MouseWheel>", self._wheel)
        self._draw()

    def _quantize(self, value):
        value = min(max(value, self.minimum), self.maximum)
        if self.resolution > 0:
            value = round(value / self.resolution) * self.resolution
        return min(max(value, self.minimum), self.maximum)

    def _set(self, value, commit=False):
        value = self._quantize(value)
        if value == self.value and not commit:
            return
        self.value = value
        self.on_change(self.value)
        self._draw()
        if commit and self.on_commit is not None:
            self.on_commit(self.value)

    def _press(self, event):
        self.drag_y = event.y
        self.drag_value = self.value

    def _drag(self, event):
        if self.drag_y is None:
            return
        # 130 px of vertical mouse movement sweeps the complete knob range.
        delta = (self.drag_y - event.y) / 130.0
        self._set(self.drag_value + delta * (self.maximum - self.minimum))

    def _release(self, event):
        self.drag_y = None
        self.drag_value = None
        if self.on_commit is not None:
            self.on_commit(self.value)

    def _wheel(self, event):
        direction = 1.0 if event.delta > 0 else -1.0
        step = self.resolution if self.resolution > 0 else (self.maximum - self.minimum) / 100.0
        self._set(self.value + direction * step, commit=True)

    def _draw(self):
        self.delete("all")
        s = self.size
        cx = s / 2
        cy = s / 2 + 2
        r = s * 0.37

        self.create_text(cx, 7, text=self.label, anchor="n")
        self.create_oval(cx-r, cy-r+7, cx+r, cy+r+7, width=2)

        norm = (self.value - self.minimum) / (self.maximum - self.minimum)
        # 225 degrees (min) -> -45 degrees (max), leaving a gap at the bottom.
        angle = math.radians(225.0 - 270.0 * norm)
        x2 = cx + (r - 7) * math.cos(angle)
        y2 = cy + 7 - (r - 7) * math.sin(angle)
        self.create_line(cx, cy+7, x2, y2, width=3)
        self.create_text(cx, s + 18, text=f"{self.value:.3f}")


# -----------------------------------------------------------------------------
# Sequencer GUI
# -----------------------------------------------------------------------------
seq_outer = ttk.LabelFrame(root, text="8-Step Sequencer — X = strike, O = rest", padding=8)
seq_outer.pack(fill="x", padx=10, pady=(8, 4))

transport = ttk.Frame(seq_outer)
transport.grid(row=0, column=0, columnspan=N_STEPS, sticky="w", pady=(0, 8))
ttk.Button(transport, text="Play", command=_sequencer_play).pack(side="left", padx=(0, 4))
ttk.Button(transport, text="Stop", command=_sequencer_stop).pack(side="left", padx=(0, 10))
ttk.Label(transport, text="Tempo (16th-note steps)").pack(side="left")
bpm_scale = ttk.Scale(transport, from_=40.0, to=240.0, orient="horizontal",
                      command=_set_bpm, length=180)
bpm_scale.pack(side="left", padx=6)
bpm_value_label = ttk.Label(transport, text=f"{sequencer['bpm']:.1f} BPM")
bpm_value_label.pack(side="left")
bpm_scale.set(sequencer["bpm"])

step_buttons = []
step_headers = []
step_knobs = []

for i in range(N_STEPS):
    step_frame = ttk.LabelFrame(seq_outer, text=f"Step {i + 1}", padding=4)
    step_frame.grid(row=1, column=i, padx=3, sticky="n")

    btn = ttk.Button(step_frame, text="X" if sequencer_steps[i]["active"] else "O",
                     width=6, command=lambda idx=i: _toggle_step(idx))
    btn.pack(pady=(0, 4))
    step_buttons.append(btn)

    knobs = []
    for key, label, lo, hi, res in (
        ("Lx", "Lx", 0.001, 2.0, 0.001),
        ("Ly", "Ly", 0.001, 2.0, 0.001),
        ("alpha_g", "αg", 0.001, 5.0, 0.001),
    ):
        def change_value(value, idx=i, param_name=key):
            sequencer_steps[idx][param_name] = float(value)

        def commit_value(value, idx=i):
            refresh_step_cache(idx)

        knob = Knob(
            step_frame, label, lo, hi, sequencer_steps[i][key],
            on_change=change_value,
            on_commit=commit_value,
            resolution=res,
        )
        knob.pack()
        knobs.append(knob)
    step_knobs.append(knobs)


# -----------------------------------------------------------------------------
# Existing manual controls / excitation position
# -----------------------------------------------------------------------------
manual_frame = ttk.LabelFrame(root, text="Manual controls (used when sequencer is stopped)", padding=8)
manual_frame.pack(fill="x", padx=10, pady=4)

manual_grid = ttk.Frame(manual_frame)
manual_grid.pack(fill="x")

lx_slider = ttk.Scale(manual_grid, from_=0.001, to=2, orient="horizontal",
                      command=lambda v: _set_manual_param("Lx", v), length=180)
ttk.Label(manual_grid, text="Lx").grid(row=0, column=0, sticky="w")
lx_slider.grid(row=0, column=1, padx=(4, 14))
lx_slider.set(params["Lx"])

ly_slider = ttk.Scale(manual_grid, from_=0.001, to=2, orient="horizontal",
                      command=lambda v: _set_manual_param("Ly", v), length=180)
ttk.Label(manual_grid, text="Ly").grid(row=0, column=2, sticky="w")
ly_slider.grid(row=0, column=3, padx=(4, 14))
ly_slider.set(params["Ly"])

alpha_slider = ttk.Scale(manual_grid, from_=0.001, to=5, orient="horizontal",
                         command=lambda v: _set_manual_param("alpha_g", v), length=180)
ttk.Label(manual_grid, text="Alpha_g").grid(row=0, column=4, sticky="w")
alpha_slider.grid(row=0, column=5, padx=(4, 14))
alpha_slider.set(params["alpha_g"])

number_slider = Scale(manual_grid, from_=1, to=18, orient="horizontal", resolution=1,
                      command=lambda v: _set_manual_param("x", v), length=180)
ttk.Label(manual_grid, text="Mode index limit").grid(row=1, column=0, sticky="w")
number_slider.grid(row=1, column=1, padx=(4, 14))
number_slider.set(params["x"])

excitation_slider = Scale(manual_grid, from_=2, to=192, orient="horizontal", resolution=1,
                          command=lambda v: params.__setitem__("N_ex", int(float(v))), length=180)
ttk.Label(manual_grid, text="Excitation Length").grid(row=1, column=2, sticky="w")
excitation_slider.grid(row=1, column=3, padx=(4, 14))
excitation_slider.set(params["N_ex"])

strike_btn = ttk.Button(manual_grid, text="Strike", command=_strike)
strike_btn.grid(row=1, column=4, padx=4)
excitation_btn = ttk.Button(
    manual_grid,
    text="Toggle Excitation",
    command=lambda: params.__setitem__("excitation", 1 - params["excitation"]),
)
excitation_btn.grid(row=1, column=5, padx=4)


def _draw_point():
    canvas_id.delete('point')
    px = int(params['x_e'] * (canvas_size - 1))
    py = int((1 - params['y_e']) * (canvas_size - 1))
    r = 6
    canvas_id.create_oval(px-r, py-r, px+r, py+r, fill='red', tags=('point',))


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


canvas_id.bind('<Button-1>', _on_press)
canvas_id.bind('<B1-Motion>', _on_move)
canvas_id.bind('<ButtonRelease-1>', _on_release)


def _gui_tick():
    # Tkinter must only be touched from the GUI thread, not the audio callback.
    update_canvas()
    _update_step_buttons()
    root.after(30, _gui_tick)


_gui_tick()
root.mainloop()