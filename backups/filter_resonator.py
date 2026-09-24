import numpy as np
import sounddevice as sd   # optional, for saving output
import os
from tkinter import *
from tkinter import ttk



fs = 48000.0
f0 = 200
alpha = 10  # damping / decay rate

params = {
    "alpha": 10,
    "base_freq": 220.0,
    "spacing": 1.4,
    "N": 1
}

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

# initial frequencies
filter_freqs = list(freqs(params["N"], params["base_freq"], params["spacing"]))

# resonator coefficients and states
Z = [np.exp(-params["alpha"]/fs) * np.exp(1j * 2*np.pi*f/fs) for f in filter_freqs]

states = [0j for _ in filter_freqs]

last_params = params.copy()

def callback(outdata, frames, time, status):
    if status:
        print(status)

    global filter_freqs, Z, states, last_params

    # Check if ANY parameter changed
    if params != last_params:
        last_params = params.copy()

        # Recompute frequencies
        filter_freqs = list(freqs(params["N"], params["base_freq"], params["spacing"]))

        # Recompute Z coefficients
        Z = [np.exp(-params["alpha"]/fs) * np.exp(1j * 2*np.pi*f/fs)
             for f in filter_freqs]

        # Reset states (or keep them if you prefer)
        states = [0j for _ in filter_freqs]

    out = np.zeros(frames)

    for i in range(frames):
        u = 1.0 if int(callback.pos) % int(fs) == 0 else 0.0

        s = 0.0
        for k in range(len(states)):
            states[k] = Z[k] * states[k] + u
            s += np.imag(states[k])

        out[i] = s
        callback.pos += 1

    outdata[:] = (out / len(states)).reshape(-1, 1)

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
alpha_slider = ttk.Scale(root, from_=0.1, to=50, orient="horizontal",
                         command=lambda v: params.__setitem__("alpha", float(v)))

ttk.Label(root, text="Alpha").pack()
alpha_slider.pack(fill="x")

# Base frequency slider
base_slider = ttk.Scale(root, from_=20, to=2000, orient="horizontal",
                        command=lambda v: params.__setitem__("base_freq", float(v)))

ttk.Label(root, text="Base Frequency").pack()
base_slider.pack(fill="x")

# Spacing slider
spacing_slider = ttk.Scale(root, from_=1.01, to=2.0, orient="horizontal",
                           command=lambda v: params.__setitem__("spacing", float(v)))

ttk.Label(root, text="Spacing").pack()
spacing_slider.pack(fill="x")

# Number of filters slider
number_slider = Scale(root, from_= 1, to=50, orient="horizontal", resolution=1,
                          command=lambda v: params.__setitem__("N", int(float(v))))

ttk.Label(root, text="Number of Filters").pack()
number_slider.pack(fill="x")



ttk.Label(frm, text="Filter GUI").grid(column=0, row=0)
root.mainloop()