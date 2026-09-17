from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .audio_engine import AudioEngine


class SynthUI:
    """Phase-1 UI. Geometry is still rectangular; backend internals stay hidden."""

    CANVAS_SIZE = 320
    DEBOUNCE_MS = 60

    def __init__(self, root: tk.Tk, engine: AudioEngine) -> None:
        self.root = root
        self.engine = engine
        self.root.title("Plate Resonator Synth - Phase 1")
        self._pending_update_id: str | None = None
        self._queued_changes: dict[str, object] = {}
        self._drag_target = tk.StringVar(value="strike")
        self._build_layout()
        self._draw_plate_and_points()
        self._poll_status()

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)
        left = ttk.Frame(container)
        right = ttk.Frame(container)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right.grid(row=0, column=1, sticky="ns")
        container.columnconfigure(0, weight=1)

        ttk.Label(left, text="Plate / spatial controls", font=("", 11, "bold")).pack(anchor="w")
        self.canvas = tk.Canvas(
            left,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            bg="white",
            highlightthickness=1,
            highlightbackground="#888",
        )
        self.canvas.pack(pady=6)
        self.canvas.bind("<Button-1>", self._on_canvas_drag)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)

        target = ttk.Frame(left)
        target.pack(anchor="w")
        ttk.Label(target, text="Drag:").pack(side="left")
        ttk.Radiobutton(target, text="Strike", value="strike", variable=self._drag_target).pack(side="left", padx=6)
        ttk.Radiobutton(target, text="Pickup", value="pickup", variable=self._drag_target).pack(side="left")

        buttons = ttk.Frame(left)
        buttons.pack(anchor="w", pady=(8, 0))
        ttk.Button(buttons, text="Strike", command=self.engine.strike).pack(side="left")
        ttk.Button(buttons, text="Clear", command=self.engine.clear).pack(side="left", padx=6)

        self.pickup_enabled = tk.BooleanVar(value=self.engine.params.pickup_enabled)
        ttk.Checkbutton(
            left,
            text="Spatial pickup (off = v12 equal-weight readout)",
            variable=self.pickup_enabled,
            command=lambda: self._queue_update(pickup_enabled=bool(self.pickup_enabled.get())),
        ).pack(anchor="w", pady=(8, 0))

        ttk.Label(right, text="Geometry / tuning", font=("", 11, "bold")).pack(anchor="w")
        self._add_slider(right, "Lx [m]", 0.05, 2.0, "length_x_m")
        self._add_slider(right, "Ly [m]", 0.05, 2.0, "length_y_m")
        self._add_int_slider(right, "Number of modes", 1, 128, "n_modes")
        self._add_slider(right, "Frequency scale", 0.25, 4.0, "frequency_scale")

        ttk.Separator(right).pack(fill="x", pady=8)
        ttk.Label(right, text="Material / damping", font=("", 11, "bold")).pack(anchor="w")
        self._add_slider(right, "D [N m]", 100.0, 40_000.0, "flexural_rigidity")
        self._add_slider(right, "rho [kg/m3]", 500.0, 12_000.0, "density")
        self._add_slider(right, "H [m]", 0.001, 0.05, "thickness_m")
        self._add_slider(right, "alpha_g", 0.001, 5.0, "alpha_g")
        self._add_slider(right, "alpha_r", 0.0, 2e-4, "alpha_r")

        ttk.Separator(right).pack(fill="x", pady=8)
        ttk.Label(right, text="Excitation / nonlinearity", font=("", 11, "bold")).pack(anchor="w")
        self._add_int_slider(right, "Excitation length [samples]", 2, 512, "excitation_length_samples")
        self._add_slider(right, "tau", 0.0, 2.0, "tau")
        self._add_slider(right, "eta", 0.0, 0.10, "eta")
        self._add_slider(right, "lambda", 0.0, 0.10, "lamb")

        self.status_label = ttk.Label(right, text="")
        self.status_label.pack(anchor="w", pady=(8, 0))

    def _add_slider(self, parent, label: str, start: float, end: float, field: str) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(4, 0))
        value = float(getattr(self.engine.params, field))
        variable = tk.DoubleVar(value=value)
        value_label = ttk.Label(frame, text=f"{value:.4g}")
        ttk.Label(frame, text=label).pack(anchor="w")

        def changed(raw: str) -> None:
            v = float(raw)
            value_label.configure(text=f"{v:.4g}")
            self._queue_update(**{field: v})

        ttk.Scale(frame, from_=start, to=end, variable=variable, command=changed).pack(fill="x")
        value_label.pack(anchor="e")

    def _add_int_slider(self, parent, label: str, start: int, end: int, field: str) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(4, 0))
        ttk.Label(frame, text=label).pack(anchor="w")
        variable = tk.IntVar(value=int(getattr(self.engine.params, field)))
        tk.Scale(
            frame,
            from_=start,
            to=end,
            orient="horizontal",
            resolution=1,
            variable=variable,
            command=lambda raw: self._queue_update(**{field: int(float(raw))}),
        ).pack(fill="x")

    def _queue_update(self, **changes: object) -> None:
        self._queued_changes.update(changes)
        if self._pending_update_id is not None:
            self.root.after_cancel(self._pending_update_id)
        self._pending_update_id = self.root.after(self.DEBOUNCE_MS, self._flush_updates)

    def _flush_updates(self) -> None:
        self._pending_update_id = None
        changes = self._queued_changes
        self._queued_changes = {}
        if changes:
            self.engine.update_parameters(**changes)
            self._draw_plate_and_points()

    def _plate_bounds(self) -> tuple[float, float, float, float]:
        p = self.engine.params
        margin = 28.0
        available = self.CANVAS_SIZE - 2.0 * margin
        aspect = p.length_x_m / p.length_y_m
        if aspect >= 1.0:
            width, height = available, available / aspect
        else:
            width, height = available * aspect, available
        cx = cy = self.CANVAS_SIZE * 0.5
        return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2

    def _normalized_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        x0, y0, x1, y1 = self._plate_bounds()
        return x0 + x * (x1 - x0), y1 - y * (y1 - y0)

    def _canvas_to_normalized(self, px: float, py: float) -> tuple[float, float]:
        x0, y0, x1, y1 = self._plate_bounds()
        x = (px - x0) / max(x1 - x0, 1e-12)
        y = (y1 - py) / max(y1 - y0, 1e-12)
        return min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)

    def _draw_plate_and_points(self) -> None:
        self.canvas.delete("all")
        x0, y0, x1, y1 = self._plate_bounds()
        self.canvas.create_rectangle(x0, y0, x1, y1, fill="#d8e7f3", outline="#222", width=2)
        p = self.engine.params
        for label, x, y, fill in (
            ("S", p.strike_x, p.strike_y, "#d62728"),
            ("P", p.pickup_x, p.pickup_y, "#2ca02c"),
        ):
            px, py = self._normalized_to_canvas(x, y)
            r = 6
            self.canvas.create_oval(px-r, py-r, px+r, py+r, fill=fill, outline="")
            self.canvas.create_text(px+10, py-10, text=label, anchor="sw")

    def _on_canvas_drag(self, event) -> None:
        x, y = self._canvas_to_normalized(event.x, event.y)
        if self._drag_target.get() == "pickup":
            self._queue_update(pickup_x=x, pickup_y=y)
        else:
            self._queue_update(strike_x=x, strike_y=y)

    def _poll_status(self) -> None:
        freqs = self.engine.current_frequencies_hz
        if len(freqs):
            summary = f"Modes: {len(freqs)}\nf min/max: {freqs.min():.1f} / {freqs.max():.1f} Hz"
        else:
            summary = "Modes: 0"
        if self.engine.last_status:
            summary += f"\nAudio status: {self.engine.last_status}"
        self.status_label.configure(text=summary)
        self.root.after(250, self._poll_status)


def run_ui() -> None:
    root = tk.Tk()
    engine = AudioEngine()
    SynthUI(root, engine)

    def on_close() -> None:
        engine.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    engine.start()
    root.mainloop()
