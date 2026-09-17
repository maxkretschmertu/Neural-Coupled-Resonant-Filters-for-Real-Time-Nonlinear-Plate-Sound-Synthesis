from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import numpy as np

from .audio_engine import AudioEngine
from .material_coordinates import material_to_physical, physical_to_material


class SynthUI:
    CANVAS_SIZE = 420
    DEBOUNCE_MS = 50

    def __init__(self, root: tk.Tk, engine: AudioEngine) -> None:
        self.root = root
        self.engine = engine
        self.root.title("Neural Modal Plate Synth - Geometry Phase")
        self._queued_changes: dict[str, object] = {}
        self._pending_update_id: str | None = None
        self._drag_target = tk.StringVar(value="strike")
        self._build()
        self._draw_geometry()
        self._poll()

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        left = ttk.Frame(outer)
        right = ttk.Frame(outer)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right.grid(row=0, column=1, sticky="ns")

        ttk.Label(
            left,
            text="Plate geometry / material positions",
            font=("", 11, "bold"),
        ).pack(anchor="w")
        self.canvas = tk.Canvas(
            left,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            bg="white",
        )
        self.canvas.pack(pady=6)
        self.canvas.bind("<Button-1>", self._on_drag)
        self.canvas.bind("<B1-Motion>", self._on_drag)

        ttk.Label(
            left,
            text=(
                "Filled shape = current geometry preview. "
                "Dashed outline = active audio geometry when they differ."
            ),
            wraplength=self.CANVAS_SIZE,
        ).pack(anchor="w", pady=(0, 6))

        row = ttk.Frame(left)
        row.pack(anchor="w")
        ttk.Radiobutton(
            row,
            text="Strike",
            value="strike",
            variable=self._drag_target,
        ).pack(side="left")
        ttk.Radiobutton(
            row,
            text="Pickup",
            value="pickup",
            variable=self._drag_target,
        ).pack(side="left", padx=8)
        ttk.Button(
            row,
            text="Strike",
            command=self.engine.strike,
        ).pack(side="left", padx=(16, 4))
        ttk.Button(
            row,
            text="Clear",
            command=self.engine.clear,
        ).pack(side="left")

        self.pickup_enabled = tk.BooleanVar(
            value=self.engine.params.pickup_enabled
        )
        ttk.Checkbutton(
            left,
            text="Spatial pickup",
            variable=self.pickup_enabled,
            command=lambda: self._queue(
                pickup_enabled=self.pickup_enabled.get()
            ),
        ).pack(anchor="w", pady=6)

        self._section(right, "GEOMETRY")
        self._slider(
            right,
            "Morph   Rectangle → Ellipse → Triangle",
            0.0,
            1.0,
            self.engine.params.morph,
            "morph",
        )
        self._slider(
            right,
            "Shape Mod   compact → slender",
            0.0,
            1.0,
            self.engine.params.shape_mod,
            "shape_mod",
        )
        self._slider(
            right,
            "Size √area [m]",
            0.1,
            2.0,
            self.engine.params.size_m,
            "size_m",
        )
        self._int_slider(
            right,
            "Number of modes",
            4,
            64,
            self.engine.params.n_modes,
            "n_modes",
        )

        ttk.Label(
            right,
            text=(
                "Boundary condition: "
                f"{self.engine.params.boundary_condition}"
            ),
        ).pack(anchor="w", pady=(4, 0))

        self._section(right, "MATERIAL / DAMPING")
        self._slider(
            right,
            "Flexural rigidity D",
            1000.0,
            40000.0,
            self.engine.params.flexural_rigidity,
            "flexural_rigidity",
        )
        self._slider(
            right,
            "Density ρ",
            500.0,
            10000.0,
            self.engine.params.density,
            "density",
        )
        self._slider(
            right,
            "Thickness H [m]",
            0.001,
            0.05,
            self.engine.params.thickness_m,
            "thickness_m",
        )
        self._slider(
            right,
            "alpha_g",
            0.001,
            5.0,
            self.engine.params.alpha_g,
            "alpha_g",
        )
        self._slider(
            right,
            "alpha_r",
            0.0,
            2e-4,
            self.engine.params.alpha_r,
            "alpha_r",
        )

        self._section(right, "TUNING / NONLINEAR")
        self._slider(
            right,
            "Frequency scale",
            0.25,
            4.0,
            self.engine.params.frequency_scale,
            "frequency_scale",
        )
        self._slider(
            right,
            "tau",
            0.0,
            2.0,
            self.engine.params.tau,
            "tau",
        )
        self._slider(
            right,
            "eta",
            0.0,
            0.1,
            self.engine.params.eta,
            "eta",
        )
        self._slider(
            right,
            "lambda",
            0.0,
            0.1,
            self.engine.params.lamb,
            "lamb",
        )

        self.status = ttk.Label(right, text="", wraplength=340)
        self.status.pack(anchor="w", pady=(10, 0))

    @staticmethod
    def _section(parent, text: str) -> None:
        ttk.Separator(parent).pack(fill="x", pady=(8, 4))
        ttk.Label(
            parent,
            text=text,
            font=("", 10, "bold"),
        ).pack(anchor="w")

    def _slider(self, parent, label, low, high, value, name) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        ttk.Label(frame, text=label).pack(anchor="w")
        var = tk.DoubleVar(value=value)
        value_label = ttk.Label(frame, text=f"{value:.4g}")
        ttk.Scale(
            frame,
            from_=low,
            to=high,
            variable=var,
            command=lambda raw: (
                value_label.configure(text=f"{float(raw):.4g}"),
                self._queue(**{name: float(raw)}),
            ),
        ).pack(fill="x")
        value_label.pack(anchor="e")

    def _int_slider(self, parent, label, low, high, value, name) -> None:
        ttk.Label(parent, text=label).pack(anchor="w")
        scale = tk.Scale(
            parent,
            from_=low,
            to=high,
            orient="horizontal",
            resolution=1,
            command=lambda raw: self._queue(
                **{name: int(float(raw))}
            ),
        )
        scale.set(value)
        scale.pack(fill="x")

    def _queue(self, **changes: object) -> None:
        self._queued_changes.update(changes)
        if self._pending_update_id is not None:
            self.root.after_cancel(self._pending_update_id)
        self._pending_update_id = self.root.after(
            self.DEBOUNCE_MS,
            self._flush,
        )

    def _flush(self) -> None:
        self._pending_update_id = None
        changes, self._queued_changes = self._queued_changes, {}
        if changes:
            self.engine.update_parameters(**changes)
            self._draw_geometry()

    def _bounds(self) -> float:
        preview = self.engine.current_geometry.boundary_xy
        active = self.engine.active_audio_geometry.boundary_xy
        max_abs = max(
            float(np.max(np.abs(preview))),
            float(np.max(np.abs(active))),
            1e-9,
        )
        margin = 30.0
        return (self.CANVAS_SIZE / 2 - margin) / max_abs

    def _physical_to_canvas(self, x, y):
        scale = self._bounds()
        c = self.CANVAS_SIZE / 2
        return c + np.asarray(x) * scale, c - np.asarray(y) * scale

    def _canvas_to_physical(self, px, py):
        scale = self._bounds()
        c = self.CANVAS_SIZE / 2
        return (px - c) / scale, (c - py) / scale

    @staticmethod
    def _same_geometry(a, b) -> bool:
        return (
            abs(float(a.morph) - float(b.morph)) < 1e-12
            and abs(float(a.shape_mod) - float(b.shape_mod)) < 1e-12
        )

    def _draw_boundary(self, geometry, *, fill, outline, dash=None) -> None:
        x, y = self._physical_to_canvas(
            geometry.boundary_xy[:, 0],
            geometry.boundary_xy[:, 1],
        )
        coords = np.column_stack((x, y)).ravel().tolist()
        self.canvas.create_polygon(
            *coords,
            fill=fill,
            outline=outline,
            width=2,
            dash=dash,
        )

    def _draw_geometry(self) -> None:
        self.canvas.delete("all")
        preview = self.engine.current_geometry
        active = self.engine.active_audio_geometry

        if not self._same_geometry(preview, active):
            self._draw_boundary(
                active,
                fill="",
                outline="#777",
                dash=(6, 4),
            )

        self._draw_boundary(
            preview,
            fill="#dbe8f4",
            outline="#222",
        )

        p = self.engine.params
        for u, v, fill, label in (
            (p.strike_u, p.strike_v, "#d62728", "S"),
            (p.pickup_u, p.pickup_v, "#2ca02c", "P"),
        ):
            px0, py0 = material_to_physical(preview, u, v)
            cx, cy = self._physical_to_canvas(float(px0), float(py0))
            r = 6
            self.canvas.create_oval(
                cx-r,
                cy-r,
                cx+r,
                cy+r,
                fill=fill,
                outline="",
            )
            self.canvas.create_text(cx+10, cy-8, text=label)

    def _on_drag(self, event) -> None:
        x, y = self._canvas_to_physical(event.x, event.y)
        u, v = physical_to_material(
            self.engine.current_geometry,
            x,
            y,
        )
        if self._drag_target.get() == "pickup":
            self._queue(pickup_u=float(u), pickup_v=float(v))
        else:
            self._queue(strike_u=float(u), strike_v=float(v))

    def _poll(self) -> None:
        freqs = self.engine.current_frequencies_hz
        preview = self.engine.current_geometry
        active = self.engine.active_audio_geometry

        text = (
            "Preview: "
            f"morph={preview.morph:.3f}, shape={preview.shape_mod:.3f}, "
            f"area={preview.area:.5f}\n"
            "Active audio: "
            f"morph={active.morph:.3f}, shape={active.shape_mod:.3f}\n"
            f"Audio modes: {len(freqs)} | "
            f"f={freqs.min():.1f}…{freqs.max():.1f} Hz\n"
            f"Modal status: {self.engine.model_status}"
        )
        if not self.engine.control_model_valid:
            text += "\nAudio parameter updates are frozen until the preview returns to a supported geometry."
        if self.engine.last_status:
            text += f"\nAudio: {self.engine.last_status}"

        self.status.configure(text=text)
        self.root.after(250, self._poll)


def run_ui() -> None:
    root = tk.Tk()
    engine = AudioEngine()
    SynthUI(root, engine)

    def close() -> None:
        engine.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    engine.start()
    root.mainloop()
