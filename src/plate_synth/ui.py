from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .audio_engine import AudioEngine


class SynthUI:
    """Small Tk interface for the Phase-1 refactored rectangle synth."""

    CANVAS_SIZE = 320
    DEBOUNCE_MS = 60

    def __init__(self, root: tk.Tk, engine: AudioEngine) -> None:
        self.root = root
        self.engine = engine
        self.root.title("Plate Resonator Synth - Phase 1 Refactor")

        self._pending_update_id: str | None = None
        self._drag_target = tk.StringVar(value="strike")

        self._build_layout()
        self._draw_plate_and_points()
        self._poll_status()

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)

        left = ttk.Frame(container)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        right = ttk.Frame(container)
        right.grid(row=0, column=1, sticky="ns")

        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        ttk.Label(
            left,
            text="Plate / spatial controls",
            font=("", 11, "bold"),
        ).pack(anchor="w")

        self.canvas = tk.Canvas(
            left,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            bg="white",
            highlightthickness=1,
            highlightbackground="#888",
        )
        self.canvas.pack(pady=(6, 6))

        target_row = ttk.Frame(left)
        target_row.pack(anchor="w", pady=(0, 8))
        ttk.Label(target_row, text="Drag target:").pack(side="left")
        ttk.Radiobutton(
            target_row,
            text="Strike",
            value="strike",
            variable=self._drag_target,
        ).pack(side="left", padx=(8, 0))
        ttk.Radiobutton(
            target_row,
            text="Pickup",
            value="pickup",
            variable=self._drag_target,
        ).pack(side="left", padx=(8, 0))

        self.canvas.bind("<Button-1>", self._on_canvas_drag)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)

        button_row = ttk.Frame(left)
        button_row.pack(anchor="w")
        ttk.Button(
            button_row,
            text="Strike",
            command=self.engine.strike,
        ).pack(side="left")
        ttk.Button(
            button_row,
            text="Clear",
            command=self.engine.clear,
        ).pack(side="left", padx=(6, 0))

        self.pickup_enabled = tk.BooleanVar(
            value=self.engine.params.pickup_enabled
        )
        ttk.Checkbutton(
            left,
            text="Use spatial pickup (off = v12 equal-weight output)",
            variable=self.pickup_enabled,
            command=lambda: self._queue_update(
                pickup_enabled=bool(self.pickup_enabled.get())
            ),
        ).pack(anchor="w", pady=(8, 0))

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=(0, 8))
        ttk.Label(
            right,
            text="Model controls",
            font=("", 11, "bold"),
        ).pack(anchor="w")

        self.length_x = self._add_slider(
            right,
            "Lx [m]",
            0.05,
            2.0,
            self.engine.params.length_x_m,
            "length_x_m",
        )
        self.length_y = self._add_slider(
            right,
            "Ly [m]",
            0.05,
            2.0,
            self.engine.params.length_y_m,
            "length_y_m",
        )
        self.alpha_g = self._add_slider(
            right,
            "alpha_g",
            0.001,
            5.0,
            self.engine.params.alpha_g,
            "alpha_g",
        )
        self.frequency_scale = self._add_slider(
            right,
            "Frequency scale",
            0.25,
            4.0,
            self.engine.params.frequency_scale,
            "frequency_scale",
        )

        mode_frame = ttk.Frame(right)
        mode_frame.pack(fill="x", pady=(5, 0))
        ttk.Label(mode_frame, text="Legacy mode order").pack(anchor="w")
        self.mode_order = tk.IntVar(
            value=self.engine.params.legacy_mode_order
        )
        mode_scale = tk.Scale(
            mode_frame,
            from_=2,
            to=18,
            orient="horizontal",
            resolution=1,
            variable=self.mode_order,
            command=lambda value: self._queue_update(
                legacy_mode_order=int(float(value))
            ),
        )
        mode_scale.pack(fill="x")

        self.excitation_length = self._add_int_slider(
            right,
            "Excitation length [samples]",
            2,
            512,
            self.engine.params.excitation_length_samples,
            "excitation_length_samples",
        )

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(
            right,
            text="Nonlinear coupling",
            font=("", 11, "bold"),
        ).pack(anchor="w")

        self.tau = self._add_slider(
            right,
            "tau",
            0.0,
            2.0,
            self.engine.params.tau,
            "tau",
        )
        self.eta = self._add_slider(
            right,
            "eta",
            0.0,
            0.10,
            self.engine.params.eta,
            "eta",
        )
        self.lamb = self._add_slider(
            right,
            "lambda",
            0.0,
            0.10,
            self.engine.params.lamb,
            "lamb",
        )

        self.status_label = ttk.Label(right, text="")
        self.status_label.pack(anchor="w", pady=(8, 0))

    def _add_slider(
        self,
        parent: ttk.Frame,
        label: str,
        start: float,
        end: float,
        value: float,
        parameter_name: str,
    ) -> tk.DoubleVar:
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(5, 0))
        variable = tk.DoubleVar(value=value)
        value_label = ttk.Label(frame, text=f"{value:.4g}")

        ttk.Label(frame, text=label).pack(anchor="w")
        scale = ttk.Scale(
            frame,
            from_=start,
            to=end,
            variable=variable,
            command=lambda raw: (
                value_label.configure(text=f"{float(raw):.4g}"),
                self._queue_update(**{parameter_name: float(raw)}),
            ),
        )
        scale.pack(fill="x")
        value_label.pack(anchor="e")
        return variable

    def _add_int_slider(
        self,
        parent: ttk.Frame,
        label: str,
        start: int,
        end: int,
        value: int,
        parameter_name: str,
    ) -> tk.IntVar:
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(5, 0))
        ttk.Label(frame, text=label).pack(anchor="w")
        variable = tk.IntVar(value=value)
        scale = tk.Scale(
            frame,
            from_=start,
            to=end,
            orient="horizontal",
            resolution=1,
            variable=variable,
            command=lambda raw: self._queue_update(
                **{parameter_name: int(float(raw))}
            ),
        )
        scale.pack(fill="x")
        return variable

    def _queue_update(self, **changes: object) -> None:
        # Coalesce fast slider motion. Model building still happens on the GUI
        # thread, never in the audio callback.
        if not hasattr(self, "_queued_changes"):
            self._queued_changes: dict[str, object] = {}
        self._queued_changes.update(changes)

        if self._pending_update_id is not None:
            self.root.after_cancel(self._pending_update_id)

        self._pending_update_id = self.root.after(
            self.DEBOUNCE_MS,
            self._flush_updates,
        )

    def _flush_updates(self) -> None:
        self._pending_update_id = None
        changes = getattr(self, "_queued_changes", {})
        self._queued_changes = {}
        if changes:
            self.engine.update_parameters(**changes)
            self._draw_plate_and_points()

    def _plate_bounds(self) -> tuple[float, float, float, float]:
        params = self.engine.params
        margin = 28.0
        available = self.CANVAS_SIZE - 2.0 * margin

        aspect = params.length_x_m / params.length_y_m
        if aspect >= 1.0:
            width = available
            height = available / aspect
        else:
            height = available
            width = available * aspect

        cx = self.CANVAS_SIZE * 0.5
        cy = self.CANVAS_SIZE * 0.5
        return (
            cx - width * 0.5,
            cy - height * 0.5,
            cx + width * 0.5,
            cy + height * 0.5,
        )

    def _normalized_to_canvas(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        x0, y0, x1, y1 = self._plate_bounds()
        px = x0 + x * (x1 - x0)
        py = y1 - y * (y1 - y0)
        return px, py

    def _canvas_to_normalized(
        self,
        px: float,
        py: float,
    ) -> tuple[float, float]:
        x0, y0, x1, y1 = self._plate_bounds()
        x = (px - x0) / max(x1 - x0, 1e-12)
        y = (y1 - py) / max(y1 - y0, 1e-12)
        return (
            min(max(x, 0.0), 1.0),
            min(max(y, 0.0), 1.0),
        )

    def _draw_plate_and_points(self) -> None:
        self.canvas.delete("all")
        x0, y0, x1, y1 = self._plate_bounds()
        self.canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            fill="#d8e7f3",
            outline="#222",
            width=2,
        )

        p = self.engine.params
        sx, sy = self._normalized_to_canvas(p.strike_x, p.strike_y)
        px, py = self._normalized_to_canvas(p.pickup_x, p.pickup_y)

        radius = 6
        self.canvas.create_oval(
            sx - radius,
            sy - radius,
            sx + radius,
            sy + radius,
            fill="#d62728",
            outline="",
        )
        self.canvas.create_text(
            sx + 10,
            sy - 10,
            text="S",
            anchor="sw",
        )

        self.canvas.create_oval(
            px - radius,
            py - radius,
            px + radius,
            py + radius,
            fill="#2ca02c",
            outline="",
        )
        self.canvas.create_text(
            px + 10,
            py - 10,
            text="P",
            anchor="sw",
        )

    def _on_canvas_drag(self, event) -> None:
        x, y = self._canvas_to_normalized(event.x, event.y)
        if self._drag_target.get() == "pickup":
            self._queue_update(pickup_x=x, pickup_y=y)
        else:
            self._queue_update(strike_x=x, strike_y=y)

        # Draw immediately for responsive interaction, even before the
        # debounced control-rate update reaches the engine.
        params = self.engine.params
        if self._drag_target.get() == "pickup":
            preview = params.updated(pickup_x=x, pickup_y=y)
        else:
            preview = params.updated(strike_x=x, strike_y=y)

        # Temporarily draw using the preview values without mutating engine
        # state; the actual update is published after the debounce interval.
        self.canvas.delete("preview")
        cx, cy = self._normalized_to_canvas(
            preview.pickup_x
            if self._drag_target.get() == "pickup"
            else preview.strike_x,
            preview.pickup_y
            if self._drag_target.get() == "pickup"
            else preview.strike_y,
        )
        radius = 8
        self.canvas.create_oval(
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
            outline="#000",
            width=2,
            tags="preview",
        )

    def _poll_status(self) -> None:
        freqs = self.engine.current_frequencies_hz
        status = self.engine.last_status
        summary = (
            f"Modes: {len(freqs)}\n"
            f"f min/max: {freqs.min():.1f} / {freqs.max():.1f} Hz"
            if len(freqs)
            else "Modes: 0"
        )
        if status:
            summary += f"\nAudio status: {status}"
        self.status_label.configure(text=summary)
        self.root.after(250, self._poll_status)


def run_ui() -> None:
    root = tk.Tk()
    engine = AudioEngine()
    app = SynthUI(root, engine)

    def on_close() -> None:
        engine.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    engine.start()
    root.mainloop()
