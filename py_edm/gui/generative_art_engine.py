#!/usr/bin/env python3
"""
generative_art_engine.py
Concentric‑clustered hexagon animation — reusable PyQt6 widget (c) 2025
------------------------------------------------------------------------
• Provides **HexClusterWidget**, a drop‑in Qt widget you can embed in any
  PyQt6 layout.
• Everything renders live; no video is ever written.
• `set_amplitude(value: float)` lets a host widget modulate the waves
  (0 ≤ value ≤ 1), so the pattern can pulse with audio.
• Running this file directly (`python generative_art_engine.py`) pops up
  a demo window for quick testing.
"""

from __future__ import annotations

import sys
from math import cos, sin, pi, sqrt

import numpy as np
import matplotlib

# Choose Qt‑backed Matplotlib canvas *before* importing pyplot
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget  # noqa: E402

# ──────────────────────────────────────────────────────────────
# Tunables (same defaults as original script)
# ------------------------------------------------------------------
GRID_RADIUS = 5               # axial grid radius → density of hexes
BASE_SIZE = 0.85              # nominal hexagon “radius”
FREQ = 0.55                   # controls wavelength of the concentric rings
SPEED = 0.15                  # radians per frame → wave speed
FPS = 30                      # target frame‑rate
LINEWIDTH_MINMAX = (0.2, 2.7) # (thin, thick) stroke widths
# ------------------------------------------------------------------

_SQRT3 = sqrt(3)


def _axial_to_xy(q: int, r: int, size: float):
    """Convert axial hex‑grid coords (q, r) → Cartesian (x, y)."""
    x = size * (3 / 2 * q)
    y = size * (_SQRT3 * (r + q / 2))
    return x, y


def _hexagon_vertices(center_xy: tuple[float, float], size: float):
    """Return six (x, y) vertices of a regular hexagon."""
    cx, cy = center_xy
    return [
        (
            cx + size * cos(pi / 3 * i + pi / 6),
            cy + size * sin(pi / 3 * i + pi / 6),
        )
        for i in range(6)
    ]


class HexClusterWidget(QWidget):
    """Matplotlib‑powered widget animating concentric hex‑ring waves."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        grid_radius: int = GRID_RADIUS,
        base_size: float = BASE_SIZE,
        freq: float = FREQ,
        speed: float = SPEED,
        fps: int = FPS,
        linewidth_minmax: tuple[float, float] = LINEWIDTH_MINMAX,
    ) -> None:
        super().__init__(parent)

        self._grid_radius = grid_radius
        self._base_size = base_size
        self._freq = freq
        self._speed = speed
        self._fps = fps
        self._lw_min, self._lw_max = linewidth_minmax
        self._amp = 0.0  # external amplitude (0–1)

        # -------------------- Matplotlib figure / canvas --------------------
        self._fig, self._ax = plt.subplots(figsize=(4, 4))
        self._canvas = FigureCanvas(self._fig)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._ax.set_aspect("equal")
        self._ax.axis("off")

        # -------------------- Build the static hex grid ---------------------
        self._centers: list[tuple[float, float]] = []
        self._patches: list[Polygon] = []
        for q in range(-self._grid_radius, self._grid_radius + 1):
            for r in range(-self._grid_radius, self._grid_radius + 1):
                if abs(q + r) > self._grid_radius:
                    continue  # roughly circular footprint
                cx, cy = _axial_to_xy(q, r, self._base_size * 1.01)
                self._centers.append((cx, cy))

        for (cx, cy) in self._centers:
            poly = Polygon(
                _hexagon_vertices((cx, cy), self._base_size),
                closed=True,
                fill=False,
                edgecolor="k",
                linewidth=self._lw_min,
            )
            self._ax.add_patch(poly)
            self._patches.append(poly)

        xs, ys = zip(*self._centers)
        margin = self._base_size * 2
        self._ax.set_xlim(min(xs) - margin, max(xs) + margin)
        self._ax.set_ylim(min(ys) - margin, max(ys) + margin)

        # ------------------------- Drive animation --------------------------
        # Explicitly disable frame‑data caching to avoid UserWarning
        self._anim = FuncAnimation(
            self._fig,
            self._animate,
            frames=None,              # infinite generator
            blit=True,
            interval=1_000 / self._fps,
            cache_frame_data=False,   # suppress warning & save memory
        )

    # ------------------------------------------------------------------
    def _animate(self, frame: int):
        """Matplotlib callback that updates each frame."""
        t = self._speed * frame
        for (cx, cy), patch in zip(self._centers, self._patches):
            dist = sqrt(cx * cx + cy * cy)
            base = (np.sin(self._freq * dist - t) + 1) / 2  # 0‥1
            scale = base * (0.4 + 0.6 * self._amp)          # blend in audio
            size = self._base_size * (0.25 + 0.85 * scale)
            lw = self._lw_min + (self._lw_max - self._lw_min) * scale
            patch.set_xy(_hexagon_vertices((cx, cy), size))
            patch.set_linewidth(lw)
        return self._patches  # type: ignore[return-value]

    # ------------------------------------------------------------------
    def canvas(self) -> FigureCanvas:  # pragma: no cover
        """Expose the underlying *FigureCanvas* (rarely needed)."""
        return self._canvas

    # --------------------------------------------------------------
    # Public audio‑reactive hook
    # --------------------------------------------------------------
    def set_amplitude(self, value: float) -> None:
        """Normalised RMS (0–1) → visual pulse."""
        self._amp = max(0.0, min(float(value), 1.0))


# ──────────────────────────────────────────────────────────────
# Demo harness (only executed when run as a script)
# ------------------------------------------------------------------

def _demo() -> None:  # pragma: no cover
    app = QApplication(sys.argv)
    widget = HexClusterWidget()
    widget.resize(600, 600)
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    _demo()
