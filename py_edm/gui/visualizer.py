from __future__ import annotations
"""
VisualizerWidget

Key features
============ --------------------------------------------------------------
• Two concentric *blob layers* (inner + outer)  ────────────┐
  – Each blob is a polar curve  r(θ) = R₀ + A·sin(kθ+φ)     │
  – Inner layer : faster phase, more waves (turbulent)      │
  – Outer layer : slower phase, fewer waves (placid)        ├─⇒   mesmerizing
• Amplitude (RMS) of incoming PCM drives blob inflation     │    molten blobs
• HSV wheel spins for colour-cycling neon glow              │
• Additive blend  +  α-fade overlay  = motion-blur trails   ┘
=============================================================================
Public API remains identical (`start`, `stop`) so you can drop this file
into the larger GUI project without touching other code.
"""

from typing import Optional
import math
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import (
    QPainter,
    QPainterPath,
    QColor,
    QRadialGradient,
)
from PyQt6.QtWidgets import QWidget


class VisualizerWidget(QWidget):
    """Self-contained QWidget for the blob visual."""

    # Optional[] attributes are annotated for MyPy / IDE helpers only
    _samples: Optional[np.ndarray]

    # ──────────────────────────────────────────────────────────────
    # Construction & set-up
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(200)           # leave enough vertical estate

        # Audio state
        self._samples = None                 # int16 → float32 PCM buffer
        self._sr = 44_100
        self._pos = 0                        # current playback head (samples)
        self._step = 735                     # step per frame → ~60 FPS

        # Visual tuning
        self._trail_alpha = 32               # 0-255; lower = longer motion-blur
        self._inner_waves = 9                # sine lobes on the inner blob
        self._outer_waves = 5                # sine lobes on the outer blob
        self._inner_speed = 2.0              # rad/s of phase drift
        self._outer_speed = 0.6
        self._base_inner_ratio = 0.18        # base size ratios (vs min(w,h))
        self._base_outer_ratio = 0.42

        # Animation timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

        # ── pick a “Screen” composition-mode that exists (Qt version safe) ──
        self._screen_mode = None
        try:
            self._screen_mode = QPainter.CompositionMode.Screen                # Qt ≥ 6.5
        except AttributeError:
            try:
                self._screen_mode = QPainter.CompositionMode.CompositionMode_Screen  # strict enum
            except AttributeError:
                self._screen_mode = getattr(QPainter, "CompositionMode_Screen", None)
        if self._screen_mode is None:                                          # ultimate fallback
            self._screen_mode = QPainter.CompositionMode_SourceOver

    # ──────────────────────────────────────────────────────────────
    # Public control surface
    def start(self, pcm: np.ndarray, sample_rate: int) -> None:
        """
        Begin animating to *pcm* (int16 mono).

        Parameters
        ----------
        pcm : np.ndarray[int16]
            Raw signed-16-bit mono samples.
        sample_rate : int
            Samples per second, e.g. 44100.
        """
        self._samples = pcm.astype(np.float32) / 32_767.0      # norm to −1…+1
        self._sr = sample_rate
        self._pos = 0
        self._step = max(int(self._sr / 60), 1)                # aim ≈60 FPS
        self._timer.start(int(1_000 / 60))                     # 60 Hz Qt timer
        self.update()                                          # paint first frame

    def stop(self) -> None:
        """Halt animation and clear the canvas."""
        self._timer.stop()
        self._samples = None
        self.update()

    # ──────────────────────────────────────────────────────────────
    # Frame advance
    def _advance(self) -> None:
        """Move playback head and schedule repaint."""
        if self._samples is None:
            return
        self._pos += self._step
        if self._pos >= len(self._samples):
            self.stop()                                        # auto-stop at end
        else:
            self.update()

    # ──────────────────────────────────────────────────────────────
    # Painting
    def paintEvent(self, _evt):  # noqa: N802 D401
        """
        Qt paint callback – draws one frame of the animation.

        * Motion-blur is achieved by filling the rect with a translucent black.
        * Two QPainterPaths (inner & outer) are constructed in polar coords.
        """
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # ───────────────────────────────
            # 1) Dark-mode motion-blur fade
            # ------------------------------------------------------------------
            fade = QColor(0, 0, 0, self._trail_alpha)          # translucent blk
            painter.fillRect(self.rect(), fade)

            if self._samples is None:                          # idle state
                return

            # ───────────────────────────────
            # 2) Extract current window & amplitude
            # ------------------------------------------------------------------
            slice_end = min(self._pos + self._step, len(self._samples))
            window = self._samples[self._pos:slice_end]
            amp_rms = float(np.sqrt(np.mean(window**2)))       # 0-1
            # “blob inflation” via sigmoid – feels smoother than linear
            amp_growth = 1 / (1 + math.exp(-10 * (amp_rms - 0.1)))

            # ───────────────────────────────
            # 3) Geometry & timing helpers
            # ------------------------------------------------------------------
            w, h = self.width(), self.height()
            size = min(w, h)
            centre = QPointF(w / 2, h / 2)
            t_sec = self._pos / self._sr                       # time cursor [s]

            # Base radii (before amplitude growth)
            base_inner = size * self._base_inner_ratio
            base_outer = size * self._base_outer_ratio

            # Growth factors – outer grows less to keep nice spacing
            inner_r = base_inner * (1.0 + amp_growth * 0.35)
            outer_r = base_outer * (1.0 + amp_growth * 0.25)

            # ───────────────────────────────
            # 4) Build blob paths
            # ------------------------------------------------------------------
            path_inner = self._build_blob_path(
                centre,
                inner_r,
                waves=self._inner_waves,
                phase=t_sec * self._inner_speed,
                mod_depth=inner_r * 0.25,          # modulation amplitude
            )
            path_outer = self._build_blob_path(
                centre,
                outer_r,
                waves=self._outer_waves,
                phase=t_sec * self._outer_speed,
                mod_depth=outer_r * 0.18,
            )

            # ───────────────────────────────
            # 5) Colour gradients
            # ------------------------------------------------------------------
            # Overall hue wheel spins 30°/s
            hue = (t_sec * 30) % 360

            grad_inner = QRadialGradient(centre, inner_r * 1.1)
            grad_inner.setColorAt(0.0, QColor.fromHsvF(hue / 360, 1, 1, 0.95))
            grad_inner.setColorAt(1.0, QColor.fromHsvF(hue / 360, 1, 0.4, 0.0))

            # Outer hue is offset +120° for complementary contrast
            hue_outer = (hue + 120) % 360
            grad_outer = QRadialGradient(centre, outer_r * 1.1)
            grad_outer.setColorAt(0.0, QColor.fromHsvF(hue_outer / 360, 1, 0.9, 0.85))
            grad_outer.setColorAt(1.0, QColor.fromHsvF(hue_outer / 360, 1, 0.3, 0.0))

            # Enable additive blend for tasty neon glow
            painter.setCompositionMode(self._screen_mode)

            # ───────────────────────────────
            # 6) Draw layers (inner above outer for “lava” depth)
            # ------------------------------------------------------------------
            painter.setPen(Qt.PenStyle.NoPen)

            painter.setBrush(grad_outer)
            painter.drawPath(path_outer)

            painter.setBrush(grad_inner)
            painter.drawPath(path_inner)

        finally:
            painter.end()  # guarantee painter state is torn down

    # ──────────────────────────────────────────────────────────────
    # Helper: build a radial “blob” as QPainterPath
    # ------------------------------------------------------------------
    def _build_blob_path(
        self,
        centre: QPointF,
        radius_base: float,
        *,
        waves: int,
        phase: float,
        mod_depth: float,
        points: int = 200,
    ) -> QPainterPath:
        """
        Create a polar-modulated blob path.

        Parameters
        ----------
        centre : QPointF
            Centre of the blob.
        radius_base : float
            Base radius (no modulation).
        waves : int
            Number of sine lobes around 360°.
        phase : float
            Phase offset in *radians* – multiples of π · 2 = full rotation.
        mod_depth : float
            Peak deviation of radius from `radius_base`.
        points : int, default 200
            Angular resolution (more ⇒ smoother blob).
        """
        path = QPainterPath()
        # First point
        angle0 = 0.0
        r0 = radius_base + mod_depth * math.sin(waves * angle0 + phase)
        x0 = centre.x() + r0 * math.cos(angle0)
        y0 = centre.y() + r0 * math.sin(angle0)
        path.moveTo(x0, y0)

        # Remaining points
        for i in range(1, points + 1):
            angle = (i / points) * (2 * math.pi)
            r = radius_base + mod_depth * math.sin(waves * angle + phase)
            x = centre.x() + r * math.cos(angle)
            y = centre.y() + r * math.sin(angle)
            path.lineTo(x, y)

        path.closeSubpath()
        return path
