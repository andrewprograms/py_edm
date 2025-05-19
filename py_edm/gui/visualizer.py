from __future__ import annotations
"""
VisualizerWidget - Ferrofluid Edition
===========================================================
Some PyQt6 wheels expose *integer* constants for composition modes instead
of proper `QPainter.CompositionMode` enum members.  Passing a bare `int`
into `painter.setCompositionMode()` triggers::

    TypeError: ... unexpected type 'int'

Fix
---
* Detect raw-int constants and **cast** them to the enum if the enum type
  exists (`QPainter.CompositionMode(<int>)`).
* If we cannot resolve a valid enum member, **skip** the call entirely -
  regular `SourceOver` blending still looks fine.

Public API (`start`, `stop`) remains unchanged.
"""

from typing import Optional, Sequence
import math
import random
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QPainter, QPainterPath, QColor, QRadialGradient
from PyQt6.QtWidgets import QWidget


class _RingSpec:
    __slots__: Sequence[str] = (
        "base_ratio",
        "m",
        "n1",
        "n2",
        "n3",
        "phase_speed",
        "hue_offset",
    )

    def __init__(
        self,
        *,
        base_ratio: float,
        m: int,
        n1: float,
        n2: float,
        n3: float,
        phase_speed: float,
        hue_offset: float,
    ) -> None:
        self.base_ratio = base_ratio
        self.m = m
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.phase_speed = phase_speed
        self.hue_offset = hue_offset


class VisualizerWidget(QWidget):
    """Concentric ferrofluid‑style audio visualiser."""

    _samples: Optional[np.ndarray]

    # ──────────────────────────────────────────────────────────────
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(200)

        # Audio state
        self._samples: Optional[np.ndarray] = None
        self._sr = 44_100
        self._pos = 0
        self._step = 735  # ~60 FPS for 44.1 kHz

        # Visual persistence (lower alpha → longer trails)
        self._trail_alpha = 24

        # Four rings - inner → outer
        self._rings: list[_RingSpec] = [
            _RingSpec(base_ratio=0.12, m=9, n1=0.30, n2=0.20, n3=0.20, phase_speed=2.4, hue_offset=0),
            _RingSpec(base_ratio=0.24, m=7, n1=0.35, n2=0.25, n3=0.25, phase_speed=1.6, hue_offset=60),
            _RingSpec(base_ratio=0.36, m=5, n1=0.45, n2=0.30, n3=0.30, phase_speed=1.0, hue_offset=120),
            _RingSpec(base_ratio=0.48, m=4, n1=0.60, n2=0.35, n3=0.35, phase_speed=0.6, hue_offset=180),
        ]

        # Animation timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

        # Detect additive blend mode constant safely
        self._blend_mode = self._detect_blend_mode()

    # ------------------------------------------------------------------
    @staticmethod
    def _detect_blend_mode():
        """Return a `QPainter.CompositionMode` for 'Screen' if possible."""
        candidates = [
            # Preferred additive
            ("CompositionMode_Screen", None),
            ("CompositionMode", "Screen"),
            ("Screen", None),
            # Fallback - normal paint
            ("CompositionMode_SourceOver", None),
            ("CompositionMode", "SourceOver"),
            ("SourceOver", None),
        ]
        for top, child in candidates:
            parent = QPainter
            if child is not None:
                parent = getattr(QPainter, top, None)
                attr_name = child
            else:
                attr_name = top
            if parent is None or not hasattr(parent, attr_name):
                continue
            val = getattr(parent, attr_name)
            # If val is a bare int, wrap it in the enum if we can
            if isinstance(val, int) and hasattr(QPainter, "CompositionMode"):
                try:
                    val = QPainter.CompositionMode(val)  # type: ignore[arg-type]
                except ValueError:
                    continue  # invalid enum value
            return val
        return None  # give up - use default blending

    # ──────────────────────────────────────────────────────────────
    # Public control
    # ------------------------------------------------------------------
    def start(self, pcm: np.ndarray, sample_rate: int) -> None:
        self._samples = pcm.astype(np.float32) / 32_767.0
        self._sr = sample_rate
        self._pos = 0
        self._step = max(int(self._sr / 60), 1)
        self._timer.start(int(1_000 / 60))
        self.update()

    def stop(self) -> None:
        self._timer.stop()
        self._samples = None
        self.update()

    # ------------------------------------------------------------------
    def _advance(self) -> None:
        if self._samples is None:
            return
        self._pos += self._step
        if self._pos >= len(self._samples):
            self.stop()
        else:
            self.update()

    # ──────────────────────────────────────────────────────────────
    def paintEvent(self, _evt):  # noqa: N802 D401
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), QColor(0, 0, 0, self._trail_alpha))
            if self._samples is None:
                return

            # Audio envelope
            slice_end = min(self._pos + self._step, len(self._samples))
            window = self._samples[self._pos : slice_end]
            amp_rms = float(np.sqrt(np.mean(window ** 2)))
            amp_growth = 1.0 / (1.0 + math.exp(-9.0 * (amp_rms - 0.08)))

            # Geometry helpers
            w, h = self.width(), self.height()
            size = min(w, h)
            centre = QPointF(w / 2, h / 2)
            t_sec = self._pos / self._sr
            base_hue = (t_sec * 24) % 360

            # Switch to additive if we resolved a valid enum
            if self._blend_mode is not None:
                painter.setCompositionMode(self._blend_mode)  # type: ignore[arg-type]
            painter.setPen(Qt.PenStyle.NoPen)

            # Draw rings outer→inner
            for idx, ring in reversed(list(enumerate(self._rings))):
                base_r = size * ring.base_ratio
                radius = base_r * (1.0 + amp_growth * (0.45 - 0.08 * idx))
                n1_dyn = max(
                    0.05,
                    ring.n1 / (1.0 + amp_growth * 2.2) + (random.random() - 0.5) * 0.02,
                )
                path = self._build_superformula_path(
                    centre,
                    radius_base=radius,
                    m=ring.m,
                    n1=n1_dyn,
                    n2=ring.n2,
                    n3=ring.n3,
                    phase=t_sec * ring.phase_speed,
                )
                hue = (base_hue + ring.hue_offset) % 360
                grad = QRadialGradient(centre, radius * 1.10)
                grad.setColorAt(0.0, QColor.fromHsvF(hue / 360, 1.0, 1.0, 0.90))
                grad.setColorAt(1.0, QColor.fromHsvF(hue / 360, 1.0, 0.30, 0.0))
                painter.setBrush(grad)
                painter.drawPath(path)
        finally:
            painter.end()

    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _superformula(
        theta: float,
        *,
        m: int,
        n1: float,
        n2: float,
        n3: float,
    ) -> float:
        epsilon = 1e-9
        a = abs(math.cos(m * theta / 4.0)) ** n2
        b = abs(math.sin(m * theta / 4.0)) ** n3
        return (a + b) ** (-1.0 / max(n1, epsilon))


    def _build_superformula_path(
        self,
        centre: QPointF,
        *,
        radius_base: float,
        m: int,
        n1: float,
        n2: float,
        n3: float,
        phase: float,
        points: int = 400,
    ) -> QPainterPath:
        """Construct a closed QPainterPath for a superformula curve."""
        path = QPainterPath()
        for i in range(points + 1):
            theta = (i / points) * (2 * math.pi)
            theta_p = theta + phase
            r = radius_base * self._superformula(theta_p, m=m, n1=n1, n2=n2, n3=n3)
            x = centre.x() + r * math.cos(theta_p)
            y = centre.y() + r * math.sin(theta_p)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        path.closeSubpath()
        return path