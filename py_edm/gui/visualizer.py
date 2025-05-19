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
from PyQt6.QtCore import Qt, QTimer, QPointF, QSize, QEvent
from PyQt6.QtGui import QPainter, QPainterPath, QColor, QRadialGradient
from PyQt6.QtWidgets import QWidget, QVBoxLayout

from .generative_art_engine import HexClusterWidget



class VisualizerWidget(QWidget):
    """
    Drop-in replacement that shows the concentric-hex animation
    instead of the old ferrofluid painter.
    """
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setMinimumHeight(200)

        # ── Audio state ────────────────────────────────────────────
        self._samples: Optional[np.ndarray] = None
        self._sr     = 44_100
        self._pos    = 0
        self._step   = 735          # ≈60 FPS @ 44.1 kHz
        self._timer  = QTimer(self)
        self._timer.timeout.connect(self._advance)

        # ➋  Embed the generative-art engine
        self._hex = HexClusterWidget(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._hex)

    # --------------------------------------------------------------
    # Frame-advance: feed amplitude into HexClusterWidget
    # --------------------------------------------------------------
    def _advance(self) -> None:
        if self._samples is None:
            return
        end  = min(self._pos + self._step, len(self._samples))
        wind = self._samples[self._pos:end]
        self._pos = end

        # RMS → logistic → 0-1
        rms = float(np.sqrt(np.mean(wind ** 2)))
        amp = 1.0 / (1.0 + math.exp(-9.0 * (rms - 0.08)))
        self._hex.set_amplitude(amp)

        if self._pos >= len(self._samples):
            self.stop()

    # ------------------------------------------------------------------
    # The original API is still here so the rest of your app compiles.
    # The hex animation runs autonomously, so start/stop become no-ops.
    # ------------------------------------------------------------------
    def start(self, pcm: np.ndarray, sample_rate: int):  # type: ignore[override]
        """Begin audio-reactive mode."""
        self._samples = pcm.astype(np.float32) / 32_767.0
        self._sr      = sample_rate
        self._pos     = 0
        self._step    = max(int(self._sr / 60), 1)
        self._timer.start(int(1_000 / 60))

    def stop(self) -> None:
        self._timer.stop()
        self._samples = None
        # Reset amplitude to base level so animation settles
        self._hex.set_amplitude(0.0)

    def sizeHint(self) -> QSize:                # optional – gives a nice default
        return QSize(400, 400)

    def resizeEvent(self, ev: QEvent) -> None:   # keep it square
        side = min(self.width(), self.height())
        self._hex.setFixedSize(side, side)       # _hex is the HexClusterWidget
        super().resizeEvent(ev)

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


