
"""Distortion processors for Laser Synth.

This module provides a small collection of **time–domain distortion effects**
that can be applied to any *mono* NumPy‑array signal.  It is completely
self‑contained so it can be imported by the existing ``LaserSynth`` back‑end or
used stand‑alone in other contexts (e.g. the forthcoming distortion GUI
popup).

Key features
------------
* **Multiple algorithms** – hard‑clip, soft‑clip, tanh, fold‑back, and a simple
  bit‑crusher.
* **Drive / amount control** – scales the input into the non‑linear section.
* **Wet/dry mix** – blend the distorted signal back with the original.
* **Per‑sample envelope** – optionally modulates the *drive* with any 0‑to‑1
  array (e.g. an ADSR curve) so the intensity can change over the note’s
  lifetime.

All processing is done on *float32* arrays in the range ``‑1 .. 1`` to avoid
integer‑overflow artefacts.  Integer input is converted automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _to_float(sig: np.ndarray) -> np.ndarray:
    """Return *sig* as float32 in the range −1 .. 1."""
    if sig.dtype.kind in {"i", "u"}:  # int/uint
        max_val = np.iinfo(sig.dtype).max
        sig = sig.astype(np.float32) / max_val
    else:
        sig = sig.astype(np.float32)
    return sig


def _to_int16(sig: np.ndarray) -> np.ndarray:
    """Convert a −1..1 float array back to int16."""
    return np.int16(np.clip(sig, -1.0, 1.0) * 32_767)


# ---------------------------------------------------------------------------
# Distortion algorithms (all assume −1 .. 1 float input)
# ---------------------------------------------------------------------------

def _hard_clip(x: np.ndarray) -> np.ndarray:  # noqa: D401
    """Simple hard limiter at ±1."""
    return np.clip(x, -1.0, 1.0)


def _soft_clip(x: np.ndarray) -> np.ndarray:
    """Soft‑clipper using a cubic transfer curve (≈ 3rd‑order approximation)."""
    out = np.empty_like(x)
    mask = np.abs(x) < 1.0
    out[mask] = x[mask] - (x[mask] ** 3) / 3.0
    out[~mask] = np.sign(x[~mask]) * 2.0 / 3.0
    return out


def _tanh(x: np.ndarray) -> np.ndarray:
    """Classic *tanh* saturation."""
    return np.tanh(x)


def _foldback(x: np.ndarray, threshold: float = 0.75) -> np.ndarray:
    """Fold‑back distortion ala analog wave‑folders."""
    x = x.copy()
    above = np.abs(x) > threshold
    x[above] = (
        np.abs(np.abs(x[above]) - threshold) * np.sign(x[above]) - threshold
    )
    return x


def _bitcrush(x: np.ndarray, bits: int = 8) -> np.ndarray:
    """Very simple bit‑crusher by quantising to *bits* bits."""
    levels = 2 ** bits
    return np.round(x * levels) / levels


# Map user‑facing algorithm names to callables ------------------------------
_AlgoName = Literal[
    "hard_clip",
    "soft_clip",
    "tanh",
    "foldback",
    "bitcrush",
]

_ALGOS: dict[_AlgoName, Callable[[np.ndarray], np.ndarray]] = {
    "hard_clip": _hard_clip,
    "soft_clip": _soft_clip,
    "tanh": _tanh,
    "foldback": _foldback,
    "bitcrush": _bitcrush,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class DistortionSettings:
    """User‑tweakable distortion parameters."""

    algorithm: _AlgoName = "tanh"
    drive: float = 2.5  # pre‑gain before non‑linearity (1.0 = no extra drive)
    mix: float = 1.0  # wet (1.0) / dry (0.0) blend
    envelope: np.ndarray | None = None  # 0‑1 curve the same length as the input

    def __post_init__(self):
        if not 0.0 <= self.mix <= 1.0:
            raise ValueError("mix must be between 0 and 1")
        if self.drive <= 0.0:
            raise ValueError("drive must be > 0")


class DistortionProcessor:
    """Apply non‑linear distortion to an audio buffer."""

    def __init__(self, settings: DistortionSettings | None = None):
        self.settings = settings or DistortionSettings()

    # ------------------------------------------------------------------
    def process(self, samples: np.ndarray) -> np.ndarray:
        """Return distorted *samples* (same dtype as input).

        *samples* can be ``float32`` in −1..1 *or* any integer PCM format.
        """
        # Convert to float for maths
        x = _to_float(samples)

        # Envelope
        if self.settings.envelope is not None:
            if len(self.settings.envelope) < len(x):
                raise ValueError("Envelope shorter than input signal")
            drive_curve = self.settings.drive * self.settings.envelope[: len(x)]
        else:
            drive_curve = self.settings.drive

        # Pre‑gain / drive
        x_d = x * drive_curve

        # Non‑linear transfer
        algo = _ALGOS[self.settings.algorithm]
        y = algo(x_d)

        # Mix wet/dry
        out = (1.0 - self.settings.mix) * x + self.settings.mix * y

        # Return in original dtype
        if samples.dtype == np.float32:
            return out.astype(np.float32)
        return _to_int16(out)


# Convenience functional interface -----------------------------------------

def distort(
    samples: np.ndarray,
    *,
    algorithm: _AlgoName = "tanh",
    drive: float = 2.5,
    mix: float = 1.0,
    envelope: np.ndarray | None = None,
) -> np.ndarray:  # noqa: D401
    """One‑shot helper – see :class:`DistortionProcessor`."""
    settings = DistortionSettings(
        algorithm=algorithm, drive=drive, mix=mix, envelope=envelope
    )
    return DistortionProcessor(settings).process(samples)


__all__ = [
    "DistortionSettings",
    "DistortionProcessor",
    "distort",
]
