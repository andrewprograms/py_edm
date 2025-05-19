"""
Very small-footprint reverb for Laser Synth.

A single-channel Schroeder–Moore style reverb using four comb filters in
parallel followed by two all-pass filters.  All maths is done in float32
(-1 .. 1).  Integer input is converted transparently.

This is **not** a studio-grade algorithm – it is deliberately lightweight so
the synth stays snappy, but it sounds “room-like” enough for SFX work.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np

from config.config import SAMPLE_RATE

# ---------------------------------------------------------------------------

def _to_float(sig: np.ndarray) -> np.ndarray:
    if sig.dtype.kind in {"i", "u"}:
        sig = sig.astype(np.float32) / np.iinfo(sig.dtype).max
    else:
        sig = sig.astype(np.float32)
    return sig


def _to_int16(sig: np.ndarray) -> np.ndarray:
    return np.int16(np.clip(sig, -1.0, 1.0) * 32_767)

# ---------------------------------------------------------------------------

@dataclass
class ReverbSettings:
    """Tweakable parameters."""
    decay: float = 2.2          # seconds until reverb tail reaches -60 dB
    mix:   float = 0.35         # 0 = dry, 1 = 100 % wet
    predelay_ms: int = 20       # small predelay before early reflections

    def __post_init__(self):
        if not 0.0 <= self.mix <= 1.0:
            raise ValueError("mix must be between 0 and 1")
        if self.decay <= 0.0:
            raise ValueError("decay must be > 0")

# ---------------------------------------------------------------------------

class _Comb:
    """Simple feedback comb."""
    def __init__(self, delay_s: float, fb_gain: float, sample_rate: int):
        self.buf = np.zeros(int(delay_s * sample_rate), dtype=np.float32)
        self.idx = 0
        self.fb_gain = fb_gain

    def process(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        for n, s in enumerate(x):
            y = self.buf[self.idx]
            out[n] = y
            self.buf[self.idx] = s + y * self.fb_gain
            self.idx = (self.idx + 1) % self.buf.size
        return out

class _AllPass:
    """1st-order all-pass for diffusion."""
    def __init__(self, delay_s: float, gain: float, sr: int):
        self.buf = np.zeros(int(delay_s * sr), dtype=np.float32)
        self.idx = 0
        self.gain = gain

    def process(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        for n, s in enumerate(x):
            buf_out = self.buf[self.idx]
            y = -self.gain * s + buf_out
            self.buf[self.idx] = s + buf_out * self.gain
            out[n] = y
            self.idx = (self.idx + 1) % self.buf.size
        return out

# ---------------------------------------------------------------------------

class ReverbProcessor:
    """Mono reverb with wet/dry mix."""
    def __init__(
        self,
        settings: ReverbSettings | None = None,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.settings = settings or ReverbSettings()
        sr = sample_rate

        # Gains chosen so RT60 ≈ settings.decay
        fb = 10 ** (-3.0 * self.settings.predelay_ms / (self.settings.decay * 1000))

        # Parallel combs (prime-length delays help decorrelate)
        self.combs = [
            _Comb(0.0297, fb, sr),
            _Comb(0.0371, fb, sr),
            _Comb(0.0411, fb, sr),
            _Comb(0.0437, fb, sr),
        ]

        # Serial all-passes
        self.allpasses = [
            _AllPass(0.005, 0.7, sr),
            _AllPass(0.0017, 0.7, sr),
        ]

        # Predelay line
        self._predelay_buf = np.zeros(int(self.settings.predelay_ms * sr / 1000), dtype=np.float32)
        self._predelay_idx = 0

    # ------------------------------------------------------------------
    def process(self, samples: np.ndarray) -> np.ndarray:
        x = _to_float(samples)

        # ── predelay ───────────────────────────────────────────────────
        predelayed = np.empty_like(x)
        for n, s in enumerate(x):
            predelayed[n] = self._predelay_buf[self._predelay_idx]
            self._predelay_buf[self._predelay_idx] = s
            self._predelay_idx = (self._predelay_idx + 1) % self._predelay_buf.size

        # ── combs in parallel ─────────────────────────────────────────
        y = sum(c.process(predelayed) for c in self.combs) / len(self.combs)

        # ── all-passes in series ──────────────────────────────────────
        for ap in self.allpasses:
            y = ap.process(y)

        # ── wet/dry ───────────────────────────────────────────────────
        out = (1.0 - self.settings.mix) * x + self.settings.mix * y

        # Back to original dtype
        if samples.dtype == np.float32:
            return out.astype(np.float32)
        return _to_int16(out)

# Convenience one-shot ------------------------------------------------------

def reverb(
    samples: np.ndarray,
    *,
    decay: float = 2.2,
    mix: float = 0.35,
    predelay_ms: int = 20,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    settings = ReverbSettings(decay=decay, mix=mix, predelay_ms=predelay_ms)
    return ReverbProcessor(settings, sample_rate).process(samples)

__all__ = ["ReverbSettings", "ReverbProcessor", "reverb"]
