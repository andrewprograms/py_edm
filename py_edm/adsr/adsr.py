from __future__ import annotations

"""ADSR envelope utilities."""

from dataclasses import dataclass

import numpy as np

from config.config import SAMPLE_RATE


@dataclass
class ADSRSettings:
    """Container for ADSR parameters in milliseconds and sustain level (0‑1)."""

    attack_ms: int = 50
    decay_ms: int = 100
    sustain_level: float = 0.5
    release_ms: int = 300

    # --- Convenience in seconds -------------------------------------------------
    @property
    def attack_s(self) -> float:  # seconds
        return self.attack_ms / 1_000.0

    @property
    def decay_s(self) -> float:
        return self.decay_ms / 1_000.0

    @property
    def release_s(self) -> float:
        return self.release_ms / 1_000.0


class ADSREnvelope:
    """Generate an ADSR envelope curve as a NumPy array in the range 0‑1."""

    def __init__(self, settings: ADSRSettings):
        self.settings = settings

    # -------------------------------------------------------------------------
    def generate(self, sustain_time: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Return ADSR curve long enough for *sustain_time* seconds of sustain."""
        a, d, r = (
            self.settings.attack_s,
            self.settings.decay_s,
            self.settings.release_s,
        )
        s = sustain_time

        # Number of samples per segment
        a_n = int(a * sample_rate)
        d_n = int(d * sample_rate)
        s_n = int(s * sample_rate)
        r_n = int(r * sample_rate)

        # Build segments
        attack_curve = np.linspace(0, 1, a_n, endpoint=False)
        decay_curve = np.linspace(1, self.settings.sustain_level, d_n, endpoint=False)
        sustain_curve = np.full(s_n, self.settings.sustain_level, dtype=np.float32)
        release_curve = np.linspace(
            self.settings.sustain_level, 0, r_n, endpoint=True, dtype=np.float32
        )

        envelope = np.concatenate(
            [attack_curve, decay_curve, sustain_curve, release_curve], dtype=np.float32
        )
        return envelope
