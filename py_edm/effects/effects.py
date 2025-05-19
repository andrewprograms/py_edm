from __future__ import annotations

"""Sound‑generation back‑end ("effects")."""

from pathlib import Path
import wave
from typing import Literal

import numpy as np
import simpleaudio as sa

from adsr.adsr import ADSRSettings, ADSREnvelope
from config.config import SAMPLE_RATE

from effects.distortion import DistortionProcessor, DistortionSettings


DEFAULT_DURATION = 1.2  # seconds before release kicks in
BASE_FREQ_START = 2_000  # Hz (start of laser sweep)
BASE_FREQ_END = 200  # Hz (end of laser sweep)

# -----------------------------------------------------------------------------
class LaserSynth:
    """Generate and play laser‑style sound effects with ADSR envelope."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate

        # leave None to disable distortion
        self._dist_proc: DistortionProcessor | None = None

    # ---------------------------------------------------------------------
    def _generate_saw(self, duration: float) -> np.ndarray:
        """Generate a sweeping sawtooth wave / laser chirp."""
        t = np.linspace(0, duration, int(duration * self.sample_rate), endpoint=False)
        freqs = np.logspace(
            np.log10(BASE_FREQ_START), np.log10(BASE_FREQ_END), t.size, base=10
        )
        phase = 2 * np.pi * np.cumsum(freqs) / self.sample_rate
        return 2 * (phase / (2 * np.pi) % 1) - 1  # range −1..1

    def _generate_sine(self, duration: float) -> np.ndarray:
        """Sine variant of the sweep (softer laser)."""
        t = np.linspace(0, duration, int(duration * self.sample_rate), endpoint=False)
        freqs = np.logspace(
            np.log10(BASE_FREQ_START), np.log10(BASE_FREQ_END), t.size, base=10
        )
        phase = 2 * np.pi * np.cumsum(freqs) / self.sample_rate
        return np.sin(phase)

    # ---------------------------------------------------------------------
    def generate_wave(self, duration: float, waveform: Literal["saw", "sine"] = "saw") -> np.ndarray:
        """Return raw floating‑point waveform for *duration* seconds."""
        if waveform == "sine":
            return self._generate_sine(duration).astype(np.float32)
        return self._generate_saw(duration).astype(np.float32)

    # --------------------------------------------------------------
    def set_distortion(self, settings: DistortionSettings | None) -> None:
        """
        Enable or update distortion.  
        Pass *None* to disable.
        """
        self._dist_proc = (
            None if settings is None else DistortionProcessor(settings)
        )

    # ---------------------------------------------------------------------
    def synthesize(self, adsr: ADSRSettings, volume: float = 1.0, *, waveform: Literal["saw", "sine"] = "saw") -> np.ndarray:  # noqa: D401
        """Return 16‑bit integer samples ready for playback or saving."""
        sustain_time = max(DEFAULT_DURATION - adsr.attack_s - adsr.decay_s, 0.05)
        base_wave = self.generate_wave(DEFAULT_DURATION + adsr.release_s, waveform=waveform)
        envelope = ADSREnvelope(adsr).generate(sustain_time)
        # Ensure equal lengths
        length = min(len(base_wave), len(envelope))

        # 1) apply volume envelope first
        wave = base_wave[:length] * envelope[:length]

        # 2) optional distortion
        if self._dist_proc is not None:
            # if caller didn’t supply an envelope to the distortion,
            # just reuse the synth’s ADSR so drive follows the note
            if self._dist_proc.settings.envelope is None:
                self._dist_proc.settings.envelope = envelope[:length]
            wave = self._dist_proc.process(wave)


        # Apply volume (0‑1) before conversion
        wave = np.clip(wave * volume, -1.0, 1.0)
        return np.int16(wave * 32_767)

    # ---------------------------------------------------------------------
    def play(
        self, adsr: ADSRSettings, volume: float = 1.0, *, waveform: str = "saw"
    ) -> sa.PlayObject:
        samples = self.synthesize(adsr, volume, waveform=waveform)
        return sa.play_buffer(samples, 1, 2, self.sample_rate)   # non-blocking


    # ---------------------------------------------------------------------
    def save(self, path: str | Path, adsr: ADSRSettings, volume: float = 1.0, *, waveform: Literal["saw", "sine"] = "saw") -> Path:
        """Save the synthesised sound to *path* (.wav). Returns resolved path."""
        path = Path(path).with_suffix(".wav").expanduser().resolve()
        samples = self.synthesize(adsr, volume, waveform=waveform)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16‑bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(samples.tobytes())
        return path
