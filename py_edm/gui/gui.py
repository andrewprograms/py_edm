from __future__ import annotations

"""PyQt6 front‑end for the laser synth.
"""

from pathlib import Path
from typing import Optional

import simpleaudio as sa
from PyQt6.QtCore import Qt, QTemporaryFile, QUrl, QTimer
from PyQt6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QDialog,
    QComboBox,
    QDialogButtonBox,
)
from PyQt6.QtMultimedia import QSoundEffect

from adsr.adsr import ADSRSettings
from effects.effects import LaserSynth
from effects.distortion import DistortionSettings, _ALGOS  # _ALGOS gives us the names


import wave
import os
import tempfile


class ADSRSlider(QGroupBox):
    """Reusable slider widget with a read‑out label."""

    def __init__(
        self, label: str, min_val: int, max_val: int, init_val: int, unit: str
    ):
        super().__init__()
        self.setTitle(label)
        self.setLayout(QVBoxLayout())

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(min_val, max_val)
        self.slider.setValue(init_val)
        self.value_label = QLabel()
        self.unit = unit

        self.layout().addWidget(self.slider)
        self.layout().addWidget(self.value_label)
        self._update_label(init_val)
        self.slider.valueChanged.connect(self._update_label)

    # ------------------------------------------------------------------
    def _update_label(self, value: int) -> None:  # noqa: D401
        self.value_label.setText(f"{value} {self.unit}")

    def value(self) -> int:
        return self.slider.value()

class DistortionDialog(QDialog):
    """Small popup that lets the user tweak distortion parameters."""

    def __init__(self, parent=None, current: DistortionSettings | None = None):
        super().__init__(parent)
        self.setWindowTitle("Distortion Settings")
        self.setLayout(QVBoxLayout())

        # Algorithm picker ------------------------------------------------
        self.algo_box = QComboBox()
        self.algo_box.addItems(_ALGOS.keys())
        self.layout().addWidget(QLabel("Algorithm"))
        self.layout().addWidget(self.algo_box)

        # Drive slider ----------------------------------------------------
        self.drive_slider = ADSRSlider("Drive (pre-gain)", 1, 50, 25, "×")
        self.layout().addWidget(self.drive_slider)

        # Mix slider ------------------------------------------------------
        self.mix_slider = ADSRSlider("Wet / Dry", 0, 100, 100, "%")
        self.layout().addWidget(self.mix_slider)

        # OK / Cancel -----------------------------------------------------
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout().addWidget(buttons)

        # Load current settings if passed
        if current:
            self.algo_box.setCurrentText(current.algorithm)
            self.drive_slider.slider.setValue(int(current.drive * 10))
            self.mix_slider.slider.setValue(int(current.mix * 100))

    # ------------------------------------------------------------------
    def settings(self) -> DistortionSettings:
        """Return a DistortionSettings instance from current widget state."""
        return DistortionSettings(
            algorithm=self.algo_box.currentText(),          # type: ignore[arg-type]
            drive=self.drive_slider.value() / 10.0,         # slider = 1-50 → 0.1-5.0
            mix=self.mix_slider.value() / 100.0,
            envelope=None,  # let LaserSynth inject its ADSR curve automatically
        )


class SynthUI(QWidget):
    """Main application window."""

    _play_obj: Optional[sa.PlayObject]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Laser Synth with ADSR")
        self.resize(420, 300)

        self.synth = LaserSynth()
        self._play_obj = None
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        grid = QGridLayout()

        # ADSR sliders ---------------------------------------------------
        self.attack_slider = ADSRSlider("Attack", 1, 1_000, 50, "ms")
        self.decay_slider = ADSRSlider("Decay", 1, 1_000, 100, "ms")
        self.sustain_slider = ADSRSlider("Sustain", 0, 100, 50, "%")
        self.release_slider = ADSRSlider("Release", 10, 2_000, 300, "ms")

        sliders = [
            self.attack_slider,
            self.decay_slider,
            self.sustain_slider,
            self.release_slider,
        ]
        for i, slider in enumerate(sliders):
            grid.addWidget(slider, i // 2, i % 2)

        # Volume slider --------------------------------------------------
        self.volume_slider = ADSRSlider("Volume", 0, 100, 100, "%")
        grid.addWidget(self.volume_slider, 2, 0, 1, 2)

        layout.addLayout(grid)

        # Control buttons ------------------------------------------------
        self.play_btn = QPushButton("Play ☄️")
        self.dist_btn = QPushButton("Distortion…")
        self.save_btn = QPushButton("Save WAV")

        self.play_btn.clicked.connect(self._play_sound)
        self.dist_btn.clicked.connect(self._configure_distortion)
        self.save_btn.clicked.connect(self._save_sound)

        layout.addWidget(self.play_btn)
        layout.addWidget(self.dist_btn)
        layout.addWidget(self.save_btn)

    # ------------------------------------------------------------------
    def _collect_adsr(self) -> ADSRSettings:
        return ADSRSettings(
            attack_ms=self.attack_slider.value(),
            decay_ms=self.decay_slider.value(),
            sustain_level=self.sustain_slider.value() / 100.0,
            release_ms=self.release_slider.value(),
        )

    def _volume(self) -> float:
        return self.volume_slider.value() / 100.0

    # ------------------------------------------------------------------
    def _play_sound(self) -> None:
        adsr, volume = self._collect_adsr(), self._volume()
        samples = self.synth.synthesize(adsr, volume)

        # 1️⃣  create a real temporary file in the system temp dir
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)  # we’ll reopen it with wave
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.synth.sample_rate)
            wf.writeframes(samples.tobytes())

        # 2️⃣  spin up a fresh QSoundEffect for this one playback
        effect = QSoundEffect(self)  # parent = self; GC safe
        effect.setSource(QUrl.fromLocalFile(wav_path))
        effect.setVolume(volume)
        effect.play()

        # 3️⃣  when playback *finishes*, destroy the effect *first*,
        #     then delete the file a moment later
        def _tidy():
            if not effect.isPlaying():  # fires twice; guard it
                path = Path(wav_path)  # capture before effect dies
                effect.deleteLater()  # releases the handle soon
                QTimer.singleShot(0, lambda: path.unlink(missing_ok=True))

        effect.playingChanged.connect(_tidy)

    # ------------------------------------------------------------------
    def _save_sound(self) -> None:
        adsr = self._collect_adsr()
        volume = self._volume()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save laser sound",
            str(Path.home() / "laser.wav"),
            "WAV files (*.wav)",
        )
        if filename:
            self.synth.save(filename, adsr, volume)


    # ------------------------------------------------------------------
    def _configure_distortion(self) -> None:
        """Open the distortion popup and push the settings into the synth."""
        current = (
            self.synth._dist_proc.settings  # pyright: ignore[reportPrivateUsage]
            if self.synth._dist_proc       # noqa: SLF001
            else None
        )
        dlg = DistortionDialog(self, current)
        if dlg.exec():
            self.synth.set_distortion(dlg.settings())