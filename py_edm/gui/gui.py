from __future__ import annotations

"""PyQt6 front‑end for pyedm.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import simpleaudio as sa  # noqa: F401  — kept for potential future use
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QFont, QPalette, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QStyleFactory,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QDial,
    QVBoxLayout,
    QWidget,
    QDialog,
    QComboBox,
    QDialogButtonBox,
)
from PyQt6.QtMultimedia import QSoundEffect

from adsr.adsr import ADSRSettings
from effects.effects import LaserSynth
from effects.distortion import DistortionSettings, _ALGOS
from effects.reverb import ReverbSettings
from .visualizer import VisualizerWidget

import wave
import os
import tempfile


# -----------------------------------------------------------------------------
# ⚙️  Custom widgets
# -----------------------------------------------------------------------------
class KnobWidget(QGroupBox):
    """Circular knob with read‑out label"""

    def __init__(
        self,
        label: str,
        min_val: int,
        max_val: int,
        init_val: int,
        unit: str,
        step: int = 1,
    ):
        super().__init__()
        self._ensure_dark_mode()
        self.setTitle(label)
        self.setLayout(QVBoxLayout())

        self.dial = DragDial()
        self.dial.setRange(min_val, max_val)
        self.dial.setSingleStep(step)
        self.dial.setValue(init_val)
        self.dial.setNotchesVisible(True)

        # — dark dial with bright handle
        self.dial.setStyleSheet(
            """
            QDial {
                background-color: #272727;
                border: 2px solid #444;
                border-radius: 30px;    /* dial size set later by layout */
            }
            QDial::handle {
                background: #e0e0e0;
                width: 6px;
                height: 14px;
                margin: 0px;
            }
            """
        )

        self.value_label = QLabel()
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        font = QFont()
        font.setPointSize(9)
        self.value_label.setFont(font)

        self.layout().addWidget(self.dial)
        self.layout().addWidget(self.value_label)
        self.unit = unit
        self._update_label(init_val)
        self.dial.valueChanged.connect(self._update_label)

    # ------------------------------------------------------------------
    def _update_label(self, value: int) -> None:
        self.value_label.setText(f"{value} {self.unit}")

    def value(self) -> int:
        return self.dial.value()

    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_dark_mode() -> None:
        """Apply a Fusion dark palette to the QApplication (idempotent)."""
        app = QApplication.instance()
        if app is None or getattr(app, "_dark_mode_applied", False):
            return

        app.setStyle(QStyleFactory.create("Fusion"))

        pal = QPalette()
        pal.setColor(QPalette.ColorRole.Window,           QColor("#121212"))
        pal.setColor(QPalette.ColorRole.WindowText,       QColor("#e0e0e0"))
        pal.setColor(QPalette.ColorRole.Base,             QColor("#1e1e1e"))
        pal.setColor(QPalette.ColorRole.AlternateBase,    QColor("#212121"))
        pal.setColor(QPalette.ColorRole.ToolTipBase,      QColor("#212121"))
        pal.setColor(QPalette.ColorRole.ToolTipText,      QColor("#e0e0e0"))
        pal.setColor(QPalette.ColorRole.Text,             QColor("#e0e0e0"))
        pal.setColor(QPalette.ColorRole.Button,           QColor("#1f1f1f"))
        pal.setColor(QPalette.ColorRole.ButtonText,       QColor("#e0e0e0"))
        pal.setColor(QPalette.ColorRole.BrightText,       QColor("#ff5252"))
        pal.setColor(QPalette.ColorRole.Link,             QColor("#64b5f6"))
        pal.setColor(QPalette.ColorRole.Highlight,        QColor("#2962ff"))
        pal.setColor(QPalette.ColorRole.HighlightedText,  QColor("#ffffff"))
        app.setPalette(pal)
        app._dark_mode_applied = True

# ── NEW:   dial that reacts to mouse-drag instead of circular motion ──────────
class DragDial(QDial):
    """
    QDial whose value changes with *linear* mouse movement rather than rotation.

    Up-or-right  →  value ↑  
    Down-or-left →  value ↓
    """
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._last = None          # remember last mouse-pos while dragging
        self.setWrapping(False)    # no wrap-around
        self.setNotchesVisible(True)

    # ------------------------------------------------------------------ mouse
    def mousePressEvent(self, ev):
        # Begin linear drag – capture start position **without** invoking
        # QDial’s default angle-based handling (which caused snap-back).
        if ev.button() == Qt.MouseButton.LeftButton:
            self._last = ev.position()
            ev.accept()
            return                    # ← stop event propagation
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        # While custom-dragging, override value changes completely.
        if self._last is None:
            super().mouseMoveEvent(ev)
            return

        delta = ev.position() - self._last
        step   = self.singleStep() or 1
        change = int((-delta.y() + delta.x()) * step / 3)  # divide → nice feel

        if change:
            self.setValue(max(self.minimum(),
                         min(self.maximum(), self.value() + change)))
            self._last = ev.position()
        ev.accept()

    def mouseReleaseEvent(self, ev):
        # Finish drag and keep the exact value set during movement.
        self._last = None
        ev.accept()
        return 


# -----------------------------------------------------------------------------
# Legacy sliders for the effect dialogs (unchanged)
# -----------------------------------------------------------------------------
class ADSRSlider(QGroupBox):
    """Reusable horizontal slider with value read‑out (used in dialogs)."""

    def __init__(
        self, label: str, min_val: int, max_val: int, init_val: int, unit: str
    ):
        super().__init__()
        from PyQt6.QtWidgets import QSlider  # imported here to avoid clutter

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


# -----------------------------------------------------------------------------
# Distortion & Reverb dialogs (same functionality, UI tweaked only slightly)
# -----------------------------------------------------------------------------
class ReverbDialog(QDialog):
    def __init__(self, parent=None, current: ReverbSettings | None = None):
        super().__init__(parent)
        self.setWindowTitle("Reverb Settings")
        self.setLayout(QVBoxLayout())

        # Decay slider (0.3–8 s)
        self.decay_knob = KnobWidget("Decay (RT60)", 300, 8000, 2200, "ms", 50)
        self.mix_knob   = KnobWidget("Wet / Dry",    0,   100,   35,  "%")
        self.layout().addWidget(self.decay_knob)
        self.layout().addWidget(self.mix_knob)

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout().addWidget(buttons)

        if current:
           self.decay_knob.dial.setValue(int(current.decay * 1000))
           self.mix_knob.dial.setValue(int(current.mix * 100))

    def settings(self) -> ReverbSettings:
        return ReverbSettings(
           decay=self.decay_knob.value() / 1000.0,
           mix=self.mix_knob.value()     / 100.0,
           predelay_ms=20,
       )


class DistortionDialog(QDialog):
    """Popup to tweak distortion parameters."""

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
        self.drive_slider = ADSRSlider("Drive (pre‑gain)", 1, 50, 25, "×")
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
            algorithm=self.algo_box.currentText(),  # type: ignore[arg-type]
            drive=self.drive_slider.value() / 10.0,  # slider = 1‑50 → 0.1‑5.0
            mix=self.mix_slider.value() / 100.0,
            envelope=None,  # LaserSynth will inject its ADSR curve automatically
        )


# -----------------------------------------------------------------------------
# 🎛  Main UI
# -----------------------------------------------------------------------------
class SynthUI(QWidget):
    """Main application window."""

    _play_obj: Optional[sa.PlayObject]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PyEDM Synth")
        self.resize(460, 480)

        self.synth = LaserSynth()
        self._play_obj = None
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        
        # -- visualizer sits on top ----
        self.visualizer = VisualizerWidget(self)
        layout.addWidget(self.visualizer)

        # --------------------------------------------------------------
        # Knobs (Attack, Decay, Sustain, Release, Volume)
        # --------------------------------------------------------------
        knob_grid = QGridLayout()
        self.attack_knob  = KnobWidget("Attack",  1,   1_000,  50,  "ms", 10)
        self.decay_knob   = KnobWidget("Decay",   1,   1_000, 100, "ms", 10)
        self.sustain_knob = KnobWidget("Sustain", 0, 100, 50, "%")
        self.release_knob = KnobWidget("Release", 10,   2_000, 300, "ms", 20)
        self.volume_knob = KnobWidget("Volume", 0, 100, 100, "%")

        knobs = [
            self.attack_knob,
            self.decay_knob,
            self.sustain_knob,
            self.release_knob,
            self.volume_knob,
        ]
        for i, knob in enumerate(knobs):
            knob_grid.addWidget(knob, i // 3, i % 3)

        layout.addLayout(knob_grid)

        # --------------------------------------------------------------
        # ADSR envelope preview
        # --------------------------------------------------------------
        self.adsr_plot = pg.PlotWidget(title="ADSR Envelope")
        self.adsr_plot.setBackground("#1e1e1e")
        self.adsr_plot.setYRange(0, 1.05)
        self.adsr_plot.getPlotItem().hideAxis("bottom")
        layout.addWidget(self.adsr_plot)

        # Update preview whenever ADSR knobs move
        for knob in (
            self.attack_knob,
            self.decay_knob,
            self.sustain_knob,
            self.release_knob,
        ):
            knob.dial.valueChanged.connect(self._refresh_adsr_plot)

        # --------------------------------------------------------------
        # Waveform graph (PyQtGraph)
        # --------------------------------------------------------------
        self.wave_plot = pg.PlotWidget(title="Output Waveform")
        self.wave_plot.setYRange(-1.05, 1.05)
        self.wave_plot.setBackground("#1e1e1e")
        self.wave_plot.getPlotItem().hideAxis("bottom")
        self.wave_plot.getPlotItem().hideAxis("left")
        layout.addWidget(self.wave_plot)

        # --------------------------------------------------------------
        # Control buttons
        # --------------------------------------------------------------
        self.play_btn = QPushButton("Play  ☄️")
        self.save_btn = QPushButton("Save WAV")
        self.dist_btn = QPushButton("Distortion…")
        self.reverb_btn = QPushButton("Reverb…")

        self.play_btn.clicked.connect(self._play_sound)
        self.save_btn.clicked.connect(self._save_sound)
        self.dist_btn.clicked.connect(self._configure_distortion)
        self.reverb_btn.clicked.connect(self._configure_reverb)

        layout.addWidget(self.play_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.dist_btn)
        layout.addWidget(self.reverb_btn)

        # Initial ADSR draw
        self._refresh_adsr_plot()

    # ------------------------------------------------------------------
    # 🎚  helpers
    # ------------------------------------------------------------------
    def _collect_adsr(self) -> ADSRSettings:
        return ADSRSettings(
            attack_ms=self.attack_knob.value(),
            decay_ms=self.decay_knob.value(),
            sustain_level=self.sustain_knob.value() / 100.0,
            release_ms=self.release_knob.value(),
        )

    def _volume(self) -> float:
        return self.volume_knob.value() / 100.0

    # ------------------------------------------------------------------
    # 📈  ADSR envelope preview
    # ------------------------------------------------------------------
    def _refresh_adsr_plot(self) -> None:
        """Render an exponential-style ADSR envelope preview."""
        atk = self.attack_knob.value() / 1000.0
        dec = self.decay_knob.value() / 1000.0
        sus = self.sustain_knob.value() / 100.0
        rel = self.release_knob.value() / 1000.0

        sr = 1000  # 1 kHz resolution for plotting

        # Attack (0 → 1)
        t_a = np.linspace(0, atk, max(1, int(sr * atk)), endpoint=False)
        env_a = 1 - np.exp(-5 * t_a / atk) if atk else np.array([])

        # Decay (1 → sustain)
        t_d = np.linspace(0, dec, max(1, int(sr * dec)), endpoint=False)
        env_d = sus + (1 - sus) * np.exp(-5 * t_d / dec) if dec else np.array([])

        # Sustain (flat 200 ms)
        t_s = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
        env_s = np.full_like(t_s, sus)

        # Release (sustain → 0)
        t_r = np.linspace(0, rel, max(1, int(sr * rel)))
        env_r = sus * np.exp(-5 * t_r / rel) if rel else np.array([])

        env = np.concatenate([env_a, env_d, env_s, env_r])
        t = np.arange(env.size) / sr

        self.adsr_plot.clear()
        if env.size:
            self.adsr_plot.plot(t, env, pen=pg.mkPen(width=2))

    # ------------------------------------------------------------------
    # 🔊  playback / save
    # ------------------------------------------------------------------
    def _update_plot(self, samples: np.ndarray) -> None:
        """Render *samples* (int16) onto the plot widget."""
        self.wave_plot.clear()
        y = samples.astype(np.float32) / 32_767.0  # back to −1..1
        x = np.arange(y.size) / self.synth.sample_rate
        self.wave_plot.plot(x, y, pen=pg.mkPen(width=1))

    def _play_sound(self) -> None:
        adsr, volume = self._collect_adsr(), self._volume()
        samples = self.synth.synthesize(adsr, volume)

        # — always show graph
        self._update_plot(samples)

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

        # Kick-off the visual animation
        self.visualizer.start(samples, self.synth.sample_rate)

        # 3️⃣  when playback *finishes*, destroy the effect *first*,
        #     then delete the file a moment later
        def _tidy():
            if not effect.isPlaying():  # fires twice; guard it
                path = Path(wav_path)  # capture before effect dies
                effect.deleteLater()  # releases the handle soon
                QTimer.singleShot(0, lambda: path.unlink(missing_ok=True))

                # Stop the visualiser once the sound is done
                self.visualizer.stop()

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
    # ⚙️  effect configuration popups
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

    def _configure_reverb(self) -> None:  # NEW
        current = (
            self.synth._reverb_proc.settings  # pyright: ignore[reportPrivateUsage]
            if self.synth._reverb_proc else None
        )
        dlg = ReverbDialog(self, current)
        if dlg.exec():
            self.synth.set_reverb(dlg.settings())
