from __future__ import annotations

"""PyQt6 front‑end for pyedm.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import simpleaudio as sa  # noqa: F401  — kept for potential future use
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
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
        self.setMinimumHeight(100)
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



class _DraggablePoint(pg.ScatterPlotItem):
    """
    A single draggable control point that lives in plot-data coordinates.
    Emits *sigMoved(self)* while it is being dragged.
    """
    sigMoved = pyqtSignal(object)          # self

    def __init__(self, **spot_kw):
        super().__init__(pen=pg.mkPen("#64b5f6"),
                         brush=pg.mkBrush("#64b5f6"),
                         size=10,
                         symbol="o",
                         **spot_kw)
        self.setAcceptHoverEvents(True)

    # proper data-space dragging ------------------------------------------------
    def mouseDragEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore()
            return

        view = self.getViewBox()
        if view is None:                    # should never happen
            ev.ignore()
            return

        if ev.isStart():                    # remember starting points
            # Both “start” refs must be in *data* coordinates
            self._start_view = view.mapSceneToView(ev.scenePos())   # click-pos
            self._start_data = self._start_view                     # point-pos
            ev.accept()
            return

        if ev.isFinish():
            ev.accept()
            return

        # Δ in *data* coordinates, not pixels
        curr_view = view.mapSceneToView(ev.scenePos())
        # pointer delta in data-space
        delta = curr_view - self._start_view
        new_p = self._start_data + delta

        # clamp to sensible ADSR bounds
        new_x = max(0.0, new_p.x())
        new_y = min(1.2, max(0.0, new_p.y()))

        self.setData(pos=[(new_x, new_y)])
        self.sigMoved.emit(self)
        ev.accept()


class ADSRGraph(pg.PlotWidget):
    """
    Interactive ADSR display.  
    Four movable break-points control the same parameters as the
    Attack / Decay / Sustain / Release knobs.

    Signals (all int):
        attackChanged     ms
        decayChanged      ms
        sustainChanged    0-100  %
        releaseChanged    ms
    """
    attackChanged  = pyqtSignal(int)
    decayChanged   = pyqtSignal(int)
    sustainChanged = pyqtSignal(int)
    releaseChanged = pyqtSignal(int)

    def __init__(self, settings: ADSRSettings, *a, **kw):
        super().__init__(*a, **kw)
        self.setBackground("#1e1e1e")
        self.showGrid(x=False, y=True, alpha=0.2)
        self.getPlotItem().hideAxis("bottom")
        self.setMouseEnabled(x=False, y=False)

        self._settings = settings
        self._curve    = self.plot([], [], pen=pg.mkPen(width=2))
        self._make_points()
        self._replot()

    # ─────────────────────────────────────────────────────────────── points
    def _make_points(self):
        """Create / connect three time-points + one sustain-level point."""
        self._p_atk   = _DraggablePoint()
        self._p_dec   = _DraggablePoint()
        self._p_rel   = _DraggablePoint()
        self._p_susLV = _DraggablePoint()          # vertical move only

        for p in (self._p_atk, self._p_dec, self._p_rel, self._p_susLV):
            self.addItem(p)

        # Hook up drag-notifications
        self._p_atk.sigMoved.connect(self._on_attack_drag)
        self._p_dec.sigMoved.connect(self._on_decay_drag)
        self._p_rel.sigMoved.connect(self._on_release_drag)
        self._p_susLV.sigMoved.connect(self._on_sustain_drag)

    # ─────────────────────────────────────────────────────── drag handlers
    def _on_attack_drag(self, p):
        atk_ms = max(1, int(p.pos().x() * 1000))
        self._settings.attack_ms = atk_ms
        self.attackChanged.emit(atk_ms)
        self._replot()

    def _on_decay_drag(self, p):
        # decay duration  =  end-pos – attack-end-pos
        dec_ms = max(1, int((p.pos().x() -
                             self._settings.attack_s) * 1000))
        self._settings.decay_ms = dec_ms
        self.decayChanged.emit(dec_ms)
        # also maybe sustain level if y changed
        self._maybe_emit_sustain(p.pos().y())
        self._replot()

    def _on_release_drag(self, p):
        rel_ms = max(10, int((p.pos().x() -
                              (self._settings.attack_s +
                               self._settings.decay_s + 0.2)) * 1000))
        self._settings.release_ms = rel_ms
        self.releaseChanged.emit(rel_ms)
        self._replot()

    def _on_sustain_drag(self, p):
        self._maybe_emit_sustain(p.pos().y())
        self._replot()

    def _maybe_emit_sustain(self, y):
        sus = max(0, min(1, y))
        self._settings.sustain_level = sus
        self.sustainChanged.emit(int(round(sus * 100)))

    # ─────────────────────────────────────────────────────────────── UI sync
    def set_from_knobs(self, settings: ADSRSettings):
        """Called by knobs → refresh graph."""
        self._settings = settings
        self._replot()

    # ───────────────────────────────────────────────────────── draw curve
    def _replot(self):
        s = self._settings
        atk, dec, sus, rel = s.attack_s, s.decay_s, s.sustain_level, s.release_s

        # (same maths you already had, shortened)
        sr = 1000
        t_a = np.linspace(0, atk, max(1, int(sr*atk)), endpoint=False)
        env_a = 1 - np.exp(-5 * t_a / atk) if atk else np.array([])
        t_d = np.linspace(0, dec, max(1, int(sr*dec)), endpoint=False)
        env_d = sus + (1 - sus)*np.exp(-5 * t_d / dec) if dec else np.array([])
        t_s = np.linspace(0, 0.2, int(sr*0.2), endpoint=False)
        env_s = np.full_like(t_s, sus)
        t_r = np.linspace(0, rel, max(1, int(sr*rel)))
        env_r = sus*np.exp(-5 * t_r / rel) if rel else np.array([])

        env = np.concatenate([env_a, env_d, env_s, env_r])
        t   = np.arange(env.size)/sr
        self._curve.setData(t, env)

        # place / clamp the four draggable points
        self._p_atk.setData(pos=[(atk, 1)])
        self._p_dec.setData(pos=[(atk+dec, sus)])
        self._p_susLV.setData(pos=[(atk+dec+0.001, sus)])   # tiny offset → visible
        self._p_rel.setData(pos=[(atk+dec+0.2+rel, 0)])
        self.setXRange(0, atk+dec+0.2+rel*1.05, padding=0)
        self.setYRange(0, 1.05, padding=0)




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
        self.resize(640, 640)
        self.setMinimumSize(480, 480)

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
        self.volume_knob = KnobWidget("Volume", 0, 100, 50, "%")

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
        # ADSR envelope preview  (now interactive)
        # --------------------------------------------------------------
        self.adsr_plot = ADSRGraph(self._collect_adsr(), title="ADSR Envelope")
        layout.addWidget(self.adsr_plot)       # add **once** – ADSRGraph styles itself

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

        # knobs –→ graph
        for knob, setter in (
            (self.attack_knob,  lambda v: self.adsr_plot.set_from_knobs(self._collect_adsr())),
            (self.decay_knob,   lambda v: self.adsr_plot.set_from_knobs(self._collect_adsr())),
            (self.sustain_knob, lambda v: self.adsr_plot.set_from_knobs(self._collect_adsr())),
            (self.release_knob, lambda v: self.adsr_plot.set_from_knobs(self._collect_adsr())),
        ):
            knob.dial.valueChanged.connect(setter)

        # graph –→ knobs
        def _safe_set(dial, val):
            if dial.value() == val:                # already there → nothing to do
                return

            knob = dial.parent()                   # KnobWidget
            dial.blockSignals(True)                # stop loops
            dial.setValue(val)
            dial.blockSignals(False)

            # we suppressed valueChanged → update label manually
            if hasattr(knob, "_update_label"):
                knob._update_label(val)

        self.adsr_plot.attackChanged .connect(lambda v: _safe_set(self.attack_knob.dial,  v))
        self.adsr_plot.decayChanged  .connect(lambda v: _safe_set(self.decay_knob.dial,   v))
        self.adsr_plot.releaseChanged.connect(lambda v: _safe_set(self.release_knob.dial, v))
        self.adsr_plot.sustainChanged.connect(lambda v: _safe_set(self.sustain_knob.dial, v))

        # 🔄 make sure graph & knobs are linked *immediately*
        self.adsr_plot.set_from_knobs(self._collect_adsr())

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
