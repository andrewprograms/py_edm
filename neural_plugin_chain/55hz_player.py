import sys
import math
import struct
import torch
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
)
from PyQt6.QtCore import Qt, QIODevice, QTimer
from PyQt6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices, QAudio
from neural_waveshaping_synthesis.models.newt import FastNEWT  # ← Fast‑NEWT CNN


class SineWaveIODevice(QIODevice):
    """Continuous sine‑wave generator with attack/release envelope
    and a *Fast‑NEWT* neural waveshaping distortion (learned DDSP‑style).
    """

    def __init__(
        self,
        frequency: float,
        sample_rate: int,
        channel_count: int,
        sample_format: QAudioFormat.SampleFormat,
        attack_ms: int = 10,
        release_ms: int = 10,
        dist_amount: float = 0.0,  # 0‒1 wet/dry for the neural distortion
    ):
        super().__init__()
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.channel_count = channel_count
        self.sample_format = sample_format

        # Phase accumulator -------------------------------------------
        self.phase = 0.0
        self.phase_inc = 2 * math.pi * self.frequency / self.sample_rate

        # Envelope parameters ----------------------------------------
        self.attack_samples = max(1, int(self.sample_rate * attack_ms / 1000.0))
        self.release_samples = max(1, int(self.sample_rate * release_ms / 1000.0))
        self.env_state: str = "attack"  # attack → sustain → release
        self.env_index = 0

        # Distortion --------------------------------------------------
        self.dist_amount = dist_amount  # 0‑1 wet/dry
        self._init_newt()

        # Sample packing ---------------------------------------------
        if self.sample_format == QAudioFormat.SampleFormat.Int16:
            self.pack_fmt = "<h"
            self.bytes_per_sample = 2
            self.amp = int(0.8 * 32767)
        elif self.sample_format == QAudioFormat.SampleFormat.Float:
            self.pack_fmt = "<f"
            self.bytes_per_sample = 4
            self.amp = 0.8
        else:
            raise ValueError("Unsupported sample format – only Int16 and Float.")

        self.open(QIODevice.OpenModeFlag.ReadOnly)

    # ------------------------------------------------------------------
    # NEWT initialisation ----------------------------------------------
    # ------------------------------------------------------------------
    def _init_newt(self):
        """Load Fast‑NEWT checkpoint from local file."""
        self.newt = FastNEWT().to("cpu").eval()
        ckpt = torch.load("FastNEWT_fuzz.pt", map_location="cpu")
        self.newt.load_state_dict(ckpt)
        # Re‑usable tensor to avoid alloc every call
        self._newt_in = torch.zeros((1, 1, 1), dtype=torch.float32)

    # ------------------------------------------------------------------
    # Public control ----------------------------------------------------
    # ------------------------------------------------------------------
    def start_release(self):
        if self.env_state == "release":
            return
        self.env_state = "release"
        self.env_index = 0

    def set_distortion_amount(self, amt: float):
        """Set wet/dry mix (0‒1)."""
        self.dist_amount = max(0.0, min(1.0, amt))

    # ------------------------------------------------------------------
    # Neural distortion per‑sample -------------------------------------
    # ------------------------------------------------------------------
    def _apply_distortion(self, dry_sample: float) -> float:
        """Run one sample through Fast‑NEWT and blend wet/dry."""
        if self.dist_amount <= 0.0:
            return dry_sample

        # Scale to [-1,1] float32 for the network
        self._newt_in[0, 0, 0] = dry_sample / self.amp
        with torch.no_grad():
            wet_norm = self.newt(self._newt_in).item()
        wet_sample = wet_norm * self.amp

        # Linear cross‑fade wet/dry
        return (1 - self.dist_amount) * dry_sample + self.dist_amount * wet_sample

    # ------------------------------------------------------------------
    # QIODevice overrides ----------------------------------------------
    # ------------------------------------------------------------------
    def bytesAvailable(self):  # noqa: D401
        return 4096 + super().bytesAvailable()

    def readData(self, maxlen):  # noqa: N802
        frame_bytes = self.bytes_per_sample * self.channel_count
        n_frames = maxlen // frame_bytes
        if n_frames == 0:
            return bytes()

        data = bytearray(n_frames * frame_bytes)
        offset = 0

        for _ in range(n_frames):
            # Envelope --------------------------------------------------
            if self.env_state == "attack":
                env = self.env_index / self.attack_samples
                self.env_index += 1
                if self.env_index >= self.attack_samples:
                    self.env_state = "sustain"
                    self.env_index = 0
            elif self.env_state == "sustain":
                env = 1.0
            else:  # "release"
                env = max(0.0, 1 - self.env_index / self.release_samples)
                self.env_index += 1

            # Base sine sample -----------------------------------------
            dry = math.sin(self.phase) * self.amp * env
            sample_val = self._apply_distortion(dry)

            for _ in range(self.channel_count):
                struct.pack_into(self.pack_fmt, data, offset, sample_val)
                offset += self.bytes_per_sample

            # Phase advance -------------------------------------------
            self.phase += self.phase_inc
            if self.phase >= 2 * math.pi:
                self.phase -= 2 * math.pi

        return bytes(data)

    def writeData(self, _: bytes):  # noqa: N802 – not used
        return 0


class SineWavePlayer(QWidget):
    """Tiny Qt6 tone generator with pop‑free attack/release and Fast‑NEWT neural distortion."""

    ENV_MS = 10  # attack & release length in milliseconds

    def __init__(self):
        super().__init__()
        self.setWindowTitle("55 Hz Sine‑Wave Player (Fast‑NEWT Distortion)")
        self.resize(340, 240)

        # ----------------------------------------------------------
        # Audio format selection
        # ----------------------------------------------------------
        self.device_out = QMediaDevices.defaultAudioOutput()
        desired = QAudioFormat()
        desired.setSampleRate(44_100)
        desired.setChannelCount(2)
        desired.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        self.audio_format = (
            desired
            if self.device_out.isFormatSupported(desired)
            else self.device_out.preferredFormat()
        )

        self.sample_rate = self.audio_format.sampleRate()
        self.channel_count = self.audio_format.channelCount()
        self.sample_format = self.audio_format.sampleFormat()

        self.sink = QAudioSink(self.device_out, self.audio_format)
        self.io_device: SineWaveIODevice | None = None

        # ----------------------------------------------------------
        # UI setup
        # ----------------------------------------------------------
        self.play_btn = QPushButton("▶︎ Play")
        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.setEnabled(False)

        # Volume slider -------------------------------------------
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(50)
        self.vol_label = QLabel("Volume: 50 %", alignment=Qt.AlignmentFlag.AlignCenter)

        # Distortion slider ---------------------------------------
        self.dist_slider = QSlider(Qt.Orientation.Horizontal)
        self.dist_slider.setRange(0, 100)
        self.dist_slider.setValue(0)
        self.dist_label = QLabel(
            "Distortion: 0 %", alignment=Qt.AlignmentFlag.AlignCenter
        )

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.play_btn)
        btn_row.addWidget(self.stop_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(btn_row)
        layout.addWidget(self.vol_slider)
        layout.addWidget(self.vol_label)
        layout.addWidget(self.dist_slider)
        layout.addWidget(self.dist_label)

        self.play_btn.clicked.connect(self.start_playback)
        self.stop_btn.clicked.connect(self.stop_playback)
        self.vol_slider.valueChanged.connect(self.set_volume)
        self.dist_slider.valueChanged.connect(self.set_distortion)

        # Initial volume directly on sink (steady‑state)
        self.sink.setVolume(self.vol_slider.value() / 100.0)
        self.current_dist_amt = 0.0

    # ----------------------------------------------------------
    # Playback control
    # ----------------------------------------------------------
    def start_playback(self):
        if self.sink.state() == QAudio.State.ActiveState:
            return  # already playing

        self.io_device = SineWaveIODevice(
            55.0,
            self.sample_rate,
            self.channel_count,
            self.sample_format,
            attack_ms=self.ENV_MS,
            release_ms=self.ENV_MS,
            dist_amount=self.current_dist_amt,
        )
        self.sink.start(self.io_device)

        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_playback(self):
        if self.sink.state() != QAudio.State.ActiveState:
            return

        # Trigger internal release envelope and schedule final stop
        self.io_device.start_release()
        QTimer.singleShot(self.ENV_MS, self._complete_stop)

    def _complete_stop(self):
        self.sink.stop()
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    # ----------------------------------------------------------
    # Misc helpers
    # ----------------------------------------------------------
    def set_volume(self, val: int):
        self.vol_label.setText(f"Volume: {val}\u00a0%")
        self.sink.setVolume(val / 100.0)

    def set_distortion(self, val: int):
        self.dist_label.setText(f"Distortion: {val}\u00a0%")
        self.current_dist_amt = val / 100.0
        if self.io_device is not None:
            self.io_device.set_distortion_amount(self.current_dist_amt)

    def closeEvent(self, e):  # noqa: N802
        if self.sink.state() == QAudio.State.ActiveState:
            self.io_device.start_release()
            QTimer.singleShot
