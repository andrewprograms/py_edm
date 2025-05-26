from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
    QHBoxLayout, QLabel, QFrame, QSizePolicy, QScrollArea
)
from PyQt6.QtCore import Qt
import sys


class Track(QFrame):
    """A single horizontal track with a colored background and a name label."""
    def __init__(self, name: str, color: str):
        super().__init__()
        self.setFrameShape(QFrame.Shape.Panel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setStyleSheet(f"background-color:{color}; border-radius:4px;")
        self.setMinimumHeight(40)

        lbl = QLabel(name, parent=self)
        lbl.setStyleSheet("color:white; font-weight:bold; margin-left:8px;")
        lbl.move(0, 0)  # top-left corner


class GroupHeader(QLabel):
    """A label that acts as a header for a group of tracks."""
    def __init__(self, text: str):
        super().__init__(text.upper())
        self.setStyleSheet(
            "color:#444;"
            "font-weight:600;"
            "padding:4px 0 2px 2px;"
            "border-bottom:1px solid #bbb;"
        )
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)


class DAWWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt6 Mini-DAW")
        self.resize(900, 400)

        # -------- Top transport bar --------
        transport_bar = QWidget()
        t_layout = QHBoxLayout(transport_bar)
        t_layout.setContentsMargins(10, 6, 10, 6)
        t_layout.setSpacing(10)

        play_btn = QPushButton("▶︎ Play")
        play_btn.setFixedWidth(90)
        play_btn.setStyleSheet("font-size:16px; font-weight:bold;")
        t_layout.addWidget(play_btn)
        t_layout.addStretch()

        # -------- Track area (scrollable) --------
        track_area = QScrollArea()
        track_area.setWidgetResizable(True)

        tracks_container = QWidget()
        v_tracks = QVBoxLayout(tracks_container)
        v_tracks.setContentsMargins(10, 10, 10, 10)
        v_tracks.setSpacing(6)

        # Drums group
        v_tracks.addWidget(GroupHeader("Drums"))
        v_tracks.addWidget(Track("Kick",  "#d35400"))   # orange
        v_tracks.addWidget(Track("Snare", "#f1c40f"))   # yellow
        v_tracks.addWidget(Track("Hat",   "#16a085"))   # teal

        # Synth group
        v_tracks.addSpacing(4)
        v_tracks.addWidget(GroupHeader("Synth"))
        v_tracks.addWidget(Track("Bass",  "#27ae60"))   # green
        v_tracks.addWidget(Track("Lead",  "#8e44ad"))   # purple

        v_tracks.addStretch()  # push tracks to top
        track_area.setWidget(tracks_container)

        # -------- Assemble main layout --------
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(transport_bar)
        main_layout.addWidget(track_area)

        self.setCentralWidget(central)


def main():
    app = QApplication(sys.argv)
    win = DAWWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
