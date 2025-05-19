from __future__ import annotations

"""Entry point for the Laser Synth application.
"""

import sys
from PyQt6.QtWidgets import QApplication

from gui.gui import SynthUI


def main() -> None:  # pragma: no cover
    app = QApplication(sys.argv)
    ui = SynthUI()
    ui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
    print("done...")
