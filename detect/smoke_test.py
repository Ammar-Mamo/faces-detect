from __future__ import annotations

import sys

import faiss
import onnxruntime as ort
from insightface.app import FaceAnalysis  # noqa: F401
from PySide6.QtWidgets import QApplication, QMainWindow


class SmokeTestWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Smoke Test")
        self.resize(640, 360)


def main() -> int:
    app = QApplication(sys.argv)
    window = SmokeTestWindow()

    # Touch key runtime metadata so Nuitka keeps the relevant binary bindings.
    window.setStatusTip(
        "faiss="
        f"{faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'} | "
        f"ort={ort.__version__} | device={ort.get_device()} | "
        f"providers={', '.join(ort.get_available_providers())}"
    )

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
