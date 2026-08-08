from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from engines.face_engine import FaceEngine, ImageTooLargeError, InvalidImageError


class FakeFace:
    def __init__(self, embedding: np.ndarray) -> None:
        self.embedding = embedding
        self.bbox = np.array([10, 20, 100, 140], dtype=np.float32)
        self.det_score = 0.98


def write_valid_image(path: Path) -> None:
    image = np.full((64, 64, 3), 180, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok is True
    path.write_bytes(encoded.tobytes())


def make_app_factory(*, fail_first_provider: str | None = None):
    state = {"calls": []}

    class FakeFaceAnalysis:
        def __init__(self, *, name: str, providers: list[str], root: str | None = None) -> None:
            state["calls"].append({"name": name, "providers": tuple(providers), "root": root})
            if fail_first_provider and providers and providers[0] == fail_first_provider:
                raise RuntimeError(f"Provider init failed: {providers[0]}")

            self.providers = providers
            self.prepared = False
            self.ctx_id: int | None = None
            self.det_size: tuple[int, int] | None = None

        def prepare(self, *, ctx_id: int, det_size: tuple[int, int]) -> None:
            self.prepared = True
            self.ctx_id = ctx_id
            self.det_size = det_size

        def get(self, image: np.ndarray) -> list[FakeFace]:
            assert image.shape == (64, 64, 3)
            return [FakeFace(np.array([0.1, 0.2, 0.3], dtype=np.float32))]

    return FakeFaceAnalysis, state


def test_extract_embeddings_from_unicode_image_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("engines.face_engine.ort.get_available_providers", lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    app_factory, state = make_app_factory()

    engine = FaceEngine(app_factory=app_factory, max_file_size_mb=1)

    image_path = tmp_path / "صورة_اختبار.jpg"
    write_valid_image(image_path)

    embeddings = engine.extract_embeddings(image_path)

    assert engine.active_providers == ("DmlExecutionProvider", "CPUExecutionProvider")
    assert len(embeddings) == 1
    assert embeddings[0].dtype == np.float32
    assert np.allclose(embeddings[0], np.array([0.1, 0.2, 0.3], dtype=np.float32))
    assert state["calls"][0]["providers"] == ("DmlExecutionProvider", "CPUExecutionProvider")


def test_falls_back_to_cpu_when_accelerated_provider_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("engines.face_engine.ort.get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    app_factory, state = make_app_factory(fail_first_provider="CUDAExecutionProvider")

    engine = FaceEngine(app_factory=app_factory, max_file_size_mb=1)

    image_path = tmp_path / "face.jpg"
    write_valid_image(image_path)
    embeddings = engine.extract_embeddings(image_path)

    assert len(state["calls"]) == 2
    assert state["calls"][0]["providers"] == ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert state["calls"][1]["providers"] == ("CPUExecutionProvider",)
    assert engine.active_providers == ("CPUExecutionProvider",)
    assert len(embeddings) == 1


def test_raises_invalid_image_error_for_corrupted_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("engines.face_engine.ort.get_available_providers", lambda: ["CPUExecutionProvider"])
    app_factory, _state = make_app_factory()

    engine = FaceEngine(app_factory=app_factory, max_file_size_mb=1)

    bad_path = tmp_path / "صورة_معطوبة.jpg"
    bad_path.write_bytes(b"this-is-not-a-real-image")

    with pytest.raises(InvalidImageError):
        engine.extract_embeddings(bad_path)


def test_raises_image_too_large_error_before_decoding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("engines.face_engine.ort.get_available_providers", lambda: ["CPUExecutionProvider"])
    app_factory, _state = make_app_factory()

    engine = FaceEngine(app_factory=app_factory, max_file_size_mb=1)

    large_path = tmp_path / "huge.bin"
    large_path.write_bytes(b"x" * (2 * 1024 * 1024))

    with pytest.raises(ImageTooLargeError):
        engine.extract_embeddings(large_path)
