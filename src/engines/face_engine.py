from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis


class FaceEngineError(RuntimeError):
    """Base class for all face engine errors."""


class FaceEngineInitializationError(FaceEngineError):
    """Raised when the underlying InsightFace runtime cannot be initialized."""


class InvalidImageError(FaceEngineError):
    """Raised when an image cannot be decoded safely."""


class ImageTooLargeError(FaceEngineError):
    """Raised when an image exceeds configured safety limits."""


@dataclass(frozen=True)
class FaceEmbeddingResult:
    embedding: np.ndarray
    bbox: tuple[float, float, float, float] | None
    score: float | None


class FaceEngine:
    PROVIDER_PRIORITY = (
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    )

    def __init__(
        self,
        *,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        root_dir: str | Path | None = None,
        max_file_size_mb: int = 25,
        max_image_pixels: int = 40_000_000,
        app_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.det_size = det_size
        self.root_dir = Path(root_dir).expanduser() if root_dir else None
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.max_image_pixels = max_image_pixels
        self._app_factory = app_factory or FaceAnalysis

        self.available_providers = tuple(ort.get_available_providers())
        self.preferred_providers = tuple(self._select_providers(self.available_providers))
        self.active_providers: tuple[str, ...] = ()
        self._app = self._initialize_app()

    @property
    def runtime_provider(self) -> str:
        return self.active_providers[0] if self.active_providers else "CPUExecutionProvider"

    def extract(self, image_path: str | Path) -> list[FaceEmbeddingResult]:
        image = self.load_image(image_path)
        try:
            faces = self._app.get(image) or []
        except Exception as exc:
            raise FaceEngineError(f"Face extraction failed for image: {image_path}") from exc

        results: list[FaceEmbeddingResult] = []
        for face in faces:
            embedding = getattr(face, "embedding", None)
            if embedding is None:
                continue

            bbox_value = getattr(face, "bbox", None)
            bbox = tuple(float(value) for value in np.asarray(bbox_value).tolist()) if bbox_value is not None else None
            score_value = getattr(face, "det_score", None)
            score = float(score_value) if score_value is not None else None

            results.append(
                FaceEmbeddingResult(
                    embedding=np.asarray(embedding, dtype=np.float32),
                    bbox=bbox,
                    score=score,
                )
            )

        return results

    def extract_embeddings(self, image_path: str | Path) -> list[np.ndarray]:
        return [result.embedding for result in self.extract(image_path)]

    def load_image(self, image_path: str | Path) -> np.ndarray:
        path = Path(image_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        file_size = path.stat().st_size
        if file_size <= 0:
            raise InvalidImageError(f"Image file is empty: {path}")
        if file_size > self.max_file_size_bytes:
            raise ImageTooLargeError(
                f"Image file is too large: {path} ({file_size} bytes > {self.max_file_size_bytes} bytes)"
            )

        try:
            raw_bytes = np.fromfile(str(path), dtype=np.uint8)
        except OSError as exc:
            raise InvalidImageError(f"Unable to read image bytes: {path}") from exc

        if raw_bytes.size == 0:
            raise InvalidImageError(f"Image file is unreadable: {path}")

        image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImageError(f"Image file is corrupted or unsupported: {path}")

        height, width = image.shape[:2]
        if height * width > self.max_image_pixels:
            raise ImageTooLargeError(
                f"Decoded image is too large: {path} ({width}x{height} exceeds {self.max_image_pixels} pixels)"
            )

        return image

    def _select_providers(self, available_providers: tuple[str, ...]) -> list[str]:
        ordered = [provider for provider in self.PROVIDER_PRIORITY if provider in available_providers]
        if "CPUExecutionProvider" not in ordered:
            ordered.append("CPUExecutionProvider")
        return ordered

    def _initialize_app(self) -> Any:
        provider_attempts = [list(self.preferred_providers)]
        if provider_attempts[0] != ["CPUExecutionProvider"]:
            provider_attempts.append(["CPUExecutionProvider"])

        init_errors: list[str] = []
        for providers in provider_attempts:
            try:
                app = self._create_face_analysis(providers)
                app.prepare(ctx_id=0 if providers[0] != "CPUExecutionProvider" else -1, det_size=self.det_size)
                self.active_providers = tuple(providers)
                return app
            except Exception as exc:
                init_errors.append(f"{providers}: {exc}")

        raise FaceEngineInitializationError(
            "Unable to initialize InsightFace with the available ONNX Runtime providers. "
            + " | ".join(init_errors)
        )

    def _create_face_analysis(self, providers: list[str]) -> Any:
        kwargs: dict[str, Any] = {
            "name": self.model_name,
            "providers": providers,
        }
        if self.root_dir is not None:
            kwargs["root"] = str(self.root_dir)
        return self._app_factory(**kwargs)
