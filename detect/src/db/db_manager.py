from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class ImageCacheStatus:
    image_id: int | None
    is_cached: bool
    file_size_bytes: int
    modified_time_ns: int


@dataclass(frozen=True)
class FaceRecord:
    faiss_id: int
    bbox_x1: float | None = None
    bbox_y1: float | None = None
    bbox_x2: float | None = None
    bbox_y2: float | None = None
    det_score: float | None = None
    embedding_dim: int | None = None


class DatabaseManager:
    def __init__(self, db_path: str | Path, *, timeout: float = 30.0) -> None:
        self.db_path = Path(db_path)
        self.timeout = timeout
        self._local = threading.local()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        connection = self.get_connection()
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL UNIQUE,
                file_size_bytes INTEGER NOT NULL,
                modified_time_ns INTEGER NOT NULL,
                scanned_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                faiss_id INTEGER NOT NULL UNIQUE,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                det_score REAL,
                embedding_dim INTEGER,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_faces_image_id ON faces(image_id);
            CREATE INDEX IF NOT EXISTS idx_images_path ON images(image_path);
            """
        )
        connection.commit()

    def get_connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(
                str(self.db_path),
                timeout=self.timeout,
                isolation_level=None,
                check_same_thread=True,
            )
            connection.row_factory = sqlite3.Row
            self._configure_connection(connection)
            self._local.connection = connection
        return connection

    def close_thread_connection(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.get_connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def get_image_cache_status(self, image_path: str | Path) -> ImageCacheStatus:
        path = Path(image_path).expanduser().resolve()
        stats = path.stat()

        row = self.get_connection().execute(
            """
            SELECT id, file_size_bytes, modified_time_ns
            FROM images
            WHERE image_path = ?
            """,
            (str(path),),
        ).fetchone()

        if row is None:
            return ImageCacheStatus(
                image_id=None,
                is_cached=False,
                file_size_bytes=stats.st_size,
                modified_time_ns=stats.st_mtime_ns,
            )

        is_cached = (
            int(row["file_size_bytes"]) == stats.st_size
            and int(row["modified_time_ns"]) == stats.st_mtime_ns
        )
        return ImageCacheStatus(
            image_id=int(row["id"]),
            is_cached=is_cached,
            file_size_bytes=stats.st_size,
            modified_time_ns=stats.st_mtime_ns,
        )

    def is_image_cached(self, image_path: str | Path) -> bool:
        return self.get_image_cache_status(image_path).is_cached

    def upsert_image(self, image_path: str | Path) -> int:
        path = Path(image_path).expanduser().resolve()
        stats = path.stat()

        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO images (image_path, file_size_bytes, modified_time_ns)
                VALUES (?, ?, ?)
                ON CONFLICT(image_path) DO UPDATE SET
                    file_size_bytes = excluded.file_size_bytes,
                    modified_time_ns = excluded.modified_time_ns,
                    scanned_at = CURRENT_TIMESTAMP
                """,
                (str(path), stats.st_size, stats.st_mtime_ns),
            )
            row = connection.execute(
                "SELECT id FROM images WHERE image_path = ?",
                (str(path),),
            ).fetchone()

        if row is None:
            raise RuntimeError(f"Unable to upsert image row for {path}")
        return int(row["id"])

    def replace_faces_for_image(self, image_id: int, faces: list[FaceRecord]) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM faces WHERE image_id = ?", (image_id,))
            if not faces:
                return

            connection.executemany(
                """
                INSERT INTO faces (
                    image_id,
                    faiss_id,
                    bbox_x1,
                    bbox_y1,
                    bbox_x2,
                    bbox_y2,
                    det_score,
                    embedding_dim
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        image_id,
                        face.faiss_id,
                        face.bbox_x1,
                        face.bbox_y1,
                        face.bbox_x2,
                        face.bbox_y2,
                        face.det_score,
                        face.embedding_dim,
                    )
                    for face in faces
                ],
            )

    def get_faces_for_image(self, image_id: int) -> list[sqlite3.Row]:
        rows = self.get_connection().execute(
            """
            SELECT id, image_id, faiss_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, det_score, embedding_dim
            FROM faces
            WHERE image_id = ?
            ORDER BY faiss_id ASC
            """,
            (image_id,),
        ).fetchall()
        return list(rows)

    def count_images(self) -> int:
        row = self.get_connection().execute("SELECT COUNT(*) AS count FROM images").fetchone()
        return int(row["count"])

    def count_faces(self) -> int:
        row = self.get_connection().execute("SELECT COUNT(*) AS count FROM faces").fetchone()
        return int(row["count"])

    @staticmethod
    def _configure_connection(connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        connection.execute("PRAGMA foreign_keys=ON;")
        connection.execute("PRAGMA busy_timeout=30000;")
