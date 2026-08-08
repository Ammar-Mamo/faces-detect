from __future__ import annotations

import threading
from pathlib import Path

from db.db_manager import DatabaseManager, FaceRecord


def write_fake_image(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def test_cache_validation_uses_mtime_and_file_size(tmp_path: Path) -> None:
    db = DatabaseManager(tmp_path / "cache.db")
    image_path = tmp_path / "face.jpg"
    write_fake_image(image_path, b"face-bytes-v1")

    image_id = db.upsert_image(image_path)

    assert image_id > 0
    assert db.is_image_cached(image_path) is True

    write_fake_image(image_path, b"face-bytes-v2-with-different-size")

    status = db.get_image_cache_status(image_path)

    assert status.image_id == image_id
    assert status.is_cached is False
    assert status.file_size_bytes == image_path.stat().st_size
    assert status.modified_time_ns == image_path.stat().st_mtime_ns


def test_concurrent_thread_writes_do_not_lock_database(tmp_path: Path) -> None:
    db = DatabaseManager(tmp_path / "faces.db")
    thread_count = 6
    items_per_thread = 15
    start_barrier = threading.Barrier(thread_count)
    errors: list[BaseException] = []
    connection_ids: set[int] = set()
    connection_ids_lock = threading.Lock()
    error_lock = threading.Lock()

    def worker(thread_index: int) -> None:
        try:
            connection = db.get_connection()
            with connection_ids_lock:
                connection_ids.add(id(connection))

            start_barrier.wait()

            for item_index in range(items_per_thread):
                image_path = tmp_path / f"worker_{thread_index}_{item_index}.jpg"
                write_fake_image(image_path, f"image-{thread_index}-{item_index}".encode("utf-8"))

                image_id = db.upsert_image(image_path)
                db.replace_faces_for_image(
                    image_id,
                    [
                        FaceRecord(
                            faiss_id=(thread_index * 10_000) + item_index,
                            bbox_x1=1.0,
                            bbox_y1=2.0,
                            bbox_x2=3.0,
                            bbox_y2=4.0,
                            det_score=0.99,
                            embedding_dim=512,
                        )
                    ],
                )
        except BaseException as exc:
            with error_lock:
                errors.append(exc)
        finally:
            db.close_thread_connection()

    threads = [threading.Thread(target=worker, args=(index,), daemon=True) for index in range(thread_count)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(connection_ids) == thread_count
    assert db.count_images() == thread_count * items_per_thread
    assert db.count_faces() == thread_count * items_per_thread

    pragmas = db.get_connection().execute("PRAGMA journal_mode;").fetchone()
    synchronous = db.get_connection().execute("PRAGMA synchronous;").fetchone()

    assert str(pragmas[0]).lower() == "wal"
    assert int(synchronous[0]) in {1, 2}
