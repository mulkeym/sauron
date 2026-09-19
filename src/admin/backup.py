"""Snapshot SQLite metadata so PNG backups also include committed WAL records."""
import sqlite3
import tempfile
from pathlib import Path


def add_backup_file(archive, path):
    path = Path(path)
    is_sqlite = False
    if path.is_file() and not path.is_symlink():
        with path.open("rb") as handle:
            is_sqlite = handle.read(16) == b"SQLite format 3\x00"
    if not is_sqlite:
        archive.add(str(path), arcname=str(path), recursive=False)
        return
    with tempfile.TemporaryDirectory(prefix="sauron-backup-") as temporary:
        snapshot = Path(temporary) / "snapshot.db"
        source = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        target = sqlite3.connect(snapshot)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        archive.add(str(snapshot), arcname=str(path), recursive=False)
