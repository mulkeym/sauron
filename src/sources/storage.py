"""Original bytes are opt-in, immutable, private, and never located by a URL."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path

from src.config import settings

DOC_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
REVISION = re.compile(r"[0-9a-f]{64}\Z")
CHUNK_SIZE = 256 * 1024
# Crawled web pages are converted to Markdown, which is not an original file.
ORIGINAL_EXTENSIONS = {
    ".emf",
    ".pdf",
    ".doc",
    ".docx",
    ".docm",
    ".vsd",
    ".vsdx",
    ".vsdm",
    ".vdx",
    ".vss",
    ".vssx",
    ".vst",
    ".vstx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".xlsm",
}


def validate_reference(doc_id: str, revision: str):
    if not DOC_ID.fullmatch(doc_id) or not REVISION.fullmatch(revision):
        raise ValueError("Invalid original document reference")


class OriginalStore:
    def __init__(self):
        configured = settings.source_originals_dir
        self.root = Path(configured).resolve() if configured else None

    def _directory(self, doc_id, revision):
        validate_reference(doc_id, revision)
        if self.root is None:
            raise FileNotFoundError("Original storage is not configured")
        return self.root / doc_id

    def open_verified(self, doc_id, revision):
        """Return one verified open descriptor; never reopen a checked pathname."""
        directory = self._directory(doc_id, revision)
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fd = os.open(
                revision + ".blob",
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
        finally:
            os.close(directory_fd)
        stream = os.fdopen(fd, "rb")
        try:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_size > settings.source_originals_max_mb * 1024**2
            ):
                raise ValueError("Invalid original file")
            digest = hashlib.sha256()
            while chunk := stream.read(CHUNK_SIZE):
                digest.update(chunk)
            if digest.hexdigest() != revision:
                raise ValueError("Original file hash mismatch")
            stream.seek(0)
            return stream, info.st_size
        except BaseException:
            stream.close()
            raise

    def retain(self, path, doc_id, revision, filename=""):
        """Also used for administrator backfill: bytes must match catalog SHA-256."""
        if self.root is None or (
            filename and Path(filename).suffix.lower() not in ORIGINAL_EXTENSIONS
        ):
            return False
        directory = self._directory(doc_id, revision)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if directory.is_symlink():
            raise ValueError("Original directory cannot be a symlink")
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        temporary = None
        try:
            # Same directory/filesystem makes publication atomic. Existing bytes
            # are never overwritten, including during a concurrent backfill.
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as target:
                temporary = Path(target.name)
                digest, size = hashlib.sha256(), 0
                with Path(path).open("rb") as source:
                    while chunk := source.read(CHUNK_SIZE):
                        size += len(chunk)
                        if size > settings.source_originals_max_mb * 1024**2:
                            raise ValueError(
                                "Original exceeds configured storage limit"
                            )
                        digest.update(chunk)
                        target.write(chunk)
                if digest.hexdigest() != revision:
                    raise ValueError(
                        "Original file does not match the indexed revision"
                    )
                target.flush()
                os.fsync(target.fileno())
                os.fchmod(target.fileno(), 0o400)
            try:
                os.link(
                    temporary.name,
                    revision + ".blob",
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                stream, _ = self.open_verified(doc_id, revision)
                stream.close()
            os.fsync(directory_fd)
            return True
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            os.close(directory_fd)

    def delete_document(self, doc_id):
        if not self.root:
            return
        directory = self._directory(doc_id, "0" * 64)
        if directory.is_symlink():
            directory.unlink()
        elif directory.exists():
            shutil.rmtree(directory)
