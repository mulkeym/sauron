"""Bounded PNG assets. Decoding stays in the disposable extraction process."""
from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from src.config import settings

FIGURE_ROOT = Path("data/figures")
_active_writes = 0


def track_ingestion(fn):
    from functools import wraps
    @wraps(fn)
    async def wrapped(*args, **kwargs):
        global _active_writes
        _active_writes += 1
        try:
            return await fn(*args, **kwargs)
        finally:
            _active_writes -= 1
    return wrapped


def writes_active():
    return _active_writes > 0
_asset_warnings = ContextVar("figure_asset_warnings", default=None)
_asset_sink = ContextVar("figure_asset_sink", default=None)
PNG = b"\x89PNG\r\n\x1a\n"


@contextmanager
def extraction_assets(path):
    token = _asset_sink.set(Path(path))
    notices = []
    warning_token = _asset_warnings.set(notices)
    try:
        yield notices
    finally:
        _asset_sink.reset(token)
        _asset_warnings.reset(warning_token)


def asset_warning(message):
    notices = _asset_warnings.get()
    if notices is not None and message not in notices:
        notices.append(message)


def save_region(region):
    """Called only by extraction; return a JSON manifest, not image bytes."""
    root = _asset_sink.get()
    if root is None or not settings.figure_store_enabled:
        return {}
    from PIL import Image
    with Image.open(io.BytesIO(region.image_bytes)) as source:
        if source.width * source.height > settings.figure_max_pixels:
            raise ValueError("Figure exceeds the decoded pixel limit")
        source.load()
        full = source.convert("RGB")
    root.mkdir(parents=True, exist_ok=True)
    assets = {}
    for variant, image, limit in (
        ("full", full, settings.figure_full_max_mb * 1024**2),
        ("preview", full.copy(), settings.figure_preview_max_mb * 1024**2),
    ):
        try:
            if variant == "preview":
                image.thumbnail((settings.figure_preview_max_edge,) * 2, Image.Resampling.LANCZOS)
            while True:
                stream = io.BytesIO()
                image.save(stream, format="PNG")
                raw = stream.getvalue()
                if len(raw) <= limit:
                    break
                if variant == "full" or min(image.size) < 512:
                    raw = None
                    break
                image.thumbnail((max(1, int(image.width * .8)), max(1, int(image.height * .8))), Image.Resampling.LANCZOS)
            if raw is None:
                asset_warning(f"A {variant} figure variant exceeded its byte limit and was omitted.")
                continue
            digest = hashlib.sha256(raw).hexdigest()
            path = root / (digest + ".png")
            if not path.exists():
                used = sum(p.stat().st_size for p in root.glob("*.png"))
                if used + len(raw) > settings.figure_store_max_doc_mb * 1024**2:
                    asset_warning("The document figure storage budget was reached; additional variants were omitted.")
                    continue
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(path)
            assets[variant] = {"key": path.name, "sha256": digest, "bytes": len(raw),
                               "width": image.width, "height": image.height, "mime_type": "image/png"}
        finally:
            image.close()
    return assets


class FigureStore:
    """Opaque document-scoped asset keys; no public/static filesystem mount."""
    def __init__(self, root=None):
        self.root = Path(root or FIGURE_ROOT)

    @staticmethod
    def component(value):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
            raise ValueError("Invalid figure document ID")
        return value

    def asset_path(self, doc_id, key):
        self.component(doc_id)
        if not re.fullmatch(r"[0-9a-f]{64}\.png", key):
            raise ValueError("Invalid figure asset key")
        directory = self.root / doc_id
        path = directory / key
        if directory.is_symlink() or path.is_symlink():
            raise ValueError("Invalid figure asset path")
        return path

    def handoff(self, source):
        if not source.exists():
            return ""
        staging = self.root / ".staging"
        staging.mkdir(parents=True, exist_ok=True)
        target = Path(tempfile.mkdtemp(prefix="figures-", dir=staging))
        try:
            total = 0
            for path in source.iterdir():
                if path.is_symlink() or not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}\.png", path.name):
                    raise ValueError("Invalid extraction asset")
                size = path.stat().st_size
                total += size
                if total > settings.figure_store_max_doc_mb * 1024**2:
                    raise ValueError("Figure storage exceeds document limit")
                # Validate bytes incrementally; never decode pixels in the API.
                self.validate_file(path, path.stem, size)
                shutil.copyfile(path, target / path.name)
            return str(target.resolve())
        except BaseException:
            shutil.rmtree(target, ignore_errors=True)
            raise

    @staticmethod
    def validate_file(path, digest, size, *, serving=False):
        max_mb = 64 if serving else max(settings.figure_full_max_mb, settings.figure_preview_max_mb)
        if size > max_mb * 1024**2:
            raise ValueError("Figure exceeds image byte limit")
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise ValueError("Missing or invalid figure file")
        with path.open("rb") as handle:
            header = handle.read(24)
            if header[:8] != PNG or header[12:16] != b"IHDR":
                raise ValueError("Invalid PNG header")
            width, height = int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")
            if not width or not height or width * height > (64000000 if serving else settings.figure_max_pixels):
                raise ValueError("Figure exceeds pixel limit")
            handle.seek(0)
            if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                raise ValueError("Figure hash mismatch")

    def staging_path(self, value):
        path = Path(value)
        if path.is_symlink() or path.resolve().parent != (self.root / ".staging").resolve():
            raise ValueError("Invalid figure staging directory")
        return path

    def publish(self, doc_id, records, staging):
        if not staging:
            return []
        import dataclasses
        source = self.staging_path(staging)
        result = []
        for record in records:
            if not record.assets:
                continue
            details = dataclasses.asdict(record)
            for asset in record.assets.values():
                destination = self.asset_path(doc_id, asset["key"])
                path = source / asset["key"]
                self.validate_file(path, asset["sha256"], asset["bytes"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                tmp = destination.with_suffix(".tmp")
                shutil.copyfile(path, tmp)
                tmp.replace(destination)
            result.append(details)
        return result

    async def publish_async(self, doc_id, records, staging):
        # A cancelled await must not leave a background copy writing after
        # the caller has rolled back this document and discarded its staging.
        import asyncio
        task = asyncio.create_task(asyncio.to_thread(self.publish, doc_id, records, staging))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    def discard(self, staging):
        if staging:
            shutil.rmtree(self.staging_path(staging), ignore_errors=True)

    def delete_document(self, doc_id):
        self.component(doc_id)
        directory = self.root / doc_id
        if directory.is_symlink():
            raise ValueError("Invalid figure directory")
        shutil.rmtree(directory, ignore_errors=True)

    async def reconcile(self, metadata_store):
        """Startup only, before ingestion workers start; catalog read must succeed."""
        live = {d.doc_id for d in await metadata_store.list_documents()}
        await metadata_store.purge_orphan_figures()
        if not self.root.exists():
            return
        for path in self.root.iterdir():
            if path.is_symlink():
                continue
            if path.name == ".staging":
                shutil.rmtree(path)
            elif path.is_dir() and path.name not in live:
                self.delete_document(path.name)
