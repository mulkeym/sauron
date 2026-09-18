"""Spool uploads in bounded chunks instead of loading a PDF into API RAM."""
import asyncio
import tempfile
from pathlib import Path


async def save_upload(file) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "upload.bin").suffix) as stream:
        path = Path(stream.name)
        try:
            while chunk := await file.read(1024 * 1024):
                await asyncio.to_thread(stream.write, chunk)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    return path
