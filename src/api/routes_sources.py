"""Authenticated original delivery. There are no bearer links or static mounts."""

import asyncio
import logging
import re

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from src.sources.auth import require_download_identity
from src.sources.service import OriginalError, open_original, presentation
from src.sources.storage import CHUNK_SIZE

router = APIRouter(prefix="/api/v1", tags=["original documents"])
audit = logging.getLogger("sauron.source_access")
PRIVATE = {
    "Cache-Control": "private, no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Vary": "Authorization, Cookie",
    "Content-Security-Policy": "sandbox",
}


class OriginalFileResponse(StreamingResponse):
    """Close even when disconnect happens before the body iterator starts."""

    def __init__(self, content, *, stream, **kwargs):
        self.stream = stream
        super().__init__(content, **kwargs)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.stream.close()


def get_metadata_store():
    from src.api.routes_ingest import get_metadata_store as get_store

    return get_store()


def byte_range(value, size):
    if value is None:
        return 0, size - 1, 200
    match = re.fullmatch(r"bytes=(\d{0,20})-(\d{0,20})", value)
    if not match or not any(match.groups()) or size == 0:
        raise OriginalError(416, "Requested range is not satisfiable")
    first, last = match.groups()
    if not first:
        length = int(last)
        if length == 0:
            raise OriginalError(416, "Requested range is not satisfiable")
        return max(0, size - length), size - 1, 206
    start, end = int(first), min(int(last), size - 1) if last else size - 1
    if start >= size or end < start:
        raise OriginalError(416, "Requested range is not satisfiable")
    return start, end, 206


@router.get("/documents/{doc_id}/revisions/{revision}/original")
@router.head(
    "/documents/{doc_id}/revisions/{revision}/original", include_in_schema=False
)
async def original(
    request: Request,
    doc_id: str,
    revision: str,
    identity=Depends(require_download_identity),
):
    stream, size = None, None
    try:
        doc, stream, size = await open_original(
            doc_id, revision, identity.groups, get_metadata_store()
        )
        _, mime, disposition = presentation(doc.filename, stream.read(8))
        etag = '"' + revision + '"'
        requested_range = request.headers.get("range")
        if request.headers.get("if-range", etag) != etag:
            requested_range = None
        start, end, status = byte_range(requested_range, size)
        headers = {
            **PRIVATE,
            "Content-Disposition": disposition,
            "Content-Length": str(end - start + 1),
            "Accept-Ranges": "bytes",
            "ETag": etag,
        }
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        audit.info(
            "original_access subject=%r doc_id=%r revision=%r status=%d method=%s",
            identity.subject,
            doc_id,
            revision,
            status,
            request.method,
        )
        if request.method == "HEAD":
            stream.close()
            return Response(status_code=status, media_type=mime, headers=headers)
        stream.seek(start)

        async def chunks():
            remaining = end - start + 1
            try:
                while remaining:
                    chunk = await asyncio.to_thread(
                        stream.read, min(CHUNK_SIZE, remaining)
                    )
                    if not chunk:
                        raise OSError("Original ended before the advertised length")
                    remaining -= len(chunk)
                    yield chunk
            finally:
                stream.close()

        return OriginalFileResponse(
            chunks(),
            stream=stream,
            status_code=status,
            media_type=mime,
            headers=headers,
        )
    except OriginalError as exc:
        if stream:
            stream.close()
        audit.info(
            "original_access subject=%r doc_id=%r revision=%r status=%d method=%s",
            identity.subject,
            doc_id,
            revision,
            exc.status,
            request.method,
        )
        headers = dict(PRIVATE)
        if exc.status == 416 and size is not None:
            headers["Content-Range"] = f"bytes */{size}"
        return JSONResponse(
            {"detail": exc.detail}, status_code=exc.status, headers=headers
        )
    except BaseException:
        if stream:
            stream.close()
        raise
