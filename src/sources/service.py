"""Current catalog permissions precede every original read, including ranges."""

import anyio
from pathlib import PurePath
from urllib.parse import quote, urlsplit

from src.config import settings
from src.sources.storage import OriginalStore, validate_reference, ORIGINAL_EXTENSIONS


class OriginalError(Exception):
    def __init__(self, status, detail):
        self.status, self.detail = status, detail
        super().__init__(detail)


async def authorized_document(doc_id, revision, groups, metadata_store):
    try:
        validate_reference(doc_id, revision)
    except ValueError:
        raise OriginalError(404, "Document not found or not accessible") from None
    doc = await metadata_store.get_document(doc_id)
    groups = set(groups) - {"ALL"}
    if doc is None or not groups.intersection(doc.acl_groups or []):
        raise OriginalError(404, "Document not found or not accessible")
    if doc.dataset_id:
        dataset = await metadata_store.get_dataset(doc.dataset_id)
        # default_acl_groups are ingestion defaults, not a second ACL. A missing
        # or disabled dataset does deny delivery; document ACL stays authoritative.
        if dataset is None or not dataset.active:
            raise OriginalError(404, "Document not found or not accessible")
    if doc.content_hash != revision:
        raise OriginalError(409, "The requested original revision is no longer indexed")
    return doc


async def open_original(doc_id, revision, groups, metadata_store):
    doc = await authorized_document(doc_id, revision, groups, metadata_store)
    if PurePath(doc.filename).suffix.lower() not in ORIGINAL_EXTENSIONS:
        raise OriginalError(410, "No original-file provider for this document type")
    try:
        stream, size = await anyio.to_thread.run_sync(
            OriginalStore().open_verified, doc_id, revision
        )
    except (OSError, ValueError):
        raise OriginalError(
            410,
            "Original unavailable; configure private storage and backfill the exact indexed file",
        ) from None
    return doc, stream, size


def download_url(doc_id, revision):
    base = settings.source_download_webui_url.rstrip("/")
    parsed = urlsplit(base)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or (
            parsed.scheme == "http"
            and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        )
    ):
        return None
    return f"{base}/api/v1/sauron/documents/{quote(doc_id, safe='')}/revisions/{revision}/original"


def presentation(filename, prefix):
    name = PurePath(filename.replace("\\", "/")).name
    name = (
        "".join(c for c in name if c.isprintable() and c not in "\r\n")[:240]
        or "document"
    )
    extension = PurePath(name).suffix.lower()
    mime = {
        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".vsd": "application/vnd.visio",
        ".vsdx": "application/vnd.visio",
        ".vdx": "application/vnd.visio",
    }.get(extension, "application/octet-stream")
    mode = "attachment"
    if extension == ".pdf" and prefix.startswith(b"%PDF-"):
        mime, mode = "application/pdf", "inline"
    elif extension in {".docx", ".vsdx"} and not prefix.startswith(b"PK\x03\x04"):
        mime = "application/octet-stream"
    elif extension in {".doc", ".vsd"} and not prefix.startswith(
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    ):
        mime = "application/octet-stream"
    return name, mime, f"{mode}; filename*=UTF-8''{quote(name, safe='')}"


async def document_download(doc_id, revision, groups, metadata_store):
    try:
        doc, stream, size = await open_original(
            doc_id, revision, groups, metadata_store
        )
        try:
            _, mime, _ = presentation(doc.filename, stream.read(8))
        finally:
            stream.close()
        url = download_url(doc_id, revision)
        if not url:
            return {
                "available": False,
                "error": "OpenWebUI download URL is not configured",
            }
        return {
            "available": True,
            "doc_id": doc_id,
            "revision": revision,
            "filename": doc.filename,
            "bytes": size,
            "mime_type": mime,
            "download_url": url,
        }
    except OriginalError as exc:
        return {"available": False, "error": exc.detail}
