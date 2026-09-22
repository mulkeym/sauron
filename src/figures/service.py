"""Figure references stay small; permission checks precede each binary read."""
from __future__ import annotations

import asyncio
import re
from urllib.parse import quote

from src.config import settings
from src.figures.storage import FigureStore


def visual_question(question):
    return bool(re.search(r"\b(diagrams?|topolog(?:y|ies)|venn|flowcharts?|illustrations?|figures?|schematics?|images?|pictures?)\b", question, re.I))


def diagram_discovery_question(question):
    """Recognize requests to browse existing figures, not to interpret their content."""
    q = question.strip().lower()
    if not visual_question(q):
        return False
    if re.search(r"\b(explain|compare|contrast|describe|why|differences?|steps?|commands?|configure|troubleshoot|mean|means|represent)\b", q):
        return False
    q = re.sub(r"^(?:please\s+)?(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?", "", q)
    return bool(re.match(r"(?:list|show|find|display|retrieve|locate|search for)\b", q)
                or re.match(
                    r"(?:do (?:you|we|i) have|have (?:you|we|i) got|"
                    r"(?:are|is) there|(?:are|is) (?:any |the )?(?:figures?|diagrams?|images?|pictures?|topolog(?:y|ies))\b.*\bavailable|"
                    r"(?:what|which) (?:diagrams?|figures?|images?|pictures?|topolog(?:y|ies))|"
                    r"(?:what|which) .+? (?:diagrams?|figures?|images?|pictures?) (?:are|do)|"
                    r"(?:do|does) .+? (?:contain|include|have) (?:any )?(?:figures?|diagrams?|images?|pictures?))\b", q))


def image_policy(question, profile=None):
    config = (profile or {}).get("config", {})
    mode = config.get("images", "inherit")
    mode = settings.answer_images if mode == "inherit" else mode
    count = min(settings.answer_max_images, config.get("max_images", settings.answer_max_images))
    return count if mode == "auto" or (mode == "requested" and visual_question(question)) else 0


async def authorized_figure(doc_id, figure_id, groups, metadata_store):
    doc = await metadata_store.get_document(doc_id)
    if doc is None or not groups or ("ALL" not in groups and not set(groups).intersection(doc.acl_groups)):
        raise FileNotFoundError("Figure not found")
    figure = await metadata_store.get_figure(doc_id, figure_id)
    if not figure:
        raise FileNotFoundError("Figure not found")
    return doc, figure


def reference(doc, figure, relevance=None, variant=None):
    assets = figure.get("assets", {})
    variant = variant or ("preview" if "preview" in assets else "full")
    asset = assets.get(variant)
    if not asset:
        return None
    path = FigureStore().asset_path(doc.doc_id, asset["key"])
    if not path.is_file() or path.stat().st_size != asset["bytes"]:
        return None
    url = f"/api/v1/documents/{quote(doc.doc_id, safe='')}/figures/{quote(figure['figure_id'], safe='')}/content"
    return {"doc_id": doc.doc_id, "figure_id": figure["figure_id"], "filename": doc.filename,
            "kind": figure.get("kind", "other"), "caption": figure.get("caption", ""),
            "source_text": figure.get("source_text", ""),
            "ocr_text": figure.get("ocr_text", ""), "vision_description": figure.get("vision_description", ""),
            "description": figure.get("description", ""), "source_url": doc.source_url or "",
            "page": figure["page"] + 1 if figure.get("page") is not None else None,
            "slide": figure["slide"] + 1 if figure.get("slide") is not None else None,
            "render_warnings": figure.get("render_warnings", []), "source_page_id": figure.get("source_page_id", ""),
            "source_revision": getattr(doc, "content_hash", "") or "",
            "source": figure.get("source", ""), "analysis_status": figure.get("analysis_status", "complete"),
            "width": asset["width"], "height": asset["height"], "bytes": asset["bytes"],
            "mime_type": "image/png", "sha256": asset["sha256"], "variant": variant,
            "content_url": url + "?variant=" + variant, "available": True,
            "variants": list(assets), "relevance": relevance}


async def image_bytes(doc_id, figure_id, groups, metadata_store, variant="preview"):
    if variant not in ("preview", "full"):
        raise ValueError("variant must be preview or full")
    doc, figure = await authorized_figure(doc_id, figure_id, groups, metadata_store)
    asset = figure.get("assets", {}).get(variant)
    if not asset:
        raise FileNotFoundError("Figure variant not found")
    store = FigureStore()
    path = store.asset_path(doc_id, asset["key"])
    def read():
        store.validate_file(path, asset["sha256"], asset["bytes"], serving=True)
        return path.read_bytes()
    raw = await asyncio.to_thread(read)
    return raw, reference(doc, figure, variant=variant)


async def search_chunks(query, groups, vector_store, metadata_store, top_k=5, doc_id=None, kind=None, allowed_doc_ids=None, dataset_id=0):
    from src.retrieval.query_scope import resolve_query_scope
    from src.ingestion.embedder import embed_query
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    scope = await resolve_query_scope(groups, metadata_store, dataset_id=dataset_id, allowed_doc_ids=allowed_doc_ids, question=query)
    ids = [d for d in scope.doc_ids if not doc_id or d == doc_id]
    if not ids:
        return []
    records = await metadata_store.list_figures(ids)
    if not records:
        return []
    keys = {(r["doc_id"], r["figure_id"]) for r in records}
    vector = await asyncio.to_thread(embed_query, query)
    chunks = await asyncio.to_thread(vector_store.search_figures, vector, query, groups, ids, top_k, kind, authorized_doc_ids=ids)
    return [c for c in chunks if (c.metadata.doc_id, c.metadata.figure_id) in keys]


async def search_diagrams(query, groups, vector_store, metadata_store, top_k=5, doc_id=None, kind=None):
    chunks = await search_chunks(query, groups, vector_store, metadata_store, top_k, doc_id, kind)
    result, seen = [], set()
    for chunk in chunks:
        key = (chunk.metadata.doc_id, chunk.metadata.figure_id)
        if key in seen:
            continue
        seen.add(key)
        try:
            doc, figure = await authorized_figure(*key, groups, metadata_store)
            ref = reference(doc, figure, chunk.score)
        except (FileNotFoundError, ValueError, OSError):
            continue
        if ref:
            result.append(ref)
    return result


async def answer_images(question, citations, groups, metadata_store, profile=None):
    limit = image_policy(question, profile)
    if not limit:
        return []
    images, seen = [], set()
    records = [c.model_dump() if hasattr(c, "model_dump") else c for c in citations]
    for c in records:
        if not c.get("figure_id"):
            continue
        key = (c["doc_id"], c["figure_id"])
        if key in seen:
            continue
        seen.add(key)
        try:
            doc, figure = await authorized_figure(*key, groups, metadata_store)
            ref = reference(doc, figure, c.get("relevance"))
        except (FileNotFoundError, ValueError, OSError):
            continue
        if ref:
            ref["evidence_id"] = c.get("evidence_id", "")
            ref["evidence_ids"] = list(dict.fromkeys(r.get("evidence_id", "") for r in records
                if (r.get("doc_id"), r.get("figure_id")) == key))
            images.append(ref)
            if len(images) >= limit:
                break
    return images


async def mcp_result(payload, groups, metadata_store):
    """Deliver inline links when configured, otherwise native image blocks."""
    import base64
    import json
    from fastmcp import FastMCP  # Initialize package before importing tools on older releases.
    from fastmcp.tools import ToolResult
    from mcp.types import TextContent, ImageContent
    payload = dict(payload)
    blocks, delivered, used = [], [], 0
    warnings = list(payload.get("warnings", []))
    from src.figures.presentation import public_images, inline_images, image_markdown
    references = await public_images(payload.get("images", []), groups, metadata_store)
    if len(references) < len(payload.get("images", [])):
        warnings.append("A referenced diagram is no longer available.")
    for ref in references:
        if ref.get("inline_url"):
            ref["markdown"] = image_markdown(ref, ref["inline_url"])
            delivered.append(ref)
            continue
        try:
            raw, current = await image_bytes(ref["doc_id"], ref["figure_id"], groups, metadata_store, ref.get("variant", "preview"))
            size = 4 * ((len(raw) + 2) // 3)
            if used + size > settings.figure_mcp_max_mb * 1024**2:
                warnings.append("A diagram was omitted because the image response limit was reached.")
                continue
            used += size
            blocks.append(ImageContent(type="image", data=base64.b64encode(raw).decode("ascii"), mimeType="image/png"))
            delivered.append({**current, "evidence_id": ref.get("evidence_id", ""),
                              "evidence_ids": ref.get("evidence_ids", [])})
        except (FileNotFoundError, OSError, ValueError):
            warnings.append("A referenced diagram is no longer available.")
    payload["images"] = delivered
    if any(ref.get("inline_url") for ref in delivered):
        from src.citations import render_citations
        canonical = payload.get("answer_with_evidence_ids")
        if canonical is not None:
            illustrated = inline_images(canonical, delivered)
            payload["answer_with_evidence_ids"] = illustrated
            for key in ("answer", "summary", "comparison"):
                if key in payload:
                    payload[key] = render_citations(illustrated, payload.get("citations", []))
        diagram_markdown = "\n\n".join(ref["markdown"] for ref in delivered if ref.get("inline_url"))
        instructions = ("Only embed the diagrams in the images list below. Copy each complete supplied markdown "
            "line exactly, including its caption, URL and closing parenthesis, beside the paragraph discussing that diagram. "
            "Do not invent another figure, edit a token, escape the parentheses, or wrap the URL in another Markdown link. "
            "Citation text mentioning other figures does not provide an image URL. To show a different figure, first "
            "retrieve it with tool_search_diagrams/tool_get_diagram. Never construct an image URL from a document or figure ID. "
            "Do not move diagrams to a separate gallery or redraw them. URLs expire at each image's expires_at timestamp.")
        # Put the small, exact-copy image catalog before the larger evidence body.
        payload = {"display_instructions": instructions, "diagram_markdown": diagram_markdown, **payload}
    payload["warnings"] = warnings
    return ToolResult(content=[TextContent(type="text", text=json.dumps(payload)), *blocks], structured_content=payload)


def preview_html(images):
    import html
    result = []
    for ref in images:
        url = ref["content_url"].replace("/api/v1/documents/", "/admin/api/figure-documents/", 1)
        label = ref["caption"] or f"{ref['filename']} — {ref['figure_id']}"
        if ref.get("page"):
            label += f" (page {ref['page']})"
        if ref.get("slide"):
            label += f" (slide {ref['slide']})"
        notices = " ".join(ref.get("render_warnings", []))
        if notices:
            label += " — " + notices
        result.append(f'<figure><a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener"><img loading="lazy" style="max-width:100%;max-height:650px" src="{html.escape(url, quote=True)}" alt="{html.escape(label, quote=True)}"></a><figcaption>{html.escape(label)}</figcaption></figure>')
    return "".join(result)
