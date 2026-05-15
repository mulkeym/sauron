"""Web crawler — fetches pages, converts to markdown, and ingests into SAURON.

Supports:
- Single page fetch
- Multi-page crawl (follow links up to crawl_depth)
- File download detection (.pdf, .docx, etc.)
- Content hash dedup (won't re-ingest unchanged pages)
- Boilerplate stripping via the existing parser
"""
import asyncio
import hashlib
import logging
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

logger = logging.getLogger(__name__)

FILE_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".csv", ".txt", ".conf", ".cfg",
                   ".pptx", ".doc", ".xls", ".zip", ".json", ".xml", ".yaml", ".yml"}


def _is_file_url(url: str, extra_types: list[str] = None) -> bool:
    """Check if URL points to a downloadable file."""
    path = urlparse(url).path.lower()
    types = FILE_EXTENSIONS | set(extra_types or [])
    return any(path.endswith(ext) for ext in types)


def _clean_url(url: str) -> str:
    """Remove fragments and trailing slashes for dedup."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"


def _fetch_page(url: str, timeout: int = 30) -> tuple[str, str]:
    """Fetch a URL and return (html_content, content_type)."""
    resp = requests.get(url, timeout=timeout, headers={
        "User-Agent": "SAURON/1.0 (Enterprise RAG Crawler)",
    })
    resp.raise_for_status()
    return resp.text, resp.headers.get("content-type", "")


def _html_to_markdown(html: str, url: str = "") -> str:
    """Convert HTML to markdown, stripping scripts/styles."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
        tag.decompose()

    # Try to find main content
    main = soup.find("main") or soup.find("article") or soup.find(role="main")
    if main:
        html_content = str(main)
    else:
        body = soup.find("body")
        html_content = str(body) if body else str(soup)

    # Convert to markdown
    markdown = md(html_content, heading_style="ATX", strip=["img"])

    # Clean up excessive whitespace
    markdown = re.sub(r'\n{4,}', '\n\n\n', markdown)
    markdown = markdown.strip()

    return markdown


def _extract_links(html: str, base_url: str, url_pattern: str = "") -> list[str]:
    """Extract links from HTML, resolved to absolute URLs."""
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    base_domain = urlparse(base_url).netloc

    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Skip mailto, javascript, anchors
        if href.startswith(("mailto:", "javascript:", "#", "tel:")):
            continue

        full_url = _clean_url(urljoin(base_url, href))

        # Must be same domain
        if urlparse(full_url).netloc != base_domain:
            continue

        # Must match URL pattern if specified
        if url_pattern and not re.search(url_pattern, full_url):
            continue

        links.add(full_url)

    return list(links)


async def crawl_connector(connector, metadata_store, ingest_queue, vector_store) -> dict:
    """Crawl a web connector and ingest discovered pages/files.

    Returns {"pages_found": N, "pages_ingested": N, "files_downloaded": N, "errors": []}
    """
    from datetime import datetime, timezone

    base_url = connector.base_url.rstrip("/")
    crawl_depth = connector.crawl_depth
    max_pages = connector.max_pages
    url_pattern = connector.url_pattern or ""
    download_types = connector.download_file_types or []
    acl_groups = connector.acl_groups or []
    category = connector.category or ""
    app_id = connector.application_id

    visited = set()
    to_visit = [(base_url, 0)]  # (url, depth)
    for extra in (connector.additional_urls or []):
        extra = extra.strip()
        if extra:
            to_visit.append((extra.rstrip("/"), 0))
    pages_found = 0
    pages_ingested = 0
    files_downloaded = 0
    errors = []

    while to_visit and pages_found < max_pages:
        url, depth = to_visit.pop(0)
        clean = _clean_url(url)

        if clean in visited:
            continue
        visited.add(clean)

        try:
            # Check if it's a file to download
            if _is_file_url(url, download_types):
                await _download_and_ingest_file(
                    url, acl_groups, category, app_id, ingest_queue, metadata_store
                )
                files_downloaded += 1
                pages_found += 1
                continue

            # Fetch page
            html, content_type = await asyncio.to_thread(_fetch_page, url)
            if "text/html" not in content_type and "text/plain" not in content_type:
                continue

            pages_found += 1

            # Convert to markdown
            markdown = await asyncio.to_thread(_html_to_markdown, html, url)
            if len(markdown.strip()) < 50:
                continue  # too little content

            # Check if already ingested (content hash)
            content_hash = hashlib.sha256(markdown.encode()).hexdigest()
            existing = await metadata_store.find_by_content_hash(content_hash)
            if existing:
                logger.debug(f"Skipping {url} — already ingested as {existing.filename}")
                continue

            # Ingest the page
            # Create a temp .md file and enqueue
            slug = urlparse(url).path.strip("/").replace("/", "_") or "index"
            filename = f"web_{slug[:80]}.md"

            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                f.write(markdown)
                tmp_path = f.name

            ingest_queue.enqueue(
                filename=filename, file_path=tmp_path,
                acl_groups=acl_groups, uploaded_by="web-crawler",
                category=category, application_id=app_id,
                auto_categorize=not bool(category),
                build_graph=True,
            )
            pages_ingested += 1
            logger.info(f"Crawled: {url} → {filename} ({len(markdown)} chars)")

            # Follow links if within depth
            if depth < crawl_depth:
                links = await asyncio.to_thread(_extract_links, html, url, url_pattern)
                for link in links:
                    if _clean_url(link) not in visited:
                        to_visit.append((link, depth + 1))

        except Exception as e:
            errors.append(f"{url}: {str(e)}")
            logger.warning(f"Crawl error: {url}: {e}")

    # Update connector stats
    await metadata_store.update_web_connector(
        connector.id,
        last_crawl=datetime.now(timezone.utc),
        pages_found=pages_found,
        pages_ingested=pages_ingested,
    )

    return {
        "pages_found": pages_found,
        "pages_ingested": pages_ingested,
        "files_downloaded": files_downloaded,
        "errors": errors,
    }


async def _download_and_ingest_file(url, acl_groups, category, app_id, ingest_queue, metadata_store):
    """Download a file from a URL and queue it for ingestion."""
    resp = requests.get(url, timeout=60, headers={
        "User-Agent": "SAURON/1.0 (Enterprise RAG Crawler)",
    })
    resp.raise_for_status()

    # Check content hash
    content_hash = hashlib.sha256(resp.content).hexdigest()
    existing = await metadata_store.find_by_content_hash(content_hash)
    if existing:
        logger.debug(f"Skipping file {url} — already ingested")
        return

    filename = Path(urlparse(url).path).name or "download"
    suffix = Path(filename).suffix or ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(resp.content)
        tmp_path = f.name

    ingest_queue.enqueue(
        filename=filename, file_path=tmp_path,
        acl_groups=acl_groups, uploaded_by="web-crawler",
        category=category, application_id=app_id,
        auto_categorize=not bool(category),
        build_graph=True,
    )
    logger.info(f"Downloaded: {url} → {filename} ({len(resp.content)} bytes)")
