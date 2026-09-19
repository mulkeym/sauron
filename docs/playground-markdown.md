# Playground answer rendering

The Playground renders answer Markdown during streaming and after completion, including cached answers. Supported formatting includes paragraphs, headings, emphasis, ordered/unordered lists, blockquotes, tables, links, and fenced code blocks. Wide tables and code blocks scroll within the answer card. The completed server response supplies the final answer, citations, evidence warnings, and source diagram cards.

Markdown parsing and sanitization use pinned browser builds served by Sauron, with no CDN request needed for answer formatting. Licenses, versions, and checksums are recorded in `src/admin/static/vendor/README.md`. These are small static assets, not additional Python services or runtime packages.

Raw HTML is displayed as text. Parsed Markdown is sanitized, unsafe link protocols are removed, and external links open with `noopener noreferrer`. If the sanitizer cannot load, the answer falls back to plain text. Cached answers are HTML-escaped before transport so markup cannot execute before sanitization.

Source diagrams continue through the existing authenticated image endpoints. An inline Markdown image renders only if its URL matches a diagram reference already selected by the backend for the current Playground persona. API image URLs are mapped to the same-origin admin image route. Unknown, remote, and data-URL images are shown as text labels without fetching their bytes. During streaming, images remain labels until the completed response supplies its authorized references. Source figure cards retain provenance and rendering warnings below the answer.

## Verification

Backend regression tests exercise cached and direct answer escaping. Optional Chromium tests use the real Playground template and scripts with simulated query responses, verifying formatted answers, streaming completion, source image requests with session cookies, injection attempts, unsafe URLs, blocked image fetches, and sanitizer failure. Existing image service tests cover backend access control; these browser tests do not call an LLM or verify OpenWebUI.

```sh
python -m pip install playwright
python -m playwright install chromium
python -m pytest tests/test_admin/test_playground_markdown_browser.py -q
```

Alternatively set `SAURON_TEST_BROWSER` to an installed Chromium executable. Browser tests skip when Playwright or its browser is absent. `SAURON_BROWSER_SCREENSHOTS=/tmp/playground-review` saves rendered answer screenshots for review.
