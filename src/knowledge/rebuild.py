"""Graph-only recovery from retained indexed passages; never modifies source documents."""


def read_indexed_graph_text(vector_store, doc_id: str) -> str:
    # Paginate rather than using get_chunks_by_doc's 200-chunk default. Read one
    # tier only, so the same text isn't multiplied across four embedding tiers.
    chunks = []
    offset = 0
    while True:
        page, more = vector_store.read_document_page(doc_id, ["ALL"], offset=offset, limit=100)
        chunks.extend(page)
        if not more:
            break
        if not page:
            raise RuntimeError("Stored passage pagination made no progress")
        offset += len(page)
    chunks.sort(key=lambda c: (c.metadata.chunk_index, c.metadata.start_char))
    seen = set()
    parts = []
    for chunk in chunks:
        text = chunk.text
        # Remove the repeated synthetic document summary, not source content.
        if text.startswith("Document:") and "\n\n" in text:
            text = text.split("\n\n", 1)[1]
        text = text.strip()
        if text and text not in seen:
            seen.add(text)
            parts.append(text)
    return "\n\n".join(parts)
