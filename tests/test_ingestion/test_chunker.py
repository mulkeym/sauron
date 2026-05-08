import pytest
from src.ingestion.chunker import chunk_text, Chunk


def test_short_text_single_chunk():
    chunks = chunk_text("Hello world.", chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert chunks[0].start_char == 0


def test_long_text_multiple_chunks():
    text = "Word " * 200
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        end_of_current = chunks[i].text[-40:]
        assert end_of_current in chunks[i + 1].text


def test_respects_paragraph_boundaries():
    text = "First paragraph with some content.\n\nSecond paragraph with different content.\n\nThird paragraph here."
    chunks = chunk_text(text, chunk_size=60, chunk_overlap=10)
    for chunk in chunks:
        stripped = chunk.text.strip()
        assert not stripped.startswith(" ")


def test_chunk_metadata():
    text = "Some text here.\n\nMore text below."
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert chunks[0].index == 0
    assert chunks[0].start_char == 0


def test_empty_text():
    chunks = chunk_text("", chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 0
