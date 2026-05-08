from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    index: int
    start_char: int


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[Chunk]:
    if not text.strip():
        return []
    paragraphs = text.split("\n\n")
    chunks: list[Chunk] = []
    current_text = ""
    current_start = 0
    char_pos = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            char_pos += 2
            continue

        # If the paragraph itself exceeds chunk_size, split it by words
        if len(para) > chunk_size:
            # Flush any current accumulated text first
            if current_text.strip():
                chunks.append(Chunk(text=current_text, index=len(chunks), start_char=current_start))
                current_text = ""
            # Split the large paragraph into word-level sub-chunks
            words = para.split(" ")
            sub_start = char_pos
            sub_text = ""
            for word in words:
                candidate = sub_text + (" " if sub_text else "") + word
                if len(candidate) > chunk_size and sub_text:
                    chunks.append(Chunk(text=sub_text, index=len(chunks), start_char=sub_start))
                    overlap_text = sub_text[-chunk_overlap:] if len(sub_text) > chunk_overlap else sub_text
                    sub_start = sub_start + len(sub_text) - len(overlap_text)
                    sub_text = overlap_text + " " + word
                else:
                    sub_text = candidate
            if sub_text.strip():
                current_text = sub_text
                current_start = sub_start
            char_pos += len(para) + 2
            continue

        separator = "\n\n" if current_text else ""
        candidate = current_text + separator + para
        if len(candidate) > chunk_size and current_text:
            chunks.append(Chunk(text=current_text, index=len(chunks), start_char=current_start))
            overlap_text = current_text[-chunk_overlap:] if len(current_text) > chunk_overlap else current_text
            current_text = overlap_text + "\n\n" + para
            current_start = char_pos - len(overlap_text)
        else:
            if not current_text:
                current_start = char_pos
            current_text = candidate
        char_pos += len(para) + 2

    if current_text.strip():
        chunks.append(Chunk(text=current_text, index=len(chunks), start_char=current_start))
    return chunks
