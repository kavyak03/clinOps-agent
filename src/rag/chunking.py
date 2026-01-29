def chunk_text(text: str, max_chars: int = 800, overlap: int = 100):
    text = text.strip()
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+max_chars]
        chunks.append(chunk)
        i += max_chars - overlap
    return chunks
