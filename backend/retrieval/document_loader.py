"""Document loading — extract text from real files and chunk it.

Supports .pdf (via PyMuPDF), .txt and .md. Chunking is
paragraph-aware with a character cap and overlap, so a chunk keeps
local context while staying small enough to embed and cite.
"""
from __future__ import annotations

from pathlib import Path

MAX_CHUNK_CHARS = 1200
OVERLAP_CHARS = 150


def extract_text(path: str | Path) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        try:
            import pymupdf
        except ImportError:
            raise ValueError(
                "PDF support needs PyMuPDF: pip install pymupdf")
        with pymupdf.open(p) as doc:
            return "\n".join(page.get_text() for page in doc)
    if suffix in (".txt", ".md"):
        return p.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported document type: {suffix or '(none)'} "
                     "— supported: .pdf, .txt, .md")


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS,
               overlap: int = OVERLAP_CHARS) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > max_chars:
            # A single huge paragraph: flush, then hard-split it with
            # overlap so no content is lost at the seams.
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(para):
                chunks.append(para[start:start + max_chars])
                start += max_chars - overlap
            continue
        candidate = f"{current}\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def load_chunks(path: str | Path) -> list[dict]:
    """A file -> retrieval-ready chunk dicts tagged with their source."""
    p = Path(path)
    return [{"title": f"{p.name} (part {i + 1})", "text": chunk,
             "source": p.name}
            for i, chunk in enumerate(chunk_text(extract_text(p)))]
