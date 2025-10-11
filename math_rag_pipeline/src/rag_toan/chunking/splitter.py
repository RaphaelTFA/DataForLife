from typing import List
import re

def naive_token_count(text: str) -> int:
    return len(text.split())

def chunk_by_words(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Simple word-based chunking (approx tokens). chunk_size = ~words per chunk."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

def split_by_headings(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Try split by obvious headings (ALL CAPS or keywords). Fallback to chunk_by_words.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    headings_idx = []
    for i, l in enumerate(lines):
        if re.match(r'^[A-Z0-9\s]{3,}$', l) or re.match(r'^(BÀI|CHƯƠNG|PHẦN|LUYỆN TẬP)\b', l, flags=re.I):
            headings_idx.append(i)
    if not headings_idx:
        return chunk_by_words(text, chunk_size=chunk_size, overlap=overlap)
    chunks = []
    for j, idx in enumerate(headings_idx):
        start = idx
        end = headings_idx[j+1] if j+1 < len(headings_idx) else len(lines)
        section = " ".join(lines[start:end])
        if naive_token_count(section) > chunk_size * 1.5:
            chunks.extend(chunk_by_words(section, chunk_size=chunk_size, overlap=overlap))
        else:
            chunks.append(section)
    return chunks
