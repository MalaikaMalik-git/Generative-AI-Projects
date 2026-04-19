from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from rag.loaders import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    strategy: str
    chunk_index: int


class FixedChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> List[Chunk]:
        text = re.sub(r"\s+", " ", document.text).strip()
        chunks: List[Chunk] = []
        start = 0
        index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}_fixed_{index}",
                        doc_id=document.doc_id,
                        source=document.source,
                        text=chunk_text,
                        strategy="fixed",
                        chunk_index=index,
                    )
                )
            if end == len(text):
                break
            start = end - self.chunk_overlap
            index += 1

        return chunks


class RecursiveChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> List[Chunk]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", document.text) if p.strip()]
        segments: List[str] = []

        for para in paragraphs:
            if len(para) <= self.chunk_size:
                segments.append(para)
                continue

            sentences = re.split(r"(?<=[.!?])\s+", para)
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) + 1 <= self.chunk_size:
                    current = f"{current} {sentence}".strip()
                else:
                    if current:
                        segments.append(current)
                    current = sentence.strip()
            if current:
                segments.append(current)

        packed_chunks: List[str] = []
        current_chunk = ""

        for segment in segments:
            candidate = f"{current_chunk}\n\n{segment}".strip() if current_chunk else segment
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    packed_chunks.append(current_chunk)
                current_chunk = segment

        if current_chunk:
            packed_chunks.append(current_chunk)

        if self.chunk_overlap > 0 and len(packed_chunks) > 1:
            overlapped_chunks: List[str] = []
            for i, chunk in enumerate(packed_chunks):
                if i == 0:
                    overlapped_chunks.append(chunk)
                    continue
                previous_tail = packed_chunks[i - 1][-self.chunk_overlap :].strip()
                overlapped_chunks.append(f"{previous_tail}\n\n{chunk}".strip())
            packed_chunks = overlapped_chunks

        return [
            Chunk(
                chunk_id=f"{document.doc_id}_recursive_{index}",
                doc_id=document.doc_id,
                source=document.source,
                text=chunk_text,
                strategy="recursive",
                chunk_index=index,
            )
            for index, chunk_text in enumerate(packed_chunks)
            if chunk_text.strip()
        ]
