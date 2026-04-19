from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Document:
    doc_id: str
    source: str
    text: str


SUPPORTED_EXTENSIONS = {".md", ".txt"}


def load_documents(data_dir: Path) -> List[Document]:
    documents: List[Document] = []

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        doc_id = path.stem
        documents.append(Document(doc_id=doc_id, source=path.name, text=text))

    return documents
