"""
Session 2 - Ingest.

Reads every .txt file in data/raw/, splits it into overlapping chunks,
embeds each chunk with a local sentence-transformers model, and stores
everything in a persistent ChromaDB collection on disk.

First run needs internet once to download the embedding model
(~90MB, sentence-transformers/all-MiniLM-L6-v2). After that it's fully
local — no per-query API cost for retrieval.

Usage:
    python scripts/ingest.py
"""

import os
import glob

import chromadb
from chromadb.utils import embedding_functions

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
COLLECTION_NAME = "company_knowledge"

CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # characters shared between consecutive chunks
MIN_CHUNK_WORDS = 15   # skip near-empty chunks (e.g. leftover template text)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple sliding-window character chunker. Good enough for MVP-sized
    marketing/FAQ pages; splitting on paragraph boundaries first keeps
    chunks from cutting mid-sentence too often."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk, carrying overlap from the end of the previous one
            overlap_text = current[-overlap:] if current else ""
            current = f"{overlap_text}\n{para}".strip()

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c.split()) >= MIN_CHUNK_WORDS]


def load_documents() -> list[dict]:
    docs = []
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()

        source_line = text.splitlines()[0] if text else ""
        source_url = (
            source_line.replace("SOURCE:", "").strip()
            if source_line.startswith("SOURCE:")
            else os.path.basename(path)
        )

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        filename = os.path.basename(path)

        if not chunks:
            print(f"  [SKIPPED] {filename} — no usable content yet (still a template?)")
            continue

        for i, chunk in enumerate(chunks):
            docs.append(
                {
                    "id": f"{filename}::chunk-{i}",
                    "text": chunk,
                    "source": source_url,
                    "file": filename,
                }
            )
        print(f"  {filename}: {len(chunks)} chunks")

    return docs


def main():
    os.makedirs(DB_DIR, exist_ok=True)

    print("Loading + chunking documents from data/raw/ ...")
    docs = load_documents()

    if not docs:
        print(
            "\nNo usable content found. Fill in data/raw/shammarianas_home.txt "
            "(and confirm the agentixsystem_*.txt files are present) before "
            "running this again."
        )
        return

    print(f"\nTotal chunks: {len(docs)}")

    print("Loading embedding model (downloads on first run, then cached)...")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=DB_DIR)

    # Fresh collection each run so re-ingesting after editing data/raw/
    # doesn't leave stale chunks behind.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME, embedding_function=embed_fn
    )

    collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[{"source": d["source"], "file": d["file"]} for d in docs],
    )

    print(f"\nStored {len(docs)} chunks in ChromaDB at {DB_DIR}")
    print("Run `python scripts/test_retrieval.py` to sanity-check retrieval.")


if __name__ == "__main__":
    main()
