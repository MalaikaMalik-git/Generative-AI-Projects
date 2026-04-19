from __future__ import annotations

import argparse

from rag.chunkers import FixedChunker, RecursiveChunker
from rag.config import (
    CHROMA_DIR,
    DATA_DIR,
    EMBEDDING_MODEL_NAME,
    FIXED_CHUNK_OVERLAP,
    FIXED_CHUNK_SIZE,
    FIXED_COLLECTION,
    RECURSIVE_CHUNK_OVERLAP,
    RECURSIVE_CHUNK_SIZE,
    RECURSIVE_COLLECTION,
)
from rag.embedder import Embedder
from rag.loaders import load_documents
from rag.utils import ensure_dir
from rag.vector_store import VectorStore


def main(strategy: str) -> None:
    ensure_dir(CHROMA_DIR)
    documents = load_documents(DATA_DIR)
    if not documents:
        raise ValueError(f"No supported documents found in {DATA_DIR}")

    embedder = Embedder(EMBEDDING_MODEL_NAME)

    if strategy == "fixed":
        chunker = FixedChunker(chunk_size=FIXED_CHUNK_SIZE, chunk_overlap=FIXED_CHUNK_OVERLAP)
        collection_name = FIXED_COLLECTION
    elif strategy == "recursive":
        chunker = RecursiveChunker(chunk_size=RECURSIVE_CHUNK_SIZE, chunk_overlap=RECURSIVE_CHUNK_OVERLAP)
        collection_name = RECURSIVE_COLLECTION
    else:
        raise ValueError("strategy must be 'fixed' or 'recursive'")

    vector_store = VectorStore(str(CHROMA_DIR), collection_name)
    vector_store.reset_collection()

    all_chunks = []
    for document in documents:
        all_chunks.extend(chunker.chunk(document))

    embeddings = embedder.encode([chunk.text for chunk in all_chunks])
    vector_store.add_chunks(all_chunks, embeddings)

    print(f"Indexed {len(documents)} documents into collection '{collection_name}'.")
    print(f"Created {len(all_chunks)} chunks using strategy '{strategy}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Chroma.")
    parser.add_argument("--strategy", choices=["fixed", "recursive"], required=True)
    args = parser.parse_args()
    main(args.strategy)
