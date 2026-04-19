from __future__ import annotations

from typing import Dict, List

import chromadb
from chromadb.api.models.Collection import Collection

from rag.chunkers import Chunk


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name
        self.collection: Collection = self.client.get_or_create_collection(name=collection_name)

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        self.collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "doc_id": chunk.doc_id,
                    "source": chunk.source,
                    "strategy": chunk.strategy,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ],
        )

    def query(self, query_embedding: List[float], top_k: int = 3) -> Dict:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
