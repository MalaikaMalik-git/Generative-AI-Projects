from __future__ import annotations

from typing import Dict, List


def distance_to_similarity(distance: float) -> float:
    return max(0.0, 1.0 - float(distance))


class Retriever:
    def __init__(self, embedder, vector_store) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        query_embedding = self.embedder.encode([query])[0]
        raw = self.vector_store.query(query_embedding=query_embedding, top_k=top_k)

        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        results: List[Dict] = []
        for rank, (doc, meta, distance) in enumerate(zip(documents, metadatas, distances), start=1):
            results.append(
                {
                    "rank": rank,
                    "text": doc,
                    "doc_id": meta["doc_id"],
                    "source": meta["source"],
                    "strategy": meta["strategy"],
                    "chunk_index": meta["chunk_index"],
                    "distance": float(distance),
                    "similarity": round(distance_to_similarity(float(distance)), 4),
                }
            )
        return results
