"""
Retrieval helper used by the /chat endpoint.

Wraps the ChromaDB collection built by scripts/ingest.py so main.py doesn't
need to know about embedding functions or DB paths directly.
"""

import os

import chromadb
from chromadb.utils import embedding_functions

DB_DIR = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
COLLECTION_NAME = "company_knowledge"

_collection = None  # lazy-loaded singleton so the model only loads once


class RagNotReadyError(RuntimeError):
    """Raised when the ChromaDB collection hasn't been built yet."""


def get_collection():
    global _collection
    if _collection is not None:
        return _collection

    db_file = os.path.join(DB_DIR, "chroma.sqlite3")
    if not os.path.isfile(db_file):
        raise RagNotReadyError(
            "No ChromaDB data found. Run `python scripts/ingest.py` first."
        )

    try:
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        client = chromadb.PersistentClient(path=DB_DIR)
        _collection = client.get_collection(
            COLLECTION_NAME, embedding_function=embed_fn
        )
    except RagNotReadyError:
        raise
    except Exception as exc:
        raise RagNotReadyError(
            f"Could not load the RAG collection ({exc}). Make sure "
            f"`pip install -r requirements.txt` finished successfully and "
            f"`python scripts/ingest.py` completed without errors."
        ) from exc

    return _collection


def retrieve_context(question: str, top_k: int = 4) -> list[dict]:
    """Returns top_k chunks as [{"text": ..., "source": ...}, ...]."""
    collection = get_collection()
    results = collection.query(query_texts=[question], n_results=top_k)

    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []

    return [
        {"text": doc, "source": meta.get("source", "unknown")}
        for doc, meta in zip(docs, metas)
    ]
