"""
Session 2 - Retrieval sanity check.

Runs a handful of sample questions against the ChromaDB collection built
by ingest.py and prints the top matching chunks. This is a pure retrieval
test (no LLM call) — it proves RAG can find the right grounding text before
Session 3 wires it into an actual chat answer.

The project plan's Session 2 "done when" is 3 company questions returning
grounded answers — this script is how you check that before building the
chat endpoint around it.

Usage:
    python scripts/test_retrieval.py
"""

import os

import chromadb
from chromadb.utils import embedding_functions

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
COLLECTION_NAME = "company_knowledge"

SAMPLE_QUESTIONS = [
    "What does Agentix System do?",
    "What AI agents does Agentix offer?",
    "Where is Agentix System located and how do I contact them?",
    "What tools can Agentix agents integrate with?",
    "When was Agentix System founded?",
]

TOP_K = 3


def main():
    if not os.path.isdir(DB_DIR) or not os.listdir(DB_DIR):
        print(
            "No ChromaDB data found. Run `python scripts/ingest.py` first."
        )
        return

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)

    print(f"Collection has {collection.count()} chunks stored.\n")

    for question in SAMPLE_QUESTIONS:
        print("=" * 70)
        print(f"Q: {question}")
        results = collection.query(query_texts=[question], n_results=TOP_K)

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
            preview = doc.replace("\n", " ")[:160]
            print(f"\n  #{rank} (distance={dist:.3f}, source={meta.get('source')})")
            print(f"     {preview}...")
        print()

    print(
        "Sanity check: for each question above, does at least the #1 result "
        "actually contain the answer? If yes, retrieval is working and "
        "you're ready for Session 3 (wiring this into the /chat endpoint)."
    )


if __name__ == "__main__":
    main()
