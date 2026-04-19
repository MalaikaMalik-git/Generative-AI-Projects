import os
import shutil
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

from corpus_data import DOCUMENTS


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "semantic_search_demo"
MODEL_NAME = "all-MiniLM-L6-v2"


def chunk_text(text, source, title, chunk_size=1):
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    chunks = []

    for i in range(0, len(sentences), chunk_size):
        group = sentences[i:i + chunk_size]
        chunk_text_value = ". ".join(group).strip()
        if not chunk_text_value.endswith("."):
            chunk_text_value += "."

        chunk_id = f"{source}_chunk_{(i // chunk_size) + 1}"

        chunks.append({
            "id": chunk_id,
            "source": source,
            "title": title,
            "chunk_text": chunk_text_value
        })

    return chunks


def prepare_chunks(documents):
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(
            text=doc["text"],
            source=doc["source"],
            title=doc["title"],
            chunk_size=1
        )
        all_chunks.extend(chunks)
    return all_chunks


def build_index():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Preparing chunks...")
    chunks = prepare_chunks(DOCUMENTS)

    texts = [item["chunk_text"] for item in chunks]
    ids = [item["id"] for item in chunks]
    metadatas = [
        {
            "source": item["source"],
            "title": item["title"],
            "chunk_id": item["id"]
        }
        for item in chunks
    ]

    print(f"Total chunks prepared: {len(chunks)}")

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    print("Creating persistent Chroma database...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"\n✓ Index built — {len(chunks)} chunks stored in Chroma.")
    print(f"✓ Database saved in: {CHROMA_DIR}/")


if __name__ == "__main__":
    build_index()
