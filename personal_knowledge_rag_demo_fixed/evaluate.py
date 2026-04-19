from __future__ import annotations

from datetime import datetime

import pandas as pd

from rag.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL_NAME,
    FIXED_COLLECTION,
    OUTPUTS_DIR,
    QUESTIONS_FILE,
    RECURSIVE_COLLECTION,
)
from rag.embedder import Embedder
from rag.evaluation import evaluate_single_question
from rag.retriever import Retriever
from rag.utils import ensure_dir, read_json
from rag.vector_store import VectorStore


STRATEGIES = {
    "fixed": FIXED_COLLECTION,
    "recursive": RECURSIVE_COLLECTION,
}


def evaluate_strategy(strategy: str, collection_name: str, questions: list[dict]) -> pd.DataFrame:
    embedder = Embedder(EMBEDDING_MODEL_NAME)
    vector_store = VectorStore(str(CHROMA_DIR), collection_name)
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    rows = []
    for item in questions:
        query = item["question"]
        gold_doc_ids = set(item["gold_doc_ids"])
        results = retriever.retrieve(query, top_k=3)
        metrics = evaluate_single_question(results, gold_doc_ids)

        rows.append(
            {
                "strategy": strategy,
                "question_id": item["id"],
                "question": query,
                "gold_doc_ids": ", ".join(item["gold_doc_ids"]),
                "top1_doc": results[0]["doc_id"] if results else "",
                "top3_docs": " | ".join(r["doc_id"] for r in results),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    ensure_dir(OUTPUTS_DIR)
    questions = read_json(QUESTIONS_FILE)

    all_frames = []
    for strategy, collection_name in STRATEGIES.items():
        frame = evaluate_strategy(strategy, collection_name, questions)
        all_frames.append(frame)

    result_df = pd.concat(all_frames, ignore_index=True)

    summary_df = (
        result_df.groupby("strategy")[["hit@1", "hit@3", "mrr", "relevant_in_top3"]]
        .mean()
        .reset_index()
        .rename(columns={"relevant_in_top3": "avg_relevant_in_top3"})
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_path = OUTPUTS_DIR / f"retrieval_eval_detailed_{timestamp}.csv"
    summary_path = OUTPUTS_DIR / f"retrieval_eval_summary_{timestamp}.csv"

    result_df.to_csv(detailed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nDetailed Results\n")
    print(result_df.to_string(index=False))
    print("\nSummary\n")
    print(summary_df.to_string(index=False))
    print(f"\nSaved detailed report to: {detailed_path}")
    print(f"Saved summary report to: {summary_path}")


if __name__ == "__main__":
    main()
