from __future__ import annotations

from typing import Dict, List, Set


def hit_at_k(retrieved_doc_ids: List[str], gold_doc_ids: Set[str], k: int) -> int:
    return int(any(doc_id in gold_doc_ids for doc_id in retrieved_doc_ids[:k]))


def reciprocal_rank(retrieved_doc_ids: List[str], gold_doc_ids: Set[str]) -> float:
    for index, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_doc_ids:
            return 1.0 / index
    return 0.0


def relevant_in_top_k(retrieved_doc_ids: List[str], gold_doc_ids: Set[str], k: int) -> int:
    return sum(1 for doc_id in retrieved_doc_ids[:k] if doc_id in gold_doc_ids)


def evaluate_single_question(results: List[Dict], gold_doc_ids: Set[str]) -> Dict:
    retrieved_doc_ids = [item["doc_id"] for item in results]

    return {
        "hit@1": hit_at_k(retrieved_doc_ids, gold_doc_ids, 1),
        "hit@3": hit_at_k(retrieved_doc_ids, gold_doc_ids, 3),
        "mrr": reciprocal_rank(retrieved_doc_ids, gold_doc_ids),
        "relevant_in_top3": relevant_in_top_k(retrieved_doc_ids, gold_doc_ids, 3),
    }
