from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas import SingleTurnSample
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextEntityRecall,
    ContextPrecision,
    ContextUtilization,
    Faithfulness,
    FactualCorrectness,
)

from rag.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL_NAME,
    FIXED_COLLECTION,
    OUTPUTS_DIR,
    QUESTIONS_FILE,
    RECURSIVE_COLLECTION,
)
from rag.embedder import Embedder
from rag.pipeline import RAGPipeline
from rag.retriever import Retriever
from rag.vector_store import VectorStore

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini")).strip()
RAGAS_EMBEDDING_PROVIDER = os.getenv("RAGAS_EMBEDDING_PROVIDER", "openai").strip()
RAGAS_EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small").strip()

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Set it in your .env or export it in the shell.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_pipeline(collection_name: str) -> RAGPipeline:
    embedder = Embedder(EMBEDDING_MODEL_NAME)
    vector_store = VectorStore(str(CHROMA_DIR), collection_name)
    retriever = Retriever(embedder, vector_store)
    return RAGPipeline(retriever)


class MetricSuite:
    def __init__(self) -> None:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.llm = llm_factory(RAGAS_LLM_MODEL, client=client)
        self.embeddings = embedding_factory(
            RAGAS_EMBEDDING_PROVIDER,
            model=RAGAS_EMBEDDING_MODEL,
            client=client,
        )

        # LLM / embeddings based RAG metrics
        self.faithfulness = Faithfulness(llm=self.llm)
        self.answer_relevancy = AnswerRelevancy(llm=self.llm, embeddings=self.embeddings)
        self.answer_correctness = AnswerCorrectness(llm=self.llm, embeddings=self.embeddings)
        self.context_precision = ContextPrecision(llm=self.llm)
        self.context_utilization = ContextUtilization(llm=self.llm)
        self.context_entity_recall = ContextEntityRecall(llm=self.llm)
        self.factual_correctness = FactualCorrectness(llm=self.llm)

        # Deterministic retrieval-ID metrics
        self.id_context_precision = IDBasedContextPrecision()
        self.id_context_recall = IDBasedContextRecall()


async def score_row(
    metrics: MetricSuite,
    *,
    question: str,
    answer: str,
    contexts: list[str],
    reference: str,
    retrieved_context_ids: list[str],
    reference_context_ids: list[str],
) -> dict[str, float]:
    """
    Score a single sample across a broader RAGAS metric suite.

    Notes
    -----
    - Some metrics need only question/response/context.
    - Some additionally need a reference / ground truth answer.
    - ID based metrics measure retrieval quality directly from doc IDs.
    """
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        reference=reference,
        retrieved_contexts=contexts,
        retrieved_context_ids=retrieved_context_ids,
        reference_context_ids=reference_context_ids,
    )

    faithfulness_result = await metrics.faithfulness.single_turn_ascore(sample)
    answer_relevancy_result = await metrics.answer_relevancy.single_turn_ascore(sample)
    answer_correctness_result = await metrics.answer_correctness.single_turn_ascore(sample)
    context_precision_result = await metrics.context_precision.single_turn_ascore(sample)
    context_utilization_result = await metrics.context_utilization.single_turn_ascore(sample)
    context_entity_recall_result = await metrics.context_entity_recall.single_turn_ascore(sample)
    factual_correctness_result = await metrics.factual_correctness.single_turn_ascore(sample)
    id_context_precision_result = await metrics.id_context_precision.single_turn_ascore(sample)
    id_context_recall_result = await metrics.id_context_recall.single_turn_ascore(sample)

    return {
        "faithfulness": float(faithfulness_result),
        "answer_relevancy": float(answer_relevancy_result),
        "answer_correctness": float(answer_correctness_result),
        "context_precision": float(context_precision_result),
        "context_utilization": float(context_utilization_result),
        "context_entity_recall": float(context_entity_recall_result),
        "factual_correctness": float(factual_correctness_result),
        "id_context_precision": float(id_context_precision_result),
        "id_context_recall": float(id_context_recall_result),
    }


async def run_strategy_async(strategy: str, collection_name: str) -> pd.DataFrame:
    pipeline = build_pipeline(collection_name)
    questions = read_json(QUESTIONS_FILE)
    metrics = MetricSuite()

    rows = []
    for item in questions:
        result = pipeline.ask(item["question"], top_k=3)
        contexts = [r["text"] for r in result["results"]]
        retrieved_context_ids = [r["doc_id"] for r in result["results"]]
        reference_context_ids = item.get("gold_doc_ids", [])

        scores = await score_row(
            metrics,
            question=item["question"],
            answer=result["answer"],
            contexts=contexts,
            reference=item["ground_truth"],
            retrieved_context_ids=retrieved_context_ids,
            reference_context_ids=reference_context_ids,
        )

        rows.append(
            {
                "question_id": item["id"],
                "strategy": strategy,
                "question": item["question"],
                "answer": result["answer"],
                "ground_truth": item["ground_truth"],
                "retrieved_doc_ids": " | ".join(retrieved_context_ids),
                "reference_doc_ids": " | ".join(reference_context_ids),
                "contexts": " ||| ".join(contexts),
                **scores,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    strategy_map = {
        "fixed": FIXED_COLLECTION,
        "recursive": RECURSIVE_COLLECTION,
    }

    all_runs = []
    for strategy, collection_name in strategy_map.items():
        df = asyncio.run(run_strategy_async(strategy, collection_name))
        all_runs.append(df)

    final_df = pd.concat(all_runs, ignore_index=True)

    detailed_path = OUTPUTS_DIR / "ragas_eval_detailed.csv"
    final_df.to_csv(detailed_path, index=False)

    metric_columns = [
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "context_precision",
        "context_utilization",
        "context_entity_recall",
        "factual_correctness",
        "id_context_precision",
        "id_context_recall",
    ]

    summary = final_df.groupby("strategy")[metric_columns].mean().reset_index()

    summary_path = OUTPUTS_DIR / "ragas_eval_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nDetailed RAGAS Results\n")
    print(final_df.to_string(index=False))

    print("\nSummary\n")
    print(summary.to_string(index=False))

    print(f"\nSaved detailed report to: {detailed_path}")
    print(f"Saved summary report to: {summary_path}")


if __name__ == "__main__":
    main()
