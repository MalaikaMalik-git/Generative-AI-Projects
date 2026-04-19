# RAGAS Evaluation Update

This update expands `ragas_eval.py` from two metrics to a broader RAGAS suite.

## Added metrics

- Faithfulness
- Answer Relevancy
- Answer Correctness
- Context Precision
- Context Utilization
- Context Entity Recall
- Factual Correctness
- ID-Based Context Precision
- ID-Based Context Recall

## Why these were added

The original file only measured:
- Faithfulness
- Context Precision

That covered grounding and retriever ranking quality, but it did not evaluate:
- how relevant the answer is to the question,
- how correct the answer is against the ground truth,
- how much of the retrieved context is actually used,
- whether key entities were recovered,
- or ID-level retrieval precision/recall against the gold document IDs.

## Files to replace

- Replace your old `ragas_eval.py` with `ragas_eval_updated.py`
- Replace your old `requirements.txt` with `requirements_updated.txt`

## New output columns

The updated detailed CSV now includes:
- faithfulness
- answer_relevancy
- answer_correctness
- context_precision
- context_utilization
- context_entity_recall
- factual_correctness
- id_context_precision
- id_context_recall

## Run steps

```bash
pip install -r requirements.txt
python ingest.py --strategy fixed
python ingest.py --strategy recursive
python ragas_eval.py
```

## Important note

This implementation targets the newer collections-based RAGAS API and uses:
- an evaluator LLM via `llm_factory(...)`
- evaluator embeddings via `embedding_factory(...)`
- `SingleTurnSample` for the full metric bundle

If your local environment still has an older RAGAS release, update dependencies first.
