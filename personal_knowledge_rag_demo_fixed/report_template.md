# Week 4 Demo Report — Personal Knowledge RAG

## 1. Goal
Build a working RAG app on a curated real-world document set and compare two chunking strategies using measurable retrieval quality.

## 2. Dataset
- Number of documents:
- Domain:
- Why this dataset was chosen:
- Document formats:

## 3. Pipeline
- Loader
- Chunker A: fixed-size
- Chunker B: recursive / structure-aware
- Embedding model
- Vector store
- Retriever
- Generator (optional)

## 4. Chunking Setup
### Fixed Chunking
- Chunk size:
- Overlap:
- Reasoning:

### Recursive Chunking
- Chunk size:
- Overlap:
- Reasoning:

## 5. Evaluation Questions
List your five questions and their expected gold source documents.

## 6. Metrics
- Hit@1
- Hit@3
- MRR
- Average Relevant in Top-3

## 7. Results
Paste the summary table from `evaluate.py`.

## 8. Analysis
- Which chunking strategy performed better?
- On which questions did fixed chunking fail?
- On which questions did recursive chunking help?
- Were there cases where both worked equally well?
- What kinds of questions exposed weaknesses in your retriever?

## 9. Final Judgment
State which strategy you would choose for production and why.

## 10. Next Improvements
- Hybrid retrieval
- Metadata filtering
- Reranking
- Better evaluation set
- Stronger answer generation model
