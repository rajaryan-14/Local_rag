# Local RAG System

Local-first RAG with hybrid retrieval, reranking, generation, and evaluation.

## Pipeline

```text
documents → parsing → chunking → dense + sparse indexing
          → hybrid retrieval → cross-encoder reranking
          → local Ollama generation → citations → evaluation
```

The default stack is free to run locally: Python, Qdrant, Sentence
Transformers, a Cross-Encoder reranker, and Ollama. No hosted API key is
required for the retrieval or generation path.

## Local services

- Qdrant: `http://localhost:6333`
- Ollama: `http://localhost:11434`

Start Qdrant:

```powershell
docker compose up -d
```

Create an environment and install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and place source documents under `data/raw/`.

Run the current ingestion tests:

```powershell
py -m pytest
```

Build the processed chunk file:

```powershell
py -m app.ingestion.cli
```

Query the BM25 baseline:

```powershell
py -m app.retrieval.cli "refunds"
```

Dense and hybrid modes use the configured Sentence Transformers embedding model:

```powershell
py -m app.retrieval.cli "refunds" --mode dense
py -m app.retrieval.cli "refunds" --mode hybrid
```

Add `--rerank` after installing the model dependencies to apply the configured
Cross-Encoder reranker.

Generation uses Ollama and only receives the retrieved evidence. Answers are
required to cite chunk IDs or explicitly state that the evidence is insufficient.

Run the deterministic retrieval evaluation:

```powershell
py -m app.evaluation.run_retrieval
```

Evaluation cases live in `data/eval/golden.jsonl` and should contain a question,
relevant chunk IDs, and an optional reference answer.

The local judge is implemented in `app/evaluation/judge.py`. It evaluates answers
against retrieved evidence and returns structured faithfulness results.

Run the complete local pipeline:

```powershell
py -m app.run_rag "What does the sample document say?"
```

Use the persistent Qdrant-backed dense index after building it:

```powershell
py -m app.run_rag "What does the sample document say?" --persistent
```

Use `--no-rerank` or `--no-judge` when you want a faster diagnostic run.

Start the local API:

```powershell
uvicorn app.api:app --reload
```

Then call `GET /health` or `POST /ask` with:

```json
{"question":"What does the sample document say?"}
```

Build a persistent local Qdrant dense index without Docker:

```powershell
py -m app.retrieval.index_cli
```

The index is stored under `data/qdrant/` and can be rebuilt after ingestion.

Generated files under `data/processed/` and `data/qdrant/` are intentionally
ignored by Git. This keeps the repository portable and avoids committing
machine-specific indexes; rebuild them with the commands above after cloning.

The ingestion layer currently supports PDF, TXT, Markdown, and HTML files. Each
chunk retains a stable `chunk_id`, source path, and character offsets so later
answers can cite their evidence.

## Project layout

- `app/ingestion/`: loading, cleaning, chunking, indexing
- `app/retrieval/`: dense, BM25, hybrid fusion, reranking
- `app/generation/`: local model prompts and answer generation
- `app/evaluation/`: golden datasets, retrieval metrics, judge evaluation
- `data/eval/`: human-reviewed evaluation cases
- `data/raw/`: readable sample source documents and their source list
- `experiments/`: optional local run traces and comparison reports

## Evaluation loop

The evaluation harness separates retrieval quality from answer quality:

1. `app.evaluation.run_retrieval` measures whether relevant chunks are found.
2. `app.evaluation.judge` checks whether an answer is supported by retrieved
   evidence and flags unsupported claims.
3. `data/eval/golden.jsonl` provides the human-reviewed questions and expected
   evidence used for repeatable checks.

Use the deterministic tests and the golden set as a baseline before changing
chunking, hybrid fusion, reranking, or prompts.
