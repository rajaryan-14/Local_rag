# Data workflow

This directory intentionally keeps small, readable source and evaluation files
in Git. Generated chunks and vector indexes are ignored and can be rebuilt at
any time.

## Source documents

Place PDF, TXT, Markdown, or HTML files under `data/raw/`.

Then rebuild processed chunks:

```powershell
py -m app.ingestion.cli
```

Rebuild the persistent dense index:

```powershell
py -m app.retrieval.index_cli
```

## Evaluation cases

Add one JSON object per line to `data/eval/golden.jsonl`:

```json
{"question":"What is the refund period?","relevant_chunk_ids":["policy_chunk_0003"],"reference_answer":"Refunds are available within 30 days."}
```

Relevant chunk IDs can be found in `data/processed/chunks.jsonl`.

The included `golden.jsonl` file is a small starter set. Add human-reviewed
questions and expected evidence as the corpus grows; keep one JSON object per
line so evaluation runs remain diffable and easy to review.
