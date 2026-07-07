# Syntax-Aware-RAG Agent Notes

Syntax-Aware-RAG is a research portfolio repo for syntax-aware chunking,
hierarchical indexing, and multi-granularity retrieval.

Read local `CLAUDE.md` if present; in this workspace it may be untracked.

## Architecture

The core idea is to avoid fixed-size chunking when document syntax gives better
retrieval boundaries:

- chunkers: sentence, recursive, layout-aware PDF/HTML parsing.
- hierarchy: document -> section -> paragraph -> sentence nodes.
- embeddings: dense vectors at multiple hierarchy levels.
- retrieval: broad parent-level search followed by fine retrieval within top
  parents.
- context construction: explicit token budgeting and diversity optimization.
- evaluation: needle-in-a-haystack and retrieval metrics.

## Commands

```bash
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
pytest
pytest --cov=syntax_aware_rag --cov-report=html
ruff check src/ tests/
mypy src/
```

The README still mentions `black`; prefer the repo's current formatter/linter
configuration if `pyproject.toml` says otherwise.

## Editing Rules

- Keep business logic retrieval-agnostic with respect to file syntax; syntax
  should improve chunking/indexing, not leak into downstream generation.
- Preserve the two-stage retrieval model when adding new backends like Qdrant or
  Elasticsearch.
- Do not commit model downloads, generated indexes, benchmark artifacts, or
  local datasets.
- Keep the API and CLI usable as demonstration surfaces for the portfolio.
