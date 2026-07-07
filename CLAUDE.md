# CLAUDE.md — Syntax-Aware-RAG

## Research Problem

Fixed-size chunking ignores document structure — a paragraph boundary means nothing to a character splitter. **Syntax-Aware-RAG** parses documents into a hierarchical tree (document → sections → paragraphs → sentences), embeds at each level, and retrieves at the right granularity: broad search on high-level nodes, fine search within the top-k parents.

## Architecture

```
Document
  → Chunking Engine (syntax/structure-aware) — produces hierarchical tree
  → Multi-level Embedding — dense embeddings at each tree node
  → Hierarchical Index (FAISS per level)
  → Two-stage Retrieval: coarse (section-level) → fine (paragraph/sentence-level)
  → Context Construction — budget-aware, diversity-optimized context for LLM
```

Pluggable chunkers: sentence-based, recursive character splitting, layout-aware (PDF/HTML).

## Package Layout

```
src/syntax_aware_rag/
  chunking/     Pluggable chunker implementations
  embedding/    Multi-level embedders
  index/        Hierarchical FAISS index
  retrieve/     Two-stage retrieval logic
  context/      Context budgeting + diversity optimization
  eval/         Benchmark harness (needle-in-haystack, MRR, nDCG)
  api/          FastAPI service
  cli/          CLI entrypoint
```

## Setup & Commands

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# No Makefile — use CLI directly:
synrag --help

# Benchmarks
python benchmarks/scripts/run_benchmark.py    # needle-in-haystack harness
# Results land in benchmarks/results/

# Tests
pytest tests/unit/          # unit tests (no external deps)
pytest tests/integration/   # integration tests (requires embedder models)

# Lint / format / types
ruff format src/ tests/
ruff check src/ tests/
mypy src/
```

Config: `configs/` — YAML files for chunker, embedder, retriever, API settings.

## Evaluation

**Needle-in-haystack harness** — inserts a known fact at varying positions and depths in a document and measures whether retrieval finds it.

Metrics: Precision@k, Recall@k, MRR, nDCG

Run: `python benchmarks/scripts/run_benchmark.py --config configs/benchmark.yaml`

Results saved to `benchmarks/results/`.

## Key Dependencies

- `spacy` — syntax parsing for structure-aware chunking
- `sentence-transformers` — multi-level embeddings
- `faiss-cpu` — hierarchical index
- `fastapi` + `uvicorn` — serving
- Docker config in `docker/` for containerized deployment (CPU and GPU variants)

## Research Notes

The two-stage retrieval is the core contribution — coarse search reduces candidates, fine search within parent nodes adds precision without sacrificing recall. When comparing against baselines (flat chunking), always run with the same embedding model to isolate the effect of the hierarchical structure. The needle-in-haystack harness is designed specifically for this — use it rather than standard BEIR benchmarks which reward different things.
