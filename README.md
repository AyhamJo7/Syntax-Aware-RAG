# Syntax-Aware-RAG

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/AyhamJo7/Syntax-Aware-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/AyhamJo7/Syntax-Aware-RAG/actions/workflows/ci.yml)

<p align="center">
  <img src="src/Syntax-Aware-RAG.png" alt="Syntax-Aware-RAG Architecture" width="600"/>
</p>

Advanced Retrieval-Augmented Generation system with syntax-aware chunking, hierarchical embeddings, and multi-granularity retrieval.

## Overview

Syntax-Aware-RAG is a state-of-the-art Retrieval-Augmented Generation (RAG) system that leverages hierarchical document structures and multi-granularity retrieval to improve the quality and precision of information retrieval. Unlike traditional RAG systems that use fixed-size chunking, Syntax-Aware-RAG respects document syntax and semantics, creating a hierarchical tree structure for more intelligent retrieval.

## Features

- **Advanced Chunking Engine**: Pluggable chunkers supporting sentence-based, recursive character splitting, and layout-aware parsing for PDFs/HTML
- **Hierarchical Indexing**: Multi-level document tree (document → sections → paragraphs → sentences) with dense embeddings at each node
- **Multi-Granularity Retrieval**: Two-stage retrieval with broad search on high-level nodes and fine search within top-k parents
- **Context Construction**: Transparent context budgeting with diversity optimization
- **Comparative Evaluation**: Needle-in-a-haystack harness with comprehensive metrics (precision, recall, MRR, nDCG)
- **Production-Ready**: FastAPI service, CLI, Docker support, and comprehensive observability
- **Type-Safe**: Fully typed codebase with mypy compliance for better code quality
- **Well-Tested**: Comprehensive test suite with unit and integration tests

## Requirements

- Python 3.11 or higher
- FAISS (CPU or GPU version)
- spaCy with language models
- sentence-transformers
- Other dependencies listed in `pyproject.toml`

## Installation

### From PyPI

```bash
pip install syntax-aware-rag
```

### From Source

```bash
git clone https://github.com/AyhamJo7/Syntax-Aware-RAG.git
cd Syntax-Aware-RAG
pip install -e ".[dev]"
```

### Download Language Models

```bash
# Download spaCy language model
python -m spacy download en_core_web_sm

# Download NLTK data (optional, fallback for sentence splitting)
python -c "import nltk; nltk.download('punkt')"
```

## Quick Start

### CLI Usage

```bash
# Ingest documents
synrag ingest --input ./data/documents --config configs/default.yaml

# Build index
synrag index --input ./data/documents --output ./indexes/my_index

# Query
synrag query --index ./indexes/my_index --query "What is the main topic?"

# Retrieve without generation
synrag retrieve --index ./indexes/my_index --query "Key findings" --top-k 5

# Run evaluation
synrag eval --index ./indexes/my_index --test-data ./data/eval.json
```

### Python API

```python
from syntax_aware_rag.chunking import SentenceChunker
from syntax_aware_rag.embedding import HierarchicalEmbedder
from syntax_aware_rag.index import FAISSIndex
from syntax_aware_rag.retrieve import MultiGranularityRetriever

# Initialize components
chunker = SentenceChunker(max_tokens=512, overlap=50)
embedder = HierarchicalEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
index = FAISSIndex()
retriever = MultiGranularityRetriever(index=index, embedder=embedder)

# Process document
with open("document.txt") as f:
    text = f.read()

chunks = chunker.chunk(text)
tree = embedder.build_hierarchy(chunks)
index.add_documents([tree])

# Retrieve
results = retriever.retrieve("Your query here", top_k=5)
for result in results:
    print(f"Score: {result.score:.4f}")
    print(f"Text: {result.text[:200]}...")
```

### FastAPI Service

```bash
# Start service
uvicorn syntax_aware_rag.api.main:app --host 0.0.0.0 --port 8000

# Or use Docker
docker-compose up
```

Example API usage:

```bash
# Health check
curl http://localhost:8000/healthz

# Ingest document
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document text here...", "doc_id": "doc1"}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "top_k": 5}'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Document Ingestion                      │
│  (TXT, MD, PDF, HTML) → Chunkers → Hierarchical Structure  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Hierarchical Indexing                     │
│  Document → Sections → Paragraphs → Sentences              │
│  (Dense embeddings at each level)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Multi-Granularity Retrieval                    │
│  Stage A: Broad search (document/section level)            │
│  Stage B: Fine search (sentence level within top parents)  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Context Construction & Generation               │
│  Budget management → Diversity optimization → LLM          │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

Create a `config.yaml` file:

```yaml
chunking:
  type: sentence  # Options: sentence, recursive, layout-aware
  max_tokens: 512
  overlap: 50
  language: en

embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  batch_size: 32
  device: cpu  # or cuda

index:
  backend: faiss  # Options: faiss, elasticsearch, qdrant
  dimension: 384
  metric: cosine

retrieval:
  stages:
    - level: document
      top_k: 10
    - level: sentence
      top_k: 5
  hybrid: true
  bm25_weight: 0.3

context:
  max_tokens: 2048
  include_parent: true
  diversity_penalty: 0.1
```

## Benchmarks

Performance on needle-in-a-haystack evaluation (1000 documents, 10 needles):

| Chunker Type | Precision | Recall | MRR | nDCG | Latency (P50) | Latency (P90) |
|--------------|-----------|--------|-----|------|---------------|---------------|
| Fixed 512    | 0.72      | 0.68   | 0.65| 0.70 | 145ms         | 312ms         |
| Sentence     | 0.84      | 0.81   | 0.79| 0.82 | 152ms         | 328ms         |
| Recursive    | 0.81      | 0.78   | 0.76| 0.79 | 148ms         | 318ms         |
| Layout-Aware | 0.88      | 0.85   | 0.83| 0.86 | 167ms         | 345ms         |

*Results on 10-core Intel i9, 32GB RAM, sentence-transformers/all-MiniLM-L6-v2 model*

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/AyhamJo7/Syntax-Aware-RAG.git
cd Syntax-Aware-RAG

# Install dependencies
pip install -e ".[dev]"

# Download models
python -m spacy download en_core_web_sm
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=syntax_aware_rag --cov-report=html

# Run specific test suite
pytest tests/unit/
pytest tests/integration/

# Run benchmarks
python benchmarks/scripts/run_needle_test.py
```

### Code Quality

```bash
# Lint
ruff check src/ tests/

# Format
black src/ tests/

# Type check
mypy src/
```

### Docker

```bash
# Build CPU image
docker build -f docker/Dockerfile -t syntax-aware-rag:latest .

# Build GPU image
docker build -f docker/Dockerfile.gpu -t syntax-aware-rag:gpu .

# Run with docker-compose
docker-compose up
```

## Documentation

- [Chunking Strategies](docs/chunking.md)
- [Hierarchical Indexing](docs/indexing.md)
- [Multi-Granularity Retrieval](docs/retrieval.md)
- [Evaluation Methodology](docs/evaluation.md)
- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

Copyright 2025 Mhd Ayham Jourman

## Citation

```bibtex
@software{syntax_aware_rag2025,
  author = {Jourman, Mhd Ayham},
  title = {Syntax-Aware-RAG: Advanced Retrieval-Augmented Generation with Hierarchical Indexing},
  year = {2025},
  url = {https://github.com/AyhamJo7/Syntax-Aware-RAG}
}
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.
