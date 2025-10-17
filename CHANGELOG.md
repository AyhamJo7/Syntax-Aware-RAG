# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-17

### Added

- **Chunking Engine**: Three pluggable chunkers (sentence-based, recursive character, layout-aware)
- **Hierarchical Indexing**: Document tree structure with multi-level embeddings
- **FAISS Integration**: Vector index with cosine similarity and metadata storage
- **Multi-Granularity Retrieval**: Two-stage search (broad + fine-grained)
- **Context Construction**: Token budget management and diversity optimization
- **CLI Interface**: Commands for indexing, querying, and retrieval
- **Unit Tests**: Comprehensive test coverage for chunking module
- **CI/CD Pipeline**: GitHub Actions workflow for testing and linting
- **Documentation**: README, CONTRIBUTING, and inline code documentation

### Components

- `syntax_aware_rag.chunking`: Sentence, recursive, and layout-aware chunkers
- `syntax_aware_rag.embedding`: Hierarchical embedder with document trees
- `syntax_aware_rag.index`: FAISS-based vector index
- `syntax_aware_rag.retrieve`: Multi-granularity retriever
- `syntax_aware_rag.context`: Context builder
- `syntax_aware_rag.cli`: Command-line interface

### Technical Details

- Python 3.11+ support
- Sentence-transformers for embeddings
- spaCy/NLTK for sentence segmentation
- FAISS for vector similarity search
- Configurable chunking parameters
- Metadata tracking through hierarchy
- Persistent index storage

[0.1.0]: https://github.com/AyhamJo7/Syntax-Aware-RAG/releases/tag/v0.1.0
