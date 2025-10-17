# Contributing to Syntax-Aware-RAG

Thank you for your interest in contributing to Syntax-Aware-RAG! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/Syntax-Aware-RAG.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Install development dependencies: `pip install -e ".[dev]"`

## Development Workflow

### Setting Up Your Environment

```bash
# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"

# Download required models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"
```

### Code Quality

We maintain high code quality standards using automated tools:

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/

# Run all quality checks
make lint  # or run individually
```

### Testing

All contributions must include tests:

```bash
# Run all tests
pytest

# Run with coverage (must be ≥85%)
pytest --cov=syntax_aware_rag --cov-report=html --cov-report=term

# Run specific test types
pytest tests/unit/
pytest tests/integration/
pytest -m "not slow"  # Skip slow tests

# Run property-based tests
pytest tests/unit/test_chunking_properties.py
```

### Writing Tests

- Unit tests go in `tests/unit/`
- Integration tests go in `tests/integration/`
- Use descriptive test names: `test_sentence_chunker_preserves_unicode`
- Include property-based tests for invariants (e.g., chunk boundaries)
- Mock external dependencies appropriately

Example test structure:

```python
import pytest
from syntax_aware_rag.chunking import SentenceChunker

class TestSentenceChunker:
    def test_basic_chunking(self):
        chunker = SentenceChunker(max_tokens=100)
        text = "First sentence. Second sentence."
        chunks = chunker.chunk(text)
        assert len(chunks) == 2

    @pytest.mark.parametrize("text,expected", [
        ("Single sentence.", 1),
        ("Two sentences. Here they are.", 2),
    ])
    def test_sentence_count(self, text, expected):
        chunker = SentenceChunker(max_tokens=100)
        chunks = chunker.chunk(text)
        assert len(chunks) == expected
```

## Contribution Guidelines

### Commit Messages

Write clear, descriptive commit messages:

- Use present tense: "Add feature" not "Added feature"
- Keep first line under 72 characters
- Reference issues when applicable: "Fix retrieval bug (#123)"
- Provide context in the body for complex changes

Good commit messages:
```
Add layout-aware chunker for PDF documents

Implements a new chunker that preserves document structure
from PDF files, including tables, headings, and sections.
Uses unstructured library for extraction.

Fixes #45
```

### Pull Requests

1. **Small, focused changes**: Each PR should address a single concern
2. **Descriptive titles**: Clearly state what the PR does
3. **Comprehensive description**: Explain the why and how
4. **Tests included**: All new code must have tests
5. **Documentation updated**: Update docs if behavior changes
6. **Clean history**: Squash fixup commits before merging

PR template:
```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- Bullet point list of changes

## Testing
How was this tested?

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code formatted (black)
- [ ] Linting passes (ruff)
- [ ] Type checking passes (mypy)
- [ ] All tests pass
```

### Documentation

Update documentation for:
- New features
- API changes
- Configuration options
- Breaking changes

Documentation is in:
- `README.md`: High-level overview
- `docs/`: Detailed guides
- Docstrings: All public APIs
- Type hints: All function signatures

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

## Areas for Contribution

### High Priority

- Additional chunking strategies
- Support for more embedding models
- Vector database backends (Qdrant, Milvus, etc.)
- Performance optimizations
- Additional evaluation metrics
- Multi-language support

### Good First Issues

Look for issues labeled `good-first-issue` in the issue tracker.

### Feature Requests

1. Check if the feature already exists or is planned
2. Open an issue describing the feature
3. Discuss the approach before implementing
4. Implement after consensus is reached

## Code Style

### Python Style

- Follow PEP 8
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use descriptive variable names
- Add docstrings to all public APIs

Example:

```python
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Chunk:
    """Represents a text chunk with metadata.

    Attributes:
        text: The chunk text content
        start: Start position in original document
        end: End position in original document
        metadata: Optional metadata dictionary
    """
    text: str
    start: int
    end: int
    metadata: Optional[dict] = None

def chunk_text(
    text: str,
    max_tokens: int = 512,
    overlap: int = 50
) -> List[Chunk]:
    """Split text into overlapping chunks.

    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk
        overlap: Number of overlapping tokens between chunks

    Returns:
        List of Chunk objects

    Raises:
        ValueError: If max_tokens <= overlap
    """
    if max_tokens <= overlap:
        raise ValueError("max_tokens must be greater than overlap")
    # Implementation...
```

### Testing Style

- Use descriptive test names
- Follow Arrange-Act-Assert pattern
- One assertion per test (generally)
- Use fixtures for common setup
- Mock external dependencies

## Performance Considerations

- Profile before optimizing
- Include benchmarks for performance-critical code
- Document time/space complexity for algorithms
- Consider memory usage for large-scale operations

## Security

- Never commit credentials or secrets
- Validate all user inputs
- Use parameterized queries for databases
- Report security issues privately to the maintainers

## Getting Help

- Open an issue for bugs or features
- Use GitHub Discussions for questions
- Check existing issues before creating new ones
- Be respectful and constructive

## Recognition

Contributors will be recognized in:
- Release notes
- CHANGELOG.md
- GitHub contributors page

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
