"""Type definitions for chunking components."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChunkType(Enum):
    """Types of chunks in the hierarchy."""
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"


@dataclass
class Chunk:
    """Represents a text chunk with metadata.

    Attributes:
        text: The chunk text content
        start: Start character position in original document
        end: End character position in original document
        chunk_type: Type of chunk in hierarchy
        metadata: Optional metadata dictionary
        parent_id: ID of parent chunk (if any)
        chunk_id: Unique identifier for this chunk
    """
    text: str
    start: int
    end: int
    chunk_type: ChunkType = ChunkType.PARAGRAPH
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    chunk_id: str | None = None

    def __post_init__(self) -> None:
        """Generate chunk_id if not provided."""
        if self.chunk_id is None:
            import hashlib
            content = f"{self.text[:100]}{self.start}{self.end}"
            self.chunk_id = hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def length(self) -> int:
        """Return the length of the chunk text."""
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "chunk_type": self.chunk_type.value,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "chunk_id": self.chunk_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Create chunk from dictionary."""
        data = data.copy()
        if "chunk_type" in data and isinstance(data["chunk_type"], str):
            data["chunk_type"] = ChunkType(data["chunk_type"])
        return cls(**data)


@dataclass
class ChunkerConfig:
    """Configuration for chunkers.

    Attributes:
        max_tokens: Maximum tokens per chunk
        max_chars: Maximum characters per chunk (alternative to max_tokens)
        overlap: Number of overlapping tokens/chars between chunks
        language: Language code for language-specific processing
        preserve_sentences: Whether to avoid splitting mid-sentence
        normalize_whitespace: Whether to normalize whitespace
        normalize_unicode: Whether to normalize unicode characters
    """
    max_tokens: int = 512
    max_chars: int | None = None
    overlap: int = 50
    language: str = "en"
    preserve_sentences: bool = True
    normalize_whitespace: bool = True
    normalize_unicode: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_tokens <= self.overlap:
            raise ValueError("max_tokens must be greater than overlap")
        if self.max_chars is not None and self.max_chars <= 0:
            raise ValueError("max_chars must be positive")


@dataclass
class DocumentMetadata:
    """Metadata for a document being chunked.

    Attributes:
        doc_id: Document identifier
        source: Source path or URL
        title: Document title
        author: Document author
        date: Document date
        page: Page number (for PDFs)
        section: Section identifier
        extra: Additional metadata
    """
    doc_id: str
    source: str | None = None
    title: str | None = None
    author: str | None = None
    date: str | None = None
    page: int | None = None
    section: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "title": self.title,
            "author": self.author,
            "date": self.date,
            "page": self.page,
            "section": self.section,
            "extra": self.extra,
        }
