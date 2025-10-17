"""Base chunker interface and utilities."""

import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Any

from .types import Chunk, ChunkerConfig, DocumentMetadata


class BaseChunker(ABC):
    """Abstract base class for all chunkers.

    Chunkers split text into smaller pieces while preserving semantic boundaries
    and maintaining metadata about the chunk positions.
    """

    def __init__(self, config: ChunkerConfig | None = None):
        """Initialize the chunker.

        Args:
            config: Chunker configuration. If None, uses default config.
        """
        self.config = config or ChunkerConfig()

    @abstractmethod
    def chunk(
        self,
        text: str,
        metadata: DocumentMetadata | None = None
    ) -> list[Chunk]:
        """Split text into chunks.

        Args:
            text: Input text to chunk
            metadata: Optional document metadata to attach to chunks

        Returns:
            List of Chunk objects

        Raises:
            ValueError: If text is empty or invalid
        """
        pass

    def preprocess(self, text: str) -> str:
        """Preprocess text before chunking.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        if not text:
            return text

        # Normalize unicode if configured
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)

        # Normalize whitespace if configured
        if self.config.normalize_whitespace:
            # Replace multiple spaces with single space
            text = re.sub(r' +', ' ', text)
            # Replace multiple newlines with double newline
            text = re.sub(r'\n{3,}', '\n\n', text)
            # Strip leading/trailing whitespace
            text = text.strip()

        return text

    def validate_text(self, text: str) -> None:
        """Validate input text.

        Args:
            text: Text to validate

        Raises:
            ValueError: If text is invalid
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or whitespace-only")

        # Check for null bytes
        if '\x00' in text:
            raise ValueError("Text contains null bytes")

    def create_chunk(
        self,
        text: str,
        start: int,
        end: int,
        metadata: DocumentMetadata | None = None,
        **kwargs: Any
    ) -> Chunk:
        """Create a chunk with metadata.

        Args:
            text: Chunk text
            start: Start position
            end: End position
            metadata: Optional document metadata
            **kwargs: Additional chunk parameters

        Returns:
            Chunk object
        """
        chunk_metadata = metadata.to_dict() if metadata else {}
        return Chunk(
            text=text,
            start=start,
            end=end,
            metadata=chunk_metadata,
            **kwargs
        )


class TokenCounter:
    """Utility for counting tokens in text."""

    def __init__(self, tokenizer_name: str = "cl100k_base"):
        """Initialize token counter.

        Args:
            tokenizer_name: Name of tiktoken encoding to use
        """
        self.encoding: Any
        try:
            import tiktoken
            self.encoding = tiktoken.get_encoding(tokenizer_name)
        except ImportError:
            # Fallback to simple whitespace tokenization
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Simple fallback: split on whitespace
            return len(text.split())

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to maximum number of tokens.

        Args:
            text: Input text
            max_tokens: Maximum number of tokens

        Returns:
            Truncated text
        """
        if self.encoding:
            tokens = self.encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            truncated_tokens = tokens[:max_tokens]
            decoded: str = self.encoding.decode(truncated_tokens)
            return decoded
        else:
            # Simple fallback
            words = text.split()
            if len(words) <= max_tokens:
                return text
            return ' '.join(words[:max_tokens])
