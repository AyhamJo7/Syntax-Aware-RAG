"""Recursive character splitter with multiple fallback separators."""

from .base import BaseChunker, TokenCounter
from .types import Chunk, ChunkerConfig, ChunkType, DocumentMetadata


class RecursiveCharacterChunker(BaseChunker):
    """Recursively split text using multiple separators.

    Tries to split on paragraph boundaries first, then sentences, then words,
    and finally characters if needed. This preserves document structure while
    ensuring chunks fit within size limits.
    """

    # Default separator hierarchy
    DEFAULT_SEPARATORS = [
        "\n\n",  # Paragraph breaks
        "\n",    # Line breaks
        ". ",    # Sentence ends
        "! ",    # Exclamation sentences
        "? ",    # Question sentences
        "; ",    # Semicolons
        ", ",    # Commas
        " ",     # Words
        "",      # Characters
    ]

    def __init__(
        self,
        config: ChunkerConfig | None = None,
        separators: list[str] | None = None
    ):
        """Initialize recursive chunker.

        Args:
            config: Chunker configuration
            separators: Custom separator hierarchy (uses default if None)
        """
        super().__init__(config)
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.token_counter = TokenCounter()

    def _split_text(self, text: str, separator: str) -> list[str]:
        """Split text by separator.

        Args:
            text: Text to split
            separator: Separator to use

        Returns:
            List of split parts
        """
        if separator == "":
            # Split into characters
            return list(text)

        if separator:
            parts = text.split(separator)
            # Keep separator with parts (except last)
            result = []
            for part in parts[:-1]:
                result.append(part + separator)
            if parts[-1]:
                result.append(parts[-1])
            return result
        else:
            return [text]

    def _merge_splits(
        self,
        splits: list[str],
        separator: str
    ) -> list[str]:
        """Merge splits into chunks respecting token limits.

        Args:
            splits: List of text splits
            separator: Separator used for splitting

        Returns:
            List of merged chunks
        """
        chunks = []
        current_chunk: list[str] = []
        current_tokens = 0

        for split in splits:
            split_tokens = self.token_counter.count_tokens(split)

            # If single split exceeds max, add it separately
            if split_tokens > self.config.max_tokens:
                # Flush current chunk
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Add oversized split (will be recursively split later)
                chunks.append(split)
                continue

            # Check if adding split would exceed limit
            if current_tokens + split_tokens > self.config.max_tokens and current_chunk:
                # Create chunk
                chunks.append("".join(current_chunk))

                # Handle overlap
                if self.config.overlap > 0:
                    # Keep last part for overlap
                    overlap_chunk: list[str] = []
                    overlap_tokens = 0
                    for s in reversed(current_chunk):
                        s_tokens = self.token_counter.count_tokens(s)
                        if overlap_tokens + s_tokens <= self.config.overlap:
                            overlap_chunk.insert(0, s)
                            overlap_tokens += s_tokens
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_tokens = overlap_tokens
                else:
                    current_chunk = []
                    current_tokens = 0

            # Add split to current chunk
            current_chunk.append(split)
            current_tokens += split_tokens

        # Add final chunk
        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks

    def _recursive_split(
        self,
        text: str,
        separators: list[str]
    ) -> list[str]:
        """Recursively split text using separator hierarchy.

        Args:
            text: Text to split
            separators: List of separators to try

        Returns:
            List of text chunks
        """
        if not separators:
            # No more separators, return as-is (will be truncated if needed)
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by current separator
        splits = self._split_text(text, separator)

        # Merge splits respecting token limits
        chunks = self._merge_splits(splits, separator)

        # Recursively split any chunks that are still too large
        final_chunks = []
        for chunk in chunks:
            chunk_tokens = self.token_counter.count_tokens(chunk)
            if chunk_tokens > self.config.max_tokens and remaining_separators:
                # Recursively split with next separator
                sub_chunks = self._recursive_split(chunk, remaining_separators)
                final_chunks.extend(sub_chunks)
            else:
                # Truncate if still too large and no more separators
                if chunk_tokens > self.config.max_tokens:
                    chunk = self.token_counter.truncate_to_tokens(
                        chunk, self.config.max_tokens
                    )
                final_chunks.append(chunk)

        return final_chunks

    def chunk(
        self,
        text: str,
        metadata: DocumentMetadata | None = None
    ) -> list[Chunk]:
        """Split text into chunks using recursive splitting.

        Args:
            text: Input text to chunk
            metadata: Optional document metadata

        Returns:
            List of Chunk objects

        Raises:
            ValueError: If text is empty or invalid
        """
        self.validate_text(text)
        text = self.preprocess(text)

        # Recursively split text
        chunk_texts = self._recursive_split(text, self.separators)

        # Create Chunk objects with position tracking
        chunks = []
        pos = 0
        for chunk_text in chunk_texts:
            # Find chunk position in original text
            start = text.find(chunk_text, pos)
            if start == -1:
                # Fallback if exact match not found
                start = pos
            end = start + len(chunk_text)

            # Determine chunk type based on content
            chunk_type = ChunkType.PARAGRAPH
            if "\n\n" not in chunk_text and len(chunk_text) < 200:
                chunk_type = ChunkType.SENTENCE

            chunks.append(
                self.create_chunk(
                    text=chunk_text.strip(),
                    start=start,
                    end=end,
                    metadata=metadata,
                    chunk_type=chunk_type
                )
            )
            pos = end

        return chunks
