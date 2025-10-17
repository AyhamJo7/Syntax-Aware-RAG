"""Unit tests for chunking components."""

import pytest
from syntax_aware_rag.chunking import (
    Chunk,
    ChunkType,
    ChunkerConfig,
    SentenceChunker,
    RecursiveCharacterChunker,
    DocumentMetadata,
)


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunk = Chunk(
            text="Test text",
            start=0,
            end=9,
            chunk_type=ChunkType.SENTENCE
        )
        assert chunk.text == "Test text"
        assert chunk.start == 0
        assert chunk.end == 9
        assert chunk.length == 9

    def test_chunk_id_generation(self):
        """Test automatic chunk ID generation."""
        chunk = Chunk(text="Test", start=0, end=4)
        assert chunk.chunk_id is not None
        assert len(chunk.chunk_id) == 16

    def test_chunk_to_dict(self):
        """Test chunk serialization."""
        chunk = Chunk(text="Test", start=0, end=4, chunk_type=ChunkType.PARAGRAPH)
        data = chunk.to_dict()
        assert data["text"] == "Test"
        assert data["chunk_type"] == "paragraph"


class TestChunkerConfig:
    """Tests for ChunkerConfig."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = ChunkerConfig(max_tokens=512, overlap=50)
        assert config.max_tokens == 512
        assert config.overlap == 50

    def test_invalid_config(self):
        """Test that overlap cannot exceed max_tokens."""
        with pytest.raises(ValueError):
            ChunkerConfig(max_tokens=100, overlap=100)


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    def test_basic_chunking(self):
        """Test basic sentence chunking."""
        chunker = SentenceChunker(ChunkerConfig(max_tokens=100))
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk(text)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_empty_text_raises(self):
        """Test that empty text raises ValueError."""
        chunker = SentenceChunker()
        with pytest.raises(ValueError):
            chunker.chunk("")

    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        chunker = SentenceChunker(ChunkerConfig(normalize_whitespace=True))
        text = "Multiple    spaces   here."
        chunks = chunker.chunk(text)
        assert "    " not in chunks[0].text

    def test_metadata_propagation(self):
        """Test that metadata is propagated to chunks."""
        chunker = SentenceChunker()
        metadata = DocumentMetadata(doc_id="test_doc", source="test.txt")
        text = "Test sentence."
        chunks = chunker.chunk(text, metadata)

        assert chunks[0].metadata["doc_id"] == "test_doc"
        assert chunks[0].metadata["source"] == "test.txt"


class TestRecursiveCharacterChunker:
    """Tests for RecursiveCharacterChunker."""

    def test_paragraph_splitting(self):
        """Test splitting by paragraphs."""
        chunker = RecursiveCharacterChunker(ChunkerConfig(max_tokens=50))
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunker.chunk(text)

        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_respects_token_limit(self):
        """Test that chunks respect token limits."""
        chunker = RecursiveCharacterChunker(ChunkerConfig(max_tokens=20))
        text = " ".join(["word"] * 100)  # 100 words
        chunks = chunker.chunk(text)

        assert len(chunks) > 1  # Should be split into multiple chunks

    def test_separator_hierarchy(self):
        """Test that custom separators work."""
        separators = ["\n\n", ". ", " "]
        chunker = RecursiveCharacterChunker(
            ChunkerConfig(max_tokens=50),
            separators=separators
        )
        text = "Sentence one. Sentence two."
        chunks = chunker.chunk(text)

        assert len(chunks) > 0


@pytest.mark.parametrize("text,expected_min_chunks", [
    ("Short.", 1),
    ("First sentence. Second sentence. Third sentence.", 1),
    (" ".join(["Word"] * 200), 2),  # Should split long text
])
def test_chunking_parametrized(text, expected_min_chunks):
    """Parametrized test for various text inputs."""
    chunker = SentenceChunker(ChunkerConfig(max_tokens=100))
    chunks = chunker.chunk(text)
    assert len(chunks) >= expected_min_chunks
