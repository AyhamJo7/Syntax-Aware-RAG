"""Sentence-based chunker using spaCy."""

import logging
from typing import Any

from .base import BaseChunker, TokenCounter
from .types import Chunk, ChunkerConfig, ChunkType, DocumentMetadata

logger = logging.getLogger(__name__)


class SentenceChunker(BaseChunker):
    """Chunks text by sentences using spaCy for sentence segmentation.

    This chunker respects sentence boundaries and groups sentences together
    until reaching the token limit, with optional overlap.
    """

    def __init__(
        self,
        config: ChunkerConfig | None = None,
        use_spacy: bool = True
    ):
        """Initialize sentence chunker.

        Args:
            config: Chunker configuration
            use_spacy: Whether to use spaCy for sentence splitting (fallback to NLTK)
        """
        super().__init__(config)
        self.token_counter = TokenCounter()
        self.use_spacy = use_spacy
        self._nlp: Any = None

    @property
    def nlp(self) -> Any:
        """Lazy load spaCy model."""
        if self._nlp is None and self.use_spacy:
            try:
                import spacy
                # Try to load the model, use sentencizer if full model not available
                try:
                    self._nlp = spacy.load(
                        f"{self.config.language}_core_web_sm",
                        disable=["ner", "lemmatizer", "textcat"]
                    )
                except OSError:
                    logger.warning(
                        f"spaCy model for {self.config.language} not found, "
                        "using blank model with sentencizer"
                    )
                    self._nlp = spacy.blank(self.config.language)
                    self._nlp.add_pipe("sentencizer")
            except ImportError:
                logger.warning("spaCy not available, falling back to NLTK")
                self.use_spacy = False
        return self._nlp

    def _split_sentences_spacy(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into sentences using spaCy.

        Args:
            text: Input text

        Returns:
            List of (sentence_text, start_pos, end_pos) tuples
        """
        doc = self.nlp(text)
        sentences = []
        for sent in doc.sents:
            sentences.append((sent.text, sent.start_char, sent.end_char))
        return sentences

    def _split_sentences_nltk(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into sentences using NLTK.

        Args:
            text: Input text

        Returns:
            List of (sentence_text, start_pos, end_pos) tuples
        """
        try:
            import nltk
            from nltk.tokenize import sent_tokenize

            # Ensure punkt is downloaded
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)

            sentences = []
            sent_texts = sent_tokenize(text, language=self.config.language)
            pos = 0
            for sent_text in sent_texts:
                start = text.find(sent_text, pos)
                if start == -1:
                    # Fallback if exact match not found
                    start = pos
                end = start + len(sent_text)
                sentences.append((sent_text, start, end))
                pos = end

            return sentences

        except ImportError as err:
            logger.error("Neither spaCy nor NLTK available for sentence splitting")
            raise RuntimeError("No sentence splitter available") from err

    def _split_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of (sentence_text, start_pos, end_pos) tuples
        """
        if self.use_spacy and self.nlp:
            return self._split_sentences_spacy(text)
        else:
            return self._split_sentences_nltk(text)

    def chunk(
        self,
        text: str,
        metadata: DocumentMetadata | None = None
    ) -> list[Chunk]:
        """Split text into sentence-based chunks.

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

        # Split into sentences
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        chunks = []
        current_sentences: list[str] = []
        current_tokens = 0
        chunk_start = 0

        for sent_text, sent_start, sent_end in sentences:
            sent_tokens = self.token_counter.count_tokens(sent_text)

            # If single sentence exceeds max, split it into multiple chunks
            if sent_tokens > self.config.max_tokens:
                # Flush current chunk if any
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append(
                        self.create_chunk(
                            text=chunk_text,
                            start=chunk_start,
                            end=sent_start,
                            metadata=metadata,
                            chunk_type=ChunkType.PARAGRAPH
                        )
                    )
                    current_sentences = []
                    current_tokens = 0

                # Split long sentence into multiple chunks
                words = sent_text.split()
                current_chunk_words: list[str] = []
                current_chunk_tokens = 0
                word_start = sent_start

                for word in words:
                    word_tokens = self.token_counter.count_tokens(word)
                    if current_chunk_tokens + word_tokens > self.config.max_tokens and current_chunk_words:
                        # Create chunk from accumulated words
                        chunk_text = " ".join(current_chunk_words)
                        chunks.append(
                            self.create_chunk(
                                text=chunk_text,
                                start=word_start,
                                end=word_start + len(chunk_text),
                                metadata=metadata,
                                chunk_type=ChunkType.SENTENCE
                            )
                        )
                        word_start += len(chunk_text) + 1  # +1 for space
                        current_chunk_words = []
                        current_chunk_tokens = 0

                    current_chunk_words.append(word)
                    current_chunk_tokens += word_tokens

                # Add remaining words
                if current_chunk_words:
                    chunk_text = " ".join(current_chunk_words)
                    chunks.append(
                        self.create_chunk(
                            text=chunk_text,
                            start=word_start,
                            end=sent_end,
                            metadata=metadata,
                            chunk_type=ChunkType.SENTENCE
                        )
                    )

                chunk_start = sent_end
                continue

            # Check if adding this sentence would exceed limit
            if current_tokens + sent_tokens > self.config.max_tokens and current_sentences:
                # Create chunk from current sentences
                chunk_text = " ".join(current_sentences)
                chunk_end = sent_start
                chunks.append(
                    self.create_chunk(
                        text=chunk_text,
                        start=chunk_start,
                        end=chunk_end,
                        metadata=metadata,
                        chunk_type=ChunkType.PARAGRAPH
                    )
                )

                # Handle overlap
                if self.config.overlap > 0:
                    # Keep last few sentences for overlap
                    overlap_sentences: list[str] = []
                    overlap_tokens = 0
                    for s in reversed(current_sentences):
                        s_tokens = self.token_counter.count_tokens(s)
                        if overlap_tokens + s_tokens <= self.config.overlap:
                            overlap_sentences.insert(0, s)
                            overlap_tokens += s_tokens
                        else:
                            break
                    current_sentences = overlap_sentences
                    current_tokens = overlap_tokens
                else:
                    current_sentences = []
                    current_tokens = 0

                chunk_start = sent_start

            # Add sentence to current chunk
            current_sentences.append(sent_text)
            current_tokens += sent_tokens

        # Add final chunk if any sentences remain
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                self.create_chunk(
                    text=chunk_text,
                    start=chunk_start,
                    end=len(text),
                    metadata=metadata,
                    chunk_type=ChunkType.PARAGRAPH
                )
            )

        return chunks
