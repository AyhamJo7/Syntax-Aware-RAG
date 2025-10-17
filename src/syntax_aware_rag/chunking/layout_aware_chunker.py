"""Layout-aware chunker for structured documents (PDF, HTML)."""

from typing import List, Optional, Dict, Any
import logging

from .base import BaseChunker, TokenCounter
from .types import Chunk, ChunkerConfig, DocumentMetadata, ChunkType

logger = logging.getLogger(__name__)


class LayoutAwareChunker(BaseChunker):
    """Chunks documents while preserving layout structure.

    Uses unstructured library to extract document elements (titles, headings,
    tables, lists) and creates chunks that respect document structure.
    """

    def __init__(self, config: Optional[ChunkerConfig] = None):
        """Initialize layout-aware chunker.

        Args:
            config: Chunker configuration
        """
        super().__init__(config)
        self.token_counter = TokenCounter()
        self._check_unstructured()

    def _check_unstructured(self) -> None:
        """Check if unstructured library is available."""
        try:
            import unstructured
            self.has_unstructured = True
        except ImportError:
            logger.warning(
                "unstructured library not available. Layout-aware chunking "
                "will fall back to basic text chunking."
            )
            self.has_unstructured = False

    def _extract_elements_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract elements from file using unstructured.

        Args:
            file_path: Path to file

        Returns:
            List of element dictionaries
        """
        try:
            from unstructured.partition.auto import partition

            elements = partition(filename=file_path)
            return [
                {
                    "text": str(elem),
                    "type": elem.category if hasattr(elem, 'category') else "text",
                    "metadata": elem.metadata.to_dict() if hasattr(elem, 'metadata') else {}
                }
                for elem in elements
            ]
        except Exception as e:
            logger.error(f"Error extracting elements from {file_path}: {e}")
            return []

    def _extract_elements_from_html(self, html: str) -> List[Dict[str, Any]]:
        """Extract elements from HTML string.

        Args:
            html: HTML string

        Returns:
            List of element dictionaries
        """
        try:
            from unstructured.partition.html import partition_html
            from io import StringIO

            elements = partition_html(text=html)
            return [
                {
                    "text": str(elem),
                    "type": elem.category if hasattr(elem, 'category') else "text",
                    "metadata": elem.metadata.to_dict() if hasattr(elem, 'metadata') else {}
                }
                for elem in elements
            ]
        except Exception as e:
            logger.error(f"Error extracting elements from HTML: {e}")
            return []

    def _group_elements(
        self,
        elements: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Group elements into logical chunks.

        Args:
            elements: List of document elements

        Returns:
            List of element groups
        """
        groups = []
        current_group = []
        current_tokens = 0
        current_section = None

        for elem in elements:
            elem_type = elem.get("type", "text")
            elem_text = elem.get("text", "")
            elem_tokens = self.token_counter.count_tokens(elem_text)

            # Start new group for titles and headings
            if elem_type in ["Title", "Header", "Heading"]:
                # Flush current group
                if current_group:
                    groups.append(current_group)
                current_group = [elem]
                current_tokens = elem_tokens
                current_section = elem_text
                continue

            # Tables and figures get their own group if large
            if elem_type in ["Table", "Figure", "Image"] and elem_tokens > self.config.max_tokens // 2:
                # Flush current group
                if current_group:
                    groups.append(current_group)
                # Add table/figure as separate group
                groups.append([elem])
                current_group = []
                current_tokens = 0
                continue

            # Check if adding element would exceed limit
            if current_tokens + elem_tokens > self.config.max_tokens and current_group:
                groups.append(current_group)
                # Start new group with overlap
                if self.config.overlap > 0 and current_group:
                    # Keep last element for context
                    current_group = [current_group[-1]]
                    current_tokens = self.token_counter.count_tokens(current_group[-1]["text"])
                else:
                    current_group = []
                    current_tokens = 0

            # Add element to current group
            current_group.append(elem)
            current_tokens += elem_tokens

        # Add final group
        if current_group:
            groups.append(current_group)

        return groups

    def _create_chunk_from_group(
        self,
        group: List[Dict[str, Any]],
        start_pos: int,
        metadata: Optional[DocumentMetadata]
    ) -> Chunk:
        """Create a chunk from a group of elements.

        Args:
            group: List of elements
            start_pos: Starting position in document
            metadata: Document metadata

        Returns:
            Chunk object
        """
        # Combine element texts
        texts = [elem["text"] for elem in group if elem.get("text")]
        chunk_text = "\n\n".join(texts)

        # Determine chunk type based on elements
        types = [elem.get("type", "text") for elem in group]
        if "Title" in types or "Header" in types:
            chunk_type = ChunkType.SECTION
        elif "Table" in types:
            chunk_type = ChunkType.PARAGRAPH  # Could add TABLE type
        else:
            chunk_type = ChunkType.PARAGRAPH

        # Merge metadata from elements
        chunk_metadata = metadata.to_dict() if metadata else {}
        elem_metadata = {}
        for elem in group:
            elem_metadata.update(elem.get("metadata", {}))

        chunk_metadata["element_types"] = list(set(types))
        chunk_metadata.update(elem_metadata)

        end_pos = start_pos + len(chunk_text)

        return Chunk(
            text=chunk_text,
            start=start_pos,
            end=end_pos,
            chunk_type=chunk_type,
            metadata=chunk_metadata
        )

    def chunk_file(
        self,
        file_path: str,
        metadata: Optional[DocumentMetadata] = None
    ) -> List[Chunk]:
        """Chunk a file while preserving layout.

        Args:
            file_path: Path to file (PDF, HTML, etc.)
            metadata: Optional document metadata

        Returns:
            List of Chunk objects
        """
        if not self.has_unstructured:
            raise RuntimeError(
                "unstructured library required for layout-aware chunking. "
                "Install with: pip install unstructured"
            )

        # Extract elements
        elements = self._extract_elements_from_file(file_path)
        if not elements:
            return []

        # Group elements into chunks
        groups = self._group_elements(elements)

        # Create chunks
        chunks = []
        pos = 0
        for group in groups:
            chunk = self._create_chunk_from_group(group, pos, metadata)
            chunks.append(chunk)
            pos = chunk.end

        return chunks

    def chunk_html(
        self,
        html: str,
        metadata: Optional[DocumentMetadata] = None
    ) -> List[Chunk]:
        """Chunk HTML while preserving structure.

        Args:
            html: HTML string
            metadata: Optional document metadata

        Returns:
            List of Chunk objects
        """
        if not self.has_unstructured:
            raise RuntimeError(
                "unstructured library required for layout-aware chunking"
            )

        # Extract elements
        elements = self._extract_elements_from_html(html)
        if not elements:
            return []

        # Group elements into chunks
        groups = self._group_elements(elements)

        # Create chunks
        chunks = []
        pos = 0
        for group in groups:
            chunk = self._create_chunk_from_group(group, pos, metadata)
            chunks.append(chunk)
            pos = chunk.end

        return chunks

    def chunk(
        self,
        text: str,
        metadata: Optional[DocumentMetadata] = None
    ) -> List[Chunk]:
        """Chunk text (fallback for non-structured content).

        Args:
            text: Input text to chunk
            metadata: Optional document metadata

        Returns:
            List of Chunk objects
        """
        # For plain text, fall back to simple paragraph splitting
        self.validate_text(text)
        text = self.preprocess(text)

        # Split by double newlines (paragraphs)
        paragraphs = text.split("\n\n")

        chunks = []
        current_para = []
        current_tokens = 0
        pos = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.token_counter.count_tokens(para)

            # If paragraph exceeds max, split it
            if para_tokens > self.config.max_tokens:
                # Flush current
                if current_para:
                    chunk_text = "\n\n".join(current_para)
                    start = text.find(current_para[0], pos)
                    end = start + len(chunk_text)
                    chunks.append(
                        self.create_chunk(
                            text=chunk_text,
                            start=start,
                            end=end,
                            metadata=metadata,
                            chunk_type=ChunkType.PARAGRAPH
                        )
                    )
                    pos = end
                    current_para = []
                    current_tokens = 0

                # Truncate long paragraph
                para = self.token_counter.truncate_to_tokens(para, self.config.max_tokens)
                start = text.find(para, pos)
                end = start + len(para)
                chunks.append(
                    self.create_chunk(
                        text=para,
                        start=start,
                        end=end,
                        metadata=metadata,
                        chunk_type=ChunkType.PARAGRAPH
                    )
                )
                pos = end
                continue

            # Check if adding would exceed limit
            if current_tokens + para_tokens > self.config.max_tokens and current_para:
                chunk_text = "\n\n".join(current_para)
                start = text.find(current_para[0], pos)
                end = start + len(chunk_text)
                chunks.append(
                    self.create_chunk(
                        text=chunk_text,
                        start=start,
                        end=end,
                        metadata=metadata,
                        chunk_type=ChunkType.PARAGRAPH
                    )
                )
                pos = end

                # Handle overlap
                if self.config.overlap > 0:
                    current_para = [current_para[-1]]
                    current_tokens = self.token_counter.count_tokens(current_para[-1])
                else:
                    current_para = []
                    current_tokens = 0

            current_para.append(para)
            current_tokens += para_tokens

        # Final chunk
        if current_para:
            chunk_text = "\n\n".join(current_para)
            start = text.find(current_para[0], pos)
            end = start + len(chunk_text)
            chunks.append(
                self.create_chunk(
                    text=chunk_text,
                    start=start,
                    end=end,
                    metadata=metadata,
                    chunk_type=ChunkType.PARAGRAPH
                )
            )

        return chunks
