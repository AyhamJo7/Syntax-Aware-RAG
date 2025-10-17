"""Hierarchical embedder for creating document trees with embeddings."""

import hashlib
import logging
from typing import Any

import numpy as np

from ..chunking.types import Chunk, ChunkType
from .types import DocumentNode, DocumentTree, NodeLevel

logger = logging.getLogger(__name__)


class HierarchicalEmbedder:
    """Creates hierarchical document trees with embeddings at each level."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str = "cpu"
    ):
        """Initialize the hierarchical embedder.

        Args:
            model_name: Name of sentence-transformers model
            batch_size: Batch size for encoding
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError as err:
                raise RuntimeError(
                    "sentence-transformers required. Install with: "
                    "pip install sentence-transformers"
                ) from err
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: List of text strings

        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings

    def _chunk_type_to_level(self, chunk_type: ChunkType) -> NodeLevel:
        """Convert chunk type to node level.

        Args:
            chunk_type: Chunk type

        Returns:
            Corresponding node level
        """
        mapping = {
            ChunkType.DOCUMENT: NodeLevel.DOCUMENT,
            ChunkType.SECTION: NodeLevel.SECTION,
            ChunkType.PARAGRAPH: NodeLevel.PARAGRAPH,
            ChunkType.SENTENCE: NodeLevel.SENTENCE,
        }
        return mapping.get(chunk_type, NodeLevel.PARAGRAPH)

    def _create_node_id(self, text: str, position: int) -> str:
        """Create a unique node ID.

        Args:
            text: Node text
            position: Position in document

        Returns:
            Unique node ID
        """
        content = f"{text[:100]}{position}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def build_tree_from_chunks(
        self,
        chunks: list[Chunk],
        doc_id: str,
        metadata: dict[str, Any] | None = None
    ) -> DocumentTree:
        """Build hierarchical tree from flat list of chunks.

        Args:
            chunks: List of Chunk objects
            doc_id: Document identifier
            metadata: Document metadata

        Returns:
            DocumentTree with hierarchical structure
        """
        if not chunks:
            raise ValueError("Cannot build tree from empty chunk list")

        # Create root document node
        full_text = " ".join(chunk.text for chunk in chunks)
        root_id = self._create_node_id(full_text, 0)

        root_node = DocumentNode(
            node_id=root_id,
            text=full_text[:1000],  # Store summary in root
            level=NodeLevel.DOCUMENT,
            metadata=metadata or {},
            start=0,
            end=max(chunk.end for chunk in chunks) if chunks else 0
        )

        # Create tree
        tree = DocumentTree(
            doc_id=doc_id,
            root_id=root_id,
            metadata=metadata or {}
        )
        tree.add_node(root_node)

        # Group chunks by level
        sections = []
        paragraphs = []
        sentences = []

        for chunk in chunks:
            level = self._chunk_type_to_level(chunk.chunk_type)
            node_id = chunk.chunk_id or self._create_node_id(chunk.text, chunk.start)

            node = DocumentNode(
                node_id=node_id,
                text=chunk.text,
                level=level,
                metadata=chunk.metadata,
                start=chunk.start,
                end=chunk.end,
                parent_id=root_id
            )

            tree.add_node(node)
            root_node.add_child(node_id)

            if level == NodeLevel.SECTION:
                sections.append(node)
            elif level == NodeLevel.PARAGRAPH:
                paragraphs.append(node)
            elif level == NodeLevel.SENTENCE:
                sentences.append(node)

        return tree

    def embed_tree(self, tree: DocumentTree) -> DocumentTree:
        """Add embeddings to all nodes in the tree.

        Args:
            tree: DocumentTree to embed

        Returns:
            Same tree with embeddings added
        """
        # Collect all nodes that need embeddings
        nodes_to_embed = [node for node in tree.nodes.values() if not node.has_embedding()]

        if not nodes_to_embed:
            logger.info("All nodes already have embeddings")
            return tree

        # Extract texts
        texts = [node.text for node in nodes_to_embed]

        # Encode in batches
        logger.info(f"Encoding {len(texts)} nodes...")
        embeddings = self.encode(texts)

        # Assign embeddings
        for node, embedding in zip(nodes_to_embed, embeddings, strict=True):
            node.embedding = embedding.astype(np.float32)

        logger.info(f"Embedded {len(nodes_to_embed)} nodes")
        return tree

    def build_hierarchy(
        self,
        chunks: list[Chunk],
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        embed: bool = True
    ) -> DocumentTree:
        """Build complete hierarchical tree with embeddings.

        Args:
            chunks: List of chunks
            doc_id: Document ID (auto-generated if None)
            metadata: Document metadata
            embed: Whether to compute embeddings

        Returns:
            Complete DocumentTree with embeddings
        """
        # Generate doc_id if not provided
        if doc_id is None:
            doc_id = hashlib.sha256(chunks[0].text.encode()).hexdigest()[:16]

        # Build tree structure
        tree = self.build_tree_from_chunks(chunks, doc_id, metadata)

        # Add embeddings if requested
        if embed:
            tree = self.embed_tree(tree)

        return tree

    def create_summary_nodes(self, tree: DocumentTree) -> DocumentTree:
        """Create summary nodes for sections by aggregating children.

        Args:
            tree: DocumentTree

        Returns:
            Tree with summary nodes added
        """
        # For each section/paragraph with multiple children, create a summary
        nodes_with_children = [
            node for node in tree.nodes.values()
            if len(node.children) > 1 and node.level in [NodeLevel.SECTION, NodeLevel.PARAGRAPH]
        ]

        for parent in nodes_with_children:
            children = tree.get_children(parent.node_id)
            if not children:
                continue

            # Aggregate child texts
            child_texts = [child.text for child in children[:5]]  # Limit to first 5
            summary_text = " ".join(child_texts)[:500]  # Truncate

            # Update parent text if it's too short
            if len(parent.text) < 100:
                parent.text = summary_text

        # Re-embed modified nodes
        modified_nodes = [node for node in nodes_with_children if not node.has_embedding()]
        if modified_nodes:
            texts = [node.text for node in modified_nodes]
            embeddings = self.encode(texts)
            for node, embedding in zip(modified_nodes, embeddings, strict=True):
                node.embedding = embedding.astype(np.float32)

        return tree
