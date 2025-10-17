"""Type definitions for embedding components."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt


class NodeLevel(Enum):
    """Levels in the document hierarchy."""
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"


@dataclass
class DocumentNode:
    """Node in the hierarchical document tree.

    Attributes:
        node_id: Unique identifier for the node
        text: Text content of the node
        level: Hierarchical level
        embedding: Dense embedding vector (optional)
        children: List of child node IDs
        parent_id: Parent node ID (None for root)
        metadata: Additional metadata
        start: Start position in original document
        end: End position in original document
    """
    node_id: str
    text: str
    level: NodeLevel
    embedding: npt.NDArray[np.float32] | None = None
    children: list[str] = field(default_factory=list)
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    start: int = 0
    end: int = 0

    def add_child(self, child_id: str) -> None:
        """Add a child node ID."""
        if child_id not in self.children:
            self.children.append(child_id)

    def has_embedding(self) -> bool:
        """Check if node has an embedding."""
        return self.embedding is not None

    def to_dict(self, include_embedding: bool = False) -> dict[str, Any]:
        """Convert node to dictionary.

        Args:
            include_embedding: Whether to include embedding vector

        Returns:
            Dictionary representation
        """
        data = {
            "node_id": self.node_id,
            "text": self.text,
            "level": self.level.value,
            "children": self.children,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
            "start": self.start,
            "end": self.end,
        }
        if include_embedding and self.embedding is not None:
            data["embedding"] = self.embedding.tolist()
        return data


@dataclass
class DocumentTree:
    """Hierarchical document tree structure.

    Attributes:
        doc_id: Document identifier
        root_id: ID of root node
        nodes: Dictionary mapping node_id to DocumentNode
        metadata: Document-level metadata
    """
    doc_id: str
    root_id: str
    nodes: dict[str, DocumentNode] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: DocumentNode) -> None:
        """Add a node to the tree.

        Args:
            node: DocumentNode to add
        """
        self.nodes[node.node_id] = node

    def get_node(self, node_id: str) -> DocumentNode | None:
        """Get a node by ID.

        Args:
            node_id: Node identifier

        Returns:
            DocumentNode if found, None otherwise
        """
        return self.nodes.get(node_id)

    def get_root(self) -> DocumentNode | None:
        """Get the root node."""
        return self.get_node(self.root_id)

    def get_children(self, node_id: str) -> list[DocumentNode]:
        """Get all children of a node.

        Args:
            node_id: Parent node ID

        Returns:
            List of child DocumentNodes
        """
        node = self.get_node(node_id)
        if not node:
            return []
        return [self.nodes[child_id] for child_id in node.children if child_id in self.nodes]

    def get_nodes_by_level(self, level: NodeLevel) -> list[DocumentNode]:
        """Get all nodes at a specific level.

        Args:
            level: Node level to filter by

        Returns:
            List of DocumentNodes at the specified level
        """
        return [node for node in self.nodes.values() if node.level == level]

    def traverse_dfs(
        self,
        start_node_id: str | None = None
    ) -> list[DocumentNode]:
        """Depth-first traversal of the tree.

        Args:
            start_node_id: Node to start from (default: root)

        Returns:
            List of nodes in DFS order
        """
        start_id = start_node_id or self.root_id
        start_node = self.get_node(start_id)
        if not start_node:
            return []

        result = [start_node]
        for child_id in start_node.children:
            result.extend(self.traverse_dfs(child_id))
        return result

    def to_dict(self, include_embeddings: bool = False) -> dict[str, Any]:
        """Convert tree to dictionary.

        Args:
            include_embeddings: Whether to include embedding vectors

        Returns:
            Dictionary representation
        """
        return {
            "doc_id": self.doc_id,
            "root_id": self.root_id,
            "nodes": {
                node_id: node.to_dict(include_embedding=include_embeddings)
                for node_id, node in self.nodes.items()
            },
            "metadata": self.metadata,
        }
