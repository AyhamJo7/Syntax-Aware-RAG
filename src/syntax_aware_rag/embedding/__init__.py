"""Embedding module for hierarchical document representation."""

from .hierarchical_embedder import HierarchicalEmbedder
from .types import DocumentNode, DocumentTree, NodeLevel

__all__ = [
    "DocumentNode",
    "DocumentTree",
    "NodeLevel",
    "HierarchicalEmbedder",
]
