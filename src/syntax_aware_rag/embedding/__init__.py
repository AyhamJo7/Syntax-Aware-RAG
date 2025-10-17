"""Embedding module for hierarchical document representation."""

from .types import DocumentNode, DocumentTree, NodeLevel
from .hierarchical_embedder import HierarchicalEmbedder

__all__ = [
    "DocumentNode",
    "DocumentTree",
    "NodeLevel",
    "HierarchicalEmbedder",
]
