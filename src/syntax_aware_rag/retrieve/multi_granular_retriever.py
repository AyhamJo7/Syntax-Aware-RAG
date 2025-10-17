"""Multi-granularity retriever with two-stage search."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
import numpy as np

from ..embedding.hierarchical_embedder import HierarchicalEmbedder
from ..embedding.types import NodeLevel
from ..index.faiss_index import FAISSIndex

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval.

    Attributes:
        text: Retrieved text
        score: Relevance score
        doc_id: Document ID
        node_id: Node ID
        level: Hierarchical level
        metadata: Additional metadata
    """
    text: str
    score: float
    doc_id: str
    node_id: str
    level: str
    metadata: Dict[str, Any]


class MultiGranularityRetriever:
    """Two-stage retriever with broad and fine-grained search."""

    def __init__(
        self,
        index: FAISSIndex,
        embedder: HierarchicalEmbedder,
        stage_a_level: NodeLevel = NodeLevel.PARAGRAPH,
        stage_a_top_k: int = 10,
        stage_b_top_k: int = 5
    ):
        """Initialize multi-granularity retriever.

        Args:
            index: FAISS index
            embedder: Hierarchical embedder
            stage_a_level: Level for broad search
            stage_a_top_k: Number of results in stage A
            stage_b_top_k: Number of results in stage B
        """
        self.index = index
        self.embedder = embedder
        self.stage_a_level = stage_a_level
        self.stage_a_top_k = stage_a_top_k
        self.stage_b_top_k = stage_b_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        level_filter: Optional[NodeLevel] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant passages for query.

        Args:
            query: Query string
            top_k: Number of results (overrides stage_b_top_k)
            level_filter: Optional level filter for single-stage search

        Returns:
            List of RetrievalResult objects
        """
        # Encode query
        query_embedding = self.embedder.encode([query])[0]

        # Single-stage search if level filter provided
        if level_filter:
            return self._single_stage_search(
                query_embedding,
                top_k or self.stage_b_top_k,
                level_filter
            )

        # Two-stage search
        return self._two_stage_search(query_embedding, top_k or self.stage_b_top_k)

    def _single_stage_search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        level: NodeLevel
    ) -> List[RetrievalResult]:
        """Single-stage search at specific level.

        Args:
            query_embedding: Query embedding
            top_k: Number of results
            level: Level to search

        Returns:
            List of RetrievalResult objects
        """
        results = self.index.search(
            query_embedding,
            top_k=top_k,
            level_filter=level
        )

        return [
            RetrievalResult(
                text=meta["text"],
                score=score,
                doc_id=meta["doc_id"],
                node_id=meta["node_id"],
                level=meta["level"],
                metadata=meta.get("metadata", {})
            )
            for score, meta in results
        ]

    def _two_stage_search(
        self,
        query_embedding: np.ndarray,
        final_top_k: int
    ) -> List[RetrievalResult]:
        """Two-stage hierarchical search.

        Stage A: Broad search at paragraph/section level
        Stage B: Fine search in children of top parents

        Args:
            query_embedding: Query embedding
            final_top_k: Final number of results

        Returns:
            List of RetrievalResult objects
        """
        # Stage A: Broad search
        stage_a_results = self.index.search(
            query_embedding,
            top_k=self.stage_a_top_k,
            level_filter=self.stage_a_level
        )

        if not stage_a_results:
            logger.warning("No results from stage A")
            return []

        # Collect parent node IDs
        parent_nodes = set()
        for _, meta in stage_a_results:
            parent_nodes.add((meta["doc_id"], meta["node_id"]))

        # Stage B: Search children of top parents
        candidate_results = []

        for doc_id, parent_node_id in parent_nodes:
            # Get parent node
            parent = self.index.get_node(doc_id, parent_node_id)
            if not parent or not parent.children:
                # No children, use parent itself
                for score, meta in stage_a_results:
                    if meta["doc_id"] == doc_id and meta["node_id"] == parent_node_id:
                        candidate_results.append(
                            RetrievalResult(
                                text=meta["text"],
                                score=score,
                                doc_id=doc_id,
                                node_id=parent_node_id,
                                level=meta["level"],
                                metadata=meta.get("metadata", {})
                            )
                        )
                continue

            # Search among children
            for child_id in parent.children:
                child = self.index.get_node(doc_id, child_id)
                if not child or not child.has_embedding():
                    continue

                # Compute similarity
                if child.embedding is not None:
                    similarity = float(np.dot(query_embedding, child.embedding))
                    candidate_results.append(
                        RetrievalResult(
                            text=child.text,
                            score=similarity,
                            doc_id=doc_id,
                            node_id=child_id,
                            level=child.level.value,
                            metadata=child.metadata
                        )
                    )

        # Sort by score and return top-k
        candidate_results.sort(key=lambda x: x.score, reverse=True)
        return candidate_results[:final_top_k]

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5,
        include_parent: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve passages with parent context.

        Args:
            query: Query string
            top_k: Number of results
            include_parent: Whether to include parent context

        Returns:
            List of result dictionaries with context
        """
        results = self.retrieve(query, top_k=top_k)

        enriched_results = []
        for result in results:
            data = {
                "text": result.text,
                "score": result.score,
                "doc_id": result.doc_id,
                "node_id": result.node_id,
                "level": result.level,
                "metadata": result.metadata,
            }

            if include_parent and result.metadata.get("parent_id"):
                parent = self.index.get_node(result.doc_id, result.metadata["parent_id"])
                if parent:
                    data["parent_text"] = parent.text
                    data["parent_level"] = parent.level.value

            enriched_results.append(data)

        return enriched_results
