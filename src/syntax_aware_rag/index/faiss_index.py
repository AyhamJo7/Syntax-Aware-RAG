"""FAISS-based vector index for hierarchical retrieval."""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ..embedding.types import DocumentNode, DocumentTree, NodeLevel

logger = logging.getLogger(__name__)


class FAISSIndex:
    """FAISS index for storing and retrieving document embeddings."""

    def __init__(
        self,
        dimension: int | None = None,
        metric: str = "cosine",
        use_gpu: bool = False
    ):
        """Initialize FAISS index.

        Args:
            dimension: Embedding dimension (auto-detected from first doc)
            metric: Distance metric ('cosine', 'l2', 'ip')
            use_gpu: Whether to use GPU acceleration
        """
        self.dimension = dimension
        self.metric = metric
        self.use_gpu = use_gpu
        self._index: Any = None
        self._metadata: list[dict[str, Any]] = []
        self._node_map: dict[int, str] = {}  # index_id -> node_id
        self._doc_trees: dict[str, DocumentTree] = {}  # doc_id -> tree

    @property
    def index(self) -> Any:
        """Lazy initialize FAISS index."""
        if self._index is None and self.dimension:
            self._initialize_index()
        return self._index

    def _initialize_index(self) -> None:
        """Initialize the FAISS index."""
        try:
            import faiss
        except ImportError as err:
            raise RuntimeError("faiss required. Install with: pip install faiss-cpu") from err

        if self.metric == "cosine":
            # Normalize vectors and use inner product
            self._index = faiss.IndexFlatIP(self.dimension)
        elif self.metric == "l2":
            self._index = faiss.IndexFlatL2(self.dimension)
        elif self.metric == "ip":
            self._index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
                logger.info("Using GPU-accelerated FAISS index")
            except Exception as e:
                logger.warning(f"GPU not available, falling back to CPU: {e}")

        logger.info(f"Initialized FAISS index: dim={self.dimension}, metric={self.metric}")

    def _normalize_vectors(self, vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Normalize vectors for cosine similarity.

        Args:
            vectors: Input vectors

        Returns:
            Normalized vectors
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms

    def add_documents(self, trees: list[DocumentTree]) -> None:
        """Add document trees to the index.

        Args:
            trees: List of DocumentTree objects
        """
        if not trees:
            return

        # Auto-detect dimension from first tree
        if self.dimension is None:
            first_node = next(
                (node for tree in trees for node in tree.nodes.values() if node.has_embedding()),
                None
            )
            if first_node and first_node.embedding is not None:
                self.dimension = len(first_node.embedding)
            else:
                raise ValueError("No embeddings found in trees")

        # Initialize index if needed
        if self._index is None:
            self._initialize_index()

        # Collect all nodes with embeddings
        embeddings_list = []
        metadata_list = []

        for tree in trees:
            # Store tree
            self._doc_trees[tree.doc_id] = tree

            for node in tree.nodes.values():
                if not node.has_embedding() or node.embedding is None:
                    continue

                # Get current index ID
                index_id = len(self._metadata)

                # Store mapping
                self._node_map[index_id] = node.node_id

                # Store metadata
                metadata_list.append({
                    "doc_id": tree.doc_id,
                    "node_id": node.node_id,
                    "text": node.text,
                    "level": node.level.value,
                    "parent_id": node.parent_id,
                    "metadata": node.metadata,
                })

                embeddings_list.append(node.embedding)

        if not embeddings_list:
            logger.warning("No embeddings to index")
            return

        # Convert to numpy array
        embeddings = np.vstack(embeddings_list).astype(np.float32)

        # Normalize if using cosine similarity
        if self.metric == "cosine":
            embeddings = self._normalize_vectors(embeddings)

        # Add to index
        self.index.add(embeddings)
        self._metadata.extend(metadata_list)

        logger.info(f"Added {len(embeddings)} vectors to index. Total: {self.index.ntotal}")

    def search(
        self,
        query_embedding: npt.NDArray[np.float32],
        top_k: int = 10,
        level_filter: NodeLevel | None = None
    ) -> list[tuple[float, dict[str, Any]]]:
        """Search the index.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            level_filter: Optional level to filter by

        Returns:
            List of (score, metadata) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # Reshape query
        query = query_embedding.reshape(1, -1).astype(np.float32)

        # Normalize if using cosine
        if self.metric == "cosine":
            query = self._normalize_vectors(query)

        # Search - get more results if filtering
        search_k = top_k * 5 if level_filter else top_k
        scores, indices = self.index.search(query, min(search_k, self.index.ntotal))

        # Collect results
        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self._metadata):
                continue

            metadata = self._metadata[idx]

            # Apply level filter
            if level_filter and metadata["level"] != level_filter.value:
                continue

            results.append((float(score), metadata))

            if len(results) >= top_k:
                break

        return results

    def get_node(self, doc_id: str, node_id: str) -> DocumentNode | None:
        """Retrieve a specific node.

        Args:
            doc_id: Document ID
            node_id: Node ID

        Returns:
            DocumentNode if found
        """
        tree = self._doc_trees.get(doc_id)
        if tree:
            return tree.get_node(node_id)
        return None

    def get_tree(self, doc_id: str) -> DocumentTree | None:
        """Retrieve a document tree.

        Args:
            doc_id: Document ID

        Returns:
            DocumentTree if found
        """
        return self._doc_trees.get(doc_id)

    def save(self, directory: Path) -> None:
        """Save index and metadata to disk.

        Args:
            directory: Directory to save to
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            import faiss
            # Save FAISS index
            faiss.write_index(self.index, str(directory / "index.faiss"))

        # Save metadata and trees
        with open(directory / "metadata.pkl", "wb") as f:
            pickle.dump({
                "dimension": self.dimension,
                "metric": self.metric,
                "metadata": self._metadata,
                "node_map": self._node_map,
                "doc_trees": self._doc_trees,
            }, f)

        logger.info(f"Saved index to {directory}")

    def load(self, directory: Path) -> None:
        """Load index and metadata from disk.

        Args:
            directory: Directory to load from
        """
        directory = Path(directory)

        # Load FAISS index
        try:
            import faiss
            self._index = faiss.read_index(str(directory / "index.faiss"))

            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
                except Exception as e:
                    logger.warning(f"Could not move index to GPU: {e}")
        except FileNotFoundError:
            logger.warning("No index.faiss found")

        # Load metadata
        with open(directory / "metadata.pkl", "rb") as f:
            data = pickle.load(f)
            self.dimension = data["dimension"]
            self.metric = data["metric"]
            self._metadata = data["metadata"]
            self._node_map = data["node_map"]
            self._doc_trees = data["doc_trees"]

        logger.info(f"Loaded index from {directory}. Total vectors: {self.index.ntotal if self.index else 0}")

    @property
    def size(self) -> int:
        """Get number of vectors in index."""
        return self.index.ntotal if self.index else 0
