"""Context builder for managing token budgets and diversity."""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    """Configuration for context construction.

    Attributes:
        max_tokens: Maximum tokens in final context
        include_parent: Whether to include parent context
        diversity_penalty: Penalty for similar passages (0-1)
        min_score_threshold: Minimum relevance score
    """
    max_tokens: int = 2048
    include_parent: bool = True
    diversity_penalty: float = 0.1
    min_score_threshold: float = 0.0


class ContextBuilder:
    """Builds context from retrieval results with budget management."""

    def __init__(self, config: ContextConfig | None = None):
        """Initialize context builder.

        Args:
            config: Context configuration
        """
        self.config = config or ContextConfig()

    def count_tokens(self, text: str) -> int:
        """Estimate token count.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Simple approximation: ~4 chars per token
        return len(text) // 4

    def build_context(
        self,
        results: list[dict[str, Any]],
        query: str | None = None
    ) -> str:
        """Build context string from retrieval results.

        Args:
            results: List of retrieval results
            query: Optional query string to include

        Returns:
            Formatted context string
        """
        # Filter by score threshold
        filtered = [
            r for r in results
            if r.get("score", 0) >= self.config.min_score_threshold
        ]

        if not filtered:
            return ""

        # Build context parts
        context_parts = []
        total_tokens = 0

        # Add query if provided
        if query:
            query_text = f"Query: {query}\n\n"
            query_tokens = self.count_tokens(query_text)
            if query_tokens < self.config.max_tokens // 4:
                context_parts.append(query_text)
                total_tokens += query_tokens

        # Add passages
        for i, result in enumerate(filtered):
            # Format passage
            text = result.get("text", "")
            score = result.get("score", 0)

            # Include parent if available
            if self.config.include_parent and "parent_text" in result:
                parent = result["parent_text"]
                passage = f"[Context] {parent[:200]}...\n\n{text}"
            else:
                passage = text

            passage_text = f"[Passage {i+1}] (score: {score:.3f})\n{passage}\n\n"
            passage_tokens = self.count_tokens(passage_text)

            # Check budget
            if total_tokens + passage_tokens > self.config.max_tokens:
                # Try without parent context
                passage_text = f"[Passage {i+1}] (score: {score:.3f})\n{text}\n\n"
                passage_tokens = self.count_tokens(passage_text)

                if total_tokens + passage_tokens > self.config.max_tokens:
                    break

            context_parts.append(passage_text)
            total_tokens += passage_tokens

        return "".join(context_parts)

    def build_context_with_diversity(
        self,
        results: list[dict[str, Any]],
        query: str | None = None
    ) -> str:
        """Build context with diversity optimization.

        Args:
            results: List of retrieval results
            query: Optional query string

        Returns:
            Formatted context string
        """
        # Simple diversity: skip very similar consecutive passages
        diverse_results = []
        prev_text = ""

        for result in results:
            text = result.get("text", "")

            # Check similarity with previous (naive: compare first 100 chars)
            if prev_text:
                overlap = sum(
                    a == b for a, b in zip(text[:100], prev_text[:100], strict=False)
                ) / 100

                if overlap > (1 - self.config.diversity_penalty):
                    continue  # Skip similar passage

            diverse_results.append(result)
            prev_text = text

        return self.build_context(diverse_results, query)
