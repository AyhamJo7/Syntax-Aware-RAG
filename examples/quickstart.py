"""Quickstart example for Syntax-Aware-RAG."""

from syntax_aware_rag.chunking import SentenceChunker, ChunkerConfig, DocumentMetadata
from syntax_aware_rag.embedding import HierarchicalEmbedder
from syntax_aware_rag.index import FAISSIndex
from syntax_aware_rag.retrieve import MultiGranularityRetriever
from syntax_aware_rag.context import ContextBuilder

# Sample documents
documents = [
    {
        "id": "doc1",
        "text": """
        Machine learning is a subset of artificial intelligence. It focuses on building
        systems that learn from data. Deep learning uses neural networks with multiple layers.
        These networks can learn complex patterns. Applications include image recognition,
        natural language processing, and recommendation systems.
        """
    },
    {
        "id": "doc2",
        "text": """
        Retrieval-Augmented Generation combines retrieval and generation. First, relevant
        documents are retrieved from a knowledge base. Then, a language model generates
        responses based on the retrieved context. This approach improves factual accuracy
        and reduces hallucinations in generated text.
        """
    },
    {
        "id": "doc3",
        "text": """
        Vector databases store embeddings for semantic search. They enable finding similar
        items based on meaning rather than exact matches. Popular vector databases include
        FAISS, Pinecone, and Weaviate. These systems are crucial for modern RAG applications.
        """
    }
]

def main():
    print("Syntax-Aware-RAG Quickstart\n")
    print("="*80 + "\n")

    # 1. Initialize components
    print("[1] Initializing components...")
    config = ChunkerConfig(max_tokens=100, overlap=20)
    chunker = SentenceChunker(config)
    embedder = HierarchicalEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = FAISSIndex()

    # 2. Process and index documents
    print("[2] Processing and indexing documents...")
    trees = []
    for doc in documents:
        metadata = DocumentMetadata(doc_id=doc["id"], source="example")
        chunks = chunker.chunk(doc["text"].strip(), metadata)
        tree = embedder.build_hierarchy(chunks, doc_id=doc["id"])
        trees.append(tree)
        print(f"  - Indexed {doc['id']}: {len(tree.nodes)} nodes")

    index.add_documents(trees)
    print(f"  Total vectors in index: {index.size}\n")

    # 3. Initialize retriever
    print("[3] Initializing retriever...")
    retriever = MultiGranularityRetriever(
        index=index,
        embedder=embedder,
        stage_b_top_k=3
    )

    # 4. Query the system
    queries = [
        "What is deep learning?",
        "How does RAG work?",
        "Tell me about vector databases"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 80)

        # Retrieve results
        results = retriever.retrieve_with_context(query, top_k=2, include_parent=True)

        for j, result in enumerate(results, 1):
            print(f"\n  Result {j} (score: {result['score']:.4f}):")
            print(f"  Doc: {result['doc_id']} | Level: {result['level']}")
            print(f"  Text: {result['text'][:150]}...")

    # 5. Build context for generation
    print("\n" + "="*80)
    print("[4] Building context for generation...")
    print("="*80 + "\n")

    query = "Explain how RAG systems use vector databases"
    results = retriever.retrieve_with_context(query, top_k=3)

    context_builder = ContextBuilder()
    context = context_builder.build_context_with_diversity(results, query=query)

    print(context)
    print(f"\nContext tokens: ~{context_builder.count_tokens(context)}")

    print("\n" + "="*80)
    print("Quickstart complete!")
    print("="*80)


if __name__ == "__main__":
    main()
