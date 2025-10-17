"""Command-line interface for Syntax-Aware-RAG."""

import logging
import sys
from pathlib import Path

import click

from ..chunking import (
    BaseChunker,
    ChunkerConfig,
    DocumentMetadata,
    RecursiveCharacterChunker,
    SentenceChunker,
)
from ..context import ContextBuilder
from ..embedding import HierarchicalEmbedder
from ..index import FAISSIndex
from ..retrieve import MultiGranularityRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """Syntax-Aware-RAG CLI - Advanced RAG with hierarchical indexing."""
    pass


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Input file or directory')
@click.option('--output', '-o', required=True, type=click.Path(), help='Output index directory')
@click.option('--chunker', '-c', type=click.Choice(['sentence', 'recursive']), default='sentence', help='Chunker type')
@click.option('--max-tokens', '-m', type=int, default=512, help='Maximum tokens per chunk')
@click.option('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='Embedding model')
def index(input: str, output: str, chunker: str, max_tokens: int, model: str) -> None:
    """Build an index from documents."""
    try:
        input_path = Path(input)
        output_path = Path(output)

        # Initialize components
        config = ChunkerConfig(max_tokens=max_tokens)
        chunker_obj: BaseChunker
        if chunker == 'sentence':
            chunker_obj = SentenceChunker(config)
        else:
            chunker_obj = RecursiveCharacterChunker(config)

        embedder = HierarchicalEmbedder(model_name=model)
        faiss_index = FAISSIndex()

        # Process files
        trees = []
        if input_path.is_file():
            files = [input_path]
        else:
            files = list(input_path.glob('**/*.txt')) + list(input_path.glob('**/*.md'))

        logger.info(f"Processing {len(files)} files...")

        for file_path in files:
            logger.info(f"Processing {file_path.name}...")
            try:
                with open(file_path, encoding='utf-8') as f:
                    text = f.read()

                metadata = DocumentMetadata(
                    doc_id=file_path.stem,
                    source=str(file_path)
                )

                chunks = chunker_obj.chunk(text, metadata)
                tree = embedder.build_hierarchy(chunks, doc_id=file_path.stem)
                trees.append(tree)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        # Build index
        if trees:
            logger.info(f"Indexing {len(trees)} documents...")
            faiss_index.add_documents(trees)

            # Save index
            output_path.mkdir(parents=True, exist_ok=True)
            faiss_index.save(output_path)

            logger.info(f"Index saved to {output_path}")
            logger.info(f"Total vectors indexed: {faiss_index.size}")
        else:
            logger.error("No documents were processed successfully")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--index-path', '-i', required=True, type=click.Path(exists=True), help='Index directory')
@click.option('--query', '-q', required=True, help='Query string')
@click.option('--top-k', '-k', type=int, default=5, help='Number of results')
@click.option('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='Embedding model')
def retrieve(index_path: str, query: str, top_k: int, model: str) -> None:
    """Retrieve relevant passages for a query."""
    try:
        # Load index
        faiss_index = FAISSIndex()
        faiss_index.load(Path(index_path))

        # Initialize embedder and retriever
        embedder = HierarchicalEmbedder(model_name=model)
        retriever = MultiGranularityRetriever(
            index=faiss_index,
            embedder=embedder,
            stage_b_top_k=top_k
        )

        # Retrieve
        logger.info(f"Retrieving top-{top_k} passages for: '{query}'")
        results = retriever.retrieve_with_context(query, top_k=top_k)

        # Display results
        click.echo("\nResults:\n")
        for i, result in enumerate(results, 1):
            click.echo(f"[{i}] Score: {result['score']:.4f}")
            click.echo(f"Doc: {result['doc_id']} | Level: {result['level']}")
            click.echo(f"Text: {result['text'][:200]}...")
            if 'parent_text' in result:
                click.echo(f"Context: {result['parent_text'][:100]}...")
            click.echo()

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--index-path', '-i', required=True, type=click.Path(exists=True), help='Index directory')
@click.option('--query', '-q', required=True, help='Query string')
@click.option('--top-k', '-k', type=int, default=5, help='Number of results')
@click.option('--max-context-tokens', '-m', type=int, default=2048, help='Maximum context tokens')
@click.option('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='Embedding model')
def query(index_path: str, query: str, top_k: int, max_context_tokens: int, model: str) -> None:
    """Query the index and build context."""
    try:
        # Load index
        faiss_index = FAISSIndex()
        faiss_index.load(Path(index_path))

        # Initialize components
        embedder = HierarchicalEmbedder(model_name=model)
        retriever = MultiGranularityRetriever(
            index=faiss_index,
            embedder=embedder,
            stage_b_top_k=top_k
        )
        context_builder = ContextBuilder()
        context_builder.config.max_tokens = max_context_tokens

        # Retrieve and build context
        logger.info(f"Querying: '{query}'")
        results = retriever.retrieve_with_context(query, top_k=top_k)
        context = context_builder.build_context_with_diversity(results, query=query)

        # Display
        click.echo("\n" + "="*80)
        click.echo("CONTEXT FOR GENERATION")
        click.echo("="*80 + "\n")
        click.echo(context)
        click.echo("\n" + "="*80)
        click.echo(f"Context tokens: ~{context_builder.count_tokens(context)}")
        click.echo("="*80)

    except Exception as e:
        logger.error(f"Query failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Input file')
@click.option('--chunker', '-c', type=click.Choice(['sentence', 'recursive']), default='sentence', help='Chunker type')
@click.option('--max-tokens', '-m', type=int, default=512, help='Maximum tokens per chunk')
def ingest(input: str, chunker: str, max_tokens: int) -> None:
    """Ingest and chunk a document (preview)."""
    try:
        # Initialize chunker
        config = ChunkerConfig(max_tokens=max_tokens)
        chunker_obj: BaseChunker
        if chunker == 'sentence':
            chunker_obj = SentenceChunker(config)
        else:
            chunker_obj = RecursiveCharacterChunker(config)

        # Read and chunk
        with open(input, encoding='utf-8') as f:
            text = f.read()

        chunks = chunker_obj.chunk(text)

        # Display
        click.echo(f"\nChunked into {len(chunks)} chunks:\n")
        for i, chunk in enumerate(chunks[:5], 1):  # Show first 5
            click.echo(f"[Chunk {i}] ({chunk.start}-{chunk.end})")
            click.echo(f"Type: {chunk.chunk_type.value}")
            click.echo(f"Text: {chunk.text[:150]}...")
            click.echo()

        if len(chunks) > 5:
            click.echo(f"... and {len(chunks) - 5} more chunks")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    cli()
