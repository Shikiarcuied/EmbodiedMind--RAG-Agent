#!/usr/bin/env python3
"""
Full ingestion: load all configured sources into ChromaDB.
Run check_compliance.py first!
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler(rich_tracebacks=True)],
    format="%(message)s",
)

console = Console()


async def main() -> None:
    from embodiedmind.ingestion.pipeline import IngestionPipeline
    from embodiedmind.vectorstore import get_vector_store

    console.rule("[bold cyan]EmbodiedMind — Full Ingestion")

    vs = get_vector_store()
    pipeline = IngestionPipeline(vector_store=vs)

    console.print("Starting ingestion of all configured sources...")
    results = await pipeline.ingest_all()

    console.print("\n[bold green]Ingestion complete![/bold green]")
    for source, count in results.items():
        console.print(f"  {source}: {count} chunks")

    stats = vs.collection_stats()
    console.print(f"\nTotal chunks in DB: [bold]{stats.get('total_chunks', 0)}[/bold]")


if __name__ == "__main__":
    asyncio.run(main())
