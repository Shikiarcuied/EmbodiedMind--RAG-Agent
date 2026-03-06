#!/usr/bin/env python3
"""
Ingest only GitHub-based sources (Lumina Embodied-AI-Guide).
Uses GitHub REST API or git clone — never scrapes github.com web pages.
"""

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


def main() -> None:
    from embodiedmind.config import settings
    from embodiedmind.ingestion.loaders import (
        load_github_repo_docs,
        load_github_repo_via_clone,
    )
    from embodiedmind.ingestion.chunker import chunk_documents
    from embodiedmind.vectorstore import get_vector_store

    console.rule("[bold cyan]EmbodiedMind — GitHub Ingestion")

    if not settings.github_token or settings.github_token == "ghp_...":
        console.print(
            "[yellow]GITHUB_TOKEN not set — using git clone fallback[/yellow]"
        )
        docs = load_github_repo_via_clone(
            repo_url="https://github.com/TianxingChen/Embodied-AI-Guide",
            clone_dir="data/repos/lumina",
            ext=".md",
            license_str="unknown",
            source_name="lumina_embodied_ai_guide",
        )
    else:
        console.print("[green]Using GitHub REST API[/green]")
        docs = load_github_repo_docs(
            repo_full_name="TianxingChen/Embodied-AI-Guide",
            token=settings.github_token,
            ext=".md",
            license_str="unknown",
        )

    console.print(f"Loaded [bold]{len(docs)}[/bold] documents")

    chunks = chunk_documents(docs)
    console.print(f"Created [bold]{len(chunks)}[/bold] chunks")

    vs = get_vector_store()
    vs.add_documents(chunks)

    stats = vs.collection_stats()
    console.print(
        f"\n[green]Done![/green] Total chunks in DB: [bold]{stats.get('total_chunks', 0)}[/bold]"
    )


if __name__ == "__main__":
    main()
