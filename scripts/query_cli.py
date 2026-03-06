#!/usr/bin/env python3
"""
CLI query interface.
Usage: poetry run python scripts/query_cli.py --query "What is Diffusion Policy?"
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.markdown import Markdown

logging.basicConfig(level=logging.WARNING)
console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="EmbodiedMind CLI query tool")
    parser.add_argument("--query", "-q", required=True, help="Question to ask")
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Use full ReAct agent (slower, uses web search & arXiv)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    from embodiedmind.agent.executor import EmbodiedMindAgent
    from embodiedmind.vectorstore import get_vector_store

    console.rule("[bold cyan]EmbodiedMind Query")
    console.print(f"Question: [bold]{args.query}[/bold]\n")

    vs = get_vector_store()
    agent = EmbodiedMindAgent(vs)

    if args.agent:
        console.print("[dim]Using ReAct agent...[/dim]")
        answer = agent.ask(args.query, use_agent=True)
        console.print(Markdown(answer))
    else:
        console.print("[dim]Using direct RAG chain...[/dim]")
        result = agent.ask_with_citations(args.query)
        console.print(Markdown(result.format()))

        if result.citations:
            console.print("\n[bold]Sources:[/bold]")
            seen: set[str] = set()
            for i, meta in enumerate(result.citations, 1):
                url = meta.get("source_url", "")
                if url in seen:
                    continue
                seen.add(url)
                console.print(f"  {i}. {url}")


if __name__ == "__main__":
    main()
