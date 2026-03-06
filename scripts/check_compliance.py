#!/usr/bin/env python3
"""
Pre-ingestion compliance check. Must pass before running ingest_all.py.
Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed.
"""

import os
import sys
import time

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.table import Table

console = Console()


def check_github_token() -> tuple[bool, str]:
    token = os.getenv("GITHUB_TOKEN", "")
    if not token or token == "ghp_...":
        return False, "GITHUB_TOKEN not set or is placeholder"

    try:
        from github import Github, GithubException

        g = Github(token)
        user = g.get_user()
        rl = g.get_rate_limit()
        return True, (
            f"Authenticated as {user.login} | "
            f"API remaining: {rl.core.remaining}/{rl.core.limit}"
        )
    except Exception as exc:
        return False, f"GitHub API call failed: {exc}"


def check_bot_email() -> tuple[bool, str]:
    email = os.getenv("BOT_CONTACT_EMAIL", "")
    if not email or email == "your@email.com":
        return False, "BOT_CONTACT_EMAIL not set or is placeholder"
    if "@" not in email:
        return False, f"BOT_CONTACT_EMAIL does not look like a valid email: {email}"
    return True, f"Contact email: {email}"


def check_github_rate_limit() -> tuple[bool, str]:
    token = os.getenv("GITHUB_TOKEN", "")
    if not token or token == "ghp_...":
        return False, "Cannot check rate limit — GITHUB_TOKEN not set"
    try:
        from github import Github

        g = Github(token)
        rl = g.get_rate_limit()
        remaining = rl.core.remaining
        if remaining < 500:
            return False, f"GitHub API rate limit too low: {remaining} remaining (need > 500)"
        return True, f"GitHub API rate limit OK: {remaining} remaining"
    except Exception as exc:
        return False, f"Rate limit check failed: {exc}"


def check_robots_txt(site_url: str) -> tuple[bool, str]:
    import httpx

    email = os.getenv("BOT_CONTACT_EMAIL", "your@email.com")
    user_agent = f"EmbodiedMindBot/1.0 (research; contact: {email})"

    robots_url = f"{site_url}/robots.txt"
    try:
        resp = httpx.get(
            robots_url,
            headers={"User-Agent": user_agent},
            follow_redirects=True,
            timeout=10.0,
        )
        if resp.status_code == 200:
            lines = resp.text.splitlines()
            # Show first 20 non-empty lines as summary
            summary_lines = [l for l in lines if l.strip()][:20]
            summary = "\n    ".join(summary_lines)
            return True, f"robots.txt fetched ({len(lines)} lines):\n    {summary}"
        elif resp.status_code == 404:
            return True, f"No robots.txt at {robots_url} (404) — treating as allow-all"
        else:
            return False, f"Unexpected HTTP {resp.status_code} fetching {robots_url}"
    except Exception as exc:
        return False, f"Failed to fetch {robots_url}: {exc}"


def main() -> int:
    console.rule("[bold cyan]EmbodiedMind Compliance Pre-Check")

    checks = [
        ("GITHUB_TOKEN configured & API accessible", check_github_token),
        ("BOT_CONTACT_EMAIL configured", check_bot_email),
        ("GitHub API rate limit > 500", check_github_rate_limit),
        ("xbotics-embodied.site robots.txt", lambda: check_robots_txt("https://xbotics-embodied.site")),
        ("huggingface.co robots.txt", lambda: check_robots_txt("https://huggingface.co")),
    ]

    table = Table(title="Compliance Check Results", show_lines=True)
    table.add_column("Check", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    all_passed = True
    for name, fn in checks:
        try:
            passed, detail = fn()
        except Exception as exc:
            passed, detail = False, f"Unexpected error: {exc}"

        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(name, status, detail)
        if not passed:
            all_passed = False

    console.print(table)

    if all_passed:
        console.print(
            "\n[bold green]✅ All compliance checks passed. Safe to proceed.[/bold green]"
        )
        return 0
    else:
        console.print(
            "\n[bold red]❌ One or more compliance checks failed. "
            "Fix the issues above before running ingestion.[/bold red]"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
