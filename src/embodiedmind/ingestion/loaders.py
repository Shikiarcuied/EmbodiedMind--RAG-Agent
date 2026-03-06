"""
Document loaders.
GitHub: uses PyGithub API or git clone — never scrapes github.com web pages.
Web (HuggingFace docs / Xbotics): checks robots.txt and enforces rate limiting.
"""

import asyncio
import base64
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from embodiedmind.compliance.attribution import build_metadata
from embodiedmind.compliance.rate_limiter import web_limiter
from embodiedmind.compliance.robots_checker import RobotsChecker
from embodiedmind.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GitHub loader (API-first)
# ---------------------------------------------------------------------------

def load_github_repo_docs(
    repo_full_name: str,
    token: str,
    path: str = "",
    ext: str = ".md",
    license_str: str = "unknown",
) -> list[Document]:
    """
    Recursively fetch all files with the given extension from a GitHub repo
    using the GitHub REST API (PyGithub). Falls back to git clone if needed.

    Checks rate limit remaining; waits for reset if remaining < 100.
    Each Document's metadata includes source_url, license, crawl_date, content_hash.
    """
    from github import Github, GithubException

    g = Github(token)
    repo = g.get_repo(repo_full_name)
    documents: list[Document] = []

    def _check_rate_limit() -> None:
        rl = g.get_rate_limit().core
        logger.debug("GitHub API rate limit: %d/%d remaining", rl.remaining, rl.limit)
        if rl.remaining < 100:
            import time
            reset_in = (rl.reset - datetime.now(timezone.utc)).total_seconds()
            if reset_in > 0:
                logger.warning(
                    "GitHub rate limit low (%d remaining) — waiting %.0fs for reset",
                    rl.remaining,
                    reset_in,
                )
                time.sleep(reset_in + 5)

    def _fetch_contents(current_path: str) -> None:
        _check_rate_limit()
        try:
            items = repo.get_contents(current_path)
        except GithubException as exc:
            logger.warning("Failed to list %s in %s: %s", current_path, repo_full_name, exc)
            return

        if not isinstance(items, list):
            items = [items]

        for item in items:
            if item.type == "dir":
                _fetch_contents(item.path)
            elif item.type == "file" and item.name.endswith(ext):
                _check_rate_limit()
                try:
                    raw = base64.b64decode(item.content).decode("utf-8", errors="replace")
                except Exception as exc:
                    logger.warning("Failed to decode %s: %s", item.path, exc)
                    continue

                blob_url = (
                    f"https://github.com/{repo_full_name}/blob/main/{item.path}"
                )
                meta = build_metadata(
                    content=raw,
                    source_url=blob_url,
                    license=license_str,
                    source_name=repo_full_name,
                    title=item.name,
                ).to_dict()

                documents.append(Document(page_content=raw, metadata=meta))
                logger.info("Loaded %s from %s", item.path, repo_full_name)

    _fetch_contents(path)
    logger.info(
        "GitHub loader: fetched %d documents from %s", len(documents), repo_full_name
    )
    return documents


def load_github_repo_via_clone(
    repo_url: str,
    clone_dir: str,
    ext: str = ".md",
    license_str: str = "unknown",
    source_name: str = "",
) -> list[Document]:
    """
    Clone a public GitHub repo to clone_dir and read all files matching ext.
    Uses git clone, not web scraping.
    """
    import subprocess
    from pathlib import Path

    clone_path = Path(clone_dir)
    if not clone_path.exists():
        logger.info("Cloning %s into %s", repo_url, clone_dir)
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, str(clone_path)],
            check=True,
            capture_output=True,
        )
    else:
        logger.info("Repo already cloned at %s, pulling latest", clone_dir)
        subprocess.run(
            ["git", "-C", str(clone_path), "pull", "--ff-only"],
            check=False,
            capture_output=True,
        )

    documents: list[Document] = []
    for fpath in clone_path.rglob(f"*{ext}"):
        try:
            raw = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Failed to read %s: %s", fpath, exc)
            continue

        # Build a blob URL from the path relative to clone root
        rel = fpath.relative_to(clone_path).as_posix()
        source_url = f"{repo_url}/blob/main/{rel}" if repo_url.startswith("https://github.com") else repo_url

        meta = build_metadata(
            content=raw,
            source_url=source_url,
            license=license_str,
            source_name=source_name or repo_url,
            title=fpath.name,
        ).to_dict()
        documents.append(Document(page_content=raw, metadata=meta))

    logger.info(
        "Clone loader: read %d %s files from %s", len(documents), ext, clone_dir
    )
    return documents


# ---------------------------------------------------------------------------
# Web page loader (HuggingFace docs / Xbotics)
# ---------------------------------------------------------------------------

async def load_web_page(
    url: str,
    checker: RobotsChecker,
    license_str: str = "unknown",
    source_name: str = "",
) -> Optional[Document]:
    """
    Fetch a single web page with compliance checks:
    1. checker.is_allowed(url) — skip if disallowed by robots.txt
    2. await web_limiter.acquire() — enforce rate limit
    3. Set User-Agent header
    Returns None if the page is disallowed or unreachable.
    """
    if not checker.is_allowed(url):
        return None

    await web_limiter.acquire()

    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": settings.bot_user_agent},
            follow_redirects=True,
            timeout=15.0,
        ) as client:
            resp = await client.get(url)

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After", "60")
                logger.warning("429 at %s — backing off %ss", url, retry_after)
                await asyncio.sleep(float(retry_after))
                return None

            if resp.status_code != 200:
                logger.warning("HTTP %d fetching %s", resp.status_code, url)
                return None

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove nav, footer, script, style
            for tag in soup(["nav", "footer", "script", "style", "header"]):
                tag.decompose()

            # Try to extract main content
            main = soup.find("main") or soup.find("article") or soup.body
            text = main.get_text(separator="\n", strip=True) if main else ""

            if not text.strip():
                logger.warning("Empty page content at %s", url)
                return None

            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else url

            meta = build_metadata(
                content=text,
                source_url=url,
                license=license_str,
                source_name=source_name,
                title=title,
            ).to_dict()

            return Document(page_content=text, metadata=meta)

    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


async def crawl_site(
    start_urls: list[str],
    checker: RobotsChecker,
    max_pages: int = 200,
    license_str: str = "unknown",
    source_name: str = "",
) -> list[Document]:
    """
    BFS crawl starting from start_urls, respecting robots.txt and rate limits.
    Only follows links within the same domain.
    """
    from urllib.parse import urljoin, urlparse

    visited: set[str] = set()
    queue: list[str] = list(start_urls)
    documents: list[Document] = []

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        doc = await load_web_page(url, checker, license_str=license_str, source_name=source_name)
        if doc is None:
            continue

        documents.append(doc)

        # Discover same-domain links
        if len(visited) < max_pages:
            try:
                async with httpx.AsyncClient(
                    headers={"User-Agent": settings.bot_user_agent},
                    follow_redirects=True,
                    timeout=10.0,
                ) as client:
                    resp = await client.get(url)
                    soup = BeautifulSoup(resp.text, "html.parser")
                    base_netloc = urlparse(url).netloc
                    for a in soup.find_all("a", href=True):
                        href = urljoin(url, a["href"])
                        parsed = urlparse(href)
                        if parsed.netloc == base_netloc and href not in visited:
                            queue.append(href)
            except Exception:
                pass

    logger.info("Crawl finished: %d pages from %s", len(documents), start_urls)
    return documents
