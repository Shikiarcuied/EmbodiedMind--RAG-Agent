"""
Every ingested Document must carry full attribution metadata:
  - source_url: original URL
  - license: content license (MIT / Apache-2.0 / CC-BY / unknown)
  - crawl_date: ingestion date (ISO 8601)
  - content_hash: SHA256 for deduplication and provenance
Answers must display source_url; content must not be presented as proprietary knowledge.
"""

import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional


@dataclass
class AttributionMetadata:
    source_url: str
    license: str
    crawl_date: str
    content_hash: str
    source_name: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_metadata(
    content: str,
    source_url: str,
    license: str = "unknown",
    source_name: Optional[str] = None,
    author: Optional[str] = None,
    title: Optional[str] = None,
) -> AttributionMetadata:
    return AttributionMetadata(
        source_url=source_url,
        license=license,
        crawl_date=datetime.now(timezone.utc).isoformat(),
        content_hash=compute_hash(content),
        source_name=source_name,
        author=author,
        title=title,
    )


def format_citation(metadata: dict) -> str:
    """Format a citation string suitable for displaying in answers."""
    url = metadata.get("source_url", "")
    title = metadata.get("title", "")
    source = metadata.get("source_name", "")
    parts = []
    if title:
        parts.append(title)
    if source:
        parts.append(f"[{source}]")
    if url:
        parts.append(url)
    return " — ".join(parts) if parts else url
