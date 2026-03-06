"""
Metadata schema for documents stored in the vector store.
Every document must include: source_url, license, crawl_date, content_hash.
"""

from typing import TypedDict, Optional


class DocumentMetadata(TypedDict, total=False):
    source_url: str          # Original URL (required)
    license: str             # e.g. "MIT", "Apache-2.0", "CC-BY", "unknown"
    crawl_date: str          # ISO 8601
    content_hash: str        # SHA-256 hex digest for dedup
    source_name: str         # Logical source name (e.g. "lumina_embodied_ai_guide")
    title: str               # Document/page title
    author: str              # Author name if known
    # Markdown header hierarchy fields (added by MarkdownHeaderTextSplitter)
    h1: Optional[str]
    h2: Optional[str]
    h3: Optional[str]
    h4: Optional[str]


REQUIRED_METADATA_FIELDS = {"source_url", "license", "crawl_date", "content_hash"}


def validate_metadata(metadata: dict) -> list[str]:
    """Return a list of missing required fields."""
    return [f for f in REQUIRED_METADATA_FIELDS if not metadata.get(f)]
