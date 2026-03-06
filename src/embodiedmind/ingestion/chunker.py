"""
Hybrid chunking strategy:
- Markdown-aware splitting for .md files
- Recursive character splitting for plain text / HTML
- Preserves attribution metadata on each chunk
"""

import logging
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from embodiedmind.config import settings

logger = logging.getLogger(__name__)

_MD_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks, preserving all metadata fields."""
    chunks: list[Document] = []

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MD_HEADERS,
        strip_headers=False,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for doc in documents:
        title = doc.metadata.get("title", "")
        is_markdown = title.endswith(".md") or title.endswith(".rst")

        try:
            if is_markdown:
                md_chunks = md_splitter.split_text(doc.page_content)
                # md_splitter returns Documents with header metadata
                for chunk in md_chunks:
                    # Merge parent metadata (attribution) with header metadata
                    merged_meta = {**doc.metadata, **chunk.metadata}
                    sub_chunks = text_splitter.split_documents(
                        [Document(page_content=chunk.page_content, metadata=merged_meta)]
                    )
                    chunks.extend(sub_chunks)
            else:
                sub_chunks = text_splitter.split_documents([doc])
                chunks.extend(sub_chunks)
        except Exception as exc:
            logger.warning(
                "Chunking failed for %s: %s — using whole document",
                doc.metadata.get("source_url", "?"),
                exc,
            )
            chunks.append(doc)

    logger.info("Chunker: %d documents -> %d chunks", len(documents), len(chunks))
    return chunks
