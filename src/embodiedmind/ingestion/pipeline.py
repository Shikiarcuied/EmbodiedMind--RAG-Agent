"""
Ingestion pipeline.
Loads documents from all configured sources, chunks them, and upserts into the vector store.
Compliance modules are invoked at every step.
"""

import asyncio
import logging
from typing import Optional

from langchain_core.documents import Document

from embodiedmind.compliance.robots_checker import RobotsChecker
from embodiedmind.config import settings
from embodiedmind.config.sources import KNOWLEDGE_SOURCES, KnowledgeSource
from embodiedmind.ingestion.chunker import chunk_documents
from embodiedmind.ingestion.loaders import (
    crawl_site,
    load_github_repo_docs,
    load_github_repo_via_clone,
)

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, vector_store=None):
        self._vector_store = vector_store
        self._checker = RobotsChecker()

    def set_vector_store(self, vector_store) -> None:
        self._vector_store = vector_store

    # ------------------------------------------------------------------
    # Per-source loaders
    # ------------------------------------------------------------------

    def _load_github_source(self, source: KnowledgeSource) -> list[Document]:
        repo_full_name = source.extra.get("repo_full_name", "")
        exts = source.extra.get("target_extensions", [".md"])
        clone_dir = source.extra.get("clone_dir", "")

        if not settings.github_token:
            logger.warning(
                "GITHUB_TOKEN not set — falling back to git clone for %s", source.name
            )
            if clone_dir:
                docs = []
                for ext in exts:
                    docs.extend(
                        load_github_repo_via_clone(
                            repo_url=source.base_url,
                            clone_dir=clone_dir,
                            ext=ext,
                            license_str=source.license,
                            source_name=source.name,
                        )
                    )
                return docs
            else:
                logger.error("No clone_dir configured for %s — skipping", source.name)
                return []

        docs = []
        for ext in exts:
            docs.extend(
                load_github_repo_docs(
                    repo_full_name=repo_full_name,
                    token=settings.github_token,
                    ext=ext,
                    license_str=source.license,
                )
            )
        return docs

    async def _load_web_source(self, source: KnowledgeSource) -> list[Document]:
        start_urls = source.extra.get("start_urls", [source.base_url])
        return await crawl_site(
            start_urls=start_urls,
            checker=self._checker,
            license_str=source.license,
            source_name=source.name,
        )

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    async def ingest_source(self, source: KnowledgeSource) -> list[Document]:
        logger.info("Ingesting source: %s (%s)", source.name, source.access_method)

        if source.access_method in ("github_api", "git_clone"):
            docs = self._load_github_source(source)
        elif source.access_method == "web_crawl":
            docs = await self._load_web_source(source)
        else:
            logger.error("Unknown access_method '%s' for %s", source.access_method, source.name)
            return []

        chunks = chunk_documents(docs)
        logger.info(
            "Source %s: %d raw docs -> %d chunks", source.name, len(docs), len(chunks)
        )

        if self._vector_store is not None and chunks:
            self._vector_store.add_documents(chunks)
            logger.info("Upserted %d chunks from %s", len(chunks), source.name)

        return chunks

    async def ingest_all(
        self,
        source_names: Optional[list[str]] = None,
    ) -> dict[str, int]:
        """
        Ingest all (or a subset of) configured sources.
        Returns a dict mapping source name -> chunk count.
        """
        sources = KNOWLEDGE_SOURCES
        if source_names:
            sources = [s for s in sources if s.name in source_names]

        results: dict[str, int] = {}
        for source in sources:
            chunks = await self.ingest_source(source)
            results[source.name] = len(chunks)

        total = sum(results.values())
        logger.info("Ingestion complete: %d total chunks across %d sources", total, len(results))
        return results

    def ingest_all_sync(self, source_names: Optional[list[str]] = None) -> dict[str, int]:
        return asyncio.run(self.ingest_all(source_names=source_names))
