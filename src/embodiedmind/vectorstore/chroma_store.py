"""
ChromaDB vector store wrapper.
Handles persistence, deduplication via content_hash, and metadata filtering.
"""

import logging
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from embodiedmind.config import settings
from embodiedmind.vectorstore.schema import validate_metadata

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self._persist_dir = persist_dir or settings.chroma_persist_dir
        self._collection_name = collection_name or settings.chroma_collection_name
        self._embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self._store: Optional[Chroma] = None

    def _ensure_store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embeddings,
                persist_directory=self._persist_dir,
            )
        return self._store

    @property
    def store(self) -> Chroma:
        return self._ensure_store()

    def add_documents(self, documents: list[Document]) -> None:
        """
        Add documents to the store, skipping duplicates based on content_hash.
        Validates metadata before insertion.
        """
        store = self._ensure_store()

        # Filter documents with missing required metadata
        valid_docs = []
        for doc in documents:
            missing = validate_metadata(doc.metadata)
            if missing:
                logger.warning(
                    "Document missing required metadata fields %s — skipping: %s",
                    missing,
                    doc.metadata.get("source_url", "?"),
                )
                continue
            valid_docs.append(doc)

        if not valid_docs:
            return

        # Dedup by content_hash
        existing_hashes = self._get_existing_hashes()
        new_docs = [
            d for d in valid_docs
            if d.metadata.get("content_hash") not in existing_hashes
        ]
        skipped = len(valid_docs) - len(new_docs)
        if skipped:
            logger.info("Skipped %d duplicate documents (content_hash match)", skipped)

        if new_docs:
            store.add_documents(new_docs)
            logger.info("Added %d new documents to ChromaDB", len(new_docs))

    def _get_existing_hashes(self) -> set[str]:
        store = self._ensure_store()
        try:
            result = store.get(include=["metadatas"])
            return {
                m.get("content_hash", "")
                for m in (result.get("metadatas") or [])
                if m.get("content_hash")
            }
        except Exception as exc:
            logger.warning("Could not fetch existing hashes: %s", exc)
            return set()

    def as_retriever(self, k: int = 5, **kwargs):
        return self._ensure_store().as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 3, **kwargs},
        )

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        return self._ensure_store().similarity_search(query, k=k)

    def collection_stats(self) -> dict:
        store = self._ensure_store()
        try:
            result = store.get(include=["metadatas"])
            total = len(result.get("ids") or [])
            sources: dict[str, int] = {}
            for meta in (result.get("metadatas") or []):
                src = meta.get("source_name", "unknown")
                sources[src] = sources.get(src, 0) + 1
            return {"total_chunks": total, "by_source": sources}
        except Exception as exc:
            logger.warning("Could not get collection stats: %s", exc)
            return {}


# Module-level singleton
_store_instance: Optional[ChromaVectorStore] = None


def get_vector_store() -> ChromaVectorStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = ChromaVectorStore()
    return _store_instance
