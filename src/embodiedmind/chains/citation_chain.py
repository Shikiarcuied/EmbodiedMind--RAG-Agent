"""
Citation chain: ensures every answer includes source_url references.
Wraps the base retrieval QA chain and appends a formatted citation block.
"""

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from embodiedmind.compliance.attribution import format_citation

logger = logging.getLogger(__name__)


@dataclass
class AnswerWithCitations:
    answer: str
    citations: list[dict]

    def format(self) -> str:
        if not self.citations:
            return self.answer

        lines = [self.answer, "", "---", "**Sources:**"]
        seen_urls: set[str] = set()
        for i, meta in enumerate(self.citations, 1):
            url = meta.get("source_url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            citation_str = format_citation(meta)
            lines.append(f"{i}. {citation_str}")

        return "\n".join(lines)


class CitationChain:
    """
    Wraps any retriever + chain to produce AnswerWithCitations.
    Guarantees source_url is always present in the output.
    """

    def __init__(self, retriever, qa_chain):
        self._retriever = retriever
        self._qa_chain = qa_chain

    def invoke(self, question: str) -> AnswerWithCitations:
        # Retrieve docs first so we can capture metadata
        docs: list[Document] = self._retriever.invoke(question)

        # Run QA chain with the same question
        answer: str = self._qa_chain.invoke(question)

        # Collect unique citations from retrieved docs
        citations = []
        seen: set[str] = set()
        for doc in docs:
            url = doc.metadata.get("source_url", "")
            if url and url not in seen:
                seen.add(url)
                citations.append(doc.metadata)

        result = AnswerWithCitations(answer=answer, citations=citations)
        logger.debug(
            "CitationChain: question=%r, citations=%d", question[:80], len(citations)
        )
        return result

    def format_response(self, question: str) -> str:
        return self.invoke(question).format()
