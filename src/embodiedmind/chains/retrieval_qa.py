"""
RAG retrieval-QA chain.
Retrieves relevant chunks, formats context with source attribution, and generates answers.
"""

import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from embodiedmind.config import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are EmbodiedMind, an expert assistant specializing in embodied AI, \
robotics, and related technologies. You answer questions based on retrieved knowledge from \
curated sources including Lumina Embodied-AI-Guide, HuggingFace LeRobot, and Xbotics community.

IMPORTANT RULES:
1. Base your answers on the provided context documents.
2. Always cite your sources using the source_url from the context.
3. If the context doesn't contain enough information, say so clearly.
4. Do NOT present retrieved content as your own knowledge — always attribute it.
5. Answer in the same language as the user's question (Chinese or English).

Context documents:
{context}
"""

_HUMAN_PROMPT = "{question}"


def _format_context(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        url = doc.metadata.get("source_url", "unknown")
        title = doc.metadata.get("title", "")
        source = doc.metadata.get("source_name", "")
        header = f"[Doc {i}] {title} | {source} | {url}"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_retrieval_qa_chain(retriever):
    """
    Build a LCEL retrieval-QA chain.
    Returns a runnable that accepts {"question": str} and returns str.
    """
    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        openai_api_key=settings.openai_api_key,
        temperature=0.1,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", _HUMAN_PROMPT),
    ])

    chain = (
        RunnableParallel(
            context=retriever | _format_context,
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
