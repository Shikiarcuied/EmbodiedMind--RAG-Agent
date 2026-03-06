"""
Conversation memory management for multi-turn dialogue.
Uses LangChain's in-memory chat history.
"""

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from embodiedmind.config import settings


class InMemoryChatHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages: list[BaseMessage] = []

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


# Session store: session_id -> InMemoryChatHistory
_session_store: dict[str, InMemoryChatHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatHistory:
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatHistory()
    return _session_store[session_id]


def build_conversational_chain(retriever):
    """
    Build a conversational retrieval chain with message history.
    Returns a RunnableWithMessageHistory instance.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        openai_api_key=settings.openai_api_key,
        temperature=0.1,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are EmbodiedMind, an embodied AI expert. "
         "Answer questions based on retrieved context. "
         "Always cite sources. Context: {context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    def retrieve_and_format(inputs: dict) -> dict:
        docs = retriever.invoke(inputs["question"])
        context = "\n\n".join(
            f"[{d.metadata.get('source_url', '')}]\n{d.page_content}" for d in docs
        )
        return {**inputs, "context": context}

    chain = (
        RunnablePassthrough.assign(context=lambda x: retrieve_and_format(x)["context"])
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
