from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-large"
    openai_chat_model: str = "gpt-4o"

    # GitHub
    github_token: str = ""

    # Vector DB
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "embodied_knowledge"

    # Pinecone (optional)
    pinecone_api_key: str = ""
    pinecone_index_name: str = "embodiedmind"

    # Search
    tavily_api_key: str = ""

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 200

    # Rate limiting
    crawl_delay_seconds: float = 1.0
    github_api_max_per_hour: int = 4500

    # Compliance
    bot_contact_email: str = "your@email.com"

    @property
    def bot_user_agent(self) -> str:
        return f"EmbodiedMindBot/1.0 (research; contact: {self.bot_contact_email})"


settings = Settings()
