"""Knowledge source definitions with compliance metadata."""

from dataclasses import dataclass, field


@dataclass
class KnowledgeSource:
    name: str
    description: str
    access_method: str  # "github_api" | "git_clone" | "web_crawl"
    base_url: str
    allowed_paths: list[str]
    license: str
    commercial_use: bool = False
    crawl_delay: float = 1.0
    extra: dict = field(default_factory=dict)


KNOWLEDGE_SOURCES: list[KnowledgeSource] = [
    KnowledgeSource(
        name="lumina_embodied_ai_guide",
        description="Lumina Embodied-AI-Guide — comprehensive embodied AI learning resources",
        access_method="github_api",
        base_url="https://github.com/TianxingChen/Embodied-AI-Guide",
        allowed_paths=["/"],
        license="unknown",
        commercial_use=False,
        extra={
            "repo_full_name": "TianxingChen/Embodied-AI-Guide",
            "target_extensions": [".md", ".rst", ".txt"],
            "clone_dir": "data/repos/lumina",
        },
    ),
    KnowledgeSource(
        name="lerobot_docs",
        description="HuggingFace LeRobot official documentation",
        access_method="web_crawl",
        base_url="https://huggingface.co",
        allowed_paths=["/docs/lerobot"],
        license="Apache-2.0",
        commercial_use=False,
        crawl_delay=1.0,
        extra={
            "start_urls": [
                "https://huggingface.co/docs/lerobot/index",
            ],
        },
    ),
    KnowledgeSource(
        name="xbotics_community",
        description="Xbotics embodied intelligence community",
        access_method="web_crawl",
        base_url="https://xbotics-embodied.site",
        allowed_paths=["/"],  # will be validated against robots.txt at runtime
        license="unknown",
        commercial_use=False,
        crawl_delay=1.0,
        extra={
            "start_urls": [
                "https://xbotics-embodied.site",
            ],
        },
    ),
]
