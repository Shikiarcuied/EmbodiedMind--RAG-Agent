# EmbodiedMind

> **具身智能垂直领域 RAG 知识问答 Agent**
>
> 基于 LangChain 构建，整合 Lumina Embodied-AI-Guide、HuggingFace LeRobot 文档和 Xbotics-Embodied-Guide三大知识库，支持中英文自然语言问答，每条回答强制附带原始来源链接。

---

## 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [技术架构](#技术架构)
- [目录结构](#目录结构)
- [安装教程（保姆级）](#安装教程保姆级)
  - [0. 环境要求](#0-环境要求)
  - [1. 克隆项目](#1-克隆项目)
  - [2. 安装 Poetry](#2-安装-poetry)
  - [3. 安装 Python 依赖](#3-安装-python-依赖)
  - [4. 安装 Playwright 浏览器](#4-安装-playwright-浏览器)
  - [5. 配置环境变量](#5-配置环境变量)
  - [6. 合规预检](#6-合规预检)
  - [7. 摄取知识库](#7-摄取知识库)
  - [8. 启动 Web UI](#8-启动-web-ui)
- [使用教程](#使用教程)
  - [Web 界面](#web-界面)
  - [命令行查询](#命令行查询)
  - [REST API](#rest-api)
  - [Python SDK 调用](#python-sdk-调用)
- [知识库来源配置](#知识库来源配置)
- [合规说明](#合规说明)
- [Docker 部署](#docker-部署)
- [开发指南](#开发指南)
- [常见问题](#常见问题)
- [版权与引用声明](#版权与引用声明)

---

## 项目简介

EmbodiedMind 是一个面向**具身智能（Embodied AI）**领域的垂直 RAG（检索增强生成）知识问答系统。它抓取并索引具身智能领域的权威公开资料，用户可以用自然语言（中文或英文）提问，系统从本地向量数据库中检索相关知识片段，由大语言模型综合作答，并强制附带每条引用的原始 URL 链接。

**适用场景：**

- 快速了解具身智能、机器人学习的核心概念（Diffusion Policy、ACT、VLA 等）
- 查询 LeRobot、RoboAgent 等开源框架的使用方法
- 学习具身 AI 的前沿论文和社区动态
- Hackathon Demo、课程作业、个人学习研究

---

## 功能特性

| 功能 | 说明 |
|------|------|
| **多源知识整合** | 同时索引 GitHub 仓库文档、HuggingFace 官方文档、Xbotics 社区三个来源 |
| **中英文双语问答** | 自动识别提问语言，以相同语言回答 |
| **强制来源引用** | 每条回答末尾附带原始 URL，不将他人内容作为自有知识呈现 |
| **合规爬取** | 自动读取 robots.txt，限速访问，携带联系邮件的 User-Agent |
| **内容去重** | 基于 SHA-256 content_hash 去重，避免重复摄取 |
| **增量更新** | APScheduler 定时任务，每天凌晨 2 点自动增量更新知识库 |
| **ReAct Agent 模式** | 可选启用完整 Agent，联网搜索（Tavily）+ arXiv 论文检索 |
| **多轮对话** | 支持上下文记忆的连续对话 |
| **Web UI** | Gradio 界面，开箱即用 |
| **REST API** | FastAPI 接口，便于集成到其他系统 |

---

## 技术架构

```
用户提问
    │
    ▼
┌─────────────────────────────────────────────┐
│             Gradio Web UI / CLI             │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│            EmbodiedMind Agent               │
│  ┌───────────────────┐  ┌────────────────┐  │
│  │    ReAct Agent    │  │ Citation Chain │  │
│  │    (可选，慢)      │  │  (默认，快速)   │  │
│  └─────────┬─────────┘  └───────┬────────┘  │
└────────────┼───────────────────┼────────────┘
             │                    │
             ▼                    ▼
┌─────────────────────┐  ┌───────────────────┐
│        Tools        │  │ Retrieval QA Chain│
│   · KB Search       │  │  GPT-4o 生成答案  │
│   · Tavily 搜索     │  └────────┬──────────┘
│   · arXiv 论文      │           │
└─────────────────────┘           │
                                  ▼
                   ┌──────────────────────────┐
                   │      ChromaDB 向量库      │
                   │   text-embedding-3-large │
                   └─────────────┬────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
┌──────────────────┐  ┌────────────────────┐  ┌──────────────────┐
│ GitHub API/clone │  │  HuggingFace Docs  │  │  Xbotics GitHub  │
│   Lumina Guide   │  │  遵守 robots.txt   |  │ GitHub API/clone │
└──────────────────┘  └────────────────────┘  └──────────────────┘
```

---

## 目录结构

```
embodiedmind/
├── .env.example              # 环境变量模板（复制为 .env 后填写）
├── .gitignore
├── pyproject.toml            # Poetry 依赖定义
├── LEGAL.md                  # 数据来源与版权声明
├── docker-compose.yml
├── Dockerfile
│
├── src/
│   └── embodiedmind/
│       ├── config/
│       │   ├── settings.py       # 全局配置（Pydantic BaseSettings）
│       │   └── sources.py        # 知识来源定义
│       │
│       ├── compliance/           # 合规模块（最高优先级）
│       │   ├── robots_checker.py # 自动检查 robots.txt
│       │   ├── rate_limiter.py   # 异步令牌桶限速
│       │   └── attribution.py    # 内容归因与元数据
│       │
│       ├── ingestion/
│       │   ├── loaders.py        # GitHub API / git clone / 合规网页爬取
│       │   ├── chunker.py        # Markdown 感知混合分块
│       │   ├── pipeline.py       # 摄取流水线
│       │   └── scheduler.py      # 定时增量更新
│       │
│       ├── vectorstore/
│       │   ├── schema.py         # 元数据 Schema 与验证
│       │   └── chroma_store.py   # ChromaDB 封装（去重、MMR 检索）
│       │
│       ├── chains/
│       │   ├── retrieval_qa.py   # LCEL RAG 链
│       │   ├── citation_chain.py # 强制来源引用链
│       │   └── memory.py         # 多轮对话记忆
│       │
│       ├── agent/
│       │   ├── tools.py          # KB / Tavily / arXiv 工具
│       │   └── executor.py       # Agent 执行器
│       │
│       ├── api/
│       │   └── router.py         # FastAPI 路由
│       │
│       └── ui/
│           └── gradio_app.py     # Gradio Web 界面
│
├── scripts/
│   ├── check_compliance.py   # 合规预检（必须先运行）
│   ├── ingest_all.py         # 全量摄取
│   ├── ingest_github.py      # 仅摄取 GitHub 来源
│   └── query_cli.py          # 命令行问答
│
├── tests/
│   ├── conftest.py
│   ├── test_compliance.py
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_agent.py
│
└── data/
    ├── chroma_db/            # 向量数据库持久化（已 .gitignore）
    ├── repos/                # git clone 目标目录（已 .gitignore）
    └── raw_cache/            # 原始缓存（已 .gitignore）
```

---

## 安装教程（保姆级）

### 0. 环境要求

在开始之前，请确认你的机器满足以下要求：

| 要求 | 版本 | 检查命令 |
|------|------|---------|
| Python | **3.11 或更高** | `python --version` |
| Git | 任意近期版本 | `git --version` |
| 磁盘空间 | 至少 2 GB（向量库 + 模型缓存） | — |
| 网络 | 可访问 OpenAI API、GitHub、HuggingFace | — |

> **Windows 用户注意：** 推荐使用 Git Bash 或 WSL2 执行以下命令，避免路径兼容问题。

---

### 1. 克隆项目

```bash
git clone <your-repo-url> embodiedmind
cd embodiedmind
```

### 2. 安装 Poetry

Poetry 是本项目的包管理工具。如果你还没有安装：

**macOS / Linux：**

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Windows（PowerShell）：**

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

安装完成后，将 Poetry 加入 PATH（安装时会有提示），然后验证：

```bash
poetry --version
# 输出类似：Poetry (version 1.8.x)
```

> 如果不想使用 Poetry，也可以用 pip + venv：
>
> ```bash
> python -m venv .venv
> source .venv/bin/activate   # Windows: .venv\Scripts\activate
> pip install -r requirements.txt  # 需先手动导出 requirements.txt
> ```

---

### 3. 安装 Python 依赖

```bash
poetry install
```

这会自动创建虚拟环境并安装所有依赖，包括 LangChain、ChromaDB、Gradio 等。首次安装可能需要 3–5 分钟。

验证安装：

```bash
poetry run python -c "import langchain; print(langchain.__version__)"
# 输出类似：0.3.x
```

---

### 4. 安装 Playwright 浏览器

Playwright 用于爬取网页内容（HuggingFace 文档、Xbotics 社区）：

```bash
poetry run playwright install chromium
```

> 如果在国内网络环境下载缓慢，可以设置代理：
>
> ```bash
> HTTPS_PROXY=http://your-proxy:port poetry run playwright install chromium
> ```

---

### 5. 配置环境变量

复制模板文件并填写你的密钥：

```bash
cp .env.example .env
```

用文本编辑器打开 `.env`，按以下说明逐项填写：

```env
# ============================================
# 必填项（缺少任何一项将无法正常运行）
# ============================================

# OpenAI API 密钥（用于 GPT-4o 问答和 text-embedding-3-large 向量化）
# 获取地址：https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-你的密钥

# GitHub Personal Access Token（用于通过 API 读取仓库内容，避免匿名限速）
# 获取地址：GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
# 权限只需勾选：repo → public_repo（只读公开仓库）
GITHUB_TOKEN=ghp_你的token

# 爬虫联系邮件（合规要求，写入 User-Agent 头，体现善意爬取）
BOT_CONTACT_EMAIL=你的邮箱@example.com

# ============================================
# 选填项（有默认值，可根据需要修改）
# ============================================

# 使用的 OpenAI 模型（默认 gpt-4o）
OPENAI_CHAT_MODEL=gpt-4o

# 向量化模型（默认 text-embedding-3-large，支持中英文双语）
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# ChromaDB 持久化目录（默认 ./data/chroma_db）
CHROMA_PERSIST_DIR=./data/chroma_db

# Tavily 搜索 API 密钥（可选，用于 Agent 模式联网搜索）
# 获取地址：https://app.tavily.com/
TAVILY_API_KEY=tvly-你的key

# 爬取间隔（默认 1 秒，遵守 robots.txt 的 Crawl-delay 要求）
CRAWL_DELAY_SECONDS=1.0

# 分块大小（tokens，默认 512）
CHUNK_SIZE=512
CHUNK_OVERLAP=200
```

> **安全提示：** `.env` 文件已被加入 `.gitignore`，不会被 git 追踪，请勿将密钥提交到版本控制。

---

### 6. 合规预检

**这一步是强制要求，必须通过后才能进行数据摄取。**

```bash
poetry run python scripts/check_compliance.py
```

预检内容包括：

1. GITHUB_TOKEN 是否配置且 API 调用成功
2. BOT_CONTACT_EMAIL 是否配置
3. GitHub API 剩余配额是否 > 500
4. xbotics-embodied.site 的 robots.txt 摘要
5. huggingface.co 的 robots.txt 摘要

**全部通过时的输出：**

```
✅ All compliance checks passed. Safe to proceed.
```

**如果某项失败，示例输出：**

```
❌ GITHUB_TOKEN not set or is placeholder
   → 请在 .env 中填写有效的 GITHUB_TOKEN
```

按照提示修复后重新运行预检，直到全部通过。

---

### 7. 摄取知识库

预检通过后，运行全量摄取（首次运行约需 10–30 分钟，取决于网络速度）：

```bash
poetry run python scripts/ingest_all.py
```

**摄取过程说明：**

1. 通过 GitHub API 读取 Lumina Embodied-AI-Guide 的所有 Markdown 文件
2. 遵守 robots.txt 爬取 HuggingFace LeRobot 文档
3. 遵守 robots.txt 爬取 Xbotics 社区内容
4. 按 Markdown 标题层级和字符数分块
5. 调用 OpenAI Embeddings API 生成向量
6. 存入 ChromaDB，基于 SHA-256 去重

**只摄取 GitHub 来源（速度更快，适合测试）：**

```bash
poetry run python scripts/ingest_github.py
```

---

### 8. 启动 Web UI

```bash
poetry run python -m embodiedmind.ui.gradio_app
```

浏览器访问：**<http://localhost:7860>**

---

## 使用教程

### Web 界面

启动后你会看到以下界面：

```
┌─────────────────────────────────────────────────────┐
│       EmbodiedMind — 具身智能垂直领域知识问答         │
├─────────────────────────────────────┬───────────────┤
│                                     │  Knowledge DB │
│  [用户] 什么是 Diffusion Policy?     │               │
│                                     │  lumina:  800 │
│  [AI] Diffusion Policy 是一种        │  lerobot: 300 │
│  基于扩散模型的策略学习方法，          │  xbotics: 134 │
│  能高效学习机器人操作行为。            │               │
│                                     │  Total: 1234  │
│  来源：                              │               │
│  1. https://github.com/...          │  [ Refresh ]  │
│                                     │               │
├─────────────────────────────────────┴───────────────┤
│  [提问...]                                  [发送]   │
│  □ 启用 ReAct Agent（更强，但更慢）                   │
│  [清空对话]                                          │
├─────────────────────────────────────────────────────┤
│  免责声明：内容来自公开资料，仅供非商业学习使用。       │
└─────────────────────────────────────────────────────┘
```

**使用技巧：**

- **默认模式（ReAct Agent 关闭）**：直接从本地知识库检索，响应快（1–3 秒）
- **Agent 模式（ReAct Agent 开启）**：联合本地知识库 + Tavily 联网搜索 + arXiv 论文，信息更全面，但响应较慢（5–15 秒）
- 支持中英文混合提问，AI 会用相同语言回答
- 每次回答末尾都会列出引用的原始链接

**示例问题：**

```
什么是 Diffusion Policy？它和 ACT 有什么区别？
LeRobot 框架怎么安装？支持哪些机器人硬件？
具身智能中的 VLA（Vision-Language-Action）模型是什么？
How does imitation learning work in robotics?
What datasets are available for robot manipulation?
```

---

### 命令行查询

适合脚本集成和快速测试：

```bash
# 基本查询
poetry run python scripts/query_cli.py --query "什么是 Diffusion Policy？"

# 启用 ReAct Agent（联网搜索）
poetry run python scripts/query_cli.py --query "最新的具身智能论文" --agent

# 显示详细日志
poetry run python scripts/query_cli.py --query "LeRobot 安装" --verbose
```

**输出示例：**

```
─────────────── EmbodiedMind Query ───────────────
Question: 什么是 Diffusion Policy？

Diffusion Policy 是一种基于扩散模型（Diffusion Model）的机器人策略学习方法，
由 Chi 等人在 2023 年提出。它将去噪扩散概率模型（DDPM）应用于行为克隆，
能够建模多模态动作分布，在复杂操作任务中表现优异...

---
**Sources:**
1. README.md — lumina_embodied_ai_guide — https://github.com/TianxingChen/Embodied-AI-Guide/blob/main/README.md
2. https://huggingface.co/docs/lerobot/index
```

---

### REST API

启动 API 服务器：

```bash
poetry run uvicorn embodiedmind.api.router:router --host 0.0.0.0 --port 8000 --app-dir src
```

**接口列表：**

#### `GET /api/v1/health`

健康检查。

```bash
curl http://localhost:8000/api/v1/health
# {"status": "ok", "service": "EmbodiedMind"}
```

#### `POST /api/v1/query`

提问接口。

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是 Diffusion Policy？", "use_agent": false}'
```

返回：

```json
{
  "question": "什么是 Diffusion Policy？",
  "answer": "Diffusion Policy 是一种基于扩散模型的机器人策略学习方法...",
  "citations": [
    {
      "source_url": "https://github.com/TianxingChen/Embodied-AI-Guide/blob/main/README.md",
      "license": "unknown",
      "crawl_date": "2025-01-01T00:00:00+00:00",
      "source_name": "lumina_embodied_ai_guide",
      "title": "README.md"
    }
  ]
}
```

#### `GET /api/v1/stats`

知识库统计信息。

```bash
curl http://localhost:8000/api/v1/stats
# {"total_chunks": 1234, "by_source": {"lumina_embodied_ai_guide": 800, ...}}
```

---

### Python SDK 调用

```python
import sys
sys.path.insert(0, "src")

from embodiedmind.vectorstore import get_vector_store
from embodiedmind.agent.executor import EmbodiedMindAgent

# 初始化
vs = get_vector_store()
agent = EmbodiedMindAgent(vs)

# 快速 RAG 问答
result = agent.ask_with_citations("什么是 ACT？")
print(result.answer)
print("\n来源：")
for c in result.citations:
    print(f"  - {c['source_url']}")

# 格式化输出（含来源）
formatted = result.format()
print(formatted)

# ReAct Agent 模式（更强，更慢）
answer = agent.ask("最新的具身智能进展", use_agent=True)
print(answer)
```

---

## 知识库来源配置

知识来源定义在 [src/embodiedmind/config/sources.py](src/embodiedmind/config/sources.py)，你可以按以下格式添加新来源：

```python
KnowledgeSource(
    name="my_source",                          # 唯一标识符
    description="我的自定义知识来源",
    access_method="web_crawl",                 # "github_api" | "git_clone" | "web_crawl"
    base_url="https://my-site.example.com",
    allowed_paths=["/docs"],                   # 仅爬取这些路径下的页面
    license="CC-BY-4.0",                       # 内容许可证
    commercial_use=False,
    crawl_delay=2.0,                           # 礼貌爬取间隔（秒）
    extra={
        "start_urls": ["https://my-site.example.com/docs"],
    },
)
```

添加后重新运行 `ingest_all.py` 即可摄取新来源。

---

## 合规说明

本项目严格遵守所有来源平台的服务条款，合规模块位于 [`src/embodiedmind/compliance/`](src/embodiedmind/compliance/)。

### 各来源访问方式

| 来源 | 访问方式 | 限速 | 依据 |
|------|---------|------|------|
| **GitHub（Lumina 仓库）** | GitHub REST API（认证 Token）或 `git clone` | ≤ 4500 req/hr | GitHub ToS §H；robots.txt 禁止爬取 `/*/tree/` |
| **HuggingFace 文档** | 遵守 robots.txt 的 HTTP 爬取 | ≥ 1 秒/请求 | HuggingFace Content Policy |
| **Xbotics 社区** | GitHub REST API / `git clone` | ≤ 4500 req/hr | GitHub ToS |

### 合规机制

- **robots.txt 自动检查**：每次摄取前获取并解析目标站点的 robots.txt，被禁止的 URL 自动跳过
- **限速控制**：令牌桶算法，确保不超过平台允许的请求频率
- **User-Agent 标识**：所有请求携带 `EmbodiedMindBot/1.0 (research; contact: 你的邮箱)`，体现善意爬取
- **429 自动退避**：收到 429 响应时自动等待 Retry-After 时间，不强行重试
- **内容归因**：每条文档携带 `source_url`、`license`、`crawl_date`、`content_hash` 四个元数据字段
- **答案引用**：所有回答强制展示来源链接，不将他人内容作为自有知识呈现

### 数据使用范围

- ✅ 个人学习与研究（非商业）
- ✅ Hackathon Demo 展示
- ⚠️ 商业化部署前须获得各来源平台的明确书面授权
- ❌ 禁止将抓取内容转售或二次分发
- ❌ 禁止将抓取内容用于训练商业模型（未经授权）

---

## Docker 部署

```bash
# 先填写 .env 文件（参见第 5 步）

# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f embodiedmind

# 运行摄取（一次性任务）
docker-compose --profile ingest up ingest

# 停止
docker-compose down
```

Web UI 访问：`http://localhost:7860`

---

## 开发指南

### 运行测试

```bash
# 运行全部测试
poetry run pytest tests/ -v

# 只运行合规模块测试
poetry run pytest tests/test_compliance.py -v

# 带覆盖率报告
poetry run pytest tests/ --cov=src/embodiedmind --cov-report=html
```

### 代码风格检查

```bash
poetry run ruff check src/ tests/
poetry run ruff format src/ tests/
```

### 增量摄取（只更新某个来源）

```python
from embodiedmind.ingestion.pipeline import IngestionPipeline
from embodiedmind.vectorstore import get_vector_store
import asyncio

vs = get_vector_store()
pipeline = IngestionPipeline(vector_store=vs)

# 只更新 GitHub 来源
asyncio.run(pipeline.ingest_all(source_names=["lumina_embodied_ai_guide"]))
```

### 启用定时自动更新

```python
from embodiedmind.ingestion.pipeline import IngestionPipeline
from embodiedmind.ingestion.scheduler import IngestionScheduler
from embodiedmind.vectorstore import get_vector_store

vs = get_vector_store()
pipeline = IngestionPipeline(vector_store=vs)
scheduler = IngestionScheduler(pipeline)

# 每天凌晨 2 点（UTC）自动增量更新
scheduler.start(hour=2, minute=0)
```

---

## 常见问题

**Q: 运行 `ingest_all.py` 时报 `openai.AuthenticationError`**
A: 检查 `.env` 中的 `OPENAI_API_KEY` 是否正确，注意不要有多余的空格或引号。

**Q: GitHub API 报 `403 Forbidden` 或 `rate limit exceeded`**
A: 确认 `GITHUB_TOKEN` 已填写且有效。匿名访问只有 60 req/hr，认证后 5000 req/hr。

**Q: 爬取 Xbotics 时所有页面都被跳过**
A: robots.txt 可能限制了爬取路径，这是系统的合规行为。可以查看日志了解具体哪些路径被禁止。

**Q: ChromaDB 启动时报 `sqlite3` 版本错误**
A: ChromaDB 需要 SQLite ≥ 3.35。在部分旧版 Linux 上可以通过 `pysqlite3-binary` 解决：

```bash
poetry add pysqlite3-binary
```

并在 `chroma_store.py` 开头添加：

```python
import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3
```

**Q: Gradio 界面打开后知识库统计显示 0**
A: 需要先运行 `ingest_all.py` 摄取知识库，UI 只是查询已有的数据。

**Q: 回答内容不准确或知识过时**
A: 运行 `ingest_all.py` 或 `ingest_github.py` 重新摄取最新内容，或手动触发定时任务 `scheduler.run_now()`。

**Q: 能否使用其他 LLM 替换 GPT-4o？**
A: 可以，修改 `src/embodiedmind/chains/retrieval_qa.py` 中的 `ChatOpenAI` 替换为 LangChain 支持的任意模型（如 `ChatAnthropic`、`ChatOllama` 等）。

---

## 版权与引用声明

### 本项目使用的开源软件

本项目基于以下开源框架和库构建，在此致谢：

| 库 | 版本 | 许可证 | 项目地址 |
|----|------|--------|---------|
| **LangChain** | ≥ 0.3.0 | MIT | <https://github.com/langchain-ai/langchain> |
| **LangChain OpenAI** | ≥ 0.2.0 | MIT | <https://github.com/langchain-ai/langchain> |
| **LangChain Community** | ≥ 0.3.0 | MIT | <https://github.com/langchain-ai/langchain> |
| **LangChain Chroma** | ≥ 0.1.0 | MIT | <https://github.com/langchain-ai/langchain> |
| **ChromaDB** | ≥ 0.5.0 | Apache-2.0 | <https://github.com/chroma-core/chroma> |
| **OpenAI Python SDK** | ≥ 1.0.0 | MIT | <https://github.com/openai/openai-python> |
| **Gradio** | ≥ 4.0.0 | Apache-2.0 | <https://github.com/gradio-app/gradio> |
| **FastAPI** | ≥ 0.110.0 | MIT | <https://github.com/tiangolo/fastapi> |
| **Pydantic / pydantic-settings** | ≥ 2.0.0 | MIT | <https://github.com/pydantic/pydantic> |
| **Playwright** | ≥ 1.44.0 | Apache-2.0 | <https://github.com/microsoft/playwright-python> |
| **BeautifulSoup4** | ≥ 4.12.0 | MIT | <https://www.crummy.com/software/BeautifulSoup/> |
| **APScheduler** | ≥ 3.10.0 | MIT | <https://github.com/agronholm/apscheduler> |
| **PyGithub** | ≥ 2.3.0 | LGPL-3.0 | <https://github.com/PyGithub/PyGithub> |
| **GitPython** | ≥ 3.1.0 | BSD-3-Clause | <https://github.com/gitpython-developers/GitPython> |
| **httpx** | ≥ 0.27.0 | BSD-3-Clause | <https://github.com/encode/httpx> |
| **tenacity** | ≥ 8.3.0 | Apache-2.0 | <https://github.com/jd/tenacity> |
| **Rich** | ≥ 13.0.0 | MIT | <https://github.com/Textualize/rich> |
| **arxiv.py** | ≥ 2.1.0 | MIT | <https://github.com/lukasschwab/arxiv.py> |
| **pypdf** | ≥ 4.0.0 | BSD-3-Clause | <https://github.com/py-pdf/pypdf> |
| **Tavily Python** | ≥ 0.3.0 | MIT | <https://github.com/tavily-ai/tavily-python> |

### 知识库内容来源声明

本系统索引并引用了以下公开资料，**所有内容版权归原作者所有**：

#### 1. Lumina Embodied-AI-Guide

- **仓库**：<https://github.com/TianxingChen/Embodied-AI-Guide>
- **作者**：TianxingChen 及贡献者
- **访问方式**：GitHub REST API / `git clone`（非网页爬取）
- **内容性质**：具身 AI 领域综合学习指南，汇集论文、框架、数据集等资源
- **使用范围**：非商业学习与研究
- **注意**：仓库许可证以实际 LICENSE 文件为准，使用前请自行核查

#### 2. HuggingFace LeRobot 文档

- **文档地址**：<https://huggingface.co/docs/lerobot>
- **项目仓库**：<https://github.com/huggingface/lerobot>
- **版权方**：HuggingFace Inc. 及社区贡献者
- **许可证**：Apache License 2.0
- **访问方式**：遵守 huggingface.co/robots.txt 的限速 HTTP 爬取
- **使用范围**：遵守 HuggingFace Terms of Service（非商业研究）

#### 3. Xbotics 具身智能社区

- **仓库**：<https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide>
- **版权方**：Xbotics 社区及内容贡献者
- **访问方式**：GitHub REST API / `git clone`（非网页爬取）
- **使用范围**：非商业学习与研究
- **注意**：仓库许可证以实际 LICENSE 文件为准，使用前请自行核查

### AI 模型与服务声明

| 服务 | 提供方 | 用途 | 服务条款 |
|------|--------|------|---------|
| GPT-4o | OpenAI | 问答生成 | <https://openai.com/policies/terms-of-use> |
| text-embedding-3-large | OpenAI | 文本向量化 | <https://openai.com/policies/terms-of-use> |
| Tavily Search API | Tavily AI | 联网搜索（可选） | <https://tavily.com/terms> |
| arXiv API | arXiv / Cornell University | 论文搜索（可选） | <https://arxiv.org/help/api/tou> |

### 引用本项目

如果你在学术研究或项目中使用了 EmbodiedMind，欢迎引用：

```bibtex
@software{embodiedmind2025,
  title  = {EmbodiedMind: A LangChain-Powered Embodied AI Knowledge Agent},
  year   = {2026},
  url    = {https://github.com/your-username/embodiedmind},
  note   = {Non-commercial research use only}
}
```

### 免责声明

```
本系统内容来源于 Xbotics、HuggingFace LeRobot、Lumina 具身智能社区的公开资料。
所有内容版权归原作者所有，本系统仅供非商业学习研究使用。
引用内容均附原始链接，如有侵权请联系删除。

This system aggregates publicly available content from Xbotics, HuggingFace LeRobot,
and the Lumina Embodied AI community. All content copyrights belong to their original
authors. This system is for non-commercial learning and research purposes only.
All citations include original source links. Contact us for removal requests.
```

---

*EmbodiedMind — 让具身智能知识触手可及*
