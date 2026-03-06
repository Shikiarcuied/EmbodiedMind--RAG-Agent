"""
Gradio Web UI for EmbodiedMind.
Disclaimer footer is mandatory and cannot be removed.
"""

import logging

import gradio as gr

from embodiedmind.agent.executor import EmbodiedMindAgent
from embodiedmind.vectorstore import get_vector_store

logger = logging.getLogger(__name__)

# --- Mandatory disclaimer (per project spec, must not be omitted) ---
DISCLAIMER = (
    "**免责声明 / Disclaimer**\n\n"
    "本系统内容来源于 Xbotics、HuggingFace LeRobot、Lumina 具身智能社区的公开资料。\n"
    "所有内容版权归原作者所有，本系统仅供非商业学习研究使用。\n"
    "引用内容均附原始链接，如有侵权请联系删除。\n\n"
    "This system aggregates publicly available content from Xbotics, HuggingFace LeRobot, "
    "and the Lumina Embodied AI community. All content copyrights belong to their original "
    "authors. This system is for non-commercial learning and research only. "
    "All citations include original source links. Contact us for removal requests."
)

_agent: EmbodiedMindAgent | None = None


def _get_agent() -> EmbodiedMindAgent:
    global _agent
    if _agent is None:
        logger.info("Initializing EmbodiedMind agent...")
        vs = get_vector_store()
        _agent = EmbodiedMindAgent(vs)
    return _agent


def respond(
    message: str,
    history: list[tuple[str, str]],
    use_agent: bool,
) -> tuple[str, list[tuple[str, str]]]:
    if not message.strip():
        return "", history

    agent = _get_agent()
    try:
        result = agent.ask_with_citations(message)
        answer = result.format()
    except Exception as exc:
        logger.error("Agent error: %s", exc, exc_info=True)
        answer = f"Error processing your question: {exc}"

    history = history + [(message, answer)]
    return "", history


def get_stats() -> str:
    vs = get_vector_store()
    stats = vs.collection_stats()
    total = stats.get("total_chunks", 0)
    by_source = stats.get("by_source", {})
    lines = [f"Total chunks: **{total}**"]
    for src, count in by_source.items():
        lines.append(f"- {src}: {count}")
    return "\n".join(lines)


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="EmbodiedMind — 具身智能知识问答",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# EmbodiedMind\n"
            "**具身智能垂直领域 RAG 知识问答 Agent**\n\n"
            "基于 Lumina Embodied-AI-Guide、HuggingFace LeRobot 和 Xbotics 社区的知识库，"
            "回答具身智能相关问题，每个回答附带来源引用。"
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="对话 / Conversation",
                    height=520,
                    bubble_full_width=False,
                )
                with gr.Row():
                    msg = gr.Textbox(
                        label="提问 / Question",
                        placeholder="例如：什么是 Diffusion Policy？/ What is Diffusion Policy?",
                        lines=2,
                        scale=4,
                    )
                    submit_btn = gr.Button("发送 / Send", variant="primary", scale=1)

                use_agent_toggle = gr.Checkbox(
                    label="启用 ReAct Agent（更强，但更慢 / Enables web search & arXiv tools）",
                    value=False,
                )

                clear_btn = gr.Button("清空对话 / Clear", variant="secondary")

            with gr.Column(scale=1):
                gr.Markdown("### 知识库统计 / KB Stats")
                stats_display = gr.Markdown("Loading...")
                refresh_stats_btn = gr.Button("刷新 / Refresh")

        # --- Mandatory disclaimer footer ---
        gr.Markdown("---")
        gr.Markdown(DISCLAIMER)

        # --- Event handlers ---
        submit_btn.click(
            respond,
            inputs=[msg, chatbot, use_agent_toggle],
            outputs=[msg, chatbot],
        )
        msg.submit(
            respond,
            inputs=[msg, chatbot, use_agent_toggle],
            outputs=[msg, chatbot],
        )
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
        refresh_stats_btn.click(get_stats, outputs=[stats_display])
        app.load(get_stats, outputs=[stats_display])

    return app


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
