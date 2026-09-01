from __future__ import annotations

import gradio as gr

from a2r.graph import A2REngine, build_engine


def build_ui(engine: A2REngine):
    def ask(query: str, history: list[dict]):
        try:
            result = engine.query(query)
        except ValueError as exc:
            return history, f"Input error: {exc}", ""
        sources = "\n".join(f"- {source}" for source in result["sources"]) or "No internal source matched."
        badge = f"\n\n**Source: {result['source_badge']}**\n{sources}"
        history = history + [{"role": "user", "content": query}, {"role": "assistant", "content": result["answer"] + badge}]
        return history, "Ready", result["query_id"]

    def feedback(query_id: str, signal: str):
        if not query_id:
            return "Ask a question before leaving feedback."
        weight = engine.feedback(query_id, signal)
        return "Feedback already recorded or query not found." if weight is None else f"Thanks — router weight is now {weight:.2f}."

    with gr.Blocks(title="A2R — Adaptive Knowledge Assistant") as demo:
        gr.Markdown("# A2R — Adaptive Knowledge Assistant\nLocal-first RAG. Answers are always attributed to internal documents.")
        if engine.config["runtime"]["mode"] == "public_demo":
            gr.Markdown("> **Demo data only:** learning resets when the free hosting environment restarts.")
        def get_model_status() -> str:
            h = engine.health()
            health = h["model_status"]
            cache_info = f" • Cache: {h['cache']['cache_size']} items ({round(h['cache']['hit_rate'] * 100)}% hits)" if h.get("cache") else ""
            return f"Model: `{health['model']}` — {health['detail']}{cache_info}. Offline extractive fallback is available."

        model_status_bar = gr.Markdown(get_model_status)
        demo.load(get_model_status, outputs=model_status_bar)
        history = gr.Chatbot(height=420)
        query_id = gr.State("")
        with gr.Row():
            query = gr.Textbox(label="Ask about billing, product, or HR policy", max_lines=2)
            submit = gr.Button("Ask", variant="primary")
        status = gr.Markdown("Ready")
        with gr.Row():
            helpful = gr.Button("✓ Helpful")
            unhelpful = gr.Button("✗ Not helpful")
        submit.click(ask, [query, history], [history, status, query_id])
        query.submit(ask, [query, history], [history, status, query_id])
        helpful.click(lambda identifier: feedback(identifier, "accept"), query_id, status)
        unhelpful.click(lambda identifier: feedback(identifier, "reject"), query_id, status)
    return demo


if __name__ == "__main__":
    # Development convenience. Production local mode is: uvicorn a2r.serving.api:app --port 7860
    build_ui(build_engine()).launch(server_name="0.0.0.0", server_port=7860, show_error=True)
