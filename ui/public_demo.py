"""Hugging Face Space entry point. Set A2R_RUNTIME_MODE=public_demo."""
from a2r.graph import build_engine
from ui.app import build_ui

engine = build_engine()
engine.ensure_ingested()
demo = build_ui(engine)

if __name__ == "__main__":
    demo.launch()
