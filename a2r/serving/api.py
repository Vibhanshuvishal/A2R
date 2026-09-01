from __future__ import annotations

import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from a2r.graph import A2REngine, build_engine
from a2r.serving.schema import (
    CacheStatsResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SessionCreateRequest,
    SessionDetailResponse,
    SessionUpdateTitleRequest,
)


def create_app(engine: A2REngine | None = None, mount_ui: bool = True) -> FastAPI:
    engine = engine or build_engine()
    app = FastAPI(title="A2R — Adaptive Retrieval Router", version="0.2.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/query", response_model=QueryResponse)
    async def handle_query(request: QueryRequest):
        return engine.query(request.query, session_id=request.session_id)

    @app.post("/query-stream")
    async def handle_query_stream_post(request: QueryRequest):
        def event_generator():
            try:
                for event in engine.stream_query(request.query, session_id=request.session_id):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'event': 'error', 'message': str(exc)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/query-stream")
    async def handle_query_stream_get(query: str, session_id: str = ""):
        def event_generator():
            try:
                for event in engine.stream_query(query, session_id=session_id):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'event': 'error', 'message': str(exc)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/feedback", response_model=FeedbackResponse)
    async def handle_feedback(request: FeedbackRequest):
        weight = engine.feedback(request.query_id, request.signal)
        return FeedbackResponse(acknowledged=weight is not None, new_weight=weight)

    # Multi-turn session endpoints
    @app.post("/sessions")
    async def create_session(req: SessionCreateRequest):
        sess_id = engine.session_manager.create_session(user_id=req.user_id, title=req.title)
        return {"session_id": sess_id, "title": req.title}

    @app.get("/sessions")
    async def list_sessions(user_id: str = "default", limit: int = 30):
        return engine.session_manager.list_sessions(user_id=user_id, limit=limit)

    @app.get("/sessions/{session_id}", response_model=SessionDetailResponse)
    async def get_session(session_id: str):
        sess = engine.session_manager.get_session(session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        messages = engine.session_manager.load_session_messages(session_id)
        return SessionDetailResponse(session=sess, messages=messages)

    @app.patch("/sessions/{session_id}")
    async def update_session_title(session_id: str, req: SessionUpdateTitleRequest):
        ok = engine.session_manager.update_title(session_id, req.title)
        if not ok:
            raise HTTPException(status_code=404, detail="Session not found or not modified")
        return {"acknowledged": True, "title": req.title}

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str, hard: bool = False):
        ok = engine.session_manager.delete_session(session_id, soft=not hard)
        if not ok:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"acknowledged": True}

    # Semantic cache endpoints
    @app.get("/cache/stats", response_model=CacheStatsResponse)
    async def cache_stats():
        if not engine.cache:
            return {
                "cache_size": 0,
                "hits": 0,
                "misses": 0,
                "total_lookups": 0,
                "hit_rate": 0.0,
                "threshold": 0.85,
            }
        return engine.cache.stats()

    @app.post("/cache/clear")
    async def clear_cache():
        if engine.cache:
            engine.cache.clear()
        return {"acknowledged": True}

    @app.get("/weights")
    async def weights():
        return {"matrix": engine.router.matrix(), **engine.logger.stats()}

    @app.get("/stats")
    async def stats():
        return engine.logger.stats()

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return engine.health()

    # Mount UI layers
    if mount_ui:
        from fastapi.responses import HTMLResponse
        from gradio import mount_gradio_app
        from ui.app import build_ui

        # Mount static directory for modern custom SPA at /
        static_dir = Path(__file__).parents[2] / "ui" / "static"
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

            @app.get("/")
            async def serve_index():
                index_path = static_dir / "index.html"
                return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

        # Mount Gradio at /gradio
        app = mount_gradio_app(app, build_ui(engine), path="/gradio")

    return app


app = create_app()
