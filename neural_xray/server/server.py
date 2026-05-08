"""
Model Surgery MRI — FastAPI server entry point.

Starts the local API server that wraps the autopsy modules.
Run: python -m backend.server
"""

import asyncio
import logging
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .state import LogCapture, state

# ---------------------------------------------------------------------------
# Logging setup — capture all logs for WebSocket terminal streaming
# ---------------------------------------------------------------------------
log_fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s — %(message)s",
                            datefmt="%H:%M:%S")

handler_console = logging.StreamHandler(sys.stdout)
handler_console.setFormatter(log_fmt)

handler_ws = LogCapture()
handler_ws.setFormatter(log_fmt)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(handler_console)
root_logger.addHandler(handler_ws)

logger = logging.getLogger("model-surgery")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Model Surgery MRI",
    version="0.1.0",
    description="Local API server for neural model inspection and surgery",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register API routers
# ---------------------------------------------------------------------------
from .api.models import router as models_router      # noqa: E402
from .api.concepts import router as concepts_router   # noqa: E402
from .api.trace import router as trace_router         # noqa: E402
from .api.cartography import router as carto_router   # noqa: E402
from .api.cluster import router as cluster_router     # noqa: E402
from .api.sae import router as sae_router             # noqa: E402
from .api.surgery import router as surgery_router     # noqa: E402
from .api.system import router as system_router       # noqa: E402
from .api.diagnostics import router as diagnostics_router  # noqa: E402
from .api.evidence import router as evidence_router        # noqa: E402
from .api.chat import router as chat_router                # noqa: E402
from .api.attention import router as attention_router       # noqa: E402
from .api.patching import router as patching_router        # noqa: E402
from .api.abliteration import router as abliteration_router  # noqa: E402

app.include_router(models_router, prefix="/api/models", tags=["models"])
app.include_router(concepts_router, prefix="/api/concepts", tags=["concepts"])
app.include_router(trace_router, prefix="/api/trace", tags=["trace"])
app.include_router(carto_router, prefix="/api/cartography", tags=["cartography"])
app.include_router(cluster_router, prefix="/api/cluster", tags=["cluster"])
app.include_router(sae_router, prefix="/api/sae", tags=["sae"])
app.include_router(surgery_router, prefix="/api/surgery", tags=["surgery"])
app.include_router(system_router, prefix="/api/system", tags=["system"])
app.include_router(diagnostics_router, prefix="/api/diagnostics", tags=["diagnostics"])
app.include_router(evidence_router, prefix="/api/evidence", tags=["evidence"])
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(attention_router, prefix="/api/attention", tags=["attention"])
app.include_router(patching_router, prefix="/api/patching", tags=["patching"])
app.include_router(abliteration_router, prefix="/api/abliteration", tags=["abliteration"])


# ---------------------------------------------------------------------------
# WebSocket — terminal log streaming
# ---------------------------------------------------------------------------
@app.websocket("/ws/terminal")
async def ws_terminal(websocket: WebSocket):
    await websocket.accept()
    cursor = len(state.log_lines)
    try:
        while True:
            # Send any new lines since last check
            if cursor < len(state.log_lines):
                batch = state.log_lines[cursor:]
                cursor = len(state.log_lines)
                for line in batch:
                    await websocket.send_text(line)
            else:
                state.log_event.clear()
                # Wait up to 1s for new logs, then loop (heartbeat)
                try:
                    await asyncio.wait_for(state.log_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"status": "ok", "loaded_models": list(state.sessions.keys())}


# ---------------------------------------------------------------------------
# WebSocket — remote control bridge
# ---------------------------------------------------------------------------
_remote_frontends: list[WebSocket] = []
_remote_controllers: list[WebSocket] = []


@app.websocket("/ws/remote")
async def ws_remote(websocket: WebSocket):
    """
    Remote control bridge. Two roles:
      - 'controller' (Python CLI / agent) sends commands
      - 'frontend' (React app) receives commands, sends back state
    First message must be: {"role": "controller"} or {"role": "frontend"}
    """
    await websocket.accept()
    role = None
    try:
        # Wait for role handshake
        raw = await websocket.receive_text()
        import json as _json
        msg = _json.loads(raw)
        role = msg.get("role", "controller")

        if role == "frontend":
            _remote_frontends.append(websocket)
            await websocket.send_text(_json.dumps({"type": "ack", "role": "frontend"}))
            logger.info("Remote control: frontend connected")
            # Frontend relays state back — forward to all controllers
            while True:
                data = await websocket.receive_text()
                for ctrl in list(_remote_controllers):
                    try:
                        await ctrl.send_text(data)
                    except Exception:
                        _remote_controllers.remove(ctrl)
        else:
            _remote_controllers.append(websocket)
            await websocket.send_text(_json.dumps({
                "type": "ack", "role": "controller",
                "frontends": len(_remote_frontends),
            }))
            logger.info("Remote control: controller connected")
            # Controller sends commands — forward to all frontends
            while True:
                data = await websocket.receive_text()
                for fe in list(_remote_frontends):
                    try:
                        await fe.send_text(data)
                    except Exception:
                        _remote_frontends.remove(fe)
    except WebSocketDisconnect:
        pass
    finally:
        if role == "frontend" and websocket in _remote_frontends:
            _remote_frontends.remove(websocket)
            logger.info("Remote control: frontend disconnected")
        elif role == "controller" and websocket in _remote_controllers:
            _remote_controllers.remove(websocket)
            logger.info("Remote control: controller disconnected")


# ---------------------------------------------------------------------------
# Static frontend — serve the bundled React app at / (registered LAST so
# the SPA catch-all does not shadow /api/* and /ws/* routes)
# ---------------------------------------------------------------------------
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.responses import FileResponse   # noqa: E402
from fastapi import HTTPException as _HTTPException  # noqa: E402
from pathlib import Path as _Path            # noqa: E402

_static_dir = _Path(__file__).parent / "static"
if _static_dir.exists() and (_static_dir / "index.html").exists():
    _assets = _static_dir / "assets"
    if _assets.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")

    @app.get("/")
    async def _index():
        return FileResponse(str(_static_dir / "index.html"))

    @app.get("/{full_path:path}")
    async def _spa_fallback(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("ws/"):
            raise _HTTPException(status_code=404)
        f = _static_dir / full_path
        if f.is_file():
            return FileResponse(str(f))
        return FileResponse(str(_static_dir / "index.html"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(host: str = "127.0.0.1", port: int = 8000):
    logger.info(f"Starting neural-xray server on http://{host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
