"""
Model Surgery MRI — /api/models endpoints.

Load, unload, and inspect HuggingFace models.
"""

import logging
import sys
import traceback
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


from neural_xray import ModelLoader, ArchitectureMapper  # noqa: E402
from ..licensing import check_model_size  # noqa: E402
from ..state import state, ModelSession  # noqa: E402

logger = logging.getLogger("model-surgery")
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class LoadRequest(BaseModel):
    model_id: str
    quantization: Optional[str] = None  # "4bit"|"8bit"|"float16"|"float32"
    cache_dir: Optional[str] = None
    hf_token: Optional[str] = None  # For private HuggingFace repos
    max_params: Optional[int] = None  # User-specified param limit


class ModelInfo(BaseModel):
    model_id: str
    hidden_size: int
    num_layers: int
    vocab_size: int
    quantization: str
    num_params: int
    fact_layer_start: int
    fact_layer_end: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/load", response_model=ModelInfo)
async def load_model(req: LoadRequest):
    """Load a HuggingFace model into GPU memory."""
    if req.model_id in state.sessions:
        sess = state.sessions[req.model_id]
        return _session_to_info(sess)

    logger.info(f"Loading model: {req.model_id}")
    try:
        loader = ModelLoader(
            req.model_id,
            force_quantization=req.quantization,
            cache_dir=req.cache_dir,
            hf_token=req.hf_token,
        )
        loader.load()

        # Check license tier allows this model size
        num_params = sum(p.numel() for p in loader.model.parameters())
        if not check_model_size(num_params):
            loader.unload()
            raise HTTPException(
                status_code=403,
                detail=f"Model has {num_params:,} params — exceeds your license tier limit",
            )

        # Check user-specified param limit
        if req.max_params and num_params > req.max_params:
            loader.unload()
            raise HTTPException(
                status_code=400,
                detail=f"Model has {num_params:,} params — exceeds your limit of {req.max_params:,}",
            )

        # Map architecture
        mapper = ArchitectureMapper(loader)
        arch_map = mapper.map()

        sess = ModelSession(
            model_id=req.model_id,
            loader=loader,
            arch_map=arch_map,
        )
        with state.lock:
            state.sessions[req.model_id] = sess

        logger.info(f"✓ Loaded {req.model_id} — {loader.hidden_size}d, {loader.num_layers}L")
        return _session_to_info(sess)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load {req.model_id}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


class UnloadRequest(BaseModel):
    model_id: str


@router.post("/unload")
async def unload_model(req: UnloadRequest):
    """Unload a model and free GPU memory."""
    if req.model_id not in state.sessions:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not loaded")

    sess = state.sessions.pop(req.model_id)
    sess.loader.unload()
    logger.info(f"Unloaded {req.model_id}")
    return {"status": "unloaded", "model_id": req.model_id}


@router.get("/loaded")
async def list_loaded():
    """List all currently loaded models."""
    return [_session_to_info(s) for s in state.sessions.values()]


@router.get("/{model_id}/architecture")
async def get_architecture(model_id: str):
    """Get the full architecture map for a loaded model."""
    from urllib.parse import unquote
    model_id = unquote(model_id)  # path param arrives as openai-community%2Fgpt2
    try:
        sess = state.get_session(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")
    return sess.arch_map.to_dict()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _session_to_info(sess: ModelSession) -> ModelInfo:
    loader = sess.loader
    arch = sess.arch_map
    num_params = sum(p.numel() for p in loader.model.parameters())
    return ModelInfo(
        model_id=sess.model_id,
        hidden_size=loader.hidden_size,
        num_layers=loader.num_layers,
        vocab_size=arch.vocab_size,
        quantization=loader.quantization_used or "none",
        num_params=num_params,
        fact_layer_start=arch.fact_layer_start,
        fact_layer_end=arch.fact_layer_end,
    )
