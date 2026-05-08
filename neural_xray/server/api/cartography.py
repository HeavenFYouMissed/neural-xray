"""
Model Surgery MRI — /api/cartography endpoints.

LoRA-based concept mapping, layer ownership, alignment, and fast_map.
"""

import asyncio
import concurrent.futures
import logging
import sys
import traceback
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from neural_xray import LoRACartographer  # noqa: E402
from ..state import state  # noqa: E402

logger = logging.getLogger("model-surgery")
router = APIRouter()


class MapRequest(BaseModel):
    model_id: str
    concept: str
    sentences: List[str]
    rank: int = 4
    epochs: int = 3
    fast: bool = True  # Use fast_map by default


class BatchMapRequest(BaseModel):
    model_id: str
    concepts: Dict[str, List[str]]  # concept -> sentences
    rank: int = 4


def _get_cartographer(model_id: str, rank: int = 4) -> LoRACartographer:
    sess = state.get_session(model_id)
    if sess.cartographer is None or sess.cartographer.rank != rank:
        sess.cartographer = LoRACartographer(sess.loader, sess.arch_map, rank=rank)
    return sess.cartographer


@router.post("/map")
async def map_concept(req: MapRequest):
    """Map a concept through the model using LoRA cartography."""
    try:
        sess = state.get_session(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not loaded")

    try:
        carto = _get_cartographer(req.model_id, req.rank)

        # Run in thread pool with timeout to prevent GPU hangs
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            if req.fast:
                future = loop.run_in_executor(pool, carto.fast_map, req.concept, req.sentences)
            else:
                future = loop.run_in_executor(
                    pool, lambda: carto.train_concept(
                        req.concept, req.sentences, epochs=req.epochs, rank=req.rank
                    )
                )
            try:
                cmap = await asyncio.wait_for(future, timeout=120)
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"Cartography timed out after 120s for '{req.concept}'. "
                           f"Model may be too large for available VRAM."
                )

        with state.lock:
            sess.cartography_maps[req.concept] = cmap

        logger.info(f"Mapped '{req.concept}' in {req.model_id} "
                     f"— {len(cmap.layer_ownership)} layers")
        return cmap.to_dict()

    except Exception as e:
        logger.error(f"Cartography failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/map_batch")
async def map_batch(req: BatchMapRequest):
    """Map multiple concepts in batch using fast_map_batch."""
    try:
        sess = state.get_session(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not loaded")

    try:
        carto = _get_cartographer(req.model_id, req.rank)
        maps = carto.fast_map_batch(req.concepts)

        with state.lock:
            sess.cartography_maps.update(maps)

        logger.info(f"Batch mapped {len(maps)} concepts in {req.model_id}")
        return {name: m.to_dict() for name, m in maps.items()}

    except Exception as e:
        logger.error(f"Batch cartography failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/layers")
async def get_layer_ownership(
    model_id: str = Query(...),
    concept: str = Query(...),
):
    """Get layer ownership scores for a mapped concept."""
    try:
        sess = state.get_session(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")

    cmap = sess.cartography_maps.get(concept)
    if cmap is None:
        raise HTTPException(
            status_code=404,
            detail=f"Concept '{concept}' not mapped — call /api/cartography/map first",
        )

    return {
        "concept": concept,
        "model_id": model_id,
        "layer_ownership": [
            {"layer": name, "score": score}
            for name, score in cmap.layer_ownership
        ],
        "dominant_layers": [
            {"layer": name, "score": score}
            for name, score in cmap.dominant_layers(top_k=5)
        ],
    }


@router.get("/cached")
async def list_mapped_concepts(model_id: str = Query(...)):
    """List concepts that have been mapped for a model."""
    try:
        sess = state.get_session(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")

    return {
        "concepts": sorted(sess.cartography_maps.keys()),
        "count": len(sess.cartography_maps),
    }
