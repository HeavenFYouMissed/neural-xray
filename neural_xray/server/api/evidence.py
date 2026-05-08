"""
Model Surgery MRI — /api/evidence endpoints.

Capture and retrieve evidence packages (full behavioral snapshots).
"""

import logging
import sys
import traceback

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from neural_xray import EvidenceCollector  # noqa: E402
from ..state import state  # noqa: E402

logger = logging.getLogger("model-surgery")
router = APIRouter()

# In-memory evidence store (per session)
_evidence_store: dict = {}  # package_id → EvidencePackage.to_dict()


class CaptureRequest(BaseModel):
    model_id: str
    sentence: str


@router.post("/capture")
async def capture_evidence(req: CaptureRequest):
    """Capture a full evidence package: logit lens + trace graph + diagnostics."""
    try:
        sess = state.get_session(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not loaded")

    try:
        collector = EvidenceCollector(
            loader=sess.loader,
            arch_map=sess.arch_map,
            concept_vectors=sess.concept_vectors or {},
        )
        package = collector.capture(req.sentence, model_id=req.model_id)
        result = package.to_dict()

        # Store it
        _evidence_store[package.id] = result
        logger.info(
            f"Evidence captured: {package.id[:8]}... for '{req.model_id}' "
            f"(lens={'✓' if package.logit_lens else '✗'}, "
            f"graph={'✓' if package.trace_graph else '✗'}, "
            f"diag={'✓' if package.diagnostics else '✗'})"
        )
        return result
    except Exception as e:
        logger.error(f"Evidence capture failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_evidence(model_id: str = Query(...)):
    """List all captured evidence packages for a model."""
    packages = [
        p for p in _evidence_store.values()
        if p["model_id"] == model_id
    ]
    return {"packages": packages}


@router.get("/{package_id}")
async def get_evidence(package_id: str):
    """Retrieve a specific evidence package by ID."""
    pkg = _evidence_store.get(package_id)
    if not pkg:
        raise HTTPException(status_code=404, detail=f"Evidence package '{package_id}' not found")
    return pkg
