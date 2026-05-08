"""
Model Surgery MRI — /api/diagnostics endpoints.

Run 7-layer health diagnostic checks on loaded models.
"""

import logging
import sys
import traceback

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from neural_xray import ModelDiagnostics  # noqa: E402
from ..state import state  # noqa: E402

logger = logging.getLogger("model-surgery")
router = APIRouter()


class DiagnosticsRequest(BaseModel):
    model_id: str
    sentence: str


@router.post("/run")
async def run_diagnostics(req: DiagnosticsRequest):
    """Run all 7 diagnostic checks on a sentence through the loaded model."""
    try:
        sess = state.get_session(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not loaded")

    try:
        diag = ModelDiagnostics(
            loader=sess.loader,
            arch_map=sess.arch_map,
        )
        report = diag.run_checks(req.sentence)
        logger.info(
            f"Diagnostics on '{req.model_id}': "
            f"{report.num_ok} ok, {report.num_warn} warn, {report.num_critical} critical "
            f"({len(report.checks)} checks)"
        )
        return report.to_dict()
    except Exception as e:
        logger.error(f"Diagnostics failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
